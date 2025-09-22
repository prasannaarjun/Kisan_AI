"""
Enhanced RAG store with advanced text processing and agricultural domain optimization
"""
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from sentence_transformers import SentenceTransformer
from text_processor import AgriculturalTextProcessor
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedRAGStore:
    """Enhanced RAG store optimized for agricultural content"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        self.index = None
        self.documents = []
        self.metadata = []
        self.is_initialized = False
        self.text_processor = AgriculturalTextProcessor(embedding_model)
        
    def initialize_index(self):
        """Initialize FAISS index"""
        if not self.is_initialized:
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.is_initialized = True
            logger.info("FAISS index initialized")
    
    def load_processed_chunks(self, chunks_file: str) -> bool:
        """Load pre-processed chunks from file"""
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            if not chunks:
                logger.warning("No chunks found in file")
                return False
            
            # Extract data for FAISS
            texts = []
            embeddings = []
            metadata_list = []
            
            for chunk in chunks:
                texts.append(chunk['text'])
                embeddings.append(chunk['embedding'])
                metadata_list.append(chunk['metadata'])
            
            # Convert to numpy arrays
            embeddings_array = np.array(embeddings, dtype='float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Initialize and populate index
            self.initialize_index()
            self.index.add(embeddings_array)
            
            # Store documents and metadata
            self.documents = texts
            self.metadata = metadata_list
            
            logger.info(f"Loaded {len(chunks)} processed chunks from {chunks_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading processed chunks: {e}")
            return False
    
    def process_and_add_documents(self, documents: List[Dict[str, Any]]):
        """Process documents using the text processor and add to store"""
        if not documents:
            return
        
        all_chunks = []
        
        for doc in documents:
            if 'filepath' in doc:
                # Process file
                chunks = self.text_processor.process_document(doc['filepath'])
                all_chunks.extend(chunks)
            elif 'text' in doc:
                # Process text directly
                # This would need to be implemented in text_processor
                pass
        
        if all_chunks:
            # Create embeddings
            chunks_with_embeddings = self.text_processor.create_embeddings(all_chunks)
            self.add_processed_chunks(chunks_with_embeddings)
    
    def add_processed_chunks(self, chunks: List[Dict[str, Any]]):
        """Add pre-processed chunks to the store"""
        if not chunks:
            return
        
        self.initialize_index()
        
        # Extract data
        texts = [chunk['text'] for chunk in chunks]
        embeddings = [chunk['embedding'] for chunk in chunks]
        metadata_list = [chunk['metadata'] for chunk in chunks]
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype='float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend(metadata_list)
        
        logger.info(f"Added {len(chunks)} processed chunks to RAG store. Total: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents with optional filtering
        
        Args:
            query: Search query
            top_k: Number of top results to return
            filters: Optional filters for document type, crop type, etc.
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.is_initialized or len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search with higher k to allow for filtering
        search_k = top_k * 3 if filters else top_k
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = {
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score)
                }
                
                # Apply filters if provided
                if filters and not self._matches_filters(result, filters):
                    continue
                
                results.append(result)
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def _matches_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if result matches the provided filters"""
        metadata = result['metadata']
        
        for key, value in filters.items():
            if key in metadata:
                if isinstance(value, list):
                    if not any(v in str(metadata[key]).lower() for v in value):
                        return False
                else:
                    if value.lower() not in str(metadata[key]).lower():
                        return False
        
        return True
    
    def search_by_crop(self, query: str, crop_type: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for documents related to a specific crop"""
        filters = {'crop_type': crop_type}
        return self.search(query, top_k, filters)
    
    def search_by_practice(self, query: str, practice_type: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for documents related to a specific agricultural practice"""
        filters = {'practice_type': practice_type}
        return self.search(query, top_k, filters)
    
    def search_by_document_type(self, query: str, doc_type: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search within a specific document type"""
        filters = {'document_type': doc_type}
        return self.search(query, top_k, filters)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the stored documents"""
        if not self.documents:
            return {}
        
        stats = {
            'total_documents': len(self.documents),
            'document_types': {},
            'crop_types': {},
            'practice_types': {},
            'source_files': set()
        }
        
        for metadata in self.metadata:
            # Document types
            doc_type = metadata.get('document_type', 'unknown')
            stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
            
            # Crop types
            crop_type = metadata.get('crop_type')
            if crop_type:
                for crop in crop_type.split(', '):
                    stats['crop_types'][crop] = stats['crop_types'].get(crop, 0) + 1
            
            # Practice types
            practice_type = metadata.get('practice_type')
            if practice_type:
                for practice in practice_type.split(', '):
                    stats['practice_types'][practice] = stats['practice_types'].get(practice, 0) + 1
            
            # Source files
            stats['source_files'].add(metadata.get('source_file', 'unknown'))
        
        stats['source_files'] = list(stats['source_files'])
        return stats
    
    def save_index(self, filepath: str):
        """Save FAISS index and metadata to disk"""
        if not self.is_initialized:
            return
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save documents and metadata
        data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'embedding_model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim
        }
        
        with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved enhanced RAG store to {filepath}")
    
    def load_index(self, filepath: str):
        """Load FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.index")
            
            # Load documents and metadata
            with open(f"{filepath}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.is_initialized = True
            
            logger.info(f"Loaded enhanced RAG store from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading enhanced RAG store: {e}")
            return False

def main():
    """Main function to demonstrate enhanced RAG store"""
    # Initialize enhanced RAG store
    rag_store = EnhancedRAGStore()
    
    # Try to load processed chunks
    chunks_file = "processed_agricultural_chunks.json"
    if os.path.exists(chunks_file):
        success = rag_store.load_processed_chunks(chunks_file)
        if success:
            # Get statistics
            stats = rag_store.get_statistics()
            print(f"RAG Store Statistics:")
            print(f"Total documents: {stats.get('total_documents', 0)}")
            print(f"Document types: {stats.get('document_types', {})}")
            print(f"Crop types: {stats.get('crop_types', {})}")
            print(f"Practice types: {stats.get('practice_types', {})}")
            
            # Test search
            query = "organic farming practices"
            results = rag_store.search(query, top_k=3)
            print(f"\nSearch results for '{query}':")
            for i, result in enumerate(results, 1):
                print(f"{i}. Score: {result['score']:.3f}")
                print(f"   Text: {result['text'][:100]}...")
                print(f"   Metadata: {result['metadata']}")
                print()
        else:
            print("Failed to load processed chunks")
    else:
        print(f"Processed chunks file {chunks_file} not found. Run text_processor.py first.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
