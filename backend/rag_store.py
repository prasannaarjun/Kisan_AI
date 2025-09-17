"""
RAG (Retrieval-Augmented Generation) store using FAISS
"""
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
import json
import logging
from sentence_transformers import SentenceTransformer
import os

logger = logging.getLogger(__name__)

class RAGStore:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        self.index = None
        self.documents = []
        self.metadata = []
        self.is_initialized = False
        
    def initialize_index(self):
        """Initialize FAISS index"""
        if not self.is_initialized:
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.is_initialized = True
            logger.info("FAISS index initialized")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the RAG store
        
        Args:
            documents: List of dicts with 'text', 'metadata' keys
        """
        if not documents:
            return
            
        self.initialize_index()
        
        texts = [doc['text'] for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend([doc.get('metadata', {}) for doc in documents])
        
        logger.info(f"Added {len(documents)} documents to RAG store. Total: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.is_initialized or len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score)
                })
        
        return results
    
    def load_sample_knowledge_base(self):
        """Load sample agricultural knowledge base"""
        sample_docs = [
            {
                'text': 'Wheat rust disease is a fungal infection that causes yellow-orange pustules on leaves. It spreads through wind and can be controlled with fungicides like propiconazole.',
                'metadata': {'topic': 'wheat_diseases', 'crop': 'wheat', 'disease': 'rust'}
            },
            {
                'text': 'Rice blast is a serious fungal disease that causes lesions on leaves, stems, and panicles. Use resistant varieties and proper water management to prevent it.',
                'metadata': {'topic': 'rice_diseases', 'crop': 'rice', 'disease': 'blast'}
            },
            {
                'text': 'Tomato blight affects leaves and fruits, causing dark spots. Improve air circulation and avoid overhead watering to prevent this disease.',
                'metadata': {'topic': 'tomato_diseases', 'crop': 'tomato', 'disease': 'blight'}
            },
            {
                'text': 'Proper irrigation timing is crucial for crop health. Water early morning or evening to reduce evaporation and fungal growth.',
                'metadata': {'topic': 'irrigation', 'crop': 'general', 'practice': 'watering'}
            },
            {
                'text': 'Crop rotation helps break pest and disease cycles. Rotate between different plant families every season.',
                'metadata': {'topic': 'crop_management', 'crop': 'general', 'practice': 'rotation'}
            },
            {
                'text': 'Soil testing should be done before planting to determine pH, nutrients, and organic matter content.',
                'metadata': {'topic': 'soil_management', 'crop': 'general', 'practice': 'testing'}
            },
            {
                'text': 'Organic fertilizers like compost and manure improve soil structure and provide slow-release nutrients.',
                'metadata': {'topic': 'fertilization', 'crop': 'general', 'fertilizer': 'organic'}
            },
            {
                'text': 'Integrated Pest Management (IPM) combines biological, cultural, and chemical methods to control pests sustainably.',
                'metadata': {'topic': 'pest_management', 'crop': 'general', 'method': 'ipm'}
            }
        ]
        
        self.add_documents(sample_docs)
        logger.info("Loaded sample agricultural knowledge base")
    
    def save_index(self, filepath: str):
        """Save FAISS index and metadata to disk"""
        if not self.is_initialized:
            return
            
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save documents and metadata
        data = {
            'documents': self.documents,
            'metadata': self.metadata
        }
        
        with open(f"{filepath}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved RAG store to {filepath}")
    
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
            
            logger.info(f"Loaded RAG store from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading RAG store: {e}")
            return False
