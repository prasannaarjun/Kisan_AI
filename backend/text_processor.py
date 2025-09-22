"""
Advanced text processing pipeline for agricultural documents
"""
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import hashlib

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for text chunks"""
    source_file: str
    chunk_id: str
    chunk_index: int
    document_type: str
    section: Optional[str] = None
    topic: Optional[str] = None
    crop_type: Optional[str] = None
    practice_type: Optional[str] = None
    language: str = "en"
    word_count: int = 0
    token_count: int = 0

class AgriculturalTextProcessor:
    """Advanced text processor for agricultural documents"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
        # Agricultural domain keywords for better chunking
        self.agricultural_keywords = {
            'crops': ['rice', 'wheat', 'maize', 'coconut', 'rubber', 'pepper', 'cardamom', 'tea', 'coffee', 'cashew', 'arecanut'],
            'practices': ['organic farming', 'crop rotation', 'irrigation', 'fertilization', 'pest control', 'soil management'],
            'diseases': ['rust', 'blight', 'wilt', 'mosaic', 'rot', 'spot'],
            'pests': ['aphid', 'mite', 'beetle', 'caterpillar', 'thrips', 'whitefly'],
            'techniques': ['composting', 'mulching', 'intercropping', 'agroforestry', 'hydroponics']
        }
        
        # Document type patterns
        self.document_patterns = {
            'policy': r'(policy|strategy|action plan|guidelines)',
            'research': r'(abstract|introduction|methodology|results|conclusion)',
            'extension': r'(extension|advisory|training|demonstration)',
            'technical': r'(procedure|technique|method|practice|guideline)',
            'report': r'(report|analysis|assessment|evaluation)'
        }
    
    def detect_document_type(self, filename: str, content: str) -> str:
        """Detect document type based on filename and content"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Check filename patterns
        if 'policy' in filename_lower or 'strategy' in filename_lower:
            return 'policy'
        elif 'research' in filename_lower or 'ijcrt' in filename_lower:
            return 'research'
        elif 'extension' in filename_lower or 'advisory' in filename_lower:
            return 'extension'
        elif 'gap' in filename_lower or 'technical' in filename_lower:
            return 'technical'
        elif 'report' in filename_lower or 'analysis' in filename_lower:
            return 'report'
        
        # Check content patterns
        for doc_type, pattern in self.document_patterns.items():
            if re.search(pattern, content_lower):
                return doc_type
        
        return 'general'
    
    def extract_sections(self, content: str, doc_type: str) -> List[Tuple[str, str]]:
        """Extract sections based on document type"""
        sections = []
        
        if doc_type == 'policy':
            # Extract strategy sections
            strategy_pattern = r'(Strategy \d+[^\n]*\n(?:[^S]|S(?!trategy))*?)(?=Strategy \d+|$)'
            try:
                matches = re.finditer(strategy_pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    section_text = match.group(1).strip()
                    if len(section_text) > 100:  # Only include substantial sections
                        sections.append(('strategy', section_text))
            except re.error as e:
                logger.warning(f"Regex error in policy section extraction: {e}")
                # Fallback to paragraph splitting
                paragraphs = content.split('\n\n')
                for i, para in enumerate(paragraphs):
                    para = para.strip()
                    if len(para) > 100:
                        sections.append((f'section_{i+1}', para))
        
        elif doc_type == 'research':
            # Extract standard research sections
            section_patterns = {
                'abstract': r'(abstract[^\n]*\n.*?)(?=\n\s*(?:introduction|1\.|introduction|background))',
                'introduction': r'(introduction[^\n]*\n.*?)(?=\n\s*(?:methodology|2\.|method|materials))',
                'methodology': r'(methodology|method[^\n]*\n.*?)(?=\n\s*(?:results|3\.|findings))',
                'results': r'(results|findings[^\n]*\n.*?)(?=\n\s*(?:conclusion|4\.|discussion))',
                'conclusion': r'(conclusion[^\n]*\n.*?)(?=\n\s*(?:references|bibliography|$)'
            }
            
            for section_name, pattern in section_patterns.items():
                try:
                    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                    if match:
                        section_text = match.group(1).strip()
                        if len(section_text) > 100:
                            sections.append((section_name, section_text))
                except re.error as e:
                    logger.warning(f"Regex error in research section extraction for {section_name}: {e}")
                    continue
        
        else:
            # For other document types, split by paragraphs
            paragraphs = content.split('\n\n')
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) > 100:  # Only include substantial paragraphs
                    sections.append((f'section_{i+1}', para))
        
        return sections
    
    def extract_agricultural_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract agricultural terms and concepts from text"""
        text_lower = text.lower()
        extracted_terms = {}
        
        for category, terms in self.agricultural_keywords.items():
            found_terms = [term for term in terms if term in text_lower]
            if found_terms:
                extracted_terms[category] = found_terms
        
        return extracted_terms
    
    def create_semantic_chunks(self, text: str, max_tokens: int = 256, overlap_tokens: int = 25) -> List[str]:
        """Create semantic chunks with overlap - smaller chunks for better RAG"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(word_tokenize(sentence))
            
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_token_count = 0
                for sent in reversed(current_chunk):
                    sent_tokens = len(word_tokenize(sent))
                    if overlap_token_count + sent_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, sent)
                        overlap_token_count += sent_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_token_count + sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def create_sliding_window_chunks(self, text: str, window_size: int = 200, step_size: int = 100) -> List[str]:
        """Create overlapping sliding window chunks for comprehensive coverage"""
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), step_size):
            window_sentences = sentences[i:i + window_size]
            if window_sentences:
                chunk_text = ' '.join(window_sentences)
                chunks.append(chunk_text)
        
        return chunks
    
    def create_topic_based_chunks(self, text: str, max_tokens: int = 300) -> List[str]:
        """Create chunks based on topic boundaries and agricultural concepts"""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        
        for para in paragraphs:
            if len(word_tokenize(para)) <= max_tokens:
                chunks.append(para)
            else:
                # Further split long paragraphs by sentences
                sentences = sent_tokenize(para)
                current_chunk = []
                current_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens = len(word_tokenize(sentence))
                    
                    if current_tokens + sentence_tokens > max_tokens and current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sentence_tokens
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_question_answer_chunks(self, text: str, max_tokens: int = 200) -> List[Dict[str, str]]:
        """Create Q&A style chunks by identifying question-answer patterns"""
        chunks = []
        
        # Look for question patterns
        question_patterns = [
            r'What\s+is\s+([^?]+)\?',
            r'How\s+to\s+([^?]+)\?',
            r'Why\s+([^?]+)\?',
            r'When\s+([^?]+)\?',
            r'Where\s+([^?]+)\?',
            r'Which\s+([^?]+)\?'
        ]
        
        sentences = sent_tokenize(text)
        current_qa = {'question': '', 'answer': ''}
        
        for sentence in sentences:
            sentence_tokens = len(word_tokenize(sentence))
            
            # Check if sentence contains a question
            is_question = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in question_patterns)
            
            if is_question:
                # Save previous Q&A if exists
                if current_qa['question'] and current_qa['answer']:
                    chunks.append(current_qa.copy())
                
                # Start new Q&A
                current_qa = {'question': sentence, 'answer': ''}
            else:
                # Add to answer
                if current_qa['question']:
                    if current_qa['answer']:
                        current_qa['answer'] += ' ' + sentence
                    else:
                        current_qa['answer'] = sentence
                else:
                    # No question yet, treat as standalone content
                    if len(word_tokenize(sentence)) <= max_tokens:
                        chunks.append({'question': '', 'answer': sentence})
        
        # Add final Q&A
        if current_qa['question'] and current_qa['answer']:
            chunks.append(current_qa)
        
        return chunks
    
    def deduplicate_chunks(self, chunks: List[Dict[str, Any]], similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Remove duplicate or highly similar chunks"""
        if not chunks:
            return chunks
        
        # Create embeddings for all chunks
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Find similar chunks
        unique_chunks = []
        used_indices = set()
        
        for i, chunk in enumerate(chunks):
            if i in used_indices:
                continue
            
            unique_chunks.append(chunk)
            used_indices.add(i)
            
            # Find similar chunks to this one
            for j, other_chunk in enumerate(chunks[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                if similarity > similarity_threshold:
                    used_indices.add(j)
                    logger.debug(f"Removed duplicate chunk: similarity {similarity:.3f}")
        
        logger.info(f"Deduplication: {len(chunks)} -> {len(unique_chunks)} chunks")
        return unique_chunks
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    
    def process_document(self, filepath: str) -> List[Dict[str, Any]]:
        """Process a single document and return chunks with metadata using multiple chunking strategies"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            filename = Path(filepath).name
            doc_type = self.detect_document_type(filename, content)
            
            # Extract sections
            sections = self.extract_sections(content, doc_type)
            
            chunks = []
            chunk_index = 0
            
            for section_name, section_text in sections:
                # Use multiple chunking strategies for better coverage
                chunking_strategies = [
                    ('semantic', self.create_semantic_chunks(section_text, max_tokens=256)),
                    ('sliding_window', self.create_sliding_window_chunks(section_text, window_size=200, step_size=100)),
                    ('topic_based', self.create_topic_based_chunks(section_text, max_tokens=300))
                ]
                
                # Add Q&A chunks if document contains questions
                qa_chunks = self.create_question_answer_chunks(section_text, max_tokens=200)
                if qa_chunks:
                    chunking_strategies.append(('qa', qa_chunks))
                
                for strategy_name, strategy_chunks in chunking_strategies:
                    for chunk_data in strategy_chunks:
                        # Handle different chunk formats
                        if isinstance(chunk_data, dict):
                            # Q&A format
                            if chunk_data.get('question') and chunk_data.get('answer'):
                                chunk_text = f"Q: {chunk_data['question']} A: {chunk_data['answer']}"
                            else:
                                chunk_text = chunk_data.get('answer', '')
                        else:
                            # Regular text chunk
                            chunk_text = chunk_data
                        
                        # Skip empty or very short chunks
                        if len(chunk_text.strip()) < 50:
                            continue
                        
                        # Extract agricultural terms
                        agricultural_terms = self.extract_agricultural_terms(chunk_text)
                        
                        # Create metadata
                        chunk_id = hashlib.md5(f"{filepath}_{chunk_index}_{strategy_name}".encode()).hexdigest()[:12]
                        
                        metadata = ChunkMetadata(
                            source_file=filename,
                            chunk_id=chunk_id,
                            chunk_index=chunk_index,
                            document_type=doc_type,
                            section=section_name,
                            language="en",  # Assume English for now
                            word_count=len(chunk_text.split()),
                            token_count=len(word_tokenize(chunk_text))
                        )
                        
                        # Add agricultural terms to metadata
                        if agricultural_terms.get('crops'):
                            metadata.crop_type = ', '.join(agricultural_terms['crops'])
                        if agricultural_terms.get('practices'):
                            metadata.practice_type = ', '.join(agricultural_terms['practices'])
                        
                        chunks.append({
                            'text': chunk_text,
                            'metadata': {
                                'source_file': metadata.source_file,
                                'chunk_id': metadata.chunk_id,
                                'chunk_index': metadata.chunk_index,
                                'document_type': metadata.document_type,
                                'section': metadata.section,
                                'chunking_strategy': strategy_name,
                                'topic': metadata.topic,
                                'crop_type': metadata.crop_type,
                                'practice_type': metadata.practice_type,
                                'language': metadata.language,
                                'word_count': metadata.word_count,
                                'token_count': metadata.token_count,
                                'agricultural_terms': agricultural_terms
                            }
                        })
                        
                        chunk_index += 1
            
            logger.info(f"Processed {filename}: {len(chunks)} chunks created using multiple strategies")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            return []
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all text files in a directory with improved chunking"""
        all_chunks = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory {directory_path} does not exist")
            return all_chunks
        
        text_files = list(directory.glob("*.txt"))
        logger.info(f"Found {len(text_files)} text files in {directory_path}")
        
        for file_path in text_files:
            chunks = self.process_document(str(file_path))
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created before deduplication: {len(all_chunks)}")
        
        # Deduplicate chunks to remove similar content
        unique_chunks = self.deduplicate_chunks(all_chunks)
        
        logger.info(f"Total unique chunks after deduplication: {len(unique_chunks)}")
        return unique_chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create embeddings for chunks"""
        if not chunks:
            return []
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return chunks

def main():
    """Main function to process agricultural documents"""
    processor = AgriculturalTextProcessor()
    
    # Process the extracted text directory
    data_dir = "data/extracted_text/content/extracted_text"
    chunks = processor.process_directory(data_dir)
    
    if chunks:
        # Create embeddings
        chunks_with_embeddings = processor.create_embeddings(chunks)
        
        # Save processed chunks
        output_file = "processed_agricultural_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(chunks_with_embeddings)} processed chunks to {output_file}")
        
        # Print statistics
        doc_types = {}
        total_chunks = len(chunks_with_embeddings)
        
        for chunk in chunks_with_embeddings:
            doc_type = chunk['metadata']['document_type']
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print(f"\nProcessing Statistics:")
        print(f"Total chunks: {total_chunks}")
        print(f"Document types:")
        for doc_type, count in doc_types.items():
            print(f"  {doc_type}: {count} chunks")
    
    else:
        logger.warning("No chunks were processed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
