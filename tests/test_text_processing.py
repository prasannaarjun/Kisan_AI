"""
Test script for agricultural text processing pipeline
"""
import os
import sys
import unittest
import tempfile
import json
from pathlib import Path

# Add backend to path
sys.path.append('backend')

from text_processor import AgriculturalTextProcessor, ChunkMetadata
from enhanced_rag_store import EnhancedRAGStore

class TestAgriculturalTextProcessor(unittest.TestCase):
    """Test cases for AgriculturalTextProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = AgriculturalTextProcessor()
        self.sample_text = """
        Strategy 1: Ensure seed sovereignty of the farmers and the State
        
        Action 1.1: Establish seed villages exclusively for organic farming.
        Begin programmes for the production of seeds, seedlings, planting materials and,
        traditional animal breeds at the Panchayat level, so as to become self-sufficient in
        the availability of good quality local seeds, both indigenous and breeder seeds
        developed by the KAU and other institutions of agricultural research.
        
        Organic farming is a system with the broad principle of 'live and let live', 
        which was recognized nationally and internationally. It includes crop production,
        animal husbandry, dairy, fisheries, poultry, piggery, forestry, bee keeping, 
        and also uncultivated biodiversity around.
        """
    
    def test_document_type_detection(self):
        """Test document type detection"""
        # Test policy document
        doc_type = self.processor.detect_document_type("organic_farming_policy.txt", self.sample_text)
        self.assertEqual(doc_type, "policy")
        
        # Test research document
        research_text = "Abstract: This paper presents a study on organic farming practices..."
        doc_type = self.processor.detect_document_type("research_paper.txt", research_text)
        self.assertEqual(doc_type, "research")
    
    def test_agricultural_terms_extraction(self):
        """Test extraction of agricultural terms"""
        terms = self.processor.extract_agricultural_terms(self.sample_text)
        
        # Should find crop-related terms
        self.assertIn('crops', terms)
        self.assertIn('organic farming', terms['practices'])
        
        # Should find practice-related terms
        self.assertIn('practices', terms)
    
    def test_semantic_chunking(self):
        """Test semantic chunking functionality"""
        chunks = self.processor.create_semantic_chunks(self.sample_text, max_tokens=100)
        
        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should be reasonable length
        for chunk in chunks:
            self.assertGreater(len(chunk), 10)
            self.assertLess(len(chunk), 200)  # Should be less than 2x max_tokens
    
    def test_section_extraction(self):
        """Test section extraction for policy documents"""
        sections = self.processor.extract_sections(self.sample_text, "policy")
        
        # Should extract strategy sections
        self.assertGreater(len(sections), 0)
        
        # Should have strategy content
        strategy_found = any("strategy" in section[0].lower() for section in sections)
        self.assertTrue(strategy_found)

class TestEnhancedRAGStore(unittest.TestCase):
    """Test cases for EnhancedRAGStore"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rag_store = EnhancedRAGStore()
        
        # Create sample chunks
        self.sample_chunks = [
            {
                'text': 'Organic farming practices include crop rotation and composting.',
                'metadata': {
                    'document_type': 'technical',
                    'crop_type': 'general',
                    'practice_type': 'organic farming',
                    'source_file': 'test.txt'
                },
                'embedding': [0.1] * 768  # Dummy embedding
            },
            {
                'text': 'Rice cultivation requires proper water management and soil preparation.',
                'metadata': {
                    'document_type': 'technical',
                    'crop_type': 'rice',
                    'practice_type': 'cultivation',
                    'source_file': 'test.txt'
                },
                'embedding': [0.2] * 768  # Dummy embedding
            }
        ]
    
    def test_add_processed_chunks(self):
        """Test adding processed chunks to RAG store"""
        self.rag_store.add_processed_chunks(self.sample_chunks)
        
        # Should have added chunks
        self.assertEqual(len(self.rag_store.documents), 2)
        self.assertEqual(len(self.rag_store.metadata), 2)
        self.assertTrue(self.rag_store.is_initialized)
    
    def test_search_functionality(self):
        """Test search functionality"""
        self.rag_store.add_processed_chunks(self.sample_chunks)
        
        # Test basic search
        results = self.rag_store.search("organic farming", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("organic farming", results[0]['text'].lower())
    
    def test_filtered_search(self):
        """Test filtered search by crop type"""
        self.rag_store.add_processed_chunks(self.sample_chunks)
        
        # Search for rice-specific content
        results = self.rag_store.search_by_crop("cultivation", "rice", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("rice", results[0]['text'].lower())
    
    def test_statistics(self):
        """Test statistics generation"""
        self.rag_store.add_processed_chunks(self.sample_chunks)
        
        stats = self.rag_store.get_statistics()
        
        # Should have correct counts
        self.assertEqual(stats['total_documents'], 2)
        self.assertEqual(stats['document_types']['technical'], 2)
        self.assertIn('rice', stats['crop_types'])
        self.assertIn('organic farming', stats['practice_types'])

def create_test_data():
    """Create test data files for testing"""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample policy document
    policy_content = """
    Strategy 1: Ensure seed sovereignty of the farmers and the State
    
    Action 1.1: Establish seed villages exclusively for organic farming.
    Begin programmes for the production of seeds, seedlings, planting materials and,
    traditional animal breeds at the Panchayat level, so as to become self-sufficient in
    the availability of good quality local seeds, both indigenous and breeder seeds
    developed by the KAU and other institutions of agricultural research.
    
    Strategy 2: Implementation of organic farming policy in a phased manner
    
    Action 2.1: Conduct an initial assessment of the status of organic farming and farmers in the State
    including cultivated, certified and non-cultivated wild organic areas in the State.
    """
    
    with open(test_dir / "organic_policy.txt", "w", encoding="utf-8") as f:
        f.write(policy_content)
    
    # Create sample research document
    research_content = """
    Abstract
    
    This paper presents a comprehensive study on organic farming practices in Kerala.
    The study examines the current status, challenges, and opportunities for organic
    farming adoption among small and marginal farmers.
    
    Introduction
    
    Organic farming has gained significant attention in recent years due to growing
    concerns about environmental sustainability and food safety. In Kerala, the
    government has implemented various policies to promote organic farming practices.
    
    Methodology
    
    The study employed a mixed-methods approach, combining quantitative surveys
    with qualitative interviews. Data was collected from 200 farmers across 10 districts
    of Kerala.
    
    Results
    
    The findings indicate that 65% of farmers are interested in adopting organic
    farming practices, but face challenges related to certification costs and market access.
    
    Conclusion
    
    The study concludes that while there is significant potential for organic farming
    in Kerala, targeted interventions are needed to address the identified challenges.
    """
    
    with open(test_dir / "research_paper.txt", "w", encoding="utf-8") as f:
        f.write(research_content)
    
    return str(test_dir)

def test_full_pipeline():
    """Test the complete processing pipeline"""
    print("Testing complete agricultural text processing pipeline...")
    
    # Create test data
    test_dir = create_test_data()
    
    try:
        # Initialize processor
        processor = AgriculturalTextProcessor()
        
        # Process test directory
        print(f"Processing test directory: {test_dir}")
        chunks = processor.process_directory(test_dir)
        
        if not chunks:
            print("ERROR: No chunks were processed")
            return False
        
        print(f"Processed {len(chunks)} chunks")
        
        # Create embeddings
        print("Creating embeddings...")
        chunks_with_embeddings = processor.create_embeddings(chunks)
        
        # Initialize RAG store
        print("Initializing RAG store...")
        rag_store = EnhancedRAGStore()
        rag_store.add_processed_chunks(chunks_with_embeddings)
        
        # Test search
        print("Testing search functionality...")
        test_queries = [
            "organic farming practices",
            "seed sovereignty",
            "research methodology",
            "farmer challenges"
        ]
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            results = rag_store.search(query, top_k=2)
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     Text: {result['text'][:100]}...")
                print(f"     Type: {result['metadata'].get('document_type', 'unknown')}")
        
        # Print statistics
        stats = rag_store.get_statistics()
        print(f"\nStatistics:")
        print(f"Total documents: {stats.get('total_documents', 0)}")
        print(f"Document types: {stats.get('document_types', {})}")
        print(f"Crop types: {stats.get('crop_types', {})}")
        print(f"Practice types: {stats.get('practice_types', {})}")
        
        print("\nPipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Pipeline test failed: {e}")
        return False
    
    finally:
        # Clean up test data
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run full pipeline test
    print("\n" + "="*50)
    test_full_pipeline()
