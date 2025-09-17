"""
Conversation management with LangChain and Gemma 3n
"""
import logging
from typing import Dict, List, Any, Optional
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 2048
        self.conversation_history: Dict[str, List[BaseMessage]] = {}
        
        logger.info(f"Loading LLM model: {model_name} on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the Gemma model and tokenizer"""
        try:
            logger.info(f"Loading LLM model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token for Gemma models
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=False
            )
            
            # Create pipeline
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Create LangChain wrapper
            self.llm = HuggingFacePipeline(pipeline=self.llm_pipeline)
            
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            # Fallback to a simpler model if Gemma fails
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if primary model fails"""
        try:
            logger.info("Loading fallback model...")
            # Try different fallback models in order of preference
            fallback_models = [
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-small", 
                "gpt2",
                "distilgpt2"
            ]
            
            for model_name in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {model_name}")
                    self.model_name = model_name
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    
                    # Set pad token if not available
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self.llm_pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_length=512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    self.llm = HuggingFacePipeline(pipeline=self.llm_pipeline)
                    logger.info(f"Fallback model {model_name} loaded successfully")
                    return
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            # If all fallbacks fail, create a dummy model
            self._create_dummy_model()
            
        except Exception as e:
            logger.error(f"Error loading fallback model: {e}")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model that returns simple responses"""
        logger.warning("Creating dummy LLM model - responses will be basic")
        self.model = None
        self.tokenizer = None
        self.llm_pipeline = None
        self.llm = None
    
    def get_conversation_context(self, session_id: str, max_messages: int = 10) -> List[BaseMessage]:
        """Get conversation context for a session"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        # Return last max_messages messages
        return self.conversation_history[session_id][-max_messages:]
    
    def add_message_to_context(self, session_id: str, role: str, content: str):
        """Add a message to conversation context"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        if role == "user":
            self.conversation_history[session_id].append(HumanMessage(content=content))
        elif role == "assistant":
            self.conversation_history[session_id].append(AIMessage(content=content))
    
    def generate_response(self, user_input: str, session_id: str, 
                         context_docs: List[Dict] = None, language: str = "en") -> str:
        """
        Generate AI response using LangChain and Gemma
        
        Args:
            user_input: User's input text
            session_id: Session identifier
            context_docs: Retrieved documents from RAG
            language: Language of the conversation
            
        Returns:
            Generated response text
        """
        try:
            # Add user message to context
            self.add_message_to_context(session_id, "user", user_input)
            
            # Check if we have a working LLM
            if self.llm is None:
                return self._generate_dummy_response(user_input, context_docs, language)
            
            # Build context from retrieved documents
            context_text = ""
            if context_docs:
                context_text = "\n\n".join([doc['text'] for doc in context_docs[:3]])
            
            # Create prompt template based on language
            if language in ["hi", "bn", "ta", "te", "gu", "kn", "ml", "mr", "pa", "or", "as"]:
                prompt_template = self._get_hindi_prompt_template()
            else:
                prompt_template = self._get_english_prompt_template()
            
            # Get conversation history
            conversation_history = self.get_conversation_context(session_id)
            
            # Format conversation history
            history_text = ""
            for msg in conversation_history[-6:]:  # Last 6 messages
                if isinstance(msg, HumanMessage):
                    history_text += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    history_text += f"Assistant: {msg.content}\n"
            
            # Create final prompt
            prompt = prompt_template.format(
                context=context_text,
                history=history_text,
                user_input=user_input
            )
            
            # Generate response
            response = self.llm(prompt)
            
            # Extract just the response text (remove prompt)
            response_text = response.strip()
            if "Assistant:" in response_text:
                response_text = response_text.split("Assistant:")[-1].strip()
            
            # Add response to context
            self.add_message_to_context(session_id, "assistant", response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_dummy_response(user_input, context_docs, language)
    
    def _generate_dummy_response(self, user_input: str, context_docs: List[Dict] = None, language: str = "en") -> str:
        """Generate a simple dummy response when LLM is not available"""
        # Add response to context
        self.add_message_to_context("dummy", "assistant", "I'm a basic agricultural assistant.")
        
        # Simple keyword-based responses
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["disease", "pest", "problem", "issue"]):
            return "I can help with agricultural problems. Please describe the symptoms you're seeing on your crops, and I'll try to provide guidance based on common agricultural knowledge."
        
        elif any(word in user_lower for word in ["fertilizer", "fertilization", "nutrient"]):
            return "For fertilization advice, consider soil testing first. Different crops have different nutrient requirements. Organic fertilizers like compost are generally good for soil health."
        
        elif any(word in user_lower for word in ["water", "irrigation", "watering"]):
            return "Proper irrigation is crucial for crop health. Water early morning or evening to reduce evaporation. Avoid overwatering as it can lead to root rot."
        
        elif any(word in user_lower for word in ["soil", "ground", "earth"]):
            return "Healthy soil is the foundation of good agriculture. Consider crop rotation, organic matter addition, and proper pH levels for optimal soil health."
        
        else:
            return "I'm here to help with agricultural questions. Please ask about crop diseases, pests, fertilization, irrigation, or any other farming topics."
    
    def _get_english_prompt_template(self) -> str:
        """Get English prompt template"""
        return """You are KisanAI, an agricultural assistant that helps farmers with crop management, disease identification, and farming advice.

Context from knowledge base:
{context}

Previous conversation:
{history}

User: {user_input}

Assistant:"""
    
    def _get_hindi_prompt_template(self) -> str:
        """Get Hindi prompt template"""
        return """आप KisanAI हैं, एक कृषि सहायक जो किसानों को फसल प्रबंधन, रोग पहचान और कृषि सलाह में मदद करता है।

ज्ञान आधार से संदर्भ:
{context}

पिछली बातचीत:
{history}

उपयोगकर्ता: {user_input}

सहायक:"""
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared conversation history for session: {session_id}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a conversation session"""
        if session_id not in self.conversation_history:
            return {"message_count": 0, "topics": []}
        
        messages = self.conversation_history[session_id]
        user_messages = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
        
        # Simple topic extraction (first few words of user messages)
        topics = []
        for msg in user_messages[:5]:  # First 5 user messages
            words = msg.split()[:3]  # First 3 words
            topics.append(" ".join(words))
        
        return {
            "message_count": len(messages),
            "topics": topics,
            "last_activity": messages[-1].content if messages else ""
        }
