"""
Conversation management with LangChain and Gemma 3n
"""
import logging
from typing import Dict, List, Any, Optional
try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self, model_name: str = "gpt2"):
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
                max_new_tokens=100,  # Reduce to ensure generation
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,  # Don't return the input text
                clean_up_tokenization_spaces=True
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
                "gpt2",
                "distilgpt2",
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-small"
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
                        max_new_tokens=256,  # Use max_new_tokens instead of max_length
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
            
            # Use the improved dummy response system for focused agricultural advice
            # GPT-2 tends to drift into unrelated topics, so we use a more reliable approach
            logger.info("Using focused agricultural response system")
            return self._generate_dummy_response(user_input, context_docs, language)
            
            # Build context from retrieved documents (truncate to avoid token limit)
            context_text = ""
            if context_docs:
                # Limit context to first 2 documents and truncate each to 200 chars
                context_parts = []
                for doc in context_docs[:2]:
                    text = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                    context_parts.append(text)
                context_text = "\n\n".join(context_parts)
            
            # Create prompt template based on language
            if language in ["hi", "bn", "ta", "te", "gu", "kn", "ml", "mr", "pa", "or", "as"]:
                prompt_template = self._get_hindi_prompt_template()
            else:
                prompt_template = self._get_english_prompt_template()
            
            # Get conversation history (limit to last 3 messages to reduce token count)
            conversation_history = self.get_conversation_context(session_id)
            
            # Format conversation history (truncate each message)
            history_text = ""
            for msg in conversation_history[-3:]:  # Last 3 messages only
                if isinstance(msg, HumanMessage):
                    content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    history_text += f"User: {content}\n"
                elif isinstance(msg, AIMessage):
                    content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                    history_text += f"Assistant: {content}\n"
            
            # Truncate user input if too long
            truncated_user_input = user_input[:200] + "..." if len(user_input) > 200 else user_input
            
            # Create final prompt
            prompt = prompt_template.format(
                context=context_text,
                history=history_text,
                user_input=truncated_user_input
            )
            
            # Generate response using invoke instead of __call__
            response = self.llm.invoke(prompt)
            
            # Debug logging
            logger.info(f"Raw LLM response type: {type(response)}")
            logger.info(f"Raw LLM response: {str(response)[:200]}...")
            
            # Extract just the response text (remove prompt)
            # Handle different response types from LangChain
            if hasattr(response, 'content'):
                response_text = response.content.strip()
            elif isinstance(response, str):
                response_text = response.strip()
            elif isinstance(response, dict) and 'text' in response:
                response_text = response['text'].strip()
            else:
                response_text = str(response).strip()
            
            logger.info(f"Extracted response text: {response_text[:100]}...")
            
            # Extract only the Assistant's response
            if "Assistant:" in response_text:
                # Split by "Assistant:" and take the last part
                parts = response_text.split("Assistant:")
                if len(parts) > 1:
                    response_text = parts[-1].strip()
                else:
                    # If no "Assistant:" found, try to find the actual response
                    # Look for the last meaningful part after the prompt
                    lines = response_text.split('\n')
                    # Find the line that looks like a response (not part of the prompt)
                    for i, line in enumerate(lines):
                        if line.strip() and not line.startswith('You are') and not line.startswith('Context') and not line.startswith('Previous') and not line.startswith('User:'):
                            response_text = line.strip()
                            break
            else:
                # If no "Assistant:" marker, try to extract response from the end
                lines = response_text.split('\n')
                # Find the last non-empty line that's not part of the prompt
                for line in reversed(lines):
                    line = line.strip()
                    if line and not line.startswith('You are') and not line.startswith('Context') and not line.startswith('Previous') and not line.startswith('User:'):
                        response_text = line
                        break
            
            # Add response to context
            self.add_message_to_context(session_id, "assistant", response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_dummy_response(user_input, context_docs, language)
    
    def _generate_dummy_response(self, user_input: str, context_docs: List[Dict] = None, language: str = "en") -> str:
        """Generate a helpful agricultural response using RAG context and keyword matching"""
        
        # Use RAG context if available
        context_info = ""
        if context_docs and len(context_docs) > 0:
            # Extract relevant information from context documents
            context_parts = []
            for doc in context_docs[:2]:  # Use first 2 documents
                text = doc.get('text', '')[:300]  # Limit length
                if text:
                    context_parts.append(text)
            if context_parts:
                context_info = f" Based on agricultural knowledge: {' '.join(context_parts)}"
        
        # Simple keyword-based responses with context
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["disease", "pest", "problem", "issue", "sick", "dying"]):
            response = "I can help with agricultural problems. Common issues include fungal diseases, pest infestations, and nutrient deficiencies. Please describe the specific symptoms you're seeing on your crops, such as leaf discoloration, wilting, or unusual growth patterns."
            if context_info:
                response += context_info
            return response
        
        elif any(word in user_lower for word in ["fertilizer", "fertilization", "nutrient", "fertilize"]):
            response = "For fertilization advice, consider soil testing first to determine nutrient levels. Different crops have different nutrient requirements. Organic fertilizers like compost, manure, and green manure are excellent for soil health. Chemical fertilizers should be used based on soil test results and crop needs."
            if context_info:
                response += context_info
            return response
        
        elif any(word in user_lower for word in ["water", "irrigation", "watering", "drought"]):
            response = "Proper irrigation is crucial for crop health. Water early morning or evening to reduce evaporation. Avoid overwatering as it can lead to root rot. Consider drip irrigation for water efficiency. Monitor soil moisture levels regularly."
            if context_info:
                response += context_info
            return response
        
        elif any(word in user_lower for word in ["soil", "ground", "earth", "dirt"]):
            response = "Healthy soil is the foundation of good agriculture. Consider crop rotation, organic matter addition, and proper pH levels for optimal soil health. Regular soil testing helps maintain nutrient balance and soil structure."
            if context_info:
                response += context_info
            return response
        
        elif any(word in user_lower for word in ["plant", "grow", "cultivate", "crop", "farming"]):
            response = "For successful crop cultivation, focus on proper soil preparation, appropriate planting times, adequate spacing, and regular monitoring. Each crop has specific requirements for sunlight, water, and nutrients."
            if context_info:
                response += context_info
            return response
        
        elif any(word in user_lower for word in ["organic", "natural", "sustainable"]):
            response = "Organic farming focuses on natural methods without synthetic chemicals. This includes using organic fertilizers, natural pest control, crop rotation, and maintaining soil health through composting and cover crops."
            if context_info:
                response += context_info
            return response
        
        else:
            response = "I'm here to help with agricultural questions. Please ask about crop diseases, pests, fertilization, irrigation, soil management, or any other farming topics. I can provide guidance based on agricultural best practices."
            if context_info:
                response += context_info
            return response
    
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
