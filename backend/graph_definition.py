"""
LangGraph definition for the conversational AI pipeline
"""
import logging
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from transcription_engine import TranscriptionEngine
from rag_store import RAGStore
from conversation_manager import ConversationManager
from persistence import ChatPersistence
from datetime import datetime

logger = logging.getLogger(__name__)

class ConversationState(TypedDict):
    """State definition for the conversation graph"""
    # Input
    audio_data: bytes
    session_id: str
    language: str
    
    # Intermediate results
    transcription: str
    detected_language: str
    context_docs: List[Dict[str, Any]]
    
    # Output
    ai_response: str
    final_output: Dict[str, Any]

class ConversationGraph:
    def __init__(self):
        self.transcription_engine = TranscriptionEngine()
        self.rag_store = RAGStore()
        self.conversation_manager = ConversationManager()
        self.persistence = ChatPersistence()
        
        # Initialize RAG store with sample data
        self.rag_store.load_sample_knowledge_base()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph"""
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("stt_node", self.stt_node)
        workflow.add_node("lang_detect_node", self.lang_detect_node)
        workflow.add_node("rag_node", self.rag_node)
        workflow.add_node("llm_node", self.llm_node)
        workflow.add_node("conversation_node", self.conversation_node)
        workflow.add_node("persistence_node", self.persistence_node)
        workflow.add_node("output_node", self.output_node)
        
        # Define the flow
        workflow.set_entry_point("stt_node")
        
        workflow.add_edge("stt_node", "lang_detect_node")
        workflow.add_edge("lang_detect_node", "rag_node")
        workflow.add_edge("rag_node", "llm_node")
        workflow.add_edge("llm_node", "conversation_node")
        workflow.add_edge("conversation_node", "persistence_node")
        workflow.add_edge("persistence_node", "output_node")
        workflow.add_edge("output_node", END)
        
        return workflow.compile()
    
    def stt_node(self, state: ConversationState) -> ConversationState:
        """Speech-to-Text node"""
        try:
            logger.info("Processing audio with STT")
            
            # Transcribe audio
            result = self.transcription_engine.transcribe_audio(
                state["audio_data"], 
                state.get("language", "auto")
            )
            
            state["transcription"] = result["text"]
            state["detected_language"] = result["language"]
            
            logger.info(f"Transcription: {state['transcription']}")
            logger.info(f"Detected language: {state['detected_language']}")
            
        except Exception as e:
            logger.error(f"Error in STT node: {e}")
            state["transcription"] = ""
            state["detected_language"] = "en"
        
        return state
    
    def lang_detect_node(self, state: ConversationState) -> ConversationState:
        """Language detection node"""
        try:
            # Language is already detected in STT node
            # This node can be used for additional language processing
            detected_lang = state.get("detected_language", "en")
            state["language"] = detected_lang
            
            logger.info(f"Language confirmed: {detected_lang}")
            
        except Exception as e:
            logger.error(f"Error in language detection node: {e}")
            state["language"] = "en"
        
        return state
    
    def rag_node(self, state: ConversationState) -> ConversationState:
        """Retrieval-Augmented Generation node"""
        try:
            query = state.get("transcription", "")
            if not query:
                state["context_docs"] = []
                return state
            
            logger.info(f"Searching RAG store for: {query}")
            
            # Search for relevant documents
            context_docs = self.rag_store.search(query, top_k=3)
            state["context_docs"] = context_docs
            
            logger.info(f"Retrieved {len(context_docs)} context documents")
            
        except Exception as e:
            logger.error(f"Error in RAG node: {e}")
            state["context_docs"] = []
        
        return state
    
    def llm_node(self, state: ConversationState) -> ConversationState:
        """LLM generation node"""
        try:
            user_input = state.get("transcription", "")
            session_id = state.get("session_id", "default")
            context_docs = state.get("context_docs", [])
            language = state.get("language", "en")
            
            if not user_input:
                state["ai_response"] = "I didn't catch that. Could you please repeat?"
                return state
            
            logger.info("Generating AI response")
            
            # Generate response using conversation manager
            ai_response = self.conversation_manager.generate_response(
                user_input=user_input,
                session_id=session_id,
                context_docs=context_docs,
                language=language
            )
            
            state["ai_response"] = ai_response
            logger.info(f"Generated response: {ai_response[:100]}...")
            
        except Exception as e:
            logger.error(f"Error in LLM node: {e}")
            state["ai_response"] = "I apologize, but I'm having trouble processing your request right now."
        
        return state
    
    def conversation_node(self, state: ConversationState) -> ConversationState:
        """Conversation management node"""
        try:
            # This node handles conversation flow and context management
            # The conversation manager already handles this in the LLM node
            # This can be extended for more complex conversation logic
            
            logger.info("Conversation context updated")
            
        except Exception as e:
            logger.error(f"Error in conversation node: {e}")
        
        return state
    
    def persistence_node(self, state: ConversationState) -> ConversationState:
        """Persistence node for saving conversation"""
        try:
            session_id = state.get("session_id", "default")
            user_text = state.get("transcription", "")
            ai_text = state.get("ai_response", "")
            language = state.get("language", "en")
            context_docs = state.get("context_docs", [])
            
            # Create context metadata
            context_metadata = {
                "language": language,
                "context_docs": [doc.get("metadata", {}) for doc in context_docs],
                "timestamp": str(datetime.now())
            }
            
            # Save user message
            if user_text:
                self.persistence.add_message(
                    session_id=session_id,
                    role="user",
                    text=user_text,
                    language=language,
                    context=context_metadata
                )
            
            # Save AI response
            if ai_text:
                self.persistence.add_message(
                    session_id=session_id,
                    role="assistant",
                    text=ai_text,
                    language=language,
                    context=context_metadata
                )
            
            logger.info("Conversation saved to persistence")
            
        except Exception as e:
            logger.error(f"Error in persistence node: {e}")
        
        return state
    
    def output_node(self, state: ConversationState) -> ConversationState:
        """Output formatting node"""
        try:
            # Format final output
            output = {
                "session_id": state.get("session_id", "default"),
                "language": state.get("language", "en"),
                "user_text": state.get("transcription", ""),
                "ai_text": state.get("ai_response", ""),
                "context_docs": state.get("context_docs", []),
                "timestamp": str(datetime.now())
            }
            
            state["final_output"] = output
            logger.info("Output formatted successfully")
            
        except Exception as e:
            logger.error(f"Error in output node: {e}")
            state["final_output"] = {
                "session_id": state.get("session_id", "default"),
                "language": "en",
                "user_text": "",
                "ai_text": "Error processing request",
                "context_docs": [],
                "timestamp": str(datetime.now())
            }
        
        return state
    
    async def process_conversation(self, audio_data: bytes, session_id: str, 
                                 language: str = "auto") -> Dict[str, Any]:
        """
        Process a conversation turn through the entire pipeline
        
        Args:
            audio_data: Raw audio bytes
            session_id: Session identifier
            language: Language code or "auto"
            
        Returns:
            Final output dictionary
        """
        try:
            # Create initial state
            initial_state = ConversationState(
                audio_data=audio_data,
                session_id=session_id,
                language=language,
                transcription="",
                detected_language="",
                context_docs=[],
                ai_response="",
                final_output={}
            )
            
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            
            return result.get("final_output", {})
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            return {
                "session_id": session_id,
                "language": "en",
                "user_text": "",
                "ai_text": "Error processing request",
                "context_docs": [],
                "timestamp": str(datetime.now())
            }
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        return self.persistence.get_session_history(session_id)
    
    def create_session(self, session_id: str) -> bool:
        """Create a new conversation session"""
        return self.persistence.create_session(session_id)
