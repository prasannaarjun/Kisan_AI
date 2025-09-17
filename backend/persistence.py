"""
SQLite persistence for chat sessions and messages
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

class ChatPersistence:
    def __init__(self, db_path: str = "chat_sessions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create messages table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        text TEXT NOT NULL,
                        language TEXT,
                        context TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_session_id 
                    ON messages (session_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
                    ON messages (timestamp)
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def create_session(self, session_id: str) -> bool:
        """Create a new chat session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions (session_id, created_at, updated_at)
                    VALUES (?, ?, ?)
                """, (session_id, datetime.now(), datetime.now()))
                conn.commit()
                logger.info(f"Created session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False
    
    def add_message(self, session_id: str, role: str, text: str, 
                   language: Optional[str] = None, context: Optional[Dict] = None) -> bool:
        """Add a message to a session"""
        try:
            context_json = json.dumps(context) if context else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO messages (session_id, role, text, language, context, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, role, text, language, context_json, datetime.now()))
                
                # Update session timestamp
                cursor.execute("""
                    UPDATE sessions SET updated_at = ? WHERE session_id = ?
                """, (datetime.now(), session_id))
                
                conn.commit()
                logger.debug(f"Added message to session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return False
    
    def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT role, text, language, context, timestamp
                    FROM messages 
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                """, (session_id, limit))
                
                messages = []
                for row in cursor.fetchall():
                    context = json.loads(row[3]) if row[3] else None
                    messages.append({
                        'role': row[0],
                        'text': row[1],
                        'language': row[2],
                        'context': context,
                        'timestamp': row[4]
                    })
                
                return messages
                
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return []
    
    def get_all_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of all sessions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.session_id, s.created_at, s.updated_at,
                           COUNT(m.id) as message_count
                    FROM sessions s
                    LEFT JOIN messages m ON s.session_id = m.session_id
                    GROUP BY s.session_id
                    ORDER BY s.updated_at DESC
                    LIMIT ?
                """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        'session_id': row[0],
                        'created_at': row[1],
                        'updated_at': row[2],
                        'message_count': row[3]
                    })
                
                return sessions
                
        except Exception as e:
            logger.error(f"Error getting sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete messages first (foreign key constraint)
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                
                # Delete session
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                
                conn.commit()
                logger.info(f"Deleted session: {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total sessions
                cursor.execute("SELECT COUNT(*) FROM sessions")
                total_sessions = cursor.fetchone()[0]
                
                # Total messages
                cursor.execute("SELECT COUNT(*) FROM messages")
                total_messages = cursor.fetchone()[0]
                
                # Messages by role
                cursor.execute("""
                    SELECT role, COUNT(*) 
                    FROM messages 
                    GROUP BY role
                """)
                messages_by_role = dict(cursor.fetchall())
                
                # Languages used
                cursor.execute("""
                    SELECT language, COUNT(*) 
                    FROM messages 
                    WHERE language IS NOT NULL
                    GROUP BY language
                    ORDER BY COUNT(*) DESC
                """)
                languages = dict(cursor.fetchall())
                
                return {
                    'total_sessions': total_sessions,
                    'total_messages': total_messages,
                    'messages_by_role': messages_by_role,
                    'languages': languages
                }
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
