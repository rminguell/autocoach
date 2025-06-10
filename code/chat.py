import sqlite3
import warnings
from datetime import datetime
from paths import CHAT_HISTORY_DB_FPATH, APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils import load_env, load_yaml_config
from db_connection import retrieve_relevant_documents
from prompt_builder import build_prompt_from_config

warnings.filterwarnings("ignore")


class ChatWithMemory:
    def __init__(self):
        load_env()
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        model_name = app_config.get("llm")
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7)
        self.current_session = None
        self.memory = None

    def start_session(self, session_name: str = None):
        if not session_name:
            session_name = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = session_name
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=200,
            memory_key="chat_history",
            return_messages=True
        )
        print(f"Started new session '{session_name}' (summary buffer mode)")

    def chat(self, user_input: str) -> str:
        if not self.memory:
            raise ValueError("No active session. Call start_session() first.")
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        vectordb_params = app_config["vectordb"]
        relevant_documents = retrieve_relevant_documents(user_input, **vectordb_params)
        relevant_documents = f"RELEVANT DOCUMENTS:\n\n{relevant_documents}\n\n"
        print(relevant_documents)
        recent_messages = self.memory.buffer
        recent_messages = f"RECENT MESSAGES:\n\n{recent_messages}\n\n"
        print(recent_messages)
        summary = self.memory.moving_summary_buffer
        summary = f"EARLIER CONVERSATION SUMMARY:\n\n{summary}\n\n"
        print(summary)
        prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
        rag_assistant_prompt = build_prompt_from_config(
            prompt_config["rag_assistant_prompt"],
            input_data=relevant_documents
        )
        messages = [SystemMessage(content=rag_assistant_prompt)]
        messages.append(SystemMessage(content=recent_messages))
        messages.append(SystemMessage(content=summary))
        messages.append(HumanMessage(content=user_input))
        response = self.llm.invoke(messages)
        self.memory.save_context({"input": user_input}, {"output": response.content})
        return response.content

    def list_sessions(self):
        try:
            conn = sqlite3.connect(CHAT_HISTORY_DB_FPATH)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS message_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                "SELECT DISTINCT session_id FROM message_store ORDER BY session_id"
            )
            sessions = [row[0] for row in cursor.fetchall()]
            conn.close()
            return sessions
        except Exception:
            return []

    def get_session_messages(self, session_id: str) -> list:
        try:
            temp_history = SQLChatMessageHistory(
                connection=f"sqlite:///{CHAT_HISTORY_DB_FPATH}",
                session_id=session_id,
            )
            return temp_history.messages
        except Exception as e:
            print(f"Error getting messages for session {session_id}: {e}")
            return []

    def display_session_messages(self, session_id: str = None, max_messages: int = None):
        if not self.memory:
            print("No active session.")
            return
        summary = self.memory.moving_summary_buffer
        print("\n--- Conversation Summary (old messages summarized) ---")
        print(summary if summary else "(No summary yet)")
        print("\n--- Recent Messages (not summarized) ---")
        recent_messages = self.memory.buffer
        if not recent_messages:
            print("No recent messages.")
            return
        if max_messages and len(recent_messages) > max_messages:
            print(f"Showing last {max_messages} of {len(recent_messages)} messages:")
            recent_messages = recent_messages[-max_messages:]
        else:
            print(f"Total recent messages: {len(recent_messages)}")
        print("-" * 50)
        for i, msg in enumerate(recent_messages, 1):
            if hasattr(msg, "type"):
                if msg.type == "human":
                    msg_type = "👤 You"
                elif msg.type == "ai":
                    msg_type = "🤖 AI"
                else:
                    msg_type = "🛠️ System"
            else:
                msg_type = "❓ Unknown"
            content = msg.content.strip()
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"{i:2d}. {msg_type}: {content}")
            if i < len(recent_messages):
                print()
