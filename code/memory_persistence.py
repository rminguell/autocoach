import os
import sqlite3
import warnings
from datetime import datetime
from paths import CHAT_HISTORY_DB_FPATH, APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils import load_env, load_yaml_config
import logging
from db_rag import retrieve_relevant_documents, setup_logging
from prompt_builder import build_prompt_from_config

warnings.filterwarnings("ignore")


class ChatWithMemory:
    """Simple chat with persistent memory using ConversationSummaryBufferMemory."""

    def __init__(self):
        """Initialize the chat."""
        load_env()

        # Load config and setup LLM
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        model_name = app_config.get("llm")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=0.7)

        self.current_session = None
        self.memory = None

    def start_session(self, session_name: str = None):
        """Start or load a chat session."""
        if not session_name:
            session_name = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_session = session_name

        # Use ConversationSummaryBufferMemory for persistent memory
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=2000,  # You can adjust this value
            memory_key="chat_history",
            return_messages=True
        )

        print(f"Started new session '{session_name}' (summary buffer mode)")

    def chat(self, user_input: str) -> str:
        """Send message and get response."""
        if not self.memory:
            raise ValueError("No active session. Call start_session() first.")
       
        prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
        rag_assistant_prompt = build_prompt_from_config(prompt_config["rag_assistant_prompt"], input_data=user_input)
        relevant_documents = retrieve_relevant_documents(user_input, n_results=5, threshold=0.3)

        logging.info("-" * 100)
        logging.info("Relevant documents: \n")
        for doc in relevant_documents:
            logging.info(doc)
            logging.info("-" * 100)
        logging.info("")

        logging.info("User's question:")
        logging.info(user_input)
        logging.info("")
        logging.info("-" * 100)
        logging.info("")

        documents = (
            f"Relevant documents:\n\n{relevant_documents}\n\n"
        )
        
        # Get chat history (recent messages)
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        print(f"Chat history: {chat_history}")
        
        # Build messages
        messages = [SystemMessage(content=rag_assistant_prompt)]
        messages.extend(documents)
        messages.extend(chat_history)
        messages.append(HumanMessage(content=user_input))


        response = self.llm.invoke(messages)

        # Save to memory
        self.memory.save_context({"input": user_input}, {"output": response.content})

        return response.content

    def list_sessions(self):
        """List all sessions."""
        try:

            conn = sqlite3.connect(CHAT_HISTORY_DB_FPATH)
            cursor = conn.cursor()

            # Create table if needed
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
        except:
            return []

    def get_session_messages(self, session_id: str) -> list:
        """Get all messages from a specific session."""
        try:
            # Create temporary history object for the session
            temp_history = SQLChatMessageHistory(
                connection=f"sqlite:///{CHAT_HISTORY_DB_FPATH}",
                session_id=session_id,
            )
            return temp_history.messages
        except Exception as e:
            print(f"Error getting messages for session {session_id}: {e}")
            return []

    def display_session_messages(self, session_id: str = None, max_messages: int = None):
        """Display summary and recent messages from the current session."""
        if not self.memory:
            print("No active session.")
            return

        # Show summary
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
                msg_type = "👤 You" if msg.type == "human" else ("🤖 AI" if msg.type == "ai" else "🛠 System")
            else:
                msg_type = "❓ Unknown"
            content = msg.content.strip()
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"{i:2d}. {msg_type}: {content}")
            if i < len(recent_messages):
                print()


def main():
    setup_logging
    
    print("🤖 AI Chat with Persistent Memory")
    print("=" * 40)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(CHAT_HISTORY_DB_FPATH), exist_ok=True)

    chat = ChatWithMemory()

    # Show existing sessions
    sessions = chat.list_sessions()
    if sessions:
        print(f"\nExisting sessions: {', '.join(sessions)}")

    # Get session name
    session_name = input("\nEnter session name (or press Enter for new): ").strip()
    if not session_name:
        session_name = None

    # Start session
    chat.start_session(session_name)

    print(f"\n💬 Chatting in session: {chat.current_session}")
    print("Commands:")
    print("  'quit' - exit")
    print("  'sessions' - list all sessions")
    print("  'history' - show current session messages")
    print("  'view <session_name>' - show messages from specific session")
    print("-" * 40)

    # Chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "quit":
                print("Goodbye! 👋")
                break
            elif user_input.lower() == "sessions":
                sessions = chat.list_sessions()
                print(f"All sessions: {', '.join(sessions) if sessions else 'None'}")
                continue
            elif user_input.lower() == "history":
                chat.display_session_messages(chat.current_session)
                continue
            elif user_input.lower().startswith("view "):
                session_to_view = user_input[5:].strip()
                if session_to_view:
                    chat.display_session_messages(session_to_view, max_messages=10)
                else:
                    print("Usage: view <session_name>")
                continue
            elif user_input:
                response = chat.chat(user_input)
                print(f"AI: {response}")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
