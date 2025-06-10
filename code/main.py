import os
from paths import CHAT_HISTORY_DB_FPATH
from chat import ChatWithMemory
from db_connection import setup_logging

def main():
    setup_logging()
    os.makedirs(os.path.dirname(CHAT_HISTORY_DB_FPATH), exist_ok=True)
    chat = ChatWithMemory()
    sessions = chat.list_sessions()
    if sessions:
        print(f"\nExisting usernames: {', '.join(sessions)}")
    session_name = input("\nEnter username (or press Enter for guest): ").strip()
    if not session_name:
        session_name = None
    chat.start_session(session_name)
    print(f"\n💬 Chatting as: {chat.current_session}")
    print("Commands:")
    print("  'quit' - exit")
    print("  'users' - list all usernames")
    print("  'history' - show current session messages")
    print("  'view <session_name>' - show messages from specific user")
    print("-" * 40)
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "quit":
                print("Goodbye! 👋")
                break
            elif user_input.lower() == "sessions":
                sessions = chat.list_sessions()
                print(f"All users: {', '.join(sessions) if sessions else 'None'}")
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
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()