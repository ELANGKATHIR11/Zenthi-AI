import os
import json

class SessionMemory:
    def __init__(self, max_turns=10):
        self.max_turns = max_turns
        self.sessions = {}

    def get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def add_message(self, session_id, role, content):
        session = self.get_session(session_id)
        session.append({"role": role, "content": content})
        
        # Keep windowed memory: max_turns * 2 (1 turn = user + assistant) + 1 system prompt
        # Assuming system prompt is at index 0, we preserve it
        system_prompt = None
        if session and session[0]["role"] == "system":
            system_prompt = session[0]
            chat_messages = session[1:]
        else:
            chat_messages = session

        max_history_len = self.max_turns * 2
        if len(chat_messages) > max_history_len:
            chat_messages = chat_messages[-max_history_len:]
            
        if system_prompt:
            self.sessions[session_id] = [system_prompt] + chat_messages
        else:
            self.sessions[session_id] = chat_messages

    def get_messages_for_prompt(self, session_id, system_prompt=None):
        session = self.get_session(session_id)
        
        # If no system prompt is present, prepend one
        if system_prompt:
            if not session or session[0]["role"] != "system":
                self.sessions[session_id] = [{"role": "system", "content": system_prompt}] + session
            else:
                self.sessions[session_id][0] = {"role": "system", "content": system_prompt}
                
        return self.sessions[session_id]

    def clear_session(self, session_id):
        if session_id in self.sessions:
            self.sessions[session_id] = []

if __name__ == "__main__":
    memory = SessionMemory(max_turns=2)
    memory.add_message("session_1", "system", "You are an assistant.")
    memory.add_message("session_1", "user", "Hello")
    memory.add_message("session_1", "assistant", "Hi there!")
    memory.add_message("session_1", "user", "Tell me a joke")
    memory.add_message("session_1", "assistant", "Why did the chicken cross the road?")
    memory.add_message("session_1", "user", "Why?")
    memory.add_message("session_1", "assistant", "To get to the other side.")
    
    msgs = memory.get_messages_for_prompt("session_1")
    print("Windowed Session Messages:")
    print(json.dumps(msgs, indent=2))
