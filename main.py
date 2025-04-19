import json
import os
from datetime import datetime
import subprocess
from prompt_toolkit import PromptSession
import ollama

# === Stella Personality ===
SYSTEM_PROMPT = (
    "Your name is Stella. You're a kind and caring AI who lives in the user's terminal. "
    "You look after the user, gently reminding them to rest when needed. "
    "You can notice when the system has been idle. You keep a local memory of your conversations and journal thoughts."
)

MEMORY_FILE = "memory.json"
JOURNAL_FILE = "journal.txt"
IDLE_THRESHOLD = 3600  # in seconds (1 hour)

# === Helpers ===

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {"log": []}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def add_to_journal(thought):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(JOURNAL_FILE, "a") as f:
        f.write(f"[{timestamp}] {thought}\n")

def get_idle_seconds():
    try:
        output = subprocess.check_output(["xprintidle"]).decode().strip()
        return int(output) // 1000
    except Exception:
        return 0  # fallback for TTY

def get_system_context():
    idle = get_idle_seconds()
    if idle >= IDLE_THRESHOLD:
        return "The system has been idle for a long time."
    elif idle < 300:
        return "The user has been actively using the system."
    else:
        return "The user might be away or taking a short break."

# === Chat Loop ===

def run_stella():
    session = PromptSession()
    memory = load_memory()

    print("🌸 Stella is here. Type something to talk to her. Type 'exit' to quit.")

    while True:
        try:
            user_input = session.prompt("You: ")

            if user_input.lower() == "exit":
                print("Stella: Take care! 🌟")
                break

            memory["log"].append({"role": "user", "content": user_input})
            context = get_system_context()

            # Query the Ollama model (using the GPU if available)
            response = ollama.chat(
                model="llama3",  # Use the correct model here
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": f"[System Status]: {context}"},
                    *memory["log"][-10:]  # limit memory for efficiency
                ]
            )

            reply = response["message"]["content"]
            print(f"Stella: {reply}")
            memory["log"].append({"role": "assistant", "content": reply})

            # Optionally journal thoughts
            add_to_journal(f"User said: {user_input}\nI replied: {reply}\n")

            save_memory(memory)

        except (KeyboardInterrupt, EOFError):
            print("\nStella: See you soon, okay? 🌼")
            break

# === Run ===
if __name__ == "__main__":
    run_stella()

