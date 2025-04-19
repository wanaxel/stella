import json
import os
from datetime import datetime
import subprocess
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import ollama
import time
import random
import shutil

# === Stella Personality ===
SYSTEM_PROMPT = (
    "Your name is Stella. You're a kind and caring AI who lives in the user's terminal. "
    "You look after the user, gently reminding them to rest when needed. "
    "You can notice when the system has been idle. You keep a local memory of your conversations and journal thoughts."
)

MEMORY_FILE = "memory.json"
JOURNAL_FILE = "journal.txt"
IDLE_THRESHOLD = 3600  # in seconds (1 hour)

# === ASCII Art & UI Elements ===
STELLA_BANNER = r"""
╭────────────────────────────────────────────────────────╮
│                                                        │
│   ⋆｡°✩  𝓢𝓽𝓮𝓵𝓵𝓪 - Your Terminal Companion  ✩°｡⋆    │
│                                                        │
╰────────────────────────────────────────────────────────╯
"""

STELLA_FACES = [
    r"""
    ╭─────╮
    │ ^‿^ │
    ╰─────╯
    """,
    r"""
    ╭─────╮
    │ ･ω･ │
    ╰─────╯
    """,
    r"""
    ╭─────╮
    │ •◡• │
    ╰─────╯
    """
]

DIVIDER = "─" * shutil.get_terminal_size().columns

# === Styling ===
def print_colored(text, color="white", bold=False):
    """Print colored text in the terminal"""
    colors = {
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m"
    }
    
    bold_code = "\033[1m" if bold else ""
    reset = "\033[0m"
    
    print(f"{colors.get(color, '')}{bold_code}{text}{reset}")

def print_slowly(text, delay=0.01):
    """Print text with a typewriter effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

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

def get_time_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"

# === Chat Loop ===
def run_stella():
    session = PromptSession()
    memory = load_memory()
    
    # Clear screen and show welcome
    os.system('cls' if os.name == 'nt' else 'clear')
    print_colored(STELLA_BANNER, "cyan", bold=True)
    print_colored(random.choice(STELLA_FACES), "cyan")
    
    greeting = f"{get_time_greeting()}! I'm Stella, your terminal companion."
    print_slowly(f"\nStella: {greeting} 🌸")
    print_slowly("       Type something to talk to me or 'exit' to quit.\n")
    print_colored(DIVIDER, "blue")
    
    while True:
        try:
            # Create custom style for the prompt
            style = Style.from_dict({
                'prompt': 'ansicyan bold',
            })
            
            # Display prompt with formatting
            user_input = session.prompt(
                HTML("<ansicyan>You:</ansicyan> "),
                style=style
            )
            
            if user_input.lower() == "exit":
                print_colored(DIVIDER, "blue")
                print_slowly("\nStella: Take care! See you next time! 🌟\n")
                break
                
            memory["log"].append({"role": "user", "content": user_input})
            context = get_system_context()
            
            # Show "thinking" animation
            print_colored("Stella is thinking", "magenta", bold=True)
            for _ in range(3):
                print(".", end='', flush=True)
                time.sleep(0.3)
            print("\r" + " " * 20 + "\r", end='')  # Clear the thinking line
            
            # Query the Ollama model
            response = ollama.chat(
                model="llama3",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": f"[System Status]: {context}"},
                    *memory["log"][-10:]  # limit memory for efficiency
                ]
            )
            
            reply = response["message"]["content"]
            
            # Display Stella's response with formatting
            print_colored(DIVIDER, "blue")
            print_colored(random.choice(STELLA_FACES), "cyan")
            print_slowly(f"Stella: {reply}")
            print_colored(DIVIDER, "blue")
            
            memory["log"].append({"role": "assistant", "content": reply})
            
            # Journal thoughts
            add_to_journal(f"User said: {user_input}\nI replied: {reply}\n")
            save_memory(memory)
            
        except (KeyboardInterrupt, EOFError):
            print_colored(DIVIDER, "blue")
            print_slowly("\nStella: See you soon, okay? 🌼\n")
            break

# === Run ===
if __name__ == "__main__":
    run_stella()
