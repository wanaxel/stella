import json
import os
from datetime import datetime
import subprocess
import time
import random
import shutil
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import ollama

class SystemCapabilities:
    def __init__(self):
        self.cpu_threads = 2
        self.gpu_available = False
        self.gpu_type = "none"
        self.batch_size = 32
        self.context_size = 2048
        
    def get_ollama_options(self):
        return {
            "num_thread": self.cpu_threads,
            "num_ctx": self.context_size,
            "batch_size": self.batch_size,
            "seed": int(time.time()),
            "repeat_penalty": 1.1,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9
        }

class StellaUI:
    BANNER = """
╭────────────────────────────────────────────────────────╮
│                                                        │
│      ⋆｡°✩  𝓢𝓽𝓮𝓵𝓵𝓪 - Your Terminal Companion  ✩°｡⋆    │
│                         [LOW MEMORY MODE]              │
╰────────────────────────────────────────────────────────╯
"""

    FACE = """
    ╭─────╮
    │ ^‿^ │
    ╰─────╯
    """

    def __init__(self):
        self.divider = "─" * min(80, shutil.get_terminal_size().columns)
    
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_colored(self, text, color="white", bold=False):
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
    
    def print_slowly(self, text, delay=0.01):
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()
    
    def print_banner(self):
        self.print_colored(self.BANNER, "cyan", bold=True)
        self.print_colored(self.FACE, "cyan")
    
    def print_divider(self):
        self.print_colored(self.divider, "blue")
    
    def print_thinking(self):
        self.print_colored("Stella is thinking", "magenta", bold=True)
        for _ in range(3):
            print(".", end='', flush=True)
            time.sleep(0.3)
        print("\r" + " " * 20 + "\r", end='')
    
    def print_response(self, response):
        self.print_divider()
        self.print_colored(self.FACE, "cyan")
        self.print_slowly(f"Stella: {response}")
        self.print_divider()
    
    def print_error(self, error_message):
        self.print_colored(f"Error: {error_message}", "red")
        self.print_colored("Let's try again...", "yellow")
    
    def print_goodbye(self):
        self.print_divider()
        self.print_slowly("\nStella: Take care! See you next time! 🌟\n")


class StellaMemory:
    def __init__(self, memory_file="memory.json", journal_file="journal.txt"):
        self.memory_file = memory_file
        self.journal_file = journal_file
        self.memory = self.load_memory()
        self.max_history = 5
    
    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {"log": []}
        return {"log": []}
    
    def save_memory(self):
        if len(self.memory["log"]) > self.max_history * 2:
            self.memory["log"] = self.memory["log"][-self.max_history*2:]
            
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)
    
    def add_to_journal(self, thought):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.journal_file, "a") as f:
            f.write(f"[{timestamp}] {thought}\n")
    
    def add_user_message(self, message):
        self.memory["log"].append({"role": "user", "content": message})
        self._limit_memory()
        self.save_memory()
    
    def add_assistant_message(self, message):
        self.memory["log"].append({"role": "assistant", "content": message})
        self._limit_memory()
        self.save_memory()
    
    def _limit_memory(self):
        if len(self.memory["log"]) > self.max_history * 2:
            self.memory["log"] = self.memory["log"][-self.max_history*2:]
    
    def get_recent_messages(self, limit=5):
        return self.memory["log"][-limit*2:]


class StellaContext:
    @staticmethod
    def get_time_greeting():
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 18:
            return "Good afternoon"
        else:
            return "Good evening"


class Stella:
    SYSTEM_PROMPT = (
        "Your name is Stella. You're a kind and caring AI who lives in the user's terminal. "
        "You're currently running in LOW MEMORY MODE with limited capabilities. "
        "Keep your responses short and to the point. "
    )
    
    def __init__(self):
        self.system_config = SystemCapabilities()
        self.ui = StellaUI()
        self.memory = StellaMemory()
        self.session = PromptSession()
    
    def get_user_input(self):
        style = Style.from_dict({
            'prompt': 'ansicyan bold',
        })
        
        return self.session.prompt(
            HTML("<ansicyan>You:</ansicyan> "),
            style=style
        )
    
    def generate_response(self, user_input):
        self.memory.add_user_message(user_input)
        
        self.ui.print_thinking()
        
        options = self.system_config.get_ollama_options()
        
        models_to_try = ["llama3.2:3b", "llama3:8b", "llama3", "llama2:7b", "llama2"]
        
        for model in models_to_try:
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        *self.memory.get_recent_messages(5)
                    ],
                    options=options
                )
                
                reply = response["message"]["content"]
                self.memory.add_assistant_message(reply)
                
                return reply
            except Exception:
                continue
        
        return f"I'm having trouble thinking. Let's try again with a simpler question."
    
    def run(self):
        self.ui.clear_screen()
        self.ui.print_banner()
        
        greeting = f"{StellaContext.get_time_greeting()}! I'm Stella, running in low memory mode."
        self.ui.print_slowly(f"\nStella: {greeting} 🌸")
        self.ui.print_slowly("       Type something to talk to me or 'exit' to quit.\n")
        self.ui.print_divider()
        
        while True:
            try:
                user_input = self.get_user_input()
                
                if user_input.lower() == "exit":
                    self.ui.print_goodbye()
                    break
                
                reply = self.generate_response(user_input)
                self.ui.print_response(reply)
                
            except (KeyboardInterrupt, EOFError):
                self.ui.print_divider()
                self.ui.print_slowly("\nStella: See you soon, okay? 🌼\n")
                break
            except Exception as e:
                self.ui.print_error(str(e))


if __name__ == "__main__":
    print("Starting Stella in LOW MEMORY mode...")
    print("This mode uses minimal resources but has limited capabilities.")
    stella = Stella()
    stella.run()
