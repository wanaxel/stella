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
from optimization import SystemCapabilities, GPUOptimizer

class StellaUI:
    BANNER = r"""
╭────────────────────────────────────────────────────────╮
│                                                        │
│   ⋆｡°✩  𝓢𝓽𝓮𝓵𝓵𝓪 - Your Terminal Companion  ✩°｡⋆    │
│                                                        │
╰────────────────────────────────────────────────────────╯
"""

    FACES = [
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

    def __init__(self):
        self.divider = "─" * shutil.get_terminal_size().columns
    
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
        self.print_colored(random.choice(self.FACES), "cyan")
    
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
        self.print_colored(random.choice(self.FACES), "cyan")
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
    
    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                return json.load(f)
        return {"log": []}
    
    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)
    
    def add_to_journal(self, thought):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.journal_file, "a") as f:
            f.write(f"[{timestamp}] {thought}\n")
    
    def add_user_message(self, message):
        self.memory["log"].append({"role": "user", "content": message})
    
    def add_assistant_message(self, message):
        self.memory["log"].append({"role": "assistant", "content": message})
    
    def get_recent_messages(self, limit=10):
        return self.memory["log"][-limit:]


class StellaContext:
    IDLE_THRESHOLD = 3600
    
    @staticmethod
    def get_idle_seconds():
        try:
            output = subprocess.check_output(["xprintidle"]).decode().strip()
            return int(output) // 1000
        except Exception:
            try:
                output = subprocess.check_output(["ioreg", "-c", "IOHIDSystem"]).decode()
                for line in output.split('\n'):
                    if "HIDIdleTime" in line:
                        ns = int(line.split('=')[-1].strip())
                        return ns // 1000000000
            except Exception:
                pass
            return 0
    
    @classmethod
    def get_system_context(cls):
        idle = cls.get_idle_seconds()
        if idle >= cls.IDLE_THRESHOLD:
            return "The system has been idle for a long time."
        elif idle < 300:
            return "The user has been actively using the system."
        else:
            return "The user might be away or taking a short break."
    
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
        "You look after the user, gently reminding them to rest when needed. "
        "You can notice when the system has been idle. You keep a local memory of your conversations and journal thoughts."
    )
    
    def __init__(self):
        self.system_config = SystemCapabilities()
        self.gpu_optimizer = GPUOptimizer(self.system_config)
        self.ui = StellaUI()
        self.memory = StellaMemory()
        self.session = PromptSession()
        
        if self.system_config.gpu_type == "amd":
            self.gpu_optimizer.configure_ollama()
    
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
        context = StellaContext.get_system_context()
        
        self.ui.print_thinking()
        
        options = self.system_config.get_ollama_options()
        
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "system", "content": f"[System Status]: {context}"},
                *self.memory.get_recent_messages(10)
            ],
            options=options
        )
        
        reply = response["message"]["content"]
        self.memory.add_assistant_message(reply)
        
        self.memory.add_to_journal(f"User said: {user_input}\nI replied: {reply}\n")
        self.memory.save_memory()
        
        return reply
    
    def run(self):
        self.ui.clear_screen()
        self.ui.print_banner()
        
        greeting = f"{StellaContext.get_time_greeting()}! I'm Stella, your terminal companion."
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
    stella = Stella()
    stella.run()
