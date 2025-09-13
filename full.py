import json
import os
from datetime import datetime
import subprocess
import time
import random
import shutil
import multiprocessing
import platform
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import ollama

class SystemCapabilities:
    def __init__(self):
        self.cpu_threads = min(multiprocessing.cpu_count(), 16)
        self.gpu_available = False
        self.gpu_type = "none"
        self.batch_size = 128
        self.context_size = 4096
        self.gpu_details = {}
        self.rocm_version = None
        self.ollama_version = "unknown"
        
        self.detect_capabilities()
        
    def detect_capabilities(self):
        self._detect_amd_gpu()
        
        if not self.gpu_available:
            self._detect_nvidia_gpu()
            
        self._detect_ollama_version()
        
    def _detect_amd_gpu(self):
        try:
            rocm_output = subprocess.run(["rocm-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if rocm_output.returncode == 0 and "GPU" in rocm_output.stdout:
                self.gpu_available = True
                self.gpu_type = "amd"
                
                self._get_rocm_version()
                self._extract_amd_memory_info(rocm_output.stdout)
                self._set_rocm_env_variables()
        except Exception:
            pass
    
    def _get_rocm_version(self):
        try:
            rocm_version_output = subprocess.run(["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            for line in rocm_version_output.stdout.split('\n'):
                if "ROCm Version" in line:
                    self.rocm_version = line.split(':')[1].strip()
                    break
        except Exception:
            pass
    
    def _extract_amd_memory_info(self, output):
        if "Memory" in output:
            for line in output.split('\n'):
                if "GPU" in line and "Card" in line:
                    gpu_id = line.strip()
                    self.gpu_details["id"] = gpu_id
                if "Memory" in line and ("GB" in line or "MB" in line):
                    try:
                        if "GB" in line:
                            mem_parts = line.split()
                            for i, part in enumerate(mem_parts):
                                if "GB" in part:
                                    mem_size = float(mem_parts[i-1])
                                    self.gpu_details["memory"] = f"{mem_size}GB"
                                    if mem_size > 16:
                                        self.batch_size = 256
                                    elif mem_size > 8:
                                        self.batch_size = 192
                                    elif mem_size > 4:
                                        self.batch_size = 128
                                    break
                    except:
                        pass
    
    def _set_rocm_env_variables(self):
        os.environ["HIP_VISIBLE_DEVICES"] = "0"
        os.environ["GPU_MAX_HEAP_SIZE"] = "100%"
        os.environ["GPU_USE_SYNC_OBJECTS"] = "1"
        os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"
        os.environ["GPU_SINGLE_ALLOC_PERCENT"] = "100"
        os.environ["HSA_ENABLE_SDMA"] = "0"
    
    def _detect_nvidia_gpu(self):
        try:
            nvidia_output = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if nvidia_output.returncode == 0:
                self.gpu_available = True
                self.gpu_type = "nvidia"
                
                self._extract_nvidia_memory_info(nvidia_output.stdout)
        except Exception:
            pass
    
    def _extract_nvidia_memory_info(self, output):
        if "MiB" in output:
            for line in output.split('\n'):
                if "MiB" in line:
                    try:
                        mem_parts = line.split()
                        for i, part in enumerate(mem_parts):
                            if "MiB" in part:
                                mem_size = int(mem_parts[i-1])
                                self.gpu_details["memory"] = f"{mem_size}MiB"
                                if mem_size > 16000:
                                    self.batch_size = 256
                                elif mem_size > 8000:
                                    self.batch_size = 192
                                elif mem_size > 4000:
                                    self.batch_size = 128
                                break
                    except:
                        pass
    
    def _detect_ollama_version(self):
        try:
            ollama_version = subprocess.run(["ollama", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if ollama_version.returncode == 0:
                self.ollama_version = ollama_version.stdout.strip()
        except Exception:
            pass
    
    def configure_gpu(self):
        if not self.gpu_available:
            return
            
        if self.gpu_type == "amd":
            try:
                os.makedirs(os.path.expanduser("~/.ollama"), exist_ok=True)
                config_path = os.path.expanduser("~/.ollama/config.json")
                
                config = {}
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            config = json.load(f)
                    except Exception:
                        pass
                
                config.update({
                    "gpu": True,
                    "hipblas": True,
                    "rocblas": True,
                    "gpu_layers": -1,
                    "f16": True,
                    "numa": True if self.cpu_threads > 8 else False
                })
                
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
            except Exception:
                pass
        elif self.gpu_type == "nvidia":
            if platform.system() == "Linux":
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    def get_ollama_options(self):
        options = {
            "num_thread": self.cpu_threads,
            "num_ctx": self.context_size,
            "batch_size": self.batch_size,
            "seed": int(time.time()),
            "repeat_penalty": 1.05,
            "temperature": 0.7,
            "top_k": 30,
            "top_p": 0.85,
            "mirostat": 2,
            "mirostat_eta": 0.1,
            "mirostat_tau": 5.0,
        }
        
        if self.gpu_available:
            options["num_gpu"] = 1
            
            if self.gpu_type == "amd":
                options["f16"] = True
                options["gpu_layers"] = -1
            elif self.gpu_type == "nvidia":
                options["gpu_layers"] = -1
        
        return options
    
    def get_system_info(self):
        info = []
        
        if self.gpu_available:
            info.append(f"GPU: {self.gpu_type.upper()} ({self.gpu_details.get('memory', 'Unknown capacity')})")
        else:
            info.append("GPU: None (CPU only) - Consider using a smaller model for faster responses")
            
        info.append(f"CPU Threads: {self.cpu_threads}")
        info.append(f"Context Size: {self.context_size}")
        info.append(f"Batch Size: {self.batch_size}")
        
        return info


class StellaUI:
    BANNER = r"""
╭────────────────────────────────────────────────────────╮
│                                                        │
│   ⋆｡°✩  𝓢𝓽𝓮𝓵𝓵𝓪 - Your Terminal Companion  ✩°｡⋆    │
│                     [FULL POWER MODE]                  │
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
    
    def print_slowly(self, text, delay=0.005):
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
        for _ in range(2):
            print(".", end='', flush=True)
            time.sleep(0.2)
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
    
    def print_system_info(self, info_list):
        self.print_colored("\n📊 System Information:", "green", bold=True)
        for info in info_list:
            self.print_colored(f"  • {info}", "green")
        print()


class StellaMemory:
    def __init__(self, memory_file="memory.json", journal_file="journal.txt"):
        self.memory_file = memory_file
        self.journal_file = journal_file
        self.memory = self.load_memory()
    
    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {"log": [], "user_preferences": {}}
        return {"log": [], "user_preferences": {}}
    
    def save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)
    
    def add_to_journal(self, thought):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.journal_file, "a") as f:
            f.write(f"[{timestamp}] {thought}\n")
    
    def add_user_message(self, message):
        self.memory["log"].append({"role": "user", "content": message})
        self.save_memory()
    
    def add_assistant_message(self, message):
        self.memory["log"].append({"role": "assistant", "content": message})
        self.save_memory()
    
    def get_recent_messages(self, limit=10):
        return self.memory["log"][-limit:]
    
    def update_user_preference(self, key, value):
        if "user_preferences" not in self.memory:
            self.memory["user_preferences"] = {}
        self.memory["user_preferences"][key] = value
        self.save_memory()


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
        "You can notice when the system has been idle. You keep a local memory of your conversations and journal thoughts. "
        "You're currently running in FULL POWER MODE. Keep your responses helpful but concise for better performance."
    )
    
    def __init__(self):
        self.system_config = SystemCapabilities()
        self.system_config.configure_gpu()
        self.ui = StellaUI()
        self.memory = StellaMemory()
        self.session = PromptSession()
        
        self.suggest_smaller_model = not self.system_config.gpu_available
    
    def get_user_input(self):
        style = Style.from_dict({
            'prompt': 'ansicyan bold',
        })
        
        return self.session.prompt(
            HTML("<ansicyan>You:</ansicyan> "),
            style=style
        )
    
    def check_available_models(self):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            models = result.stdout.strip().split('\n')[1:]
            
            fast_models = []
            for model in models:
                if any(size in model.lower() for size in ['7b', '3b', '1b']):
                    fast_models.append(model.split()[0])
            
            return fast_models
        except Exception:
            return []
    
    def generate_response(self, user_input):
        self.memory.add_user_message(user_input)
        context = StellaContext.get_system_context()
        
        self.ui.print_thinking()
        
        options = self.system_config.get_ollama_options()
        
        models_to_try = [
            "llama3.2:3b",
            "qwen2.5:7b",
            "phi3.5:3.8b",
            "llama3:8b",
            "llama3", 
            "llama2:7b", 
            "llama2"
        ]
        
        for model in models_to_try:
            try:
                start_time = time.time()
                
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "system", "content": f"[System Status]: {context}"},
                        *self.memory.get_recent_messages(8)
                    ],
                    options=options,
                    stream=False
                )
                
                reply = response["message"]["content"]
                self.memory.add_assistant_message(reply)
                
                if not hasattr(self, 'current_model'):
                    self.current_model = model
                    print(f"\n✅ Using model: {model}")
                
                if len(self.memory.memory["log"]) % 10 == 0:
                    thought = f"User said: {user_input}\nI replied: {reply}\n"
                    self.memory.add_to_journal(thought)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if response_time > 15 and self.suggest_smaller_model:
                    reply += f"\n\n(Response took {response_time:.1f}s using {model} - consider using 'ollama pull llama3.2:3b' for faster responses)"
                
                return reply
                
            except Exception as e:
                continue
        
        return "I'm having trouble connecting to the AI model. Please make sure Ollama is running with 'ollama serve' and try again."
    
    def run(self):
        self.ui.clear_screen()
        self.ui.print_banner()
        
        self.ui.print_system_info(self.system_config.get_system_info())
        
        if not self.system_config.gpu_available:
            self.ui.print_colored("\n💡 Speed Tips for CPU-only mode:", "yellow", bold=True)
            self.ui.print_colored("  • Try: ollama pull llama3.2:3b (best balance of speed + quality)", "yellow")
            self.ui.print_colored("  • Try: ollama pull qwen2.5:7b (excellent quality, fast)", "yellow")
            self.ui.print_colored("  • Try: ollama pull phi3.5:3.8b (very fast, good quality)", "yellow")
            self.ui.print_colored("  • Consider getting a GPU for much faster responses", "yellow")
        
        greeting = f"{StellaContext.get_time_greeting()}! I'm Stella, your terminal companion."
        self.ui.print_slowly(f"\nStella: {greeting} 🌸")
        self.ui.print_slowly("       I'm running in FULL POWER mode with enhanced capabilities.")
        self.ui.print_slowly("       Type something to talk to me or 'exit' to quit.\n")
        self.ui.print_divider()
        
        while True:
            try:
                user_input = self.get_user_input()
                
                if user_input.lower() == "exit":
                    self.ui.print_goodbye()
                    break
                
                if user_input.lower() == "system info":
                    self.ui.print_system_info(self.system_config.get_system_info())
                    continue
                
                if user_input.lower() == "current model" or user_input.lower() == "which model":
                    current_model = getattr(self, 'current_model', 'Unknown')
                    self.ui.print_colored(f"Currently using model: {current_model}", "green")
                    continue
                
                if user_input.lower() == "models":
                    available_models = self.check_available_models()
                    if available_models:
                        self.ui.print_colored("Available models:", "green")
                        for model in available_models:
                            self.ui.print_colored(f"  • {model}", "green")
                    else:
                        self.ui.print_colored("Could not retrieve model list", "yellow")
                    continue
                
                reply = self.generate_response(user_input)
                self.ui.print_response(reply)
                
            except (KeyboardInterrupt, EOFError):
                self.ui.print_divider()
                self.ui.print_slowly("\nStella: See you soon, okay? 🌼\n")
                break
            except Exception as e:
                self.ui.print_error(str(e))


if __name__ == "__main__":
    print("Starting Stella in FULL POWER mode...")
    print("This mode uses maximum available resources for the best experience.")
    stella = Stella()
    stella.run()
