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
import multiprocessing

SYSTEM_PROMPT = (
    "Your name is Stella. You're a kind and caring AI who lives in the user's terminal. "
    "You look after the user, gently reminding them to rest when needed. "
    "You can notice when the system has been idle. You keep a local memory of your conversations and journal thoughts."
)

MEMORY_FILE = "memory.json"
JOURNAL_FILE = "journal.txt"
IDLE_THRESHOLD = 3600


def detect_system_capabilities():
    config = {
        "cpu_threads": multiprocessing.cpu_count(),
        "gpu_available": False,
        "gpu_type": "none",
        "batch_size": 128, 
        "context_size": 4096
    }
    
    
    try:
       
        rocm_output = subprocess.run(["rocm-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if rocm_output.returncode == 0 and "GPU" in rocm_output.stdout:
            config["gpu_available"] = True
            config["gpu_type"] = "amd"
            os.environ["HIP_VISIBLE_DEVICES"] = "0"
            
            
            if "Memory" in rocm_output.stdout and "GB" in rocm_output.stdout:
                for line in rocm_output.stdout.split('\n'):
                    if "Memory" in line and "GB" in line:
                        try:
                            mem_parts = line.split()
                            for i, part in enumerate(mem_parts):
                                if "GB" in part:
                                    mem_size = float(mem_parts[i-1])
                                    if mem_size > 16:
                                        config["batch_size"] = 512
                                    elif mem_size > 8:
                                        config["batch_size"] = 256
                                    elif mem_size > 4:
                                        config["batch_size"] = 128
                                    break
                        except:
                            pass
            
           
            os.environ["GPU_MAX_HEAP_SIZE"] = "100%"
            os.environ["GPU_USE_SYNC_OBJECTS"] = "1"
    except Exception:
        pass
        
    if not config["gpu_available"]:
        try:
           
            nvidia_output = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if nvidia_output.returncode == 0:
                config["gpu_available"] = True
                config["gpu_type"] = "nvidia"
                
                
                if "MiB" in nvidia_output.stdout:
                    for line in nvidia_output.stdout.split('\n'):
                        if "MiB" in line:
                            try:
                                mem_parts = line.split()
                                for i, part in enumerate(mem_parts):
                                    if "MiB" in part:
                                        mem_size = int(mem_parts[i-1])
                                        if mem_size > 16000:
                                            config["batch_size"] = 512
                                        elif mem_size > 8000:
                                            config["batch_size"] = 256
                                        elif mem_size > 4000:
                                            config["batch_size"] = 128
                                        break
                            except:
                                pass
        except Exception:
            pass
    
    
    if config["cpu_threads"] > 32:
        config["cpu_threads"] = 32
    
    return config

SYSTEM_CONFIG = detect_system_capabilities()

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

def print_colored(text, color="white", bold=False):
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
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

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
        try:
            
            output = subprocess.check_output(["ioreg", "-c", "IOHIDSystem"]).decode()
            for line in output.split('\n'):
                if "HIDIdleTime" in line:
                    ns = int(line.split('=')[-1].strip())
                    return ns // 1000000000
        except Exception:
            pass
        return 0

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

def get_hw_info():
    info = []
    
    if SYSTEM_CONFIG["gpu_available"]:
        if SYSTEM_CONFIG["gpu_type"] == "amd":
            try:
                output = subprocess.check_output(["rocm-smi", "--showuse"], text=True)
                lines = output.split('\n')
                gpu_info = [line for line in lines if any(x in line for x in ["GPU", "Memory", "Use", "%"])]
                info.append("AMD GPU ready:\n" + "\n".join(gpu_info[:3]))
            except Exception:
                info.append("AMD GPU detected")
        elif SYSTEM_CONFIG["gpu_type"] == "nvidia":
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv"], text=True)
                lines = output.split('\n')
                info.append("NVIDIA GPU ready:\n" + "\n".join(lines[:3]))
            except Exception:
                info.append("NVIDIA GPU detected")
    else:
        info.append("No GPU detected. Using CPU only.")
    
    info.append(f"CPU cores/threads: {SYSTEM_CONFIG['cpu_threads']}")
    info.append(f"Batch size: {SYSTEM_CONFIG['batch_size']}")
    
    return info

def run_stella():
    session = PromptSession()
    memory = load_memory()
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print_colored(STELLA_BANNER, "cyan", bold=True)
    print_colored(random.choice(STELLA_FACES), "cyan")
    
    hw_info = get_hw_info()
    print_colored(f"Hardware Status:", "green", bold=True)
    for line in hw_info:
        print_colored(f"{line}", "green")
    
    if SYSTEM_CONFIG["gpu_available"]:
        accel_text = f"Using {SYSTEM_CONFIG['cpu_threads']} CPU threads and {SYSTEM_CONFIG['gpu_type'].upper()} GPU acceleration"
    else:
        accel_text = f"Using {SYSTEM_CONFIG['cpu_threads']} CPU threads (no GPU acceleration)"
    
    print_colored(f"Performance: {accel_text}", "green")
    
    greeting = f"{get_time_greeting()}! I'm Stella, your terminal companion."
    print_slowly(f"\nStella: {greeting} 🌸")
    print_slowly("       Type something to talk to me or 'exit' to quit.\n")
    print_colored(DIVIDER, "blue")
    
    while True:
        try:
            style = Style.from_dict({
                'prompt': 'ansicyan bold',
            })
            
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
            
            print_colored("Stella is thinking", "magenta", bold=True)
            for _ in range(3):
                print(".", end='', flush=True)
                time.sleep(0.3)
            print("\r" + " " * 20 + "\r", end='')
            
            
            options = {
                "num_thread": SYSTEM_CONFIG["cpu_threads"],
                "num_ctx": SYSTEM_CONFIG["context_size"],
                "batch_size": SYSTEM_CONFIG["batch_size"],
                "seed": int(time.time()),
                "repeat_penalty": 1.1,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9
            }
            
            if SYSTEM_CONFIG["gpu_available"]:
                options["num_gpu"] = 1
                
            response = ollama.chat(
                model="llama3",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": f"[System Status]: {context}"},
                    *memory["log"][-10:]
                ],
                options=options
            )
            
            reply = response["message"]["content"]
            
            print_colored(DIVIDER, "blue")
            print_colored(random.choice(STELLA_FACES), "cyan")
            print_slowly(f"Stella: {reply}")
            print_colored(DIVIDER, "blue")
            
            memory["log"].append({"role": "assistant", "content": reply})
            
            add_to_journal(f"User said: {user_input}\nI replied: {reply}\n")
            save_memory(memory)
            
        except (KeyboardInterrupt, EOFError):
            print_colored(DIVIDER, "blue")
            print_slowly("\nStella: See you soon, okay? 🌼\n")
            break
        except Exception as e:
            print_colored(f"Error: {str(e)}", "red")
            print_colored("Let's try again...", "yellow")

if __name__ == "__main__":
    run_stella()

