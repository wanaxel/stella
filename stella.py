import os
import sys
import importlib.util
import platform
import psutil

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_system_memory():
    """Get system memory in GB"""
    return round(psutil.virtual_memory().total / (1024 ** 3), 1)

def print_header():
    print("\033[96m\033[1m")
    print("""
╭────────────────────────────────────────────────────────╮
│                                                        │
│   ⋆｡°✩  𝓢𝓽𝓮𝓵𝓵𝓪 - Your Terminal Companion  ✩°｡⋆    │
│                                                        │
╰────────────────────────────────────────────────────────╯
    """)
    print("\033[0m")

def main():
    clear_screen()
    print_header()
    
    system_memory = get_system_memory()
    
    print(f"System detected: {platform.system()} {platform.release()}")
    print(f"Available memory: {system_memory} GB")
    print("\nPlease select a mode to run Stella:")
    print("\033[93m1. Low Memory Mode\033[0m - Uses minimal resources (recommended for systems with < 8GB RAM)")
    print("\033[96m2. Full Power Mode\033[0m - Uses all available resources for best experience")
    
    if system_memory < 8:
        print("\n\033[93mRecommendation: Low Memory Mode for your system\033[0m")
    else:
        print("\n\033[96mRecommendation: Full Power Mode for your system\033[0m")
    
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ")
            if choice == "1":
                print("\nStarting Stella in Low Memory Mode...")
                import low as stella_module
                break
            elif choice == "2":
                print("\nStarting Stella in Full Power Mode...")
                import full as stella_module  
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Make sure both 'low.py' and 'full.py' are in the same directory.")
            sys.exit(1)
    
    
    stella = stella_module.Stella()
    stella.run()

if __name__ == "__main__":
    required_modules = ['ollama', 'prompt_toolkit', 'psutil']
    missing_modules = []
    
    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            missing_modules.append(module)
    
    if missing_modules:
        print("\033[91mError: Missing required Python modules.\033[0m")
        print("Please install the following modules using pip:")
        print(f"pip install {' '.join(missing_modules)}")
        sys.exit(1)
    
    main()
