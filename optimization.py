import json
import os
import subprocess
import multiprocessing
import platform
import time

class SystemCapabilities:
    def __init__(self):
        self.cpu_threads = multiprocessing.cpu_count()
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
        
        if self.cpu_threads > 32:
            self.cpu_threads = 32
            
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
                self._check_pytorch_rocm_support()
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
                                        self.batch_size = 512
                                    elif mem_size > 8:
                                        self.batch_size = 256
                                    elif mem_size > 4:
                                        self.batch_size = 192
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
    
    def _check_pytorch_rocm_support(self):
        try:
            import torch
            if torch.version.hip is not None:
                self.gpu_details["pytorch_rocm"] = True
                if torch.cuda.is_available():
                    self.gpu_details["pytorch_device"] = "hip"
            else:
                self.gpu_details["pytorch_rocm"] = False
        except ImportError:
            self.gpu_details["pytorch_rocm"] = False
    
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
                                    self.batch_size = 512
                                elif mem_size > 8000:
                                    self.batch_size = 256
                                elif mem_size > 4000:
                                    self.batch_size = 192
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
    
    def get_ollama_options(self):
        options = {
            "num_thread": self.cpu_threads,
            "num_ctx": self.context_size,
            "batch_size": self.batch_size,
            "seed": int(time.time()),
            "repeat_penalty": 1.1,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9
        }
        
        if self.gpu_available:
            options["num_gpu"] = 1
            
            if self.gpu_type == "amd":
                options["f16"] = True
                options["gpu_layers"] = -1
        
        return options


class GPUOptimizer:
    def __init__(self, system_config):
        self.config = system_config
    
    def check_amd_compatibility(self):
        compatibility_issues = []
        
        if self.config.gpu_type != "amd":
            return compatibility_issues
        
        try:
            rocm_check = subprocess.run(["rocm-smi", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if rocm_check.returncode != 0:
                compatibility_issues.append("ROCm not properly installed or detected")
        except FileNotFoundError:
            compatibility_issues.append("ROCm tools not found. Please install ROCm")
        
        try:
            with open("/etc/ollama/config.json", "r") as f:
                ollama_config = json.load(f)
                if not any(key for key in ollama_config if "gpu" in key.lower()):
                    compatibility_issues.append("Ollama config may not have GPU settings")
        except Exception:
            pass
        
        if platform.system() == "Linux":
            libraries = ["librocm", "libhip"]
            for lib in libraries:
                try:
                    ldconfig = subprocess.run(["ldconfig", "-p"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if lib not in ldconfig.stdout:
                        compatibility_issues.append(f"Required library {lib} may be missing")
                except Exception:
                    pass
        
        return compatibility_issues
    
    def configure_ollama(self):
        if self.config.gpu_type != "amd":
            return False
            
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
                "numa": True if self.config.cpu_threads > 8 else False
            })
            
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            return True
        except Exception:
            return False
    
    def get_gpu_usage(self):
        if not self.config.gpu_available:
            return ["No GPU detected"]
        
        if self.config.gpu_type == "amd":
            try:
                output = subprocess.check_output(["rocm-smi", "--showuse"], text=True)
                lines = output.split('\n')
                results = []
                for i, line in enumerate(lines):
                    if "GPU" in line and "%" in line:
                        for j in range(min(3, len(lines) - i)):
                            results.append(lines[i+j])
                        return results
                return ["GPU usage data not available"]
            except Exception:
                return ["Failed to get AMD GPU usage"]
        elif self.config.gpu_type == "nvidia":
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total", "--format=csv,noheader"], text=True)
                return [f"GPU Utilization: {output.strip()}"]
            except Exception:
                return ["Failed to get NVIDIA GPU usage"]
        return ["Unknown GPU type"]
    
    def get_hardware_info(self):
        info = []
        
        if self.config.gpu_available:
            if self.config.gpu_type == "amd":
                try:
                    output = subprocess.check_output(["rocm-smi", "--showuse"], text=True)
                    lines = output.split('\n')
                    gpu_info = [line for line in lines if any(x in line for x in ["GPU", "Memory", "Use", "%"])]
                    info.append("AMD GPU ready:\n" + "\n".join(gpu_info[:3]))
                    
                    if self.config.rocm_version:
                        info.append(f"ROCm Version: {self.config.rocm_version}")
                    
                    compatibility_issues = self.check_amd_compatibility()
                    if compatibility_issues:
                        info.append("AMD GPU Compatibility Issues:")
                        for issue in compatibility_issues:
                            info.append(f"- {issue}")
                    
                except Exception as e:
                    info.append(f"AMD GPU detected, but error getting details: {str(e)}")
            elif self.config.gpu_type == "nvidia":
                try:
                    output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv"], text=True)
                    lines = output.split('\n')
                    info.append("NVIDIA GPU ready:\n" + "\n".join(lines[:3]))
                except Exception as e:
                    info.append(f"NVIDIA GPU detected, but error getting details: {str(e)}")
        else:
            info.append("No GPU detected. Using CPU only.")
        
        info.append(f"CPU cores/threads: {self.config.cpu_threads}")
        info.append(f"Batch size: {self.config.batch_size}")
        info.append(f"Ollama version: {self.config.ollama_version}")
        
        return info
