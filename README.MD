# 🌟 Stella – Your Terminal Companion Bot
**Stella** is a kind and caring AI companion that lives in your terminal. She keeps track of your interactions through `memory.json`, gently reminds you to take breaks if you've been using your computer for too long, and chats with you like a good friend. (the reminding with you is still WIP) 

Stella is powered by the **LLaMA 3 model via Ollama**, making her fast, local, and private.

---
## 🛠 Features

🧠 Persistent memory with `memory.json`<br>
💬 Personalized chat with a gentle, thoughtful tone<br>
🖥️ Runs entirely in your terminal – no UI needed<br>

---

## 🚀 Setup Instructions
### 1. 📦 Install [Ollama](https://ollama.com)
Ollama is required to run LLaMA 3 locally.

#### **Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### **Mac**
```bash
brew install ollama
```
#### **Windows**
1. Download and install the latest [Ollama Windows installer](https://ollama.com/download/windows)
2. Make sure ollama is accessible in your terminal 

### 2. 🧠 Pull the LLaMA 3 model
```bash
ollama pull llama3
```
### 3. 🐍 Clone and run stella (i recommend you use virtual environtment)
```bash
git clone https://github.com/wanaxel/stella.git
cd stella
pip install -r requirements.txt
python stella.py
```
# Showcase 
<div align="center">
<img src="https://github.com/user-attachments/assets/c3500ba2-8eaf-437a-a6df-e9c97c555cce" width="750" height="200"/> <br>
<img src="https://github.com/user-attachments/assets/319dd4e0-77bb-46a0-af3a-ceeb97619355" width="750" height="200"/> <br>
<img src="https://github.com/user-attachments/assets/beea425a-f8a9-4a50-b981-959855c56772" width="750" height="200"/> <br>
</div>

---
<details>
  <summary>🧡 Credits</summary>

  Made with laziness and love by wanaxel.
</details>

