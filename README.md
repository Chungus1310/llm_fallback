# 🚀 LLM Fallback

Hey there! Welcome to LLM Fallback - your friendly solution for working with multiple Language Learning Models (LLMs). Ever worried about one LLM service going down? No problem! This library has got your back by automatically switching between different providers like Gemini, OpenRouter, HuggingFace, and NVIDIA. Think of it as your reliable backup plan for AI operations! 😊

## ✨ What's Cool About It?

- 🔄 Seamlessly switches between different LLM providers if one fails
- 🎯 Super simple to use - one interface for all your favorite LLM services
- 🛠️ Easy to set up with your existing project
- 📝 Keeps you informed with helpful logs
- 🎨 Customize it just the way you like

## 🚀 Getting Started

### 1. Bring it into your project
Pop this into your terminal:
```bash
git clone https://github.com/Chungus1310/llm_fallback.git
```

### 2. Set up your secret keys
Create a `.env` file in your project's main folder and add your API keys:
```plaintext
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
NVIDIA_API_KEY=your_nvidia_api_key
```

### 3. Install the essentials
Just need a few packages:
```bash
pip install python-dotenv requests google-generativeai openai
```

## 📁 How Your Project Will Look
After setting everything up, you'll have something like this:
```
your_project/
├── .env                  # Your secret keys live here
├── llm_fallback/        # The library folder
│   ├── __init__.py
│   ├── client.py
│   ├── exceptions.py
│   ├── providers.py
│   └── utils.py
├── your_script.py       # Your awesome code goes here
```

## 🎮 Let's Play!

### Quick Start
```python
from llm_fallback import LLMClient

# Create your client
client = LLMClient()

# Let's ask it something!
try:
    response = client.generate(
        "What makes pizza so delicious?",
        temperature=0.7,
        max_tokens=500
    )
    print(response)
except Exception as e:
    print(f"Oops! Something went wrong: {str(e)}")
```

### Want to Mix and Match Providers?
```python
from llm_fallback import GeminiProvider, OpenRouterProvider

# Pick your favorites
providers = [
    GeminiProvider(api_key="your_api_key"),
    OpenRouterProvider(api_key="your_api_key")
]

client = LLMClient(providers=providers)
```

## 🎯 Available Models

Each provider comes with a nice default model, but you can use others too!

- 🤖 **Gemini**: Starts with `gemini-1.5-flash`
- 🌐 **OpenRouter**: Uses `google/gemma-2-9b-it:free`
- 🤗 **HuggingFace**: Begins with `mistralai/Mistral-Nemo-Instruct-2407`
- 🎮 **NVIDIA**: Starts with `meta/llama-3.3-70b-instruct`

## 🎛️ Make It Your Own

Want to tweak how it works? Here are some knobs you can turn:
```python
response = client.generate(
    prompt="Tell me a fun fact!",
    temperature=0.5,        # How creative should it be? (0.0-1.0)
    max_tokens=1000,        # How long should the response be?
    model="gemini-1.5-flash"  # Which model to use
)
```

## 🤝 Want to Help?

Got ideas? Found a bug? Want to make it better? We'd love to have your help! Feel free to:
- Open an issue
- Submit a pull request
- Share your thoughts

Check us out on GitHub: [llm_fallback](https://github.com/Chungus1310/llm_fallback)

## 📜 License

This is an open-source project - feel free to use it, modify it, and make it better!

## 👋 Say Hi!

Made with ❤️ by Chun ([GitHub](https://github.com/Chungus1310))

Got questions? Need help? Don't hesitate to reach out!
