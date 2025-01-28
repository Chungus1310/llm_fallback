# LLM API Client

A flexible Python library for managing multiple LLM providers with automatic fallback support. This library provides a unified interface to interact with various LLM providers including Gemini, OpenRouter, HuggingFace, and NVIDIA, while gracefully handling failures and API issues.

## Features

- 🔄 Unified interface for multiple LLM APIs
- ⚡ Support for leading LLM providers:
  - Google's Gemini
  - OpenRouter
  - HuggingFace
  - NVIDIA
- 🔄 Automatic fallback between providers
- 🛠 Simple configuration through environment variables
- 📝 Built-in logging system
- 🎨 Customizable provider parameters

## Setup

1. Copy the library files into your project directory
2. Create a `.env` file in your project root:

```plaintext
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
NVIDIA_API_KEY=your_nvidia_api_key
```

3. Install the required dependencies:

```bash
pip install python-dotenv requests google-generativeai
```

## Usage

### Basic Usage

```python
from llm_api import LLMClient

# Initialize client with default providers
client = LLMClient()

try:
    response = client.generate(
        "Explain quantum computing in simple terms",
        temperature=0.5,
        max_tokens=500
    )
    print(response)
except Exception as e:
    print(f"Error: {str(e)}")
```

### Custom Provider Setup

```python
from llm_api import GeminiProvider, OpenRouterProvider

# Initialize specific providers
providers = [
    GeminiProvider(api_key="your_api_key"),
    OpenRouterProvider(api_key="your_api_key")
]

# Create client with custom providers
client = LLMClient(providers=providers)
```

### Provider Management

```python
from llm_api import LLMClient, HuggingFaceProvider

client = LLMClient()

# Add a new provider
new_provider = HuggingFaceProvider(api_key="your_api_key")
client.add_provider(new_provider)

# Replace all providers
client.set_providers([new_provider])
```

## Supported Models

### Gemini
- Default model: `gemini-1.5-flash-8b`

### OpenRouter
- Default model: `mistralai/mistral-7b-instruct:free`
- Access to various models through OpenRouter

### HuggingFace
- Default model: `mistralai/Mistral-Nemo-Instruct-2407`
- Compatible with deployed HuggingFace models

### NVIDIA
- Default model: `meta/llama-3.3-70b-instruct`

## Provider Configuration

Each provider accepts the following parameters in the `generate` method:
- `temperature`: Controls randomness (default: 0.7)
- `max_tokens`: Maximum response length
- `model`: Specific model to use

Example:
```python
response = client.generate(
    prompt="Your prompt here",
    temperature=0.5,
    max_tokens=1000,
    model="gemini-1.5-flash-8b"
)
```

## Error Handling

The library includes comprehensive error handling:
- Skips unavailable providers automatically
- Logs provider failures and successes
- Raises RuntimeError when all providers fail

## Contributing

Feel free to open issues or submit pull requests on GitHub: [llm_api](https://github.com/Chungus1310/llm_api)

## License

This project is open source - feel free to use and modify as needed!

## Author

Created by Chun ([GitHub](https://github.com/Chungus1310))
