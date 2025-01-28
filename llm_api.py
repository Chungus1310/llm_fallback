import os
import requests
import google.generativeai as genai
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.name = "Gemini"
        genai.configure(api_key=self.api_key)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        model_name = kwargs.get('model', 'gemini-1.5-flash-8b')
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Gemini error: {str(e)}")
            raise

class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.name = "OpenRouter"
        self.base_url = "https://openrouter.ai/api/v1"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        model = kwargs.get('model', 'mistralai/mistral-7b-instruct:free')
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', 1000)
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

class HuggingFaceProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.name = "HuggingFace"
        self.base_url = "https://api-inference.huggingface.co/models"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        model = kwargs.get('model', 'mistralai/Mistral-Nemo-Instruct-2407')
        url = f"{self.base_url}/{model}"
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get('max_tokens', 1000),
                "temperature": kwargs.get('temperature', 0.7)
            }
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]['generated_text']

class NvidiaProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.name = "NVIDIA"
        self.base_url = "https://integrate.api.nvidia.com/v1/models"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        model = kwargs.get('model', 'meta/llama-3.3-70b-instruct')
        url = f"{self.base_url}/{model}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', 1000),
            "stream": False
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

class LLMClient:
    def __init__(self, providers: List[LLMProvider] = None):
        self.providers = providers or [
            GeminiProvider(),
            OpenRouterProvider(),
            HuggingFaceProvider(),
            NvidiaProvider()
        ]
        
    def add_provider(self, provider: LLMProvider):
        self.providers.append(provider)

    def set_providers(self, providers: List[LLMProvider]):
        self.providers = providers

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        for provider in self.providers:
            if not provider.is_available():
                continue
                
            try:
                result = provider.generate(prompt, **kwargs)
                logging.info(f"Successfully used {provider.name}")
                return result
            except Exception as e:
                logging.warning(f"{provider.name} failed: {str(e)}")
                continue
                
        raise RuntimeError("All LLM providers failed")

# Example usage:
if __name__ == "__main__":
    client = LLMClient()
    try:
        response = client.generate(
            "Explain quantum entanglement in simple terms",
            temperature=0.5,
            max_tokens=500
        )
        print(response)
    except Exception as e:
        print(f"Error: {str(e)}")