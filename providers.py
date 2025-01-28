import os
import requests
import google.generativeai as genai
from openai import OpenAI
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from .utils import load_env

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
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        model_name = kwargs.get('model', 'gemini-1.5-flash')
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini error: {str(e)}")

class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.name = "OpenRouter"
        self.base_url = "https://openrouter.ai/api/v1"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        model = kwargs.get('model', 'google/gemma-2-9b-it:free')
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get('temperature', 0.7),
            "max_tokens": kwargs.get('max_tokens', 500)
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
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        model = kwargs.get('model', 'meta/llama-3.3-70b-instruct')
        
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000),
                top_p=kwargs.get('top_p', 0.7),
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"NVIDIA API error: {str(e)}")
