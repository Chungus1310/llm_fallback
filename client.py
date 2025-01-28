import logging
from typing import Optional, List
from .providers import LLMProvider
from .exceptions import AllProvidersFailedError
from .utils import load_env

logging.basicConfig(level=logging.INFO)

class LLMClient:
    def __init__(self, providers: List[LLMProvider] = None):
        """
        Initialize the LLM client with a list of providers.
        If no providers are specified, default providers are used.
        """
        self.providers = providers or self._get_default_providers()
        
    def _get_default_providers(self) -> List[LLMProvider]:
        """Return the default list of providers."""
        from .providers import (
            GeminiProvider,
            OpenRouterProvider,
            HuggingFaceProvider,
            NvidiaProvider
        )
        return [
            GeminiProvider(),
            OpenRouterProvider(),
            HuggingFaceProvider(),
            NvidiaProvider()
        ]
        
    def add_provider(self, provider: LLMProvider):
        """Add a new provider to the client."""
        self.providers.append(provider)

    def set_providers(self, providers: List[LLMProvider]):
        """Replace the current list of providers with a new one."""
        self.providers = providers

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the available providers.
        Raises AllProvidersFailedError if all providers fail.
        """
        for provider in self.providers:
            if not provider.is_available():
                logging.warning(f"{provider.name} is not available")
                continue
                
            try:
                result = provider.generate(prompt, **kwargs)
                logging.info(f"Successfully used {provider.name}")
                return result
            except Exception as e:
                logging.warning(f"{provider.name} failed: {str(e)}")
                continue
                
        raise AllProvidersFailedError("All LLM providers failed")