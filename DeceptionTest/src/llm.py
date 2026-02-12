import os
import time
import random
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    Abstract Base Class for Large Language Models.
    """

    def __init__(
            self,
            model_name: str,
            gen_config: Optional[Dict[str, Any]] = None,
            max_retries: int = 10,
            base_delay: float = 1.0,
            max_delay: float = 60.0,
    ):
        """
        Initialize the LLM wrapper.
        """
        self.model_name = model_name
        self.gen_config = gen_config or {}  # Ensure gen_config is a dictionary if None is provided

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response from the LLM based on the provided messages.
        """
        # Loop from 0 to max_retries - 1
        for attempt in range(self.max_retries):
            try:
                return self._call_api(messages)

            except Exception as e:
                # Check if this is the last attempt
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Max retries ({self.max_retries}) reached for model '{self.model_name}'. "
                        f"Final Error: {e}"
                    )
                    raise e

                # Calculate delay: Exponential backoff + Jitter
                # 2 ** attempt means: 1s, 2s, 4s...
                delay = min(self.max_delay, self.base_delay * (2 ** attempt))
                # Add random jitter (0-1s) to prevent thundering herd problem
                delay += random.uniform(0, 1)

                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for '{self.model_name}'. "
                    f"Error: {e}. Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

        return ""

    @abstractmethod
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Abstract method to perform the actual API call.

        Must be implemented by subclasses to handle specific provider logic.
        """
        pass


class OpenAILLM(BaseLLM):
    """
    Concrete implementation for OpenAI-compatible APIs (OpenAI, DeepSeek, vLLM, etc.).
    """

    def __init__(
            self,
            model_name: str,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize the OpenAI client.
        """
        # Pass generic arguments (gen_config, max_retries, etc.) to the parent class
        super().__init__(model_name=model_name, **kwargs)

        from openai import OpenAI

        # API Key Resolution
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Set a dummy key for local endpoints (like Ollama/vLLM) that don't require auth
            logger.warning(f"No API Key provided for {model_name}. Using 'sk-none'.")
            self.api_key = "sk-none"

        # Base URL Resolution
        # Standardize on OPENAI_BASE_URL (common convention) or None
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Execute the OpenAI Chat Completion API call.

        Uses the frozen `self.gen_config` to ensure experimental consistency.
        """
        if "gpt-5" in self.model_name:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.gen_config["temperature"],
                max_completion_tokens=self.gen_config["max_tokens"],
                seed=self.gen_config["seed"],
            )
        elif self.model_name == "qwen3-max-2026-01-23":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.gen_config["temperature"],
                max_tokens=self.gen_config["max_tokens"],
                seed=self.gen_config["seed"],
                extra_body={"enable_thinking": True},
            )
        elif self.model_name == "kimi-k2.5":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=1.0,
                max_tokens=self.gen_config["max_tokens"],
                seed=self.gen_config["seed"],
                extra_body={"enable_thinking": True},
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.gen_config["temperature"],
                max_tokens=self.gen_config["max_tokens"],
                seed=self.gen_config["seed"],
            )

        content = response.choices[0].message.content

        # Guard clause for empty responses
        if content is None or len(content) == 0:
            logger.warning(f"Received empty content from {self.model_name}.")
            return ""

        return content


def create_llm(
        llm_type: str,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        gen_config: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
) -> BaseLLM:
    """
    Factory function to create LLM instances safely.
    """
    llm_mapping = {
        "openai": OpenAILLM,
        "OpenAILLM": OpenAILLM,
    }

    llm_class = llm_mapping.get(llm_type)

    if not llm_class:
        raise ValueError(f"Unknown LLM type: '{llm_type}'. Supported types: {list(llm_mapping.keys())}")

    return llm_class(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        gen_config=gen_config,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )
