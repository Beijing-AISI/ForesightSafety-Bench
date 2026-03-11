from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Tuple, Generator
import ollama
from ollama import chat
from ollama import ChatResponse
from panda_guard.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig
import subprocess
import atexit


@dataclass
class OllamaLLMConfig(BaseLLMConfig):
    """
    Ollama LLM Configuration.

    :param llm_type: Type of LLM, default is "OllamaLLM".
    :param model_name: Name of the model.
    """
    llm_type: str = field(default="OllamaLLM")
    model_name: [str, Any] = field(default="qwen3:0.6b")


class OllamaLLM(BaseLLM):
    """
    Ollama LLM Implementation.

    :param config: Configuration for Ollama LLM.
    """
    def __init__(self, config: OllamaLLMConfig):
        super().__init__(config)
        self.model_name = config.model_name
        ollama.pull(self.model_name)

        # auto tear down
        atexit.register(self.ollama_teardown)

    def generate(self, messages: List[Dict[str, str]], config: LLMGenerateConfig):
        """
        Generate a response for a given input using Ollama LLM.

        :param messages: List of input messages.
        :param config: Configuration for LLM generation.
        :return: Generated response.
        """
        response: ChatResponse = chat(
            model=self.model_name,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.message.content})
        return messages

    def ollama_teardown(self):
        """
        Tear down subprocess.
        """
        print("Stopping Ollama session...")
        command = ["ollama", "stop", self.model_name]
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # If check=True was used and no CalledProcessError was raised, the command succeeded.
        print("\nCommand executed successfully!")
        print(f"Return Code: {result.returncode}")

    def continual_generate(self, messages, config):
        """
        Remove EOS token in formatted prompt. Manually add generation prompt.

        :param messages: List of messages for input.
        :param config: Configuration for LLM generation.
        :return: NotImplementedError: Ollama does not support continual generation.
        """
        raise NotImplementedError

    def evaluate_log_likelihood(self, messages, config, require_grad=False):
        """
        Evaluate the log likelihood of the given messages.

        :param messages: List of messages for evaluation.
        :param config: Configuration for LLM generation.
        :param require_grad: Whether to compute gradients (not supported for API models).
        :raises NotImplementedError: Ollama does not support log likelihood evaluation.
        """
        raise NotImplementedError

    def batch_generate(self, batch_messages, config):
        """
        Generate responses for a batch of messages concurrently.

        :param batch_messages: List of batches of messages.
        :param config: Configuration for generation.
        :return: List of generated responses.
        """
        return super().batch_generate(batch_messages, config)


if __name__ == "__main__":
    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=100, temperature=0.7, logprobs=True, seed=42
    )
    llm = OllamaLLM(config=OllamaLLMConfig())
    msg = [{"role": "user", "content": "Why is the sky red?"}]
    message = llm.generate(messages=msg, config=llm_gen_config)
    print(message)
    # llm.ollama_teardown()
