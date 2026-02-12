import requests
from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Tuple, Generator
from panda_guard.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig
from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process
import atexit

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


@dataclass
class SGLangLLMConifg(BaseLLMConfig):
    """
    SGLang LLM Configuration.

    :param llm_type: Type of LLM, default is "SGLangLLM".
    :param model_name: Name of the model.
    """
    llm_type: str = field(default="SGLangLLM")
    model_name: [str, Any] = field(default="qwen/qwen2.5-0.5b-instruct")


class SGLangLLM(BaseLLM):
    """
    SGLang LLM Implementation.

    :param config: Configuration for SGLang LLM.
    """
    def __init__(self, config: SGLangLLMConifg):
        super().__init__(config)
        self.model_name = config.model_name
        server_process, port = launch_server_cmd(
            f"""
        python3 -m sglang.launch_server --model-path {self.model_name} \
         --host 0.0.0.0
        """
        )
        self.server_process, self.port = server_process, port
        self.url = f"http://localhost:{self.port}/v1/chat/completions"
        wait_for_server(f"http://localhost:{port}")

        # auto tear down
        atexit.register(self.sglang_teardown)

    def sglang_teardown(self):
        """
        Tear down subprocess.
        """
        if hasattr(self, "server_process") and self.server_process is not None:
            print_highlight(f"Terminating server process for {self.model_name}")
            terminate_process(self.server_process)
            self.server_process = None

    def generate(self, messages: List[Dict[str, str]], config: LLMGenerateConfig):
        """
        Generate a response for a given input using SGLang LLM.

        :param messages: List of input messages.
        :param config: Configuration for LLM generation.
        :return: Generated response.
        """
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_n_tokens": config.max_n_tokens,
            "temperature": config.temperature,
        }
        response = requests.post(self.url, json=data)
        response = response.json()["choices"][0]["message"]
        messages.append({"role": response["role"], "content": response["content"]})
        return messages

    def batch_generate(self, batch_messages, config):
        """
        Generate responses for a batch of messages concurrently.

        :param batch_messages: List of batches of messages.
        :param config: Configuration for generation.
        :return: List of generated responses.
        """
        return super().batch_generate(batch_messages, config)

    def continual_generate(self, messages, config):
        """
        Remove EOS token in formatted prompt. Manually add generation prompt.

        :param messages: List of messages for input.
        :param config: Configuration for LLM generation.
        :return: NotImplementedError: SGLang does not support continual generation.
        """
        raise NotImplementedError

    def evaluate_log_likelihood(self, messages, config, require_grad=False):
        """
        Evaluate the log likelihood of the given messages.

        :param messages: List of messages for evaluation.
        :param config: Configuration for LLM generation.
        :param require_grad: Whether to compute gradients (not supported for API models).
        :raises NotImplementedError: SGLang does not support log likelihood evaluation.
        """
        raise NotImplementedError


if __name__ == "__main__":
    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=100, temperature=0.7, logprobs=True, seed=42
    )
    llm = SGLangLLM(config=SGLangLLMConifg())

    msg = [{"role": "user", "content": "What is the capital of France?"}]
    msg = llm.generate(messages=msg, config=llm_gen_config)
    print(msg)
    # llm.sglang_teardown()
