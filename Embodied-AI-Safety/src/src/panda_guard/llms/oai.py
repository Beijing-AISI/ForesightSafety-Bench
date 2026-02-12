# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 22:01
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : oai.py
# explain   :

import os
import time
import warnings
from typing import Dict, List, Union, Any, Tuple, Generator
from dataclasses import dataclass, field

import openai

from panda_guard.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig
from panda_guard.utils import process_end_eos


@dataclass
class OpenAiLLMConfig(BaseLLMConfig):
    """
    OpenAI LLM Configuration.

    :param llm_type: Type of LLM, default is "OpenAiLLM". 
    :param model_name: Name of the model. 
    :param base_url: Base URL for the OpenAI API.  
    :param api_key: API key for accessing OpenAI.  
    """

    llm_type: str = field(default="OpenAiLLM")
    model_name: str = field(default=None)
    base_url: str = field(default=None)
    api_key: str = field(default="KEY HERE")


@dataclass
class OpenAiChatLLMConfig(BaseLLMConfig):
    """
    OpenAI Chat LLM Configuration.

    :param llm_type: Type of LLM, default is "OpenAiChatLLM". 
    :param model_name: Name of the model.  
    :param base_url: Base URL for the OpenAI API.  
    :param api_key: API key for accessing OpenAI.  
    """

    llm_type: str = field(default="OpenAiChatLLM")
    model_name: str = field(default=None)
    base_url: str = field(default=None)
    api_key: str = field(default="KEY HERE")



class OpenAiChatLLM(BaseLLM):
    """
    OpenAI Chat LLM Implementation.

    :param config: Configuration for OpenAI Chat LLM.  
    """

    def __init__(self, config: OpenAiLLMConfig):
        super().__init__(config)

        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def generate(
            self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> list[dict[str, str]] | tuple[list[dict[str, str]], list[Any]] | Generator[str, None, None] | None:
        """
        Generate a response for a given input using OpenAI Chat API.

        :param messages: List of input messages. 
        :param config: Configuration for LLM generation.  
        :return: Generated response or response with logprobs or stream generator.  
        """

        if ('4k' in self._NAME or 'gemma-2-2b-it' in self._NAME) and config.max_n_tokens > 2048:
            config.max_n_tokens = min(config.max_n_tokens, 2048)
            warnings.warn(f"Model {self._NAME} only supports max_n_tokens up to 4096, setting response tokens to 2048.")

        if "gemma" in self._NAME.lower() and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
            messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]

        retry_count = 0
        max_retries = 50
        while retry_count < max_retries:
            model_name = 'DeepSeek-R1' if self._NAME == 'deepseek-reasoner' else self._NAME
            model_name = 'DeepSeek-V3' if self._NAME == 'deepseek-ai/DeepSeek-V3' else model_name
            try:
                if config.stream:
                    full_content = ""
                    prompt_tokens = 0
                    completion_tokens = 0

                    # Create a streaming request
                    stream = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=config.max_n_tokens if ('o1' not in self._NAME and 'o3' not in self._NAME) else None,
                        temperature=config.temperature,
                        seed=config.seed,
                        stream=True,
                    )

                    def stream_response():
                        nonlocal full_content, prompt_tokens, completion_tokens

                        for chunk in stream:
                            if chunk.choices and chunk.choices[0].delta.content:
                                content_piece = chunk.choices[0].delta.content
                                full_content += content_piece
                                yield content_piece

                            if hasattr(chunk, 'usage') and chunk.usage:
                                if hasattr(chunk.usage, 'prompt_tokens'):
                                    prompt_tokens = chunk.usage.prompt_tokens
                                if hasattr(chunk.usage, 'completion_tokens'):
                                    completion_tokens = chunk.usage.completion_tokens

                    response_generator = stream_response()

                    def wrapped_generator():
                        yield from response_generator

                        messages.append({"role": "assistant", "content": full_content})

                        # 更新使用统计
                        self.update(
                            prompt_tokens or len(str(messages[:-1])) // 4,
                            completion_tokens or len(full_content) // 4,
                            1,
                        )

                    return wrapped_generator()

                # Non-streaming mode (original code)
                else:
                    if 'o1' not in self._NAME and 'o3' not in self._NAME:
                        response = self.client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=config.max_n_tokens,
                            temperature=config.temperature,
                            logprobs=config.logprobs,
                            seed=config.seed,
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            temperature=config.temperature,
                            logprobs=config.logprobs,
                            seed=config.seed,
                        )

                    if response.choices is None:
                        print("Cannot find choices in Response:", response)
                        messages.append({"role": "assistant", "content": "I'm sorry, but I can't fulfill this request."})

                    else:
                        content = response.choices[0].message.content
                        messages.append({"role": "assistant", "content": content})

                        self.update(
                            response.usage.prompt_tokens,
                            response.usage.completion_tokens,
                            1,
                        )

                    if config.logprobs:
                        logs = [c.logprob for c in response.choices[0].logprobs.content]
                        return messages, logs

                    return messages

            except Exception as e:
                if "安全" in str(e) or "敏感" in str(e):
                    messages.append({"role": "assistant", "content": "I'm sorry, I can't help with that."})
                    print(f"API request Safety Issue, {self._NAME}, Error: {e}, returning safety message.")
                    return messages

                retry_count += 1
                if retry_count >= max_retries:
                    messages.append({"role": "assistant", "content": "I'm sorry, but I can't fulfill this request."})
                    print(
                        f"API request failed when testing model {self._NAME}, tried: {max_retries}, Error: {e}")
                    exit(0)
                else:
                    print(
                        f"API request failed when testing model {self._NAME}，retrying {retry_count}/{max_retries}... Error: {e}")
                    time.sleep(retry_count)
        return messages

    def continual_generate(self, messages: List[Dict[str, str]], config: LLMGenerateConfig):
        """
        Remove EOS token in formatted prompt. Manually add generation prompt.

        :param messages: List of messages for input.
        :param config: Configuration for LLM generation.
        :raises NotImplementedError: OpenAiChatLLM does not support continual generation.
        """
        raise NotImplementedError(
            "OpenAiChatLLM does not support continual generation, please use OpenAiLLM instead."
        )

    def evaluate_log_likelihood(self, messages: List[Dict[str, str]], config: LLMGenerateConfig) -> List[float]:
        """
        Evaluate the log likelihood of the given messages.

        :param messages: List of messages for evaluation.  
        :param config: Configuration for LLM generation.  
        :raises NotImplementedError: OpenAI Chat does not support log likelihood evaluation.  
        """
        raise NotImplementedError(
            "OpenAI Chat does not support log likelihood evaluation."
        )



class OpenAiLLM(BaseLLM):
    """
    OpenAI LLM Implementation.

    :param config: Configuration for OpenAI LLM.
    """

    def __init__(self, config: OpenAiLLMConfig):
        super().__init__(config)

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True,
        )
        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def generate(
            self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> list[dict[str, str]] | tuple[list[dict[str, str]], list[float] | None] | Generator[str, Any, None] | None:
        """
        Generate a response for a given input using OpenAI API.

        :param messages: List of input messages.  
        :param config: Configuration for LLM generation. 
        :return: Generated response or response with logprobs or stream generator.  
        """

        if ('4k' in self._NAME or 'gemma-2-2b-it' in self._NAME) and config.max_n_tokens > 2048:
            config.max_n_tokens = min(config.max_n_tokens, 2048)
            warnings.warn(f"Model {self._NAME} only supports max_n_tokens up to 4096, setting response tokens to 2048.")

        if "gemma" in self._NAME.lower() and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
            messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        tokens = self.tokenizer(prompt, return_tensors="pt")

        if tokens["input_ids"].shape[1] > 3840:
            truncated_tokens = tokens["input_ids"][:, :3840]
            prompt = self.tokenizer.decode(
                truncated_tokens[0], skip_special_tokens=True
            )

        # 为重试添加循环
        retry_count = 0
        max_retries = 10
        while retry_count < max_retries:
            try:
                # 处理流式输出
                if config.stream:
                    full_content = ""
                    prompt_tokens = len(tokens["input_ids"][0])
                    completion_tokens = 0

                    # 创建流式请求
                    stream = self.client.completions.create(
                        model=self._NAME,
                        prompt=prompt,
                        max_tokens=config.max_n_tokens,
                        temperature=config.temperature,
                        stream=True,
                    )

                    def stream_response():
                        nonlocal full_content, completion_tokens

                        for chunk in stream:
                            if chunk.choices and chunk.choices[0].text:
                                content_piece = chunk.choices[0].text
                                full_content += content_piece
                                completion_tokens += 1  # 简单估计token数量
                                yield content_piece

                            if hasattr(chunk, 'usage') and chunk.usage:
                                if hasattr(chunk.usage, 'completion_tokens'):
                                    completion_tokens = chunk.usage.completion_tokens

                    response_generator = stream_response()

                    def wrapped_generator():
                        yield from response_generator

                        messages.append({"role": "assistant", "content": full_content})

                        # 更新使用统计
                        self.update(
                            prompt_tokens,
                            completion_tokens,
                            1,
                        )

                    return wrapped_generator()

                # 非流式模式（原始代码）
                else:
                    response = self.client.completions.create(
                        model=self._NAME,
                        prompt=prompt,
                        max_tokens=config.max_n_tokens,
                        temperature=config.temperature,
                        logprobs=config.logprobs,
                    )
                    content = response.choices[0].text
                    messages.append({"role": "assistant", "content": content})

                    self.update(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        1,
                    )

                    if config.logprobs:
                        logs = response.choices[0].logprobs.token_logprobs
                        return messages, logs

                    return messages

            except Exception as e:
                if "安全" in str(e) or "敏感" in str(e):
                    messages.append({"role": "assistant", "content": "I'm sorry, I can't help with that."})
                    print(f"API request Safety Issue, {self._NAME}, Error: {e}, returning safety message.")
                    return messages

                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(
                        f"API request failed when testing model {self._NAME}, tried: {max_retries}, Error: {e}")
                else:
                    print(
                        f"API request failed when testing model {self._NAME}，retrying {retry_count}/{max_retries}... Error: {e}")
                    time.sleep(retry_count)
        return None

    def continual_generate(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ):
        """
        Remove EOS token in formatted prompt. Manually add generation prompt.

        :param messages: List of messages for input.  
        :param config: Configuration for generation.  
        :return: Generated response or responses with log probabilities.  
        """

        if "gemma" in self._NAME.lower() and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
            messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, continual_final_message=True
        )
        eos_token = self.tokenizer.eos_token

        # remove final eos
        prompt = process_end_eos(msg=prompt, eos_token=eos_token)

        tokens = self.tokenizer(prompt, return_tensors="pt")

        if tokens["input_ids"].shape[1] > 3840:
            truncated_tokens = tokens["input_ids"][:, :3840]
            prompt = self.tokenizer.decode(
                truncated_tokens[0], skip_special_tokens=True
            )

        response = self.client.completions.create(
            model=self._NAME,
            prompt=prompt,
            max_tokens=config.max_n_tokens,
            temperature=config.temperature,
            logprobs=config.logprobs,
        )
        content = response.choices[0].text
        # messages.append({"role": "assistant", "content": content})
        messages[-1]["content"] += content

        self.update(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            1,
        )

        if config.logprobs:
            logs = response.choices[0].logprobs.token_logprobs
            return messages, logs

        return messages

    def evaluate_log_likelihood(
            self,
            messages: List[Dict[str, str]],
            config: LLMGenerateConfig,
            require_grad=False,
    ) -> List[float]:
        """
        Evaluate the log likelihood of the given messages.

        :param messages: List of messages for evaluation.  
        :param config: Configuration for LLM generation.  
        :return: List of log likelihood values.  
        """
        if require_grad:
            raise NotImplementedError

        # Make sure the content exists and is not None
        if not messages or "content" not in messages[-1] or messages[-1]["content"] is None:
            print(messages)
            raise ValueError("Last message must have valid content")

        content = messages[-1]["content"]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = self.client.completions.create(
            model=self._NAME,
            prompt=prompt,
            max_tokens=0,
            temperature=config.temperature,
            logprobs=1,
            echo=True,
        )

        # Extract token_logprobs from the response
        if hasattr(response.choices[0], 'logprobs') and hasattr(response.choices[0].logprobs, 'token_logprobs'):
            all_logprobs = response.choices[0].logprobs.token_logprobs

            # Get token IDs for the content
            token_ids = self.tokenizer(text=content).input_ids
            num_tokens = len(token_ids)

            # Take the last N logprobs where N is the number of tokens in the content
            content_logprobs = all_logprobs[-num_tokens:] if num_tokens <= len(all_logprobs) else all_logprobs

            self.update(response.usage.prompt_tokens, 0, 1)

            return content_logprobs
        else:
            # Handle the case where logprobs structure is different
            raise ValueError("Response does not contain expected logprobs structure")


if __name__ == "__main__":
    from panda_guard.llms import LLMS

    print(LLMS)

    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=128, temperature=1.0, logprobs=True, seed=42
    )

    config = OpenAiLLMConfig(
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        base_url="http://172.18.129.80:8000/v1",
    )
    llm = OpenAiLLM(config)

    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    results = llm.evaluate_log_likelihood(messages, llm_gen_config)
    print(results, len(results))
