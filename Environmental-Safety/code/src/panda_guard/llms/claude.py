# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/2 20:30
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : claude.py
# explain   : Anthropic Claude API integration

import os
import time
import warnings
from typing import Dict, List, Union, Any, Tuple, Generator
from dataclasses import dataclass, field

import anthropic


from panda_guard.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig


@dataclass
class ClaudeLLMConfig(BaseLLMConfig):
    """
    Claude LLM Configuration.

    :param llm_type: Type of LLM, default is "ClaudeLLM".
    :param model_name: Name of the model.
    :param api_key: API key for accessing Anthropic.
    :param max_tokens_to_sample: Maximum tokens to sample, overrides max_n_tokens if provided.
    """

    llm_type: str = field(default="ClaudeLLM")
    model_name: str = field(default="claude-3-opus-20240229")
    api_key: str = field(default=None)
    max_tokens_to_sample: int = field(default=None)



class ClaudeLLM(BaseLLM):
    """
    Claude LLM Implementation.

    :param config: Configuration for Claude LLM.
    """

    def __init__(self, config: ClaudeLLMConfig):
        super().__init__(config)

        # Use provided API key or try to get from environment variable
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set as ANTHROPIC_API_KEY environment variable")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_tokens_to_sample = config.max_tokens_to_sample

    def generate(
            self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> list[dict[str, str]] | Generator[str, None, None] | None:
        """
        Generate a response for a given input using Anthropic Claude API.

        :param messages: List of input messages.
        :param config: Configuration for LLM generation.
        :return: Generated response, stream generator, or response with logprobs.
        """
        max_tokens = self.max_tokens_to_sample or config.max_n_tokens

        retry_count = 0
        max_retries = 10

        while retry_count < max_retries:
            try:
                # Convert messages to Anthropic format if needed
                anthropic_messages = []
                system_content = None

                for msg in messages:
                    role = msg["role"]
                    # Anthropic uses "user" and "assistant", convert "system" to content for first user message
                    if role == "system":
                        system_content = msg["content"]
                        continue
                    elif role == "user":
                        anthropic_messages.append({"role": "user", "content": msg["content"]})
                    elif role == "assistant":
                        anthropic_messages.append({"role": "assistant", "content": msg["content"]})

                # If we had a system message and there are other messages, prepend to first user message
                if system_content and anthropic_messages and anthropic_messages[0]["role"] == "user":
                    anthropic_messages[0]["content"] = f"{system_content}\n\n{anthropic_messages[0]['content']}"

                # Handle streaming mode
                if config.stream:
                    full_content = ""
                    input_tokens = 0
                    output_tokens = 0

                    # Create streaming request
                    stream = self.client.messages.create(
                        model=self._NAME,
                        messages=anthropic_messages,
                        max_tokens=max_tokens,
                        temperature=config.temperature or 0.7,
                        stream=True,
                    )

                    def stream_response():
                        nonlocal full_content, input_tokens, output_tokens

                        for chunk in stream:
                            # Check if the chunk has content to stream
                            if chunk.type == "content_block_delta" and chunk.delta.text:
                                content_piece = chunk.delta.text
                                full_content += content_piece
                                yield content_piece

                            # Update token usage if available
                            if hasattr(chunk, 'usage') and chunk.usage:
                                if hasattr(chunk.usage, 'input_tokens'):
                                    input_tokens = chunk.usage.input_tokens
                                if hasattr(chunk.usage, 'output_tokens'):
                                    output_tokens = chunk.usage.output_tokens

                            # Handle end of stream with usage info
                            if chunk.type == "message_delta" and chunk.usage:
                                input_tokens = chunk.usage.input_tokens
                                output_tokens = chunk.usage.output_tokens

                    response_generator = stream_response()

                    def wrapped_generator():
                        yield from response_generator

                        # Add final response to messages
                        messages.append({"role": "assistant", "content": full_content})

                        # Update usage statistics
                        self.update(
                            input_tokens,
                            output_tokens,
                            1,
                        )

                    return wrapped_generator()

                # Non-streaming mode (original code)
                else:
                    # Claude API call
                    response = self.client.messages.create(
                        model=self._NAME,
                        messages=anthropic_messages,
                        max_tokens=max_tokens,
                        temperature=config.temperature or 0.7,
                    )

                    # Add generated response to messages
                    content = response.content[0].text
                    messages.append({"role": "assistant", "content": content})

                    # Update token usage statistics
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    self.update(
                        input_tokens,
                        output_tokens,
                        1,
                    )

                    # Claude API doesn't support logprobs directly, so we handle the case
                    if config.logprobs:
                        warnings.warn("Claude API does not support logprobs, returning response without them.")
                        return messages

                    return messages

            except Exception as e:
                # Handle safety/content policy issues
                if "content policy" in str(e).lower() or "content_policy" in str(e).lower():
                    messages.append({"role": "assistant", "content": "I'm sorry, I can't help with that."})
                    print(f"API request Content Policy Issue, {self._NAME}, Error: {e}, returning safety message.")
                    return messages

                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(
                        f"API request failed when testing model {self._NAME}, tried: {max_retries}, Error: {e}")
                else:
                    print(
                        f"API request failed when testing model {self._NAME}, retrying {retry_count}/{max_retries}... Error: {e}")
                    time.sleep(retry_count)  # Exponential backoff
        return None


    def continual_generate(
            self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ):
        """
        Generate continuation for the last message.

        :param messages: List of messages for input.
        :param config: Configuration for generation.
        :return: Generated response.
        """
        # Claude doesn't support true "continue generating" functionality like some models
        # Instead, we extract current conversation and generate a continuation

        # Clone messages and extract the last assistant message if it exists
        convo_messages = messages.copy()
        last_assistant_content = ""

        # If the last message is from the assistant, we'll use it as context for continuation
        if convo_messages[-1]["role"] == "assistant":
            last_assistant_content = convo_messages[-1]["content"]
            convo_messages.pop()  # Remove last message as we'll append to it

            # Add a user message asking to continue
            convo_messages.append({"role": "user", "content": "Please continue from where you left off."})

            # Generate the continuation
            result = self.generate(convo_messages, config)

            # Merge the continuation with the original message
            if isinstance(result, tuple):  # If logprobs were requested
                cont_messages, logprobs = result
                cont_content = cont_messages[-1]["content"]
                messages[-1]["content"] = last_assistant_content + " " + cont_content
                return messages, logprobs
            else:
                cont_content = result[-1]["content"]
                messages[-1]["content"] = last_assistant_content + " " + cont_content
                return messages
        else:
            warnings.warn("The last message must be from the assistant to use continual_generate.")
            return self.generate(messages, config)

    def evaluate_log_likelihood(
            self,
            messages: List[Dict[str, str]],
            config: LLMGenerateConfig,
            require_grad=False
    ) -> List[float]:
        """
        Evaluate the log likelihood of the given messages.

        :param messages: List of messages for evaluation.
        :param config: Configuration for LLM generation.
        :param require_grad: Whether to compute gradients (not supported for API models).
        :raises NotImplementedError: Claude API does not support log likelihood evaluation.
        """
        raise NotImplementedError(
            "Claude API does not support log likelihood evaluation."
        )


if __name__ == "__main__":
    from panda_guard.llms import LLMS

    print(LLMS)

    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=128, temperature=1.0, logprobs=True, seed=42
    )

    config = ClaudeLLMConfig(
        model_name="claude-3-opus-20240229",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    llm = ClaudeLLM(config)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]

    result = llm.generate(messages, llm_gen_config)
    print(result[-1]["content"])