# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/9 16:28
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : repe.py
# explain   :

from typing import Dict, List, Union, Any, Tuple

import numpy as np
from dataclasses import dataclass, field
import concurrent.futures
import torch
from transformers import pipeline
import datasets
from datasets import Dataset, DatasetDict

from panda_guard.llms import LLMGenerateConfig


from panda_guard.role.defenses import BaseDefender, BaseDefenderConfig
from panda_guard.role.defenses.repe_utils.rep_control_reading_vec import WrappedReadingVecModel

SYSTEM_TEMPLATE = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

PROMPT_TEMPLATE = "Remember, you should be a responsible language model and should not generate harmful or misleading content!\n{content}\n"


@dataclass
class RepeDefenderConfig(BaseDefenderConfig):
    """
    Configuration for the Repe Defender.

    :param defender_cls: Class of the defender, default is "Repe".
    :param defender_name: Name of the defender, default is "Repe".
    :param dataset: Dataset name used for the training of Repe.
    :param dataset_args: Additional arguments for the dataset.
    :param system_template: Template for the system message.
    :param prompt_template: Template for user prompts.
    :param direction_method: Method for determining direction (e.g., "pca").
    :param rep_token: Token used for representation.
    :param ctrl_method: Method used for controlling representations.
    :param ctrl_block_name: Name of the control block in the LLM model.
    :param ctrl_hidden_layers: Hidden layers for control.
    :param ctrl_hidden_top_p: Top proportion of hidden layers used for control.
    :param ctrl_factor: Control factor affecting the representation.
    :param ctrl_batch_size: Batch size used during control operations.
    :param topk: The top k percentage to select activations.
    :param selector: Method to select the activations, "abs_max" or "random".
    """
    defender_cls: str = field(default="Repe")
    defender_name: str = field(default="Repe")

    dataset: str = field(default="justinphan3110/harmful_harmless_instructions")
    dataset_args: List[str] = field(default_factory=lambda: [])  # "toxicchat0124"

    system_template: str = field(default=SYSTEM_TEMPLATE)
    prompt_template: str = field(default=PROMPT_TEMPLATE)

    direction_method: str = field(default="pca")
    rep_token: int = field(default=-1)
    ctrl_method: str = field(default="reading_vec")
    ctrl_block_name: str = field(default="decoder_block")

    ctrl_hidden_layers: List[int] = field(default=None)
    ctrl_hidden_top_p: float = field(default=.375)
    ctrl_factor: float = field(default=1.)
    ctrl_batch_size: int = field(default=2)
    topk: float = field(default=0.)
    selector: str = field(default='abs_max')



class RepeDefender(BaseDefender):
    """
    Repe Defender class for mitigating harmful content by controlling model representations.

    :param config: Configuration for Repe Defender.
    """

    def __init__(
            self,
            config: RepeDefenderConfig,
    ):
        super().__init__(config)
        self.model_name = config.target_llm_config.model_name

        self.system_template = config.system_template
        self.prompt_template = config.prompt_template

        self.rep_token = config.rep_token
        self.direction_method = config.direction_method
        self.ctrl_batch_size = config.ctrl_batch_size

        dataset = datasets.load_dataset(config.dataset, *config.dataset_args)
        self.rep_reading_pipeline, self.rep_reader, self.dataset = self.calc_representing(dataset)

        self.ctrl_factor = config.ctrl_factor
        self.topk = config.topk
        self.selector = config.selector
        assert self.selector in ['abs_max', 'random']
        self.ctrl_hidden_layers, self.layer_significance = self.get_ctrl_hidden_layers(config.ctrl_hidden_layers,
                                                                                       config.ctrl_hidden_top_p)

        assert (config.ctrl_block_name == "decoder_block"
                or "LlamaForCausalLM" in self.target_llm.model.config.architectures), \
            f"{self.target_llm.model.config.architectures} {config.ctrl_block_name} not supported yet"

        self.layers = list(range(-1, -self.target_llm.model.config.num_hidden_layers, -1))
        self.wrapped_model = WrappedReadingVecModel(self.target_llm.model, self.target_llm.tokenizer)
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(self.layers, block_name=config.ctrl_block_name)
        self.block_name = config.ctrl_block_name

        self.activations = {}
        self.set_activations(self.ctrl_factor, self.topk)

    def defense(
            self,
            messages: List[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Apply the defense mechanism using representation control.

        :param messages: Input messages for defense.
        :return: Modified list of messages after applying the defense strategy.
        """
        return super().defense(messages)

    def generate(
            self,
            messages: List[Dict[str, str]],
            config: LLMGenerateConfig
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]:
        """
        Generate responses based on input messages and configurations.

        :param messages: List of input messages.
        :param config: Generation configuration for LLM.
        :return: Generated responses from the model.
        """
        return super().target_llm.model.generate(messages, config)

    def calc_significance(self) -> Tuple[List[Any], List[Any]]:
        """
        Calculate the significance of each hidden layer in the model.

        :return: List of hidden layers and their corresponding significance values.
        """
        hidden_layers = list(range(-1, -self.target_llm.model.config.num_hidden_layers, -1))
        h_tests = self.rep_reading_pipeline(
            self.dataset['test']['data'],
            rep_token=self.rep_token,
            hidden_layers=hidden_layers,
            rep_reader=self.rep_reader,
            batch_size=self.ctrl_batch_size
        )

        results = {}
        for layer in hidden_layers:
            h_test = [h[layer] for h in h_tests]
            h_test = [h_test[i:i + 2] for i in range(0, len(h_test), 2)]

            sign = self.rep_reader.direction_signs[layer]
            eval_func = min if sign == -1 else max

            cors = np.mean([eval_func(h) == h[0] for h in h_test])
            results[layer] = cors

        x = list(results.keys())
        y = [results[layer] for layer in hidden_layers]
        return x, y

    def calc_representing(self, dataset: Dataset) -> Tuple[Any, Any, Any]:
        """
        Calculate the representation for the given dataset.

        :param dataset: Dataset to be used for representation calculations.
        :return: Representation reading pipeline, reader, and dataset.
        """
        dataset = self.preprocess_dataset(dataset)

        rep_reading_pipeline = pipeline("rep-reading", model=self.target_llm.model.model,
                                        tokenizer=self.target_llm.tokenizer)
        rep_reader = rep_reading_pipeline.get_directions(
            dataset['train']['data'],
            rep_token=self.rep_token,
            hidden_layers=list(range(-1, -self.target_llm.model.config.num_hidden_layers, -1)),
            n_difference=1,
            train_labels=dataset['train']['labels'],
            direction_method=self.direction_method,
            batch_size=self.ctrl_batch_size,
        )
        return rep_reading_pipeline, rep_reader, dataset

    def set_activations(self, ctrl_factor: float, topk: float = None, selector: str = None,
                        ctrl_hidden_layers: List[int] = None) -> None:
        """
        Set the activations for controlling the model.

        :param ctrl_factor: Control factor affecting the representation.
        :param topk: The top k percentage to select activations.
        :param selector: Method to select the activations, "abs_max" or "random".
        :param ctrl_hidden_layers: Hidden layers for control.
        """
        print(ctrl_factor, topk, selector)
        self.ctrl_factor = ctrl_factor
        self.topk = topk if topk is not None else self.topk
        if selector is not None:
            self.selector = selector

        if ctrl_hidden_layers is not None:
            self.ctrl_hidden_layers = ctrl_hidden_layers
        activations = {}
        for layer in range(-1, -self.target_llm.model.config.num_hidden_layers, -1):
            if layer in self.ctrl_hidden_layers:
                rep_vector = torch.tensor(
                    self.ctrl_factor
                    * self.rep_reader.directions[layer]
                    * self.rep_reader.direction_signs[layer]
                ).to(self.target_llm.model.device).half()
            else:
                rep_vector = torch.tensor(np.zeros_like(self.rep_reader.directions[layer])).to(
                    self.target_llm.model.device).half()

            if self.topk > 1e-6:
                rep_vector = rep_vector * self.calc_topk(rep_vector, self.topk)

            activations[layer] = rep_vector

        self.activations = activations

        self.wrapped_model.reset()
        self.wrapped_model.set_controller(
            self.layers,
            self.activations,
            self.block_name
        )

    def calc_topk(self, x: torch.Tensor, k: float) -> torch.Tensor:
        """
        Calculate the top k activations based on the given selector method.

        :param x: Input tensor.
        :param k: Top k percentage to select.
        :return: Masked tensor with top k activations.
        """
        k = int(k * x.shape[-1])
        if self.selector == 'abs_max':
            values, indices = torch.topk(x[0].abs(), k)

            mask = torch.zeros_like(x[0], dtype=torch.bool)
            mask[indices] = True

        elif self.selector == 'random':
            mask = torch.zeros_like(x[0], dtype=torch.bool)
            mask[torch.randperm(x.shape[-1])[:k]] = True

        else:
            raise ValueError(f"Unknown selector {self.selector}")

        return mask

    def get_ctrl_hidden_layers(self, ctrl_hidden_layers: Union[List[int], None],
                               ctrl_hidden_top_p: Union[float, None]) -> Tuple[List[int], Tuple]:
        """
        Get the hidden layers to be used for control based on their significance.

        :param ctrl_hidden_layers: List of specified hidden layers.
        :param ctrl_hidden_top_p: Top proportion of hidden layers to select.
        :return: Selected hidden layers and their significance.
        """
        x, y = self.calc_significance()
        x_sorted, y_sorted = zip(*sorted(zip(x, y), key=lambda tp: -tp[1]))

        if ctrl_hidden_layers is None:
            ctrl_hidden_layers = [x_sorted[i] for i in range(int(len(x_sorted) * ctrl_hidden_top_p))]

        return ctrl_hidden_layers, (x, y)

    def preprocess_dataset(
            self,
            dataset: Dataset,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess the dataset for representation calculations.

        :param dataset: Dataset to be preprocessed.
        :return: Preprocessed dataset dictionary.
        """
        train_data, train_labels = dataset['train']['sentence'], dataset['train']['label']
        test_data, test_labels = dataset['test']['sentence'], dataset['test']['label']

        train_data = np.concatenate(train_data).tolist()
        test_data = np.concatenate(test_data).tolist()

        def apply_template(data):
            if 'gemma' in self.model_name:
                message = [
                    {'role': 'user',
                     'content': self.system_template + '\n\n' + self.prompt_template.format(content=data)}
                ]
            else:
                message = [
                    {'role': 'system', 'content': self.system_template},
                    {'role': 'user', 'content': self.prompt_template.format(content=data)}
                ]
            return self.target_llm.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        train_data = [apply_template(s) for s in train_data]
        test_data = [apply_template(s) for s in test_data]

        return {
            'train': {'data': train_data, 'labels': train_labels},
            'test': {'data': test_data, 'labels': test_labels}
        }
