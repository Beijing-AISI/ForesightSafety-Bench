# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/9 16:33
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : rep_reading_pipeline.py
# explain   : Adapted from https://github.com/andyzoujm/representation-engineering.git

from typing import List, Union, Optional
from transformers import Pipeline
import torch
import numpy as np
from .rep_readers import DIRECTION_FINDERS, RepReader


class RepReadingPipeline(Pipeline):
    """
    A pipeline for extracting and transforming hidden state representations from transformer models.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_hidden_states(
            self,
            outputs,
            rep_token: Union[str, int] = -1,
            hidden_layers: Union[List[int], int] = -1,
            which_hidden_states: Optional[str] = None):
        """
        Extract hidden states from model outputs.

        :param outputs: Model output.
        :param rep_token: Token index from which to extract hidden states.
        :param hidden_layers: Layer indices to extract hidden states from.
        :param which_hidden_states: For encoder-decoder models, specifies whether to use 'encoder' or 'decoder'.
        :return: A dictionary mapping layer indices to hidden state tensors.
        """

        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states']

        hidden_states_layers = {}
        for layer in hidden_layers:
            hidden_states = outputs['hidden_states'][layer]
            hidden_states = hidden_states[:, rep_token, :]
            # hidden_states_layers[layer] = hidden_states.cpu().to(dtype=torch.float32).detach().numpy()
            hidden_states_layers[layer] = hidden_states.detach()

        return hidden_states_layers

    def _sanitize_parameters(self,
                             rep_reader: RepReader = None,
                             rep_token: Union[str, int] = -1,
                             hidden_layers: Union[List[int], int] = -1,
                             component_index: int = 0,
                             which_hidden_states: Optional[str] = None,
                             **tokenizer_kwargs):
        """
        Sanitize and prepare pipeline parameters.

        :param rep_reader: Optional `RepReader` instance for transforming representations.
        :param rep_token: Token index used to extract hidden states.
        :param hidden_layers: Layer indices to extract hidden states from.
        :param component_index: Component index to extract after transformation.
        :param which_hidden_states: Specify 'encoder' or 'decoder' for encoder-decoder models.
        :param tokenizer_kwargs: Additional tokenizer parameters.
        :return: Tuple of (preprocess_params, forward_params, postprocess_params).
        """

        preprocess_params = tokenizer_kwargs
        forward_params = {}
        postprocess_params = {}

        forward_params['rep_token'] = rep_token

        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]

        assert rep_reader is None or len(rep_reader.directions) == len(hidden_layers), f"expect total rep_reader directions ({len(rep_reader.directions)})== total hidden_layers ({len(hidden_layers)})"
        forward_params['rep_reader'] = rep_reader
        forward_params['hidden_layers'] = hidden_layers
        forward_params['component_index'] = component_index
        forward_params['which_hidden_states'] = which_hidden_states

        return preprocess_params, forward_params, postprocess_params

    def preprocess(
            self,
            inputs: Union[str, List[str], List[List[str]]],
            **tokenizer_kwargs):
        """
        Preprocess input data using tokenizer or image processor.

        :param inputs: Input data.
        :param tokenizer_kwargs: Additional arguments for the tokenizer.
        :return: Tokenized or processed inputs.
        """

        if self.image_processor:
            return self.image_processor(inputs, add_end_of_utterance_token=False, return_tensors="pt")
        return self.tokenizer(inputs, return_tensors=self.framework, **tokenizer_kwargs)

    def postprocess(self, outputs):
        """
        Pass-through postprocessing step.

        :param outputs: Outputs from the model.
        :return: Unmodified outputs.
        """
        return outputs

    def _forward(self, model_inputs, rep_token, hidden_layers, rep_reader=None, component_index=0, which_hidden_states=None, **tokenizer_args):
        """
        Forward pass to extract or transform hidden states.

        :param model_inputs: Tokenized inputs.
        :param rep_token: Index of the token to extract from.
        :param hidden_layers: Target layers to extract hidden states from.
        :param rep_reader: Optional `RepReader` to apply transformation.
        :param component_index: Component index used in transformation.
        :param which_hidden_states: For encoder-decoder models, specify 'encoder' or 'decoder'.
        :param tokenizer_args: Additional tokenizer arguments.
        :return: Extracted or transformed hidden states.
        """
        # get model hidden states and optionally transform them with a RepReader
        with torch.no_grad():
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
                decoder_start_token = [self.tokenizer.pad_token] * model_inputs['input_ids'].size(0)
                decoder_input = self.tokenizer(decoder_start_token, return_tensors="pt").input_ids
                model_inputs['decoder_input_ids'] = decoder_input
            outputs = self.model(**model_inputs, output_hidden_states=True)
        hidden_states = self._get_hidden_states(outputs, rep_token, hidden_layers, which_hidden_states)

        if rep_reader is None:
            return hidden_states

        return rep_reader.transform(hidden_states, hidden_layers, component_index)

    def _batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args):
        """
        Extract hidden states from batches of input strings.

        :param train_inputs: List of training strings.
        :param rep_token: Token index to extract representation from.
        :param hidden_layers: List of layer indices to extract from.
        :param batch_size: Batch size for processing.
        :param which_hidden_states: Specify 'encoder' or 'decoder' for encoder-decoder models.
        :param tokenizer_args: Additional tokenizer arguments.
        :return: Dictionary of hidden states.
        """
        # Wrapper method to get a dictionary hidden states from a list of strings
        hidden_states_outputs = self(train_inputs, rep_token=rep_token,
                                     hidden_layers=hidden_layers, batch_size=batch_size, rep_reader=None, which_hidden_states=which_hidden_states, **tokenizer_args)
        hidden_states = {layer: [] for layer in hidden_layers}
        for hidden_states_batch in hidden_states_outputs:
            for layer in hidden_states_batch:
                hidden_states[layer].extend(hidden_states_batch[layer])
        return {k: np.vstack(v) for k, v in hidden_states.items()}

    def _validate_params(self, n_difference, direction_method):
        """
        Validate parameters `get_directions`.

        :param n_difference: Number of pairwise differences to compute.
        :param direction_method: Method used to find representation directions.
        :raises AssertionError: If invalid parameter combinations are provided.
        """
        # validate params for get_directions
        if direction_method == 'clustermean':
            assert n_difference == 1, "n_difference must be 1 for clustermean"

    def get_directions(
            self,
            train_inputs: Union[str, List[str], List[List[str]]],
            rep_token: Union[str, int] = -1,
            hidden_layers: Union[str, int] = -1,
            n_difference: int = 1,
            batch_size: int = 1,
            train_labels: List[int] = None,
            direction_method: str = 'pca',
            direction_finder_kwargs: dict = {},
            which_hidden_states: Optional[str] = None,
            **tokenizer_args,):
        """
        Train a RepReader on the training data.

        :param train_inputs: Input examples to train on.
        :param rep_token: Index of the token to extract hidden states from.
        :param hidden_layers: Layer indices to extract hidden states from.
        :param n_difference: Number of times to compute differences in training pairs.
        :param batch_size: Batch size for extracting hidden states.
        :param train_labels: Labels for supervised direction finding.
        :param direction_method: Method to use for finding directions (e.g., 'pca', 'clustermean').
        :param direction_finder_kwargs: Additional keyword arguments for the direction finder.
        :param which_hidden_states: For encoder-decoder models, specify 'encoder' or 'decoder'.
        :param tokenizer_args: Additional tokenizer parameters.
        :return: A trained `RepReader` containing the learned directions.
        """

        if not isinstance(hidden_layers, list):
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]

        self._validate_params(n_difference, direction_method)

        # initialize a DirectionFinder
        direction_finder = DIRECTION_FINDERS[direction_method](**direction_finder_kwargs)

        # if relevant, get the hidden state data for training set
        hidden_states = None
        relative_hidden_states = None
        if direction_finder.needs_hiddens:
            # get raw hidden states for the train inputs
            hidden_states = self._batched_string_to_hiddens(train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args)

            # get differences between pairs
            relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
            for layer in hidden_layers:
                for _ in range(n_difference):  # B N C
                    relative_hidden_states[layer] = relative_hidden_states[layer][::2] - relative_hidden_states[layer][1::2]

                # get the directions
        direction_finder.directions = direction_finder.get_rep_directions(
            self.model, self.tokenizer, relative_hidden_states, hidden_layers,
            train_choices=train_labels)
        for layer in direction_finder.directions:
            if isinstance(direction_finder.directions[layer], np.ndarray):
                direction_finder.directions[layer] = direction_finder.directions[layer].astype(np.float32)

        if train_labels is not None:
            direction_finder.direction_signs = direction_finder.get_signs(
                hidden_states, train_labels, hidden_layers)

        return direction_finder
