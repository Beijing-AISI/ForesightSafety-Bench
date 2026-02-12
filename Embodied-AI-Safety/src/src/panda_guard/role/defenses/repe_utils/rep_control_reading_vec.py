# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/9 16:37
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : rep_control_reading_vec.py
# explain   : Adapted from https://github.com/andyzoujm/representation-engineering.git


# wrapping classes
import torch
import numpy as np


class WrappedBlock(torch.nn.Module):
    """
    A wrapper around a transformer block or submodule that allows external
    activation control via injection of custom activation vectors.

    :param block: The transformer block or submodule to wrap.
    """
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.normalize = False

    def forward(self, *args, **kwargs):
        """
        Forward pass with optional activation control.

        :return: Modified output from the block.
        """
        output = self.block(*args, **kwargs)

        if isinstance(output, tuple):
            self.output = output[0]
            modified = output[0]
        else:
            self.output = output
            modified = output

        if self.controller is not None:

            norm_pre = torch.norm(modified, dim=-1, keepdim=True)

            if self.mask is not None:
                mask = self.mask

            # we should ignore the padding tokens when doing the activation addition
            # mask has ones for non padding tokens and zeros at padding tokens.
            # only tested this on left padding
            elif "position_ids" in kwargs:
                pos = kwargs["position_ids"]
                zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
                col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
                target_shape = modified.shape
                mask = (col_indices >= zero_indices).float().reshape(target_shape[0], target_shape[1], 1)
                mask = mask.to(modified.dtype)
            else:
                # print(f"Warning: block {self.block_name} does not contain information 'position_ids' about token types. When using batches this can lead to unexpected results.")
                mask = 1.0

            if len(self.controller.shape) == 1:
                self.controller = self.controller.reshape(1, 1, -1)
            assert len(self.controller.shape) == len(modified.shape), f"Shape of controller {self.controller.shape} does not match shape of modified {modified.shape}."

            self.controller = self.controller.to(modified.device)
            if isinstance(mask, torch.Tensor):
                mask = mask.to(modified.device)
            if isinstance(self.token_pos, int):
                modified[:, self.token_pos] = self.operator(modified[:, self.token_pos], self.controller * mask)
            elif isinstance(self.token_pos, list) or isinstance(self.token_pos, tuple) or isinstance(self.token_pos, np.ndarray):
                modified[:, self.token_pos] = self.operator(modified[:, self.token_pos], self.controller * mask)
            elif isinstance(self.token_pos, str):
                if self.token_pos == "end":
                    len_token = self.controller.shape[1]
                    modified[:, -len_token:] = self.operator(modified[:, -len_token:], self.controller * mask)
                elif self.token_pos == "start":
                    len_token = self.controller.shape[1]
                    modified[:, :len_token] = self.operator(modified[:, :len_token], self.controller * mask)
                else:
                    assert False, f"Unknown token position {self.token_pos}."
            else:
                modified = self.operator(modified, self.controller * mask)

            if self.normalize:
                norm_post = torch.norm(modified, dim=-1, keepdim=True)
                modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            output = (modified,) + output[1:]
        else:
            output = modified

        return output

    def set_controller(self, activations, token_pos=None, masks=None, normalize=False, operator='linear_comb'):
        """
        Sets the external controller to inject activations.

        :param activations: Activation vector(s) to apply.
        :param token_pos: Target token positions to apply the activations.
        :param masks: Optional mask for activation injection.
        :param normalize: Whether to normalize pre/post activation.
        :param operator: Operator to apply (e.g. 'linear_comb', 'piecewise_linear').
        """
        self.normalize = normalize
        self.controller = activations.squeeze()
        self.mask = masks
        self.token_pos = token_pos
        if operator == 'linear_comb':
            def op(current, controller):
                return current + controller
        elif operator == 'piecewise_linear':
            def op(current, controller):
                sign = torch.sign((current * controller).sum(-1, keepdim=True))
                return current + controller * sign
        elif operator == 'projection':
            def op(current, controller):
                raise NotImplementedError
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")
        self.operator = op

    def reset(self):
        """
        Resets all control variables to remove previously injected activations.
        """
        self.output = None
        self.controller = None
        self.mask = None
        self.token_pos = None
        self.operator = None

    def set_masks(self, masks):
        """
        Sets an external mask to apply during activation injection.

        :param masks: The binary mask for injection.
        """
        self.mask = masks


BLOCK_NAMES = [
    "self_attn",
    "mlp",
    "input_layernorm",
    "post_attention_layernorm"
    ]


class WrappedReadingVecModel(torch.nn.Module):
    """
    A wrapper for transformer models that enables dynamic injection
    and extraction of intermediate activations at specified layers and blocks.

    :param model: The pretrained transformer model.
    :param tokenizer: The tokenizer associated with the model.
    """
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, *args, **kwargs):
        """
        Forwards inputs through the model.

        :return: Model outputs.
        """
        return self.model(*args, **kwargs)

    def generate(self, **kwargs):
        """
        Calls the `generate` method on the underlying model.

        :return: Generated output.
        """
        return self.model.generate(**kwargs)

    def get_logits(self, tokens):
        """
        Computes logits from the model given input tokens.

        :param tokens: Input token tensor.
        :return: Logits from the model.
        """
        with torch.no_grad():
            logits = self.model(tokens.to(self.model.device)).logits
            return logits

    def run_prompt(self, prompt, **kwargs):
        """
        Tokenizes and runs a prompt through the model without gradients.

        :param prompt: Input prompt string.
        :return: Model output.
        """
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            output = self.model(input_ids, attention_mask=attention_mask)
            return output

    def wrap(self, layer_id, block_name):
        """
        Wraps a specific sub-block in a given layer.

        :param layer_id: Index of the transformer layer to wrap.
        :param block_name: Name of the block to wrap (e.g., 'mlp', 'self_attn').
        """
        assert block_name in BLOCK_NAMES
        if self.is_wrapped(self.model.model.layers[layer_id]):
            block = getattr(self.model.model.layers[layer_id].block, block_name)
            if not self.is_wrapped(block):
                setattr(self.model.model.layers[layer_id].block, block_name, WrappedBlock(block))
        else:
            block = getattr(self.model.model.layers[layer_id], block_name)
            if not self.is_wrapped(block):
                setattr(self.model.model.layers[layer_id], block_name, WrappedBlock(block))

    def wrap_decoder_block(self, layer_id):
        """
        Wraps the entire decoder block for a given layer.

        :param layer_id: Index of the decoder layer to wrap.
        """
        block = self.model.model.layers[layer_id]
        if not self.is_wrapped(block):
            self.model.model.layers[layer_id] = WrappedBlock(block)

    def wrap_all(self):
        """
        Wraps all known sub-blocks in every layer.
        """
        for layer_id, layer in enumerate(self.model.model.layers):
            for block_name in BLOCK_NAMES:
                self.wrap(layer_id, block_name)
            self.wrap_decoder_block(layer_id)

    def wrap_block(self, layer_ids, block_name):
        """
        Wraps specified block(s) in the given layer(s).

        :param layer_ids: Single layer ID or list of layer IDs.
        :param block_name: Block name to wrap.
        """
        def _wrap_block(layer_id, block_name):
            if block_name in BLOCK_NAMES:
                self.wrap(layer_id, block_name)
            elif block_name == 'decoder_block':
                self.wrap_decoder_block(layer_id)
            else:
                assert False, f"No block named {block_name}."

        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            for layer_id in layer_ids:
                _wrap_block(layer_id, block_name)
        else:
            _wrap_block(layer_ids, block_name)

    def get_activations(self, layer_ids, block_name='decoder_block'):
        """
        Extracts activations from wrapped blocks.

        :param layer_ids: Layer index or list of indices.
        :param block_name: Target block name.
        :return: Dictionary or tensor of activations.
        """
        def _get_activations(layer_id, block_name):
            current_layer = self.model.model.layers[layer_id]

            if self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name == 'decoder_block':
                    return current_layer.output
                elif block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_block, block_name)):
                    return getattr(current_block, block_name).output
                else:
                    assert False, f"No wrapped block named {block_name}."

            else:
                if block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_layer, block_name)):
                    return getattr(current_layer, block_name).output
                else:
                    assert False, f"No wrapped block named {block_name}."

        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            activations = {}
            for layer_id in layer_ids:
                activations[layer_id] = _get_activations(layer_id, block_name)
            return activations
        else:
            return _get_activations(layer_ids, block_name)

    def set_controller(self, layer_ids, activations, block_name='decoder_block', token_pos=None, masks=None, normalize=False, operator='linear_comb'):
        """
        Injects controller activations into specified layers/blocks.

        :param layer_ids: Layer index or list of indices.
        :param activations: Activation tensor(s) to inject.
        :param block_name: Block name to apply control to.
        :param token_pos: Target token positions.
        :param masks: Optional binary masks.
        :param normalize: Whether to normalize activations.
        :param operator: Operator name to apply.
        """
        def _set_controller(layer_id, activations, block_name, masks, normalize, operator):
            current_layer = self.model.model.layers[layer_id]

            if block_name == 'decoder_block':
                current_layer.set_controller(activations, token_pos, masks, normalize, operator)
            elif self.is_wrapped(current_layer):
                current_block = current_layer.block
                if block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_block, block_name)):
                    getattr(current_block, block_name).set_controller(activations, token_pos, masks, normalize, operator)
                else:
                    return f"No wrapped block named {block_name}."

            else:
                if block_name in BLOCK_NAMES and self.is_wrapped(getattr(current_layer, block_name)):
                    getattr(current_layer, block_name).set_controller(activations, token_pos, masks, normalize, operator)
                else:
                    return f"No wrapped block named {block_name}."

        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            assert isinstance(activations, dict), "activations should be a dictionary"
            for layer_id in layer_ids:
                _set_controller(layer_id, activations[layer_id], block_name, masks, normalize, operator)
        else:
            _set_controller(layer_ids, activations, block_name, masks, normalize, operator)

    def reset(self):
        """
        Resets all internal controllers and outputs in wrapped blocks.
        """
        for layer in self.model.model.layers:
            if self.is_wrapped(layer):
                layer.reset()
                for block_name in BLOCK_NAMES:
                    if self.is_wrapped(getattr(layer.block, block_name)):
                        getattr(layer.block, block_name).reset()
            else:
                for block_name in BLOCK_NAMES:
                    if self.is_wrapped(getattr(layer, block_name)):
                        getattr(layer, block_name).reset()

    def set_masks(self, masks):
        """
        Sets masks for all wrapped layers and blocks.

        :param masks: Binary tensor mask.
        """
        for layer in self.model.model.layers:
            if self.is_wrapped(layer):
                layer.set_masks(masks)
                for block_name in BLOCK_NAMES:
                    if self.is_wrapped(getattr(layer.block, block_name)):
                        getattr(layer.block, block_name).set_masks(masks)
            else:
                for block_name in BLOCK_NAMES:
                    if self.is_wrapped(getattr(layer, block_name)):
                        getattr(layer, block_name).set_masks(masks)

    def is_wrapped(self, block):
        """
        Checks whether a given block has been wrapped.

        :param block: A model sub-block.
        :return: True if wrapped, else False.
        """
        if hasattr(block, 'block'):
            return True
        return False

    def unwrap(self):
        """
        Restores the original model structure by removing all wrappers.
        """
        for l, layer in enumerate(self.model.model.layers):
            if self.is_wrapped(layer):
                self.model.model.layers[l] = layer.block
            for block_name in BLOCK_NAMES:
                if self.is_wrapped(getattr(self.model.model.layers[l], block_name)):
                    setattr(self.model.model.layers[l],
                            block_name,
                            getattr(self.model.model.layers[l], block_name).block)
