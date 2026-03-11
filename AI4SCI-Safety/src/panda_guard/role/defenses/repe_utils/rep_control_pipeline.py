# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/9 16:36
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : rep_control_pipeline.py
# explain   : Adapted from https://github.com/andyzoujm/representation-engineering.git


from transformers.pipelines import TextGenerationPipeline
from .rep_control_reading_vec import WrappedReadingVecModel


class RepControlPipeline(TextGenerationPipeline):
    """
    A custom text generation pipeline that allows representation-based control
    during intermediate forward passes of a transformer model.

    :param model: The language model to be used for generation.
    :param tokenizer: The tokenizer associated with the model.
    :param layers: A list of layer indices to which the control method should be applied.
    :param block_name: The name of the block to be wrapped (e.g., "decoder_block").
    :param control_method: The method used to apply control during generation.
                           Currently supports only "reading_vec".
    :param kwargs: Additional keyword arguments passed to the base pipeline.
    :type kwargs: dict
    """
    def __init__(self,
                 model,
                 tokenizer,
                 layers,
                 block_name="decoder_block",
                 control_method="reading_vec",
                 **kwargs):

        # TODO: implement different control method and supported intermediate modules for different models
        assert control_method == "reading_vec", f"{control_method} not supported yet"
        assert block_name == "decoder_block" or "LlamaForCausalLM" in model.config.architectures, f"{model.config.architectures} {block_name} not supported yet"
        self.wrapped_model = WrappedReadingVecModel(model, tokenizer)
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(layers, block_name=block_name)
        self.block_name = block_name
        self.layers = layers

        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def __call__(self, text_inputs, activations=None, **kwargs):
        """
        Generate text with optional control via intermediate activations.

        :param text_inputs: Input text(s) for generation.
        :param activations: Optional activations to control intermediate layers.
        :param kwargs: Additional generation parameters.
        """
        if activations is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name)

        outputs = super().__call__(text_inputs, **kwargs)
        self.wrapped_model.reset()

        return outputs
