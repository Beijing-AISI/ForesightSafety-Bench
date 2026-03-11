# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/9 16:33
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : pipelines.py
# explain   : Adapted from https://github.com/andyzoujm/representation-engineering.git

from transformers import AutoModel, AutoModelForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY
from .rep_reading_pipeline import RepReadingPipeline
from .rep_control_pipeline import RepControlPipeline


def repe_pipeline_registry():
    """
    Register repe-related pipelines to the global pipeline registry for later instantiation by name.
    """
    PIPELINE_REGISTRY.register_pipeline(
        "rep-reading",
        pipeline_class=RepReadingPipeline,
        pt_model=AutoModel,
    )

    PIPELINE_REGISTRY.register_pipeline(
        "rep-control",
        pipeline_class=RepControlPipeline,
        pt_model=AutoModelForCausalLM,
    )


