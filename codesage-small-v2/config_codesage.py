#!/usr/bin/env python
# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from transformers.configuration_utils import PretrainedConfig

CODESAGE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "codesage/codesage-small-v2": "https://huggingface.co/codesage/codesage-small-v2/resolve/main/config.json",
    "codesage/codesage-base-v2": "https://huggingface.co/codesage/codesage-base-v2/resolve/main/config.json",
    "codesage/codesage-large-v2": "https://huggingface.co/codesage/codesage-large-v2/resolve/main/config.json",
}


class CodeSageConfig(PretrainedConfig):
    model_type = "codesage"

    def __init__(
            self,
            vocab_size=50257,
            max_position_embeddings=1024,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            activation_function="gelu_new",
            residual_dropout_prob=0.1,
            embedding_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            position_embedding_type='absolute',
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=49153,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        assert 'gelu' in activation_function
        self.activation_function = activation_function
        self.residual_dropout_prob = residual_dropout_prob
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.position_embedding_type = position_embedding_type

        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
