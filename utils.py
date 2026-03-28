"""
Utility functions for code evaluation and model handling.

This module contains various utility functions that support the main
evaluation pipeline, including model loading, API validation, and text processing.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM

# Tree-sitter setup
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
PY_LANGUAGE = Language(tspython.language())

np.random.seed(0)


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_huggingface_model_from_pytorch_checkpoint(base_model_name: str, checkpoint_path: str) -> AutoModelForCausalLM:
    """
    Load a HuggingFace model from a PyTorch checkpoint file.
    
    Args:
        base_model_name: Name of the base model (e.g., 'bigcode/starcoderbase')
        checkpoint_path: Path to the PyTorch checkpoint file
        
    Returns:
        Loaded AutoModelForCausalLM instance
    """
    model_config = AutoConfig.from_pretrained(base_model_name)

    if base_model_name in ["bigcode/starcodebase-7b", "bigcode/starcoderbase"]:
        with init_empty_weights():
            language_model = AutoModelForCausalLM.from_config(
                model_config,
                torch_dtype=torch.float16
            )
        language_model.tie_weights()
        language_model = load_checkpoint_and_dispatch(
            language_model, checkpoint=checkpoint_path, device_map="auto",
            dtype=torch.float16
        )
    else:
        language_model = AutoModelForCausalLM.from_config(model_config)
        language_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        language_model.tie_weights()
        language_model = language_model.to("cuda")

    return language_model

# ============================================================================
# CODE ANALYSIS FUNCTIONS (MINIMAL IMPLEMENTATIONS)
# ============================================================================

def has_new_api_call(
        code: str, 
        new_scope_started: bool = False,
        imports: list = [],
) -> Tuple[bool, Optional[str]]:
    parser = Parser(PY_LANGUAGE)
    # parser = Parser()
    # parser.set_language(PY_LANGUAGE)
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    # Function to recursively search for the first expression statement that starts with an import key
    def old_find_api_call(node):
        for child in node.children:
            # Check if the substring of code starting and ending at the child's byte range starts with any import key
            if child.type == 'expression_statement' and any(code[child.start_byte:child.end_byte].startswith(import_key) for import_key in imports):
                return code[child.start_byte:child.end_byte]
            else:
                found = old_find_api_call(child)
                if found:
                    return found
        return None

    api_call = old_find_api_call(root_node)
    if api_call:
        return True, api_call
    else:
        return False, code

# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

def make_dirname_from_retrieval_config(retrieval_config: Dict) -> str:
    retrieval_type = retrieval_config["retrieval_type"]
    dirname = retrieval_type

    if retrieval_type != "no_augmentation":
        dirname += f"_num_ret_{retrieval_config.get('num_to_retrieve', 1)}"

        conf_thresh = retrieval_config.get("retrieval_confidence_threshold", None)
        if conf_thresh is not None:
            dirname += f"_conf_thresh_{conf_thresh}"

    return dirname