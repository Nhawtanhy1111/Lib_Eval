import os
import json
from typing import Dict, List, Tuple

import numpy as np
import re
import logging
np.random.seed(0)
from retriever import CodeSageRetriever


def augment_prompt(task: Dict, retrieved: List[Tuple[Dict, float]]) -> Dict:
    augmentation = "# Use the following API information as reference:\n"

    for ret_pair in retrieved:
        ret = ret_pair[0]
        api_name = ret.get("API_Call", "Unknown API")
        signature = ret.get("Signature", "")
        
        # Check for your new field here:
        detailed_desc = ret.get("Api_Description") or ret.get("Detailed_Description", "")

        block = f"{api_name} : {signature}\n"

        # Parameters (dict of {name: details/desc})
        params = ret.get("Parameters", {})
        if isinstance(params, dict) and params:
            # Keep concise, name-only list for readability
            param_names = ", ".join(params.keys())
            block += f"Input Parameters : {param_names}\n"

        # Optional short description (first non-empty line)
        if isinstance(detailed_desc, str) and detailed_desc.strip():
                    # Instead of just taking the first line, you may want the whole 
                    # instruction if it contains the "condition" mentioned in your example
            block += f"Description: {detailed_desc.strip()}\n"

        block += "\n" + ("=" * 20) + "\n"

        # Comment for readability (keeps augmentation from polluting runtime code)
        commented = "\n".join("# " + line for line in block.splitlines())
        augmentation += commented + "\n"

    # Prepend augmentation to prompt area
    task["prompt"] = f"\n{augmentation}\n\n{task['prompt']}"
    return task


def read_documents_from_disk(source: str, version: str, debug: bool, base_path: str) -> List[Dict]:
    apis = []

    if "torch" in source:
        source_ = "torch"
    elif "matplotlib" in source:
        source_ = "matplotlib"
    else:
        source_ = source

    path = os.path.join(base_path, source_, f"{version}.jsonl")
    print("Loading APIs from:", path)
    logging.info(f"Loading APIs from: {path}")

    if debug:
        this_apis = [json.loads(l) for l in open(path, "r")][:100]
    else:
        this_apis = [json.loads(l) for l in open(path, "r")]

    for d in this_apis:
        if "torch" in source:
            d["package"] = "torch"
        elif "matplotlib" in source:
            d["package"] = "matplotlib"
        else:
            d["package"] = source

    apis.extend(this_apis)
    return apis


def setup_retrieval(use_retrieval: bool, retrieval_config: Dict, documents: List[Dict]) -> Tuple:
    if not use_retrieval:
        return None, lambda _: False

    retrieval_type = retrieval_config["retrieval_type"]

    if retrieval_type == "codesage":
        retriever = CodeSageRetriever(
            model_name=retrieval_config.get("retriever_model", "./codesage-small-v2"),
            batch_size=retrieval_config.get("retriever_batch_size", 2),
            max_seq_length=retrieval_config.get("retriever_max_seq_length", 256),
            device=retrieval_config.get("retriever_device", "cpu"),
        )
        retriever.prepare_index(documents)
        return retriever, lambda _: True

    if retrieval_type == "no_augmentation":
        return None, lambda _: False

    raise NotImplementedError(f"Retrieval type '{retrieval_type}' is not supported")
