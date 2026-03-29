"""Code snippet to evaluate code generation models on the API call 
completion task"""

import argparse
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
import requests

from retriever import BaseRetriever, prepare_query
from retriever_utils import (augment_prompt, read_documents_from_disk,
                               setup_retrieval)
from utils import (has_new_api_call,
                   make_dirname_from_retrieval_config)
from eval_metric import evaluate_completions

EVAL_ROOT_DIR = ""
API_INFO_DIR = ""
MODEL_MAX_LEN = 8192
DEBUG = False
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s -- %(levelname)s: %(message)s"
)

def truncate_prompt(prompt, tokenizer, max_len=MODEL_MAX_LEN, truncate_from='left'):
    # Word-based truncation for Ollama or models without a local tokenizer
    words = prompt.split()
    num_words = len(words)
    if num_words > max_len:
        if truncate_from == 'left':
            truncated_words = words[num_words - max_len:]
        else:
            truncated_words = words[:max_len]
        return ' '.join(truncated_words)
    return prompt

def prompt_wrapper(question, right_context):
    return f"<fim_prefix>{question}<fim_suffix>{right_context}<fim_middle>"

def extract_generation(input_string, prefix_special_token: str = "<fim_middle>", suffix_special_token: str = "<|endoftext|>"):
    start_index = input_string.find(prefix_special_token)
    if start_index == -1:
        return ""
    start_index += len(prefix_special_token)
    end_index = input_string.find(suffix_special_token, start_index)
    extracted_text = input_string[start_index:end_index] if end_index != -1 else input_string[start_index:]
    if "<file_sep>" in extracted_text:
        extracted_text = extracted_text.split("<file_sep>")[0]
    return extracted_text

def generate_code_completion(
        model: Any, 
        tokenizer: Any, 
        prompt: str,
        hint_to_prompt: str,
        right_context: str,
        max_new_tokens: int = 512,
        new_scope_started: bool = False,
        imports: list = [],
        source: str = "torch",
        base_model_name: str = "mistral",
) -> Tuple:
    """Generate code completion up to the first API call."""
    hinted_prompt = prompt + hint_to_prompt
    
    # 1. Branch for Ollama
    if model == "ollama":
        url = "http://localhost:11434/api/generate"
        input_text = truncate_prompt(hinted_prompt, None, max_len=6144)
        
        # SYSTEM PROMPT: Forces Mistral to act as a completion engine, not a chatbot.
        system_instruction = (
            "You are a raw code completion engine. "
            "Continue the provided code snippet exactly where it leaves off. "
            "Do NOT explain. Do NOT use conversational language like 'It seems' or 'Note:'. "
            "Output ONLY the remaining code."
        )

        payload = {
            "model": base_model_name,
            "prompt": input_text,
            "system": system_instruction,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": 0.0,  # Zero temperature for benchmark consistency
                # Expanded stop tokens to catch conversational Mistral output
                "stop": ["\n\n", "def ", "class ", "if __name__", "It seems", "Note:", "This code"]
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            res_json = response.json()
            # Clean generation: Ollama gives only the NEW text
            raw_gen = res_json.get("response", "").strip()
            generation = hint_to_prompt + raw_gen
            
            this_has_new_api_call, ret_str = has_new_api_call(
                generation, new_scope_started=new_scope_started, imports=imports
            )
            return (ret_str, generation)
        except Exception as e:
            logging.error(f"Ollama error: {e}")
            return ("", "")

    # 2. Standard HF Transformers logic (Kept identical to core benchmark)
    if "starcoder" in base_model_name.lower():
        input_text = prompt_wrapper(hinted_prompt, right_context)
    else:
        input_text = hinted_prompt

    input_text = truncate_prompt(input_text, tokenizer=tokenizer, max_len=6144)
    tokenized = tokenizer(input_text, return_tensors="pt")
    input_ids = tokenized.input_ids.to(next(model.parameters()).device)
    attn_mask = tokenized.attention_mask.to(next(model.parameters()).device)
    
    max_new_tokens = min(max_new_tokens, MODEL_MAX_LEN - input_ids.shape[1])

    try:
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True
            )

        output_tokens = output["sequences"].cpu()[0]
        this_new_tokens = output_tokens[len(input_ids[0]):]
        
        if "starcoder" in base_model_name.lower():
            this_new_text = tokenizer.decode(output_tokens, skip_special_tokens=False)
            generation = hint_to_prompt + extract_generation(this_new_text)
        else:
            this_new_text = tokenizer.decode(this_new_tokens, skip_special_tokens=True)
            generation = hint_to_prompt + this_new_text

        this_has_new_api_call, ret_str = has_new_api_call(
            generation, new_scope_started=new_scope_started, imports=imports
        )
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        generation, ret_str = "", ""

    return (ret_str, generation)

def generate_for_sample(
        model: Any,
        tokenizer: Any,
        sample: Dict,
        max_new_tokens: int = 512,
        right_context: bool = False,
        source: str = "torch",
        base_model_name: str = "mistral",
) -> Dict:
    """Prepare and evaluate a single sample for the benchmark."""
    # Ensure all strings exist to avoid truncate errors
    prompt = sample.get("prompt", "")
    code_str = sample.get("code_str", "")
    right_context_str = sample.get("right_context_few_lines", "")

    sample["prompt"] = truncate_prompt(prompt, None, 4096)
    sample["code_str"] = truncate_prompt(code_str, None, 4096)
    sample["right_context_few_lines"] = truncate_prompt(right_context_str, None, 4096, 'right')

    response = generate_code_completion(
        model,
        tokenizer,
        sample["prompt"],
        sample.get("code_str", ""),
        sample.get("right_context_few_lines", "") if right_context else "",
        max_new_tokens,
        new_scope_started=False,
        imports=sample.get("imports", []),
        source=source,
        base_model_name=base_model_name,
    )

    sample.update({"hypothesis": response[0], "generation": response[1]})
    return sample

def generate_with_retrieval_for_sample(
        model: Any,
        tokenizer: Any,
        sample: Dict,
        max_new_tokens: int = 512,
        use_retrieval: bool = False,
        retriever: Optional[BaseRetriever] = None,
        retrieval_config: Optional[Dict] = None,
        right_context: bool = False,
        source: str = "torch",
        base_model_name: str = "mistral",
) -> Dict:
    retrieved = []
    if use_retrieval and retriever:
        # Use Api_Description field if enriched RAG is requested
        if retrieval_config.get("use_api_description"):
            query = sample.get("Api_Description", "")
        else:
            query = prepare_query(sample, use_prompt=True, use_comment_str=True, use_code_str=True)
        
        retrieved = retriever.retrieve(query, num_results=retrieval_config.get("num_to_retrieve", 1))

    conf_thresh = retrieval_config.get("retrieval_confidence_threshold", 0.0) if retrieval_config else 0.0

    # Apply RAG augmentation if we have results above threshold
    if retrieved and retrieved[0][1] >= conf_thresh:
        sample["retrieved_apis"] = [r[0].get("API_Call") for r in retrieved]
        sample = augment_prompt(sample, retrieved)
        
    return generate_for_sample(
        model, tokenizer, sample,
        max_new_tokens=max_new_tokens,
        right_context=right_context, 
        source=source,
        base_model_name=base_model_name
    )

def main_generate(
    model_name: str,
    base_model_name: str,
    model_path: Optional[str],
    task: str,
    source: str,
    use_retrieval: bool = False,
    retrieval_config: Optional[Dict] = None,
    load_pt_ckpt: bool = False,
    api_version: list = ["v1"],
    task_version: list = ["v1"],
    debug: bool = False,
    right_context: bool = False,
    setting_desc_variable: str = "results",
    use_ollama: bool = False
) -> None:
    model, tokenizer = None, None
    if use_ollama:
        model = "ollama"
        logging.info(f"Using Ollama with model: {base_model_name}")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logging.info(f"Loading HF model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
    
    for a_version, t_version in zip(api_version, task_version):
        prompts_file = os.path.join(EVAL_ROOT_DIR, f"data/{task}/{source}/{t_version}.jsonl")
        output_dir = os.path.join(
            EVAL_ROOT_DIR, setting_desc_variable,
            make_dirname_from_retrieval_config(retrieval_config),
            task, base_model_name, source
        )
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"task_{t_version}_doc_{a_version}.jsonl")

        logging.info(f"Reading from: {prompts_file}")
        tasks = [json.loads(l) for l in open(prompts_file, "r")]
        if debug: tasks = tasks[:10]

        retriever = None
        if use_retrieval:
            docs = read_documents_from_disk(base_path=API_INFO_DIR, source=source, version=a_version, debug=debug)
            retriever, _ = setup_retrieval(use_retrieval, retrieval_config, docs)

        with open(output_file, "w") as f:
            for sample in tqdm(tasks, desc="Generating"):
                res = generate_with_retrieval_for_sample(
                    model, tokenizer, sample, 
                    use_retrieval=use_retrieval,
                    retriever=retriever,
                    retrieval_config=retrieval_config,
                    base_model_name=base_model_name,
                    right_context=right_context,
                    source=source
                )
                f.write(json.dumps(res) + "\n")

def main_score_generations(
    model_name: str, 
    base_model_name: str,
    task: str,
    source: str,
    retrieval_config: Optional[Dict] = None,
    api_version: list = ["v1"],
    task_version: list = ["v1"],
    debug: bool = False,
    right_context: bool = False,
    setting_desc_variable: str = "results"
) -> None:
    # Logic to load ground truth for specific library
    lib_name = "torch" if "torch" in source else "matplotlib" if "matplotlib" in source else source
    
    for a_version, t_version in zip(api_version, task_version):
        package_api_info = {}
        # Clean enriched suffix for documentation path
        clean_version = a_version.replace("_enriched", "")
        gt_path = os.path.join(API_INFO_DIR, lib_name, f"{clean_version}.jsonl")
        
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                data = [json.loads(l) for l in f]
            package_api_info[lib_name] = {(d.get("API_Call") or d.get("api")): d for d in data}

        ret_dir = make_dirname_from_retrieval_config(retrieval_config)
        samples_path = os.path.join(EVAL_ROOT_DIR, setting_desc_variable, ret_dir, task, base_model_name, source, f"task_{t_version}_doc_{a_version}.jsonl")
        
        if not os.path.exists(samples_path):
            logging.error(f"Samples file not found: {samples_path}")
            continue

        with open(samples_path, "r") as f:
            samples = [json.loads(l) for l in f if "hypothesis" in json.loads(l)]

        detailed, summary = evaluate_completions(samples, package_api_info)

        out_path = samples_path.replace(".jsonl", "_results.jsonl")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Summary for {a_version}: EM={summary.get('em')}, ES={summary.get('es')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--mode", type=str, choices=["generate", "score"], required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--use_retrieval", action="store_true")
    parser.add_argument("--use_api_description", action="store_true")
    parser.add_argument("--use_ollama", action="store_true")
    parser.add_argument("--retrieval_type", type=str, default="no_augmentation")
    parser.add_argument("--num_to_retrieve", type=int, default=3)
    parser.add_argument("--api_version", type=str, default="v1")
    parser.add_argument("--task_version", type=str, default="v1")
    parser.add_argument("--setting_desc_variable", type=str, default="results")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval_root_directory", type=str, default=".")

    args = parser.parse_args()
    if not args.base_model: args.base_model = args.model
    EVAL_ROOT_DIR = args.eval_root_directory
    API_INFO_DIR = os.path.join(EVAL_ROOT_DIR, "data/package_apis")
    
    retrieval_config = {
        "retrieval_type": args.retrieval_type,
        "num_to_retrieve": args.num_to_retrieve,
        "use_api_description": args.use_api_description,
        "retrieval_confidence_threshold": 0.0,
        "retriever_model": "./codesage-small-v2",
    }

    if args.mode == "generate":
        main_generate(
            args.model, args.base_model, None, args.task, args.source,
            use_retrieval=args.use_retrieval, retrieval_config=retrieval_config,
            api_version=[args.api_version], task_version=[args.task_version],
            debug=args.debug, setting_desc_variable=args.setting_desc_variable,
            use_ollama=args.use_ollama
        )
    else:
        main_score_generations(
            args.model, args.base_model, args.task, args.source,
            retrieval_config=retrieval_config, api_version=[args.api_version],
            task_version=[args.task_version], setting_desc_variable=args.setting_desc_variable
        )