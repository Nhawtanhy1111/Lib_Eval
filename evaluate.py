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
                   make_dirname_from_retrieval_config,
                   load_huggingface_model_from_pytorch_checkpoint)
from eval_metric import evaluate_completions

EVAL_ROOT_DIR = ""
API_INFO_DIR = os.path.join(EVAL_ROOT_DIR, "data/package_apis")
MODEL_MAX_LEN = 8192
DEBUG = False
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s -- %(levelname)s: %(message)s"
)

#     # Tokenize the prompt
    
#     # Calculate the number of tokens
    
#     # Check if the number of tokens exceeds the maximum length
#         # Calculate the number of tokens to truncate
        
#         # Truncate the prompt from the left
        
#         # Convert the truncated tokens back to string
#     else:
def truncate_prompt(prompt, tokenizer, max_len=MODEL_MAX_LEN, truncate_from='left'):
    # Tokenize the prompt
    # Truncate by words for Ollama (no tokenizer)
    words = prompt.split()
    num_words = len(words)
    if num_words > max_len:
        if truncate_from == 'left':
            truncated_words = words[num_words - max_len:]
        else:
            truncated_words = words[:max_len]
        return ' '.join(truncated_words)
    else:
        return prompt

def prompt_wrapper(question, right_context):
    system_prompt = (
        "<fim_prefix>"
    )

    system_prompt2 = (
        "<fim_suffix>"
        "<fim_middle>"
    )

    # Combine the system prompt with the specific question provided
    full_prompt = "<fim_prefix>" + question + "<fim_suffix>" + right_context + "<fim_middle>"

    # Return the combined prompt
    return full_prompt


def extract_generation(input_string, prefix_special_token: str = "<fim_middle>", suffix_special_token: str = "<|endoftext|>"):
    # Find the starting index of the target text
    start_index = input_string.find(prefix_special_token)
    
    if start_index == -1:
        return ""
    
    start_index += len(prefix_special_token)
    
    # Find the end index; if it doesn't exist, take till the end of the string
    end_index = input_string.find(suffix_special_token, start_index)
    if end_index == -1:
        extracted_text = input_string[start_index:]
    else:
        extracted_text = input_string[start_index:end_index]
    if "<file_sep>" in extracted_text:
        extracted_text = extracted_text.split("<file_sep>")[0]
    return extracted_text

def generate_code_completion(
        model,
        tokenizer,
        prompt,
        hint_to_prompt,
        right_context,
        max_new_tokens=512,
        new_scope_started=False,
        imports=[],
        source="torch",
        base_model_name="starcoder2-7b",
):
    ollama_url = "http://localhost:11434/api/generate"

    left_context = prompt + hint_to_prompt
    if right_context:
        input_prompt = prompt_wrapper(left_context, right_context)
    else:
        input_prompt = left_context

    payload = {
        "model": "mistral",
        "prompt": input_prompt,
        "raw": True,
        "stream": False,
        "options": {
            "num_predict": 128,
            "num_ctx": 8192,
            "temperature": 0
        }
    }

    try:
        session = requests.Session()
        session.trust_env = False
        response = session.post(ollama_url, json=payload, timeout=1500)
        response.raise_for_status()

        result = response.json()
        generation = result.get("response", "")

        this_has_new_api_call, ret_str = has_new_api_call(
            generation,
            new_scope_started=new_scope_started,
            imports=imports
        )
    except Exception as e:
        logging.error(f"Ollama API error: {str(e)}", exc_info=True)
        generation = ""
        this_has_new_api_call, ret_str = False, ""

    return (ret_str, generation)

def generate_for_sample(
        model: Any,
        tokenizer: Any,
        sample: Dict,
        max_new_tokens: int = 512,
        right_context: bool = False,
        source: str = "torch",
        base_model_name: str = "starcoder2-7b",
) -> bool:
    """Evaluate the model on a single sample."""

    sample["prompt"] = truncate_prompt(sample["prompt"], None, 4096)
    sample["code_str"] = truncate_prompt(sample["code_str"], None, 4096)
    sample["right_context_few_lines"] = truncate_prompt(sample["right_context_few_lines"], None, 4096, 'right')

    # generate a response
    response = generate_code_completion(
        model,
        tokenizer,
        sample["prompt"],
        sample["code_str"],
        sample["right_context_few_lines"] if right_context else "",
        max_new_tokens,
        new_scope_started=False,
        imports=sample["imports"],
        source=source,
        base_model_name=base_model_name,

    )

    res_dict = {"hypothesis": response[0], "generation": response[1]}
    sample.update(res_dict)
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
        base_model_name: str = "starcoder2-7b",
) -> bool:
    """This function performs generation optionally with retrieval. It supports
    all use cases as in `generate_for_sample` which is used as a subroutine here.
    When `use_retrieval` and `generate_before_retrieval` are both True, it will
    perform retrieval and augment the prompt before generation.
    """

    if use_retrieval:
        # formulate the query for retrieval
        query = prepare_query(
            sample,
            normalize=False,
            use_hypothesis=False,
            use_prompt=True,
            use_comment_str=True,
            use_code_str=True,
        )
        # retrieve relevant documents
        retrieved = retriever.retrieve(
        query, num_results=retrieval_config.get("num_to_retrieve", 1)
    )

    conf_thresh = retrieval_config.get("retrieval_confidence_threshold", 0.0)

    if not retrieved or retrieved[0][1] < conf_thresh:
        retrieved = []
        
        # store retrieved APIs for analysis
        sample["retrieved_apis"] = []
        for r in retrieved:
            sample["retrieved_apis"].append(r[0]["API_Call"])

        # perform augmentation with the retrieved documents
        updated_sample = augment_prompt(sample, retrieved)
        
        # generate with augmented prompt
        generate_response = generate_for_sample(
            model, tokenizer, updated_sample,
            max_new_tokens=max_new_tokens,
            right_context=right_context, 
            source=source,
            base_model_name=base_model_name
        )
    else:
        generate_response = generate_for_sample(
            model, tokenizer, sample,
            max_new_tokens=max_new_tokens,
            right_context=right_context,
            source=source,
            base_model_name=base_model_name
        )
    return generate_response


def main_generate(
        model_name: str,
        base_model_name: str,
        model_path: str,
        task: str,
        source: str,
        load_pt_ckpt: bool = False,
        use_retrieval: bool = False,
        retrieval_config: Optional[Dict] = None,
        api_version: list = ["v1"],
        task_version: list = ["v1"],
        debug: bool = False,
        right_context: bool = False,
        setting_desc_variable: str = "results"
) -> None:
    # Ollama does not require explicit model/tokenizer loading
    model = None
    tokenizer = None
    # set up paths
    prompts_file_tplt = os.path.join(
        EVAL_ROOT_DIR, "data", task, "{datadir}/{source}/{version}.jsonl"
    )
    output_file_tplt = os.path.join(
        EVAL_ROOT_DIR, setting_desc_variable, 
        make_dirname_from_retrieval_config(retrieval_config), 
        task, 
        "{datadir}/{model_name}/{source}/{fn}.jsonl"
    )

    # load tasks
    datadir = "."
    for a_version, t_version in zip(api_version, task_version):
        # setup paths
        fn = f"task_{t_version}_doc_{a_version}"
        prompts_file = prompts_file_tplt.format(
            datadir=datadir, source=source, version=t_version
        )
        # )
        output_file = output_file_tplt.format(
            datadir=datadir, source=source, fn=fn, model_name=base_model_name
        )
        logging.info(f"Reading from: {prompts_file}")
        logging.info(f"Generating to: {output_file}")
        # load the data and set up output file
        tasks = [json.loads(l) for l in open(prompts_file, "r").readlines()]

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # set up retrieval
        retrieval_config["tokenizer"] = tokenizer # add because it"s needed
        index_documents = read_documents_from_disk(source, a_version, debug, API_INFO_DIR)

        retriever, retrieval_trigger = setup_retrieval(
            use_retrieval, retrieval_config, index_documents
        )

        if debug:
            tasks = tasks[:10]

        with open(output_file, "w") as f:
            for this_task in tqdm(tasks, desc="Generating"):
                try:
                    # generate a response
                    ret = generate_with_retrieval_for_sample(
                        model, tokenizer, this_task, 
                        use_retrieval=use_retrieval,
                        retriever=retriever,
                        retrieval_config=retrieval_config,
                        right_context=right_context,
                        source=source,
                        base_model_name=base_model_name
                    )
                    f.write(json.dumps(ret) + "\n")
                except Exception as e:
                    f.write(json.dumps({
                        "error": str(e), 
                        "prompt": this_task["prompt"]
                    }) + "\n")
                    if DEBUG:
                        raise
                f.flush()


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

    datadir = "."
    #     EVAL_ROOT_DIR, setting_desc_variable, 
    # )
    samples_path_tmplt = os.path.join(
        EVAL_ROOT_DIR, setting_desc_variable, 
        "{retrieval_type}/{task}/{datadir}/{base_model_name}/{source}/{fn}.jsonl"
    )
    #     EVAL_ROOT_DIR, setting_desc_variable,
    # )
    output_path_tmplt = os.path.join(
        EVAL_ROOT_DIR, setting_desc_variable,
        "{retrieval_type}/{task}/{datadir}/{base_model_name}/{source}/{fn}_results.jsonl"
    )
    #     EVAL_ROOT_DIR, setting_desc_variable,
    # )
    detailed_output_path_tmplt = os.path.join(
        EVAL_ROOT_DIR, setting_desc_variable,
        "{retrieval_type}/{task}/{datadir}/{base_model_name}/{source}/{fn}_detailed_results.jsonl"
    )
    #     EVAL_ROOT_DIR, setting_desc_variable,
    # )
    

    for a_version, t_version in zip(api_version, task_version):
        # load data necessary for evaluation
        package_api_info = {}
        # Map source names to their directory names
        source_mapping = {
            "torch": "torch",
            "matplotlib": "matplotlib"
        }
        
        if any(s in source for s in source_mapping.keys()):
            # Find which source matches and get its directory name
            source_ = next(source_mapping[s] for s in source_mapping.keys() if s in source)
            package_files = [os.path.join(API_INFO_DIR, f"{source_}/{a_version}.jsonl")]
        else:
            # For other sources, collect all JSONL files in the source's versioned directory
            path = os.path.join(API_INFO_DIR, source, f"{a_version}")
            package_files = glob.glob(f"{path}/*.jsonl")
        for mod in tqdm(package_files, desc="Loading API info"):
            data = [json.loads(l) for l in open(mod, "r").readlines()]
            data = {d.get("function_name") or d.get("api"): d for d in data if "function_name" in d or "api" in d}
            mod_name = mod.split("/")[-1].split(".")[0]
            package_api_info[mod_name] = data

        # setup paths
        fn = f"task_{t_version}_doc_{a_version}"
        samples_path = samples_path_tmplt.format(
            task=task,
            retrieval_type=make_dirname_from_retrieval_config(retrieval_config),
            datadir=datadir, fn=fn, base_model_name=base_model_name, source=source
        )
        output_path = output_path_tmplt.format(
            task=task, 
            retrieval_type=make_dirname_from_retrieval_config(retrieval_config),
            datadir=datadir, fn=fn, base_model_name=base_model_name, source=source
        )
        detailed_output_path = detailed_output_path_tmplt.format(
            task=task, 
            retrieval_type=make_dirname_from_retrieval_config(retrieval_config),
            datadir=datadir, fn=fn, base_model_name=base_model_name, source=source
        )


        # load model generations
        logging.info(f"Reading from {samples_path}")
        samples = [json.loads(l) for l in open(samples_path, "r").readlines()]
        for r in samples:
            right_context = ""
            if r.get("right_context") is not None:
                right_context=r["right_context"]
            elif r["meta_data"]["right_context"] is not None:
                right_context=r["meta_data"]["right_context"]
            elif r.get("right_context_few_lines") is not None:
                right_context = r["right_context_few_lines"]
            else:
                right_context = ""
            r['hypothesis'] = has_new_api_call(
                r['hypothesis'], new_scope_started=False, imports=r["imports"],
            )[1]
            

        # score generations
        detailed_results, res = evaluate_completions(
            samples,
            package_api_info,
            num_proc=1,
            eval_correct_api_call=False
        )

        # # Write the summary to the new file
        with open(detailed_output_path, "w") as f:
            for r in detailed_results:
                f.write(json.dumps(r) + "\n")
        logging.info(f"DetailedResults written to {detailed_output_path}")

        # write out results
        with open(output_path, "w") as f:
            f.write(json.dumps(res, indent=2))
            f.write("\n")  # Add a newline for readability
        logging.info(f"Results written to {output_path}")
        # Writing out detailed results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experimental settings args
    parser.add_argument("--model", type=str, help="Name of the model",
                        required=True)
    parser.add_argument("--model_path", type=str, help="Path to the model ckpt")
    parser.add_argument("--base_model", type=str, 
                        help="Base model used in case of finetuned model ckpt")
    parser.add_argument("--mode", type=str, choices=["generate", "score"], 
                        required=True)
    parser.add_argument("--task", type=str, required=True, 
                        choices=["eval_examples"])
    parser.add_argument("--source", type=str, required=True, 
                        )

    # retrieval setup args
    parser.add_argument("--use_retrieval", action="store_true")

    parser.add_argument("--eval_root_directory", type=str, default=".")
    parser.add_argument("--retrieval_type", type=str, default="no_augmentation")
    parser.add_argument("--num_to_retrieve", type=int)
    parser.add_argument("--load_pt_ckpt", action="store_true",
                        help="Load a pytorch checkpoint instead of a "
                             "HF checkpoint.")


    parser.add_argument("--api_version", type=str, help="Version of the API information", default="v1")
    parser.add_argument("--task_version", type=str, help="Version of the task data", default="v1")
    parser.add_argument("--setting_desc_variable", type=str, help="dynamic variable to change path of logging based on different variables", default="results")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_right_context", action="store_true", help="Disable right context if set")


    args = parser.parse_args()

    # else:
    EVAL_ROOT_DIR = args.eval_root_directory
    API_INFO_DIR = os.path.join(EVAL_ROOT_DIR, "data/package_apis")
    args.right_context = not args.no_right_context
    # sanity checks
    if not args.use_retrieval:
        assert args.retrieval_type == "no_augmentation"
    if args.base_model is None:
        logging.info(f"base_model not specified; setting to {args.model}")
        args.base_model = args.model
    
    # set up the retrieval config
    retrieval_config = {
        "retrieval_type": args.retrieval_type,
        "num_to_retrieve": args.num_to_retrieve,
        "retrieval_confidence_threshold": 0.0,
        "retriever_model": "./codesage-small-v2",
        "retriever_batch_size": 2,
        "retriever_max_seq_length": 256,
        "retriever_device": "cpu",
    }
    if args.retrieval_type not in ["no_augmentation", "codesage"]:
        raise NotImplementedError(f"Unsupported retrieval type: {args.retrieval_type}")
    
    if args.mode == "generate":
        main_generate(
            args.model, 
            args.base_model,
            args.model_path,
            args.task,
            args.source,
            use_retrieval=args.use_retrieval,
            retrieval_config=retrieval_config,
            load_pt_ckpt=args.load_pt_ckpt,
            api_version=[args.api_version], # new
            task_version=[args.task_version], # new
            debug=args.debug,
            right_context=args.right_context,
            setting_desc_variable=args.setting_desc_variable
        )
    else:
        main_score_generations(
            args.model, 
            args.base_model,
            args.task,
            args.source,
            retrieval_config=retrieval_config,
            api_version=[args.api_version], # new
            task_version=[args.task_version], # new
            debug=args.debug,
            right_context=args.right_context,
            setting_desc_variable=args.setting_desc_variable
        )
