"""
Evaluation metrics for code completion tasks.

This module provides the main evaluation functions for assessing
code completion quality using Edit Similarity and Exact Match metrics.
"""

import logging
from typing import List, Dict, Tuple
import torch.multiprocessing as mp
from tqdm import tqdm
from functools import partial

from tree_sitter import Language, Parser
from eval_metric_utils import (
    process_examples,
    cal_edit_sim,
    compute_id_match,
)

# Tree-sitter setup for syntax parsing
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
PY_LANGUAGE = Language(tspython.language())

def evaluate_completions(
        samples: List[Dict],
        gt_api_dicts: Dict[str, Dict],
        eval_correct_api_call: bool = False,
        num_proc: int = 1
) -> Tuple[List[Dict], Dict]:
    parser = Parser(PY_LANGUAGE)

    # 1. Standard Metrics Setup
    data = [(
        sample["prompt"],
        sample["hypothesis"],
        "" if "target" not in sample else sample["target"],
    ) for sample in samples]

    truncated_samples = []
    em_labels = []
    api_labels = []
    
    print("post-processing samples ...")
    pool = mp.Pool(mp.cpu_count() - 1)
    worker = partial(process_examples)

    with tqdm(total=len(samples)) as pbar:
        for prompt, hypothesis, target in data:
            output = worker(prompt, hypothesis, target)
            trunc_s, em_label, api_label = output
            em_labels.append(em_label)
            api_labels.append(api_label)
            truncated_samples.append(trunc_s)
            pbar.update()

    # 2. MRR Calculation Logic
    mrr_sum = 0.0
    for sample in samples:
        # We look for the ground truth API in the retrieved list
        ground_truth = sample.get("exact_call") # Original target API name
        retrieved_list = sample.get("retrieved_apis", [])
        
        rank = 0
        for i, api_name in enumerate(retrieved_list):
            if api_name == ground_truth:
                rank = i + 1
                break
        
        if rank > 0:
            mrr_sum += (1.0 / rank)
    
    mrr_score = round((mrr_sum / len(samples)) * 100, 2) if samples else 0.0

    # 3. Existing EM/ES/ID Calculation
    exact_match = sum(1 for label in em_labels if label == 1)
    
    id_em, id_true_pos, id_false_pos, id_false_neg = [], [], [], []
    edit_similarities = []
    detailed_results = []

    for idx, trunc_s in enumerate(truncated_samples):
        identifier_em = int(trunc_s["pred_ids"] == trunc_s["target_ids"])
        es = cal_edit_sim([trunc_s["target"]], [trunc_s["pred"]])
        id_tp, id_fp, id_fn = compute_id_match(trunc_s["pred_ids"], trunc_s["target_ids"])

        id_em.append(identifier_em)
        id_true_pos.append(id_tp)
        id_false_pos.append(id_fp)
        id_false_neg.append(id_fn)
        edit_similarities.append(es)

        detailed_results.append({
            "em": em_labels[idx],
            "es": es,
            "id_em": identifier_em,
            "id_precision": id_tp / (id_tp + id_fp) if (id_tp + id_fp) != 0 else 0,
            "id_recall": id_tp / (id_tp + id_fn) if (id_tp + id_fn) != 0 else 0
        })

    # Summary Dict
    em_ratio = round(exact_match / len(samples) * 100, 2)
    edit_sim = round(sum(edit_similarities) / len(edit_similarities), 2)
    id_em_ratio = round(sum(id_em) / len(truncated_samples) * 100, 2)
    
    id_prec_sum = sum(id_true_pos) / (sum(id_true_pos) + sum(id_false_pos) + 1e-9)
    id_rec_sum = sum(id_true_pos) / (sum(id_true_pos) + sum(id_false_neg) + 1e-9)
    
    id_precision = round(id_prec_sum * 100, 2)
    id_recall = round(id_rec_sum * 100, 2)
    id_f1 = round(2 * (id_precision * id_recall) / (id_precision + id_recall + 1e-9), 2)
    
    res = {
        "em": em_ratio,
        "es": edit_sim,
        "mrr": mrr_score, # New MRR Metric
        "id_em": id_em_ratio,
        "id_precision": id_precision,
        "id_recall": id_recall,
        "id_f1": id_f1,
        "total": len(truncated_samples)
    }

    # Logging updates
    logging.info(f"Code Matching: EM {em_ratio:.2f}, ES {edit_sim:.2f}, MRR {mrr_score:.2f}")
    logging.info(f"ID matching: EM {id_em_ratio:.2f}, F1 {id_f1:.2f}")

    return detailed_results, res