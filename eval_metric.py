import logging
import re
from typing import List, Dict, Tuple
import torch.multiprocessing as mp
from tqdm import tqdm
from functools import partial
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from eval_metric_utils import (
    process_examples,
    cal_edit_sim,
    compute_id_match,
)

# Tree-sitter setup
PY_LANGUAGE = Language(tspython.language())

def get_base_api_name(call_str: str) -> str:
    """Trích xuất tên hàm từ một lời gọi hàm. 
    Ví dụ: torch.std_mean(X, dim=...) -> torch.std_mean
    """
    if not call_str: return ""
    # Lấy phần chữ trước dấu mở ngoặc đầu tiên
    match = re.match(r'^([a-zA-Z0-9_\.]+)', call_str.strip())
    return match.group(1) if match else call_str.strip()

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
    
    print("post-processing samples ...")
    # Sử dụng Pool để xử lý song song
    pool = mp.Pool(mp.cpu_count() - 1)
    worker = partial(process_examples)

    with tqdm(total=len(samples)) as pbar:
        for prompt, hypothesis, target in data:
            output = worker(prompt, hypothesis, target)
            trunc_s, em_label, _ = output # api_label không dùng đến ở đây
            em_labels.append(em_label)
            truncated_samples.append(trunc_s)
            pbar.update()

    # 2. MRR Calculation Logic (Đã sửa lỗi thụt đầu dòng và so khớp)
    mrr_sum = 0.0
    for sample in samples:
        # Lấy tên gốc (ví dụ: torch.std_mean)
        full_gt = sample.get("exact_call", "")
        gt_base_name = get_base_api_name(full_gt)
        
        retrieved_list = sample.get("retrieved_apis", [])
        
        rank = 0
        for i, item in enumerate(retrieved_list):
            # CodeSage trả về Dict chứa 'API_Call', hoặc String
            if isinstance(item, dict):
                retrieved_api_name = item.get("API_Call", "")
            else:
                retrieved_api_name = str(item)
            
            # So sánh sau khi đã trích xuất tên gốc
            if retrieved_api_name.strip() == gt_base_name:
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
        # identifier_em: API-EM (đúng hoàn toàn identifier)
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

    # Tính toán các tỷ lệ cuối cùng
    em_ratio = round(exact_match / len(samples) * 100, 2)
    edit_sim = round(sum(edit_similarities) / len(edit_similarities), 2)
    id_em_ratio = round(sum(id_em) / len(truncated_samples) * 100, 2)
    
    id_prec_total = sum(id_true_pos) / (sum(id_true_pos) + sum(id_false_pos) + 1e-9)
    id_rec_total = sum(id_true_pos) / (sum(id_true_pos) + sum(id_false_neg) + 1e-9)
    
    id_precision = round(id_prec_total * 100, 2)
    id_recall = round(id_rec_total * 100, 2)
    id_f1 = round(2 * (id_precision * id_recall) / (id_precision + id_recall + 1e-9), 2)
    
    res = {
        "em": em_ratio,
        "es": edit_sim,
        "mrr": mrr_score,
        "id_em": id_em_ratio,
        "id_precision": id_precision,
        "id_recall": id_recall,
        "id_f1": id_f1, # Đây chính là API-F1 trong paper
        "total": len(truncated_samples)
    }

    # Logging
    logging.info(f"Code Matching: EM {em_ratio:.2f}, ES {edit_sim:.2f}, MRR {mrr_score:.2f}")
    logging.info(f"ID matching (API-F1): {id_f1:.2f}")

    return detailed_results, res
