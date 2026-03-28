from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import re

np.random.seed(0)


def _clean_signature(sig: str) -> str:
    if not isinstance(sig, str):
        return "No signature available."
    # Strip Matplotlib doc artifacts and normalize arrow
    return (sig
            .replace("[source]\u00b6", "")
            .replace("[source]", "")
            .replace("\u00b6", "")
            .replace("\u2192", "->")
            .replace(")#:", "):")
            .strip())

def prepare_key(api_dict: Dict, normalize: bool = False) -> str:
    """Prepare a unified key string for indexing (Torch-style for both libs)."""

    # --- Map to a unified schema (works even if matplotlib wasn't pre-converted) ---
    api_call = api_dict.get("API_Call") or api_dict.get("API_Name") or "Unknown API Call"
    signature = _clean_signature(api_dict.get("Signature", "No signature available."))
    description = api_dict.get("Detailed_Description", "No description available.")
    params = api_dict.get("Parameters") or {}

    # Ensure types we expect
    if not isinstance(description, str):
        description = str(description)
    if not isinstance(params, dict):
        params = {}

    # --- Build a consistent key body (same for torch & matplotlib) ---
    # 1) Signature line
    # 2) Inputs (names only, concise)
    # 3) Commented, line-by-line description (keeps it readable in-ctx)
    commented_description = "\n".join(f"# {line}" for line in description.split("\n"))
    inputs_section = "Inputs: " + ", ".join(params.keys()) if params else "Inputs:"

    # Keep format very close to your Torch branch, but include api_call context
    key = f"{api_call} : {signature}\n{inputs_section}\n{commented_description}"

    # Optional normalization step (if you have a normalize_text helper)
    if normalize:
        try:
            return normalize_text(key)  # assumes your normalize_text exists in scope
        except NameError:
            # Fall back gracefully if normalize_text isn't available
            return key

    return key


def process_api_full(api_full):
    """
    Process the 'api_full' string to keep only the method and attribute access,
    removing any method arguments.
    """
    # Remove parentheses and their contents, including nested ones
    # This pattern matches parentheses and their contents
    pattern = r'\([^()]*\)'
    while re.search(pattern, api_full):  # Keep removing until all parentheses and their contents are removed
        api_full = re.sub(pattern, '', api_full)
    return api_full


def prepare_query(
    generation_dict: Dict,
    normalize: bool = False,
    tree_style_normalize: bool = False,
    use_hypothesis: bool = False,
    use_prompt: bool = False,
    use_comment_str: bool = False,
    use_code_str: bool = False,
) -> str:
    prompt = "\n".join(generation_dict.get("prompt", "").split("\n")[-5:])
    hypothesis = generation_dict.get("hypothesis", "")
    comment_str = generation_dict.get("comment_str", "")
    code_str = generation_dict.get("code_str", "")

    if normalize:
        func = normalize_text
    elif tree_style_normalize:
        func = process_api_full
    else:
        func = lambda x: x

    parts = []
    if use_prompt:
        parts.append(prompt)
    if use_hypothesis:
        parts.append(hypothesis)
    if use_comment_str:
        parts.append(comment_str)
    if use_code_str:
        parts.append(code_str)

    query = "\n".join([p for p in parts if p])
    return func(query)


#     # Replace these characters with an empty string
def normalize_text(text: str) -> str:
    text = text.lower()
    import re
    # Pattern to match characters that are not letters, numbers, or spaces
    pattern_non_alphanumeric = r"[^a-zA-Z0-9\s]"
    # Replace these characters with a space
    cleaned_text = re.sub(pattern_non_alphanumeric, " ", text)
    # Pattern to match any numbers
    pattern_numbers = r"\d+"
    # Remove numbers from the cleaned text
    cleaned_text_without_numbers = re.sub(pattern_numbers, "", cleaned_text)
    return cleaned_text_without_numbers

class BaseRetriever(ABC):
    """Base class to define all types of retrievers"""
    def __init__(self):
        self.index = None

    @abstractmethod
    def prepare_index(self, documents: List[Dict]):
        pass
    
    @abstractmethod
    def retrieve(self, query: str, num_results: int = 5) -> List[Tuple[Dict, float]]:
        pass

import numpy as np
from sentence_transformers import SentenceTransformer

from sentence_transformers import SentenceTransformer
import numpy as np

class CodeSageRetriever(BaseRetriever):
    def __init__(
        self,
        model_name="./codesage-small-v2",
        batch_size=2,
        max_seq_length=256,
        device="cpu",
    ):
        super().__init__()
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
        )
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.documents = []
        self.doc_embeddings = None

    def prepare_index(self, documents):
        self.documents = documents
        doc_texts = [prepare_key(doc, normalize=False) for doc in documents]

        self.doc_embeddings = self.model.encode(
            doc_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        self.index = self.doc_embeddings

    def retrieve(self, query: str, num_results: int = 5):
        q = self.model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)

        scores = self.doc_embeddings @ q
        top_idx = np.argsort(-scores)[:num_results]
        return [(self.documents[i], float(scores[i])) for i in top_idx]