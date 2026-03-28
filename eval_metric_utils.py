"""
Evaluation metric utilities for code processing and analysis.

This module provides utility functions for processing code completions,
validating syntax, and extracting code elements for evaluation.
"""

import re
import timeout_decorator
from typing import List, Optional, Any

from tree_sitter import Language, Parser
from fuzzywuzzy import fuzz

# Optional dependencies with fallbacks
try:
    from keywords.keywordslist import get_language_keywords
except ImportError:
    def get_language_keywords(programming_language: str) -> List[str]:
        """Fallback function when keywords module is not available."""
        return []

try:
    from nltk.tokenize import RegexpTokenizer
    code_tokenizer = RegexpTokenizer(r'\w+')
except ImportError:
    # Simple fallback tokenizer when NLTK is not available
    code_tokenizer = type('SimpleTokenizer', (), {
        'tokenize': lambda text: text.split() if text else []
    })()

# Tree-sitter language setup
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
PY_LANGUAGE = Language(tspython.language())

# Constants for text processing
string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''
IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')


def cal_edit_sim(references, hypotheses):
    total = len(references)
    edit_sim = 0.0
    for pred, gt in zip(hypotheses, references):
        pred = pred.strip()
        gt = gt.strip()
        edit_sim += fuzz.ratio(pred, gt)
    return edit_sim / total


def compute_id_match(pred_ids, target_ids):
    pred_ids = list(set(pred_ids))
    target_ids = list(set(target_ids))
    tp = 0
    fp = 0
    fn = 0
    for pid in pred_ids:
        if pid in target_ids:
            tp += 1
        else:
            fp += 1
    for tid in target_ids:
        if tid not in pred_ids:
            fn += 1
    return tp, fp, fn


@timeout_decorator.timeout(5)
def parse_code_to_ast(parser: Any, source_code: str) -> Optional[Any]:
    """
    Parse source code into an Abstract Syntax Tree (AST) using Tree-sitter.
    
    Args:
        parser: Tree-sitter parser instance
        source_code: Source code to parse (str or bytes)
        
    Returns:
        Parsed AST tree or None if parsing fails
    """
    assert isinstance(source_code, str) or isinstance(source_code, bytes)
    if isinstance(source_code, str):
        source_code = bytes(source_code, "utf8")
    try:
        syntax_tree = parser.parse(source_code)
        return syntax_tree
    except Exception as parsing_error:
        return None


def is_valid_python_syntax(parser: Any, source_code: str) -> bool:
    """
    Check if the source code has valid Python syntax using Tree-sitter.
    
    Args:
        parser: Tree-sitter parser instance
        source_code: Source code to validate
        
    Returns:
        True if syntax is valid, False otherwise
    """
    def has_syntax_error(ast_node: Any) -> bool:
        """Recursively check for syntax errors in AST nodes."""
        if ast_node.type == "ERROR":
            return True
        try:
            for child_node in ast_node.children:
                if has_syntax_error(child_node):
                    return True
        except RecursionError:
            return True
        return False

    syntax_tree = parse_code_to_ast(parser, source_code)
    if syntax_tree is not None:
        return not has_syntax_error(syntax_tree.root_node)
    return False


def extract_first_valid_python_statement(prompt_context: str, code_completion: str, syntax_parser: Any) -> str:
    """
    Extract the first valid Python statement from a code completion.
    
    Args:
        prompt_context: The code context before the completion
        code_completion: The code completion to analyze
        syntax_parser: Tree-sitter parser for Python
        
    Returns:
        The first valid Python statement or the full completion if no valid statement found
    """
    for char_index in range(len(code_completion)):
        combined_code = prompt_context + code_completion[:char_index + 1]
        if not is_valid_python_syntax(syntax_parser, combined_code):
            continue
        if char_index + 1 < len(code_completion) and code_completion[char_index + 1] == "\n":
            return code_completion[:char_index + 1].rstrip()
    return code_completion


def is_valid_identifier(token: str, programming_language: str = None) -> bool:
    """
    Check if a token is a valid identifier (not a language keyword).
    
    Args:
        token: Token to check
        programming_language: Programming language for keyword checking (optional)
        
    Returns:
        True if token is a valid identifier, False otherwise
    """
    if not IDENTIFIER_REGEX.match(token):
        return False
    
    if programming_language is not None:
        return token not in get_language_keywords(programming_language)
    
    return True


def remove_code_comments(source_code: str) -> str:
    """
    Remove comments from source code (Python # and C-style // comments).
    
    Args:
        source_code: Source code with comments
        
    Returns:
        Source code with comments removed
    """
    cleaned_code = re.sub(r'#.*', '', source_code)
    cleaned_code = re.sub(r'//.*', '', cleaned_code)
    return cleaned_code


def postprocess_completion_by_language(programming_language: str, prompt_context: str, 
                                     code_completion: str, syntax_parser: Any = None) -> str:
    """
    Post-process code completion based on the programming language.
    
    Args:
        programming_language: The programming language (e.g., 'python')
        prompt_context: Code context before completion
        code_completion: The code completion to process
        syntax_parser: Parser for syntax validation (required for Python)
        
    Returns:
        Post-processed code completion
    """
    if programming_language != "python":
        raise NotImplementedError(f"Language '{programming_language}' is not supported in this release")
    
    if syntax_parser is None:
        raise ValueError("Python syntax parser is required for Python code")
    
    try:
        return extract_first_valid_python_statement(prompt_context, code_completion, syntax_parser)
    except Exception as processing_error:
        return code_completion


def extract_code_identifiers(source_code: str, programming_language: str) -> List[str]:
    """
    Extract all valid identifiers from source code, excluding strings and keywords.
    
    The process:
    1. Remove string literals to avoid extracting identifiers from strings
    2. Tokenize the code to get all words
    3. Filter for valid identifiers using regex and keyword checking
    
    Args:
        source_code: Source code to analyze
        programming_language: Programming language for keyword filtering
        
    Returns:
        List of valid identifiers found in the code
    """
    # Remove string literals to avoid extracting identifiers from strings
    code_without_strings = re.sub(string_pattern, '', source_code)
    
    # Tokenize and filter for valid identifiers
    tokens = code_tokenizer.tokenize(code_without_strings)
    identifiers = [token for token in tokens if is_valid_identifier(token, programming_language)]
    
    return identifiers


def extract_function_call_name(code_line: str) -> str:
    """
    Extract the first function call name from a line of code.
    
    A function call is identified as a word followed by an opening parenthesis.
    
    Args:
        code_line: A single line of source code
        
    Returns:
        The function name if found, empty string otherwise
    """
    function_call_match = re.search(r'\b(\w+)\(', code_line)
    return function_call_match.group(0)[:-1] if function_call_match else ''


def process_examples(prompt, hypothesis, target, lang = "python"):
    # sample, ex = args
    parser = Parser(PY_LANGUAGE)
    prediction = postprocess_completion_by_language(lang, prompt, hypothesis, parser)
    prediction = remove_code_comments(prediction)
    target = target
    target = remove_code_comments(target)

    pred_lines = [l.strip() for l in prediction.split("\n") if l.strip()]
    gt_lines = [l.strip() for l in target.split("\n") if l.strip()]
    em_label = int(pred_lines == gt_lines)
    target_api_call = extract_function_call_name(target)
    hypothesis_api_call = extract_function_call_name(hypothesis)
    api_call_match = int(target_api_call == hypothesis_api_call)

    pred_ids = extract_code_identifiers(prediction, lang)
    target_ids = extract_code_identifiers(target, lang)

    trunc_s = {
        # "task_id": sample["task_id"],
        "pred": prediction,
        "target": target,
        "pred_ids": pred_ids,
        "target_ids": target_ids
    }
    return trunc_s, em_label, api_call_match
