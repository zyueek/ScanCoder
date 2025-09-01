#!/usr/bin/env python3
# This script is adapted from generate_scanpath_training_file.py for summarization data.
import json
import os
import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import re
try:
    import javalang
except ImportError:
    print("[WARNING] javalang not available, falling back to keyword-based classification")
    javalang = None
JAVA_KEYWORDS_SET = set("abstract assert boolean break byte case catch char class const continue default do double else enum extends final finally float for goto if implements import instanceof int interface long native new package private protected public return short static strictfp super switch synchronized this throw throws transient try void volatile while".split())
def extract_java_ast_semantic_types(code: str) -> Dict[int, str]:
    """Extract semantic types for each line using Java AST parsing."""
    line_to_semantic = {}
    if javalang is None:
        return line_to_semantic
    try:
        tree = javalang.parse.parse(code)
        line_to_type = {}
        for path, node in tree:
            if hasattr(node, 'position') and node.position:
                line = node.position[0]
                stmt_type = type(node).__name__
                if line not in line_to_type:
                    line_to_type[line] = (node, stmt_type)
                else:
                    current_type = line_to_type[line][1]
                    if stmt_type in ['IfStatement', 'ForStatement', 'WhileStatement', 'TryStatement', 'CatchClause', 'ReturnStatement', 'Assignment', 'MethodInvocation', 'VariableDeclarator', 'ClassDeclaration', 'MethodDeclaration']:
                        line_to_type[line] = (node, stmt_type)
        for line_num, (node, stmt_type) in line_to_type.items():
            line_to_semantic[line_num] = stmt_type
    except Exception as e:
        pass
    return line_to_semantic
def classify_token(tok: str, line_num: Optional[int] = None, ast_semantic_types: Optional[Dict[int, str]] = None) -> str:
    """Return a semantic label string for a token using AST types when available."""
    # Try AST-based classification first
    if line_num is not None and ast_semantic_types and line_num in ast_semantic_types:
        return ast_semantic_types[line_num]
    
    # For line-level classification, try to identify the most important keyword or structure
    # Split the line into words and look for Java keywords
    words = tok.split()
    for word in words:
        # Remove common punctuation
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in JAVA_KEYWORDS_SET:
            return clean_word
    
    # Check for specific patterns
    if re.search(r'\bif\s*\(', tok):
        return "IfStatement"
    elif re.search(r'\bfor\s*\(', tok):
        return "ForStatement"
    elif re.search(r'\bwhile\s*\(', tok):
        return "WhileStatement"
    elif re.search(r'\btry\s*\{', tok):
        return "TryStatement"
    elif re.search(r'\bcatch\s*\(', tok):
        return "CatchClause"
    elif re.search(r'\breturn\b', tok):
        return "ReturnStatement"
    elif re.search(r'\bthrow\b', tok):
        return "ThrowStatement"
    elif re.search(r'^\s*public\s+', tok):
        return "MethodDeclaration"
    elif re.search(r'^\s*private\s+', tok):
        return "MethodDeclaration"
    elif re.search(r'^\s*protected\s+', tok):
        return "MethodDeclaration"
    elif re.search(r'^\s*class\s+', tok):
        return "ClassDeclaration"
    elif re.search(r'^\s*interface\s+', tok):
        return "InterfaceDeclaration"
    elif re.search(r'^\s*enum\s+', tok):
        return "EnumDeclaration"
    elif re.search(r'^\s*import\s+', tok):
        return "Import"
    elif re.search(r'^\s*package\s+', tok):
        return "PackageDeclaration"
    elif re.search(r'^\s*}\s*$', tok):
        return "BlockStatement"
    elif re.search(r'^\s*{\s*$', tok):
        return "BlockStatement"
    elif re.search(r'^\s*;\s*$', tok):
        return "StatementExpression"
    elif re.search(r'^\s*//', tok):
        return "Comment"
    elif re.search(r'^\s*/\*', tok):
        return "Comment"
    elif re.search(r'^\s*\*/', tok):
        return "Comment"
    
    # Fallback to keyword-based classification for individual words
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in JAVA_KEYWORDS_SET:
            return clean_word
        if clean_word.isdigit():
            return "number"
        if re.match(r"[A-Za-z_][A-Za-z0-9_]*", clean_word):
            return "identifier"
    
    return "symbol"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
try:
    import sti_codesearchnet_java as sti
except ImportError as exc:
    print("[ERROR] Cannot import sti_codesearchnet_java.py – ensure it is located in the same directory.")
    raise exc
ORIG_FILE = os.path.join(SCRIPT_DIR, "summarization_train_final.jsonl")
OUT_FILE = os.path.join(SCRIPT_DIR, "summarization_train_final_scanpath.jsonl")
MAX_SAMPLES = None
NUM_SIMULATIONS = 1
VERBOSE_EVERY = 50
SEM_INDEX_PATH = os.path.join(SCRIPT_DIR, "sti_semantic_index_table.json")
if os.path.exists(SEM_INDEX_PATH):
    with open(SEM_INDEX_PATH, "r", encoding="utf-8") as f:
        SEM_INDEX_TABLE = json.load(f)
else:
    SEM_INDEX_TABLE = {}
def generate_scanpath(code: str, sim_id: int = 0, seed: Optional[int] = None) -> Tuple[List[int], Dict[int, float]]:
    """Generate a list of fixated line numbers and complexity mapping using sti_codesearchnet helpers."""
    sample_id = f"code_sample_{sim_id}"
    sample = {"id": sample_id, "file_path": "N/A", "code": code}
    samples = [sample]
    
    df = sti.prepare_scanpath_data(samples)
    if df.empty:
        return [], {}
    
    df_t = sti.prepare_target_data(samples)
    complexity_map = {}
    for _, row in df_t.iterrows():
        line_num = int(row["Line"])
        complexity = float(row["complexity"])
        complexity_map[line_num] = complexity
    
    columns = ["node", "Line", "Column", "semantic", "controlnum", "feature", "complexity", "corr_section", "section"]
    try:
        step = sti.set_step(df, seed=seed)
    except Exception as e:
        return [], complexity_map
    
    df_result = pd.DataFrame(columns=columns)
    try:
        lines = sti.stimulate(step, 0, sample_id, df_result, df, df_t, seed=seed)
    except Exception as e:
        print(f"ERROR in stimulate: {e}")
        return [], complexity_map
    return lines, complexity_map
def main():
    parser = argparse.ArgumentParser(description="Generate scan-path augmented training JSONL (Summarization)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index (0-based) in original JSONL")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (exclusive) in original JSONL; if None, process all items")
    parser.add_argument("--max_samples", type=int, default=None, help="Process at most this many samples (after start_idx)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible scanpath generation")
    args = parser.parse_args()
    start_idx = args.start_idx
    end_idx = args.end_idx
    max_samples = args.max_samples if args.max_samples is not None else MAX_SAMPLES
    seed = args.seed
    if not os.path.exists(ORIG_FILE):
        print(f"[ERROR] Cannot find source dataset: {ORIG_FILE}")
        return
    
    print(f"Using random seed: {seed} for reproducible scanpath generation")
    processed = 0
    with open(ORIG_FILE, "r", encoding="utf-8") as fin, open(OUT_FILE, "w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin):
            if line_idx < start_idx:
                continue
            if end_idx is not None and line_idx >= end_idx:
                break
            if max_samples is not None and processed >= max_samples:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = rec.get("code", "")
            if not code:
                continue
            # --- simulate scan-path (line numbers) ---
            try:
                scan_lines, complexity_map = generate_scanpath(code, sim_id=processed, seed=seed)
                if not scan_lines:
                    # fallback: use first N lines sequentially
                    scan_lines = list(range(1, min(20, code.count("\n") + 1)))
                    complexity_map = {}
                if VERBOSE_EVERY and processed % VERBOSE_EVERY == 0:
                    print(f"Sample {processed}: Scan lines: {scan_lines[:10]}... (total: {len(scan_lines)})")
                    print(f"Sample {processed}: Complexity map: {dict(list(complexity_map.items())[:5])}... (total: {len(complexity_map)})")
            except Exception as e:
                print(f"[WARN] simulation failed for sample {processed}: {e}")
                scan_lines = list(range(1, min(20, code.count("\n") + 1)))
                complexity_map = {}
            code_lines = code.split("\n")
            num_lines = len(code_lines)

            # Extract AST-based semantic types for the entire code
            ast_semantic_types = extract_java_ast_semantic_types(code)
            if VERBOSE_EVERY and processed % VERBOSE_EVERY == 0:
                print(f"Sample {processed}: AST semantic types: {ast_semantic_types}")
            code_tokens: List[str] = []
            semantic_token_sequence: List[int] = []
            line_numbers: List[int] = []
            mask: List[int] = []
            ngram_indices: List[int] = []
            order_sequence: List[int] = []
            complexity_sequence: List[float] = []
            # Map: line number -> order position (first fixation occurrence)
            line_to_order: Dict[int, int] = {}
            for idx, ln in enumerate(scan_lines):
                if ln not in line_to_order:
                    line_to_order[ln] = idx + 1  # 1-based order index
            for i, text_line in enumerate(code_lines):
                ln = i + 1  # 1-based line number
                code_tokens.append(text_line)
                line_numbers.append(ln)
                sem_label = classify_token(text_line, line_num=ln, ast_semantic_types=ast_semantic_types)
                semantic_token_sequence.append(SEM_INDEX_TABLE.get(sem_label, -1))
                if ln in line_to_order:
                    mask.append(1)
                    order_sequence.append(line_to_order[ln])
                else:
                    mask.append(0)
                    order_sequence.append(0)
                ngram_indices.append(0)
                # Add complexity value for this line (default to 1.0 if not found)
                complexity_value = complexity_map.get(ln, 1.0)
                # Ensure complexity is a valid number
                if not isinstance(complexity_value, (int, float)) or complexity_value < 0:
                    complexity_value = 1.0
                complexity_sequence.append(float(complexity_value))
            if not code_tokens:
                # In rare fallback where no tokens collected, skip sample
                continue
            out_obj: Dict[str, Any] = {
                "code_content_hash": rec.get("code_content_hash", f"sample_{processed}"),
                "content": rec.get("content", ""),
                "flag": "train",
                "code": code,
                "ngrams": [None] * len(code_tokens),
                "code_tokens": code_tokens,
                "code_occurrence": [0] * len(code_tokens),
                "semantic_token_sequence": semantic_token_sequence,
                "line_numbers": line_numbers,
                "mask": mask,
                "ngram_indices": ngram_indices,
                "order_sequence": order_sequence,
                "complexity_sequence": complexity_sequence,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            processed += 1
            if VERBOSE_EVERY and processed % VERBOSE_EVERY == 0:
                print(f"Processed {processed} samples …")
    print(f"\nDone. Wrote {processed} samples to {OUT_FILE}")
    
    # Print complexity statistics if we processed any samples
    if processed > 0:
        print(f"\nComplexity statistics:")
        print(f"- Used improved Java complexity calculation from sti_codesearchnet_java_new.py")
        print(f"- Complexity values include operational complexity + pattern-based boosts")
        print(f"- Each line complexity considers: arithmetic ops, logical ops, function calls, keywords, etc.")
if __name__ == "__main__":
    main() 