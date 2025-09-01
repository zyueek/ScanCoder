# ────── imports ─────────────────────────────────────────────────────────────
import json, math, re, textwrap, os, sys
from collections import defaultdict
import numpy as np

# Third-party imports (optional with graceful fallbacks)
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    nltk = None
    NLTK_AVAILABLE = False

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    sacrebleu = None
    SACREBLEU_AVAILABLE = False

try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    Levenshtein = None
    LEVENSHTEIN_AVAILABLE = False

try:
    from nltk.translate.meteor_score import meteor_score
    METEOR_AVAILABLE = True
except Exception:
    meteor_score = None
    METEOR_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    rouge_scorer = None
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    bert_score = None
    BERTSCORE_AVAILABLE = False
try:
    from codebleu import calc_codebleu
    CODEBLEU_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        from codebleu.codebleu import calc_codebleu
        CODEBLEU_AVAILABLE = True
    except ImportError:
        CODEBLEU_AVAILABLE = False
        print("Warning: codebleu library not available. CodeBLEU metric will be skipped.", file=sys.stderr)

# Attempt to import tqdm, but don't fail if not installed unless needed
try:
    # Check if we're in a Jupyter/IPython environment
    import IPython
    if IPython.get_ipython() is not None:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except (ImportError, AttributeError):
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs):
            # If tqdm is not installed, just return the original iterable
            return iterable

# Import SentenceTransformer for local embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# ────── ensure NLTK data ────────────────────────────────────────────────────
def _ensure_nltk_data(path: str):
    """Checks for NLTK data and downloads if missing."""
    try:
        nltk.data.find(path)
    except LookupError:
        print(f"Downloading required NLTK resource: {path.split('/')[-1]}...")
        nltk.download(path.split('/')[-1], quiet=True)

if NLTK_AVAILABLE:
    for res in ("tokenizers/punkt", "corpora/wordnet", "corpora/omw-1.4"):
        _ensure_nltk_data(res)

# ────── cleaning helper (for --clean) ───────────────────────────────────────
_WS = re.compile(r"\s+")
def clean(text: str, task: str) -> str:
    """Cleans text based on the task type."""
    if not text: return ""
    if task in {"completion", "translation"}:
        # Remove comments and extra whitespace for code tasks
        text = re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.S | re.M)
        text = re.sub(r"\s+\n", "\n", text)
    elif task == "summarization":
        # Remove common instruction artifacts
        text = re.sub(r"^\s*(?:answer|summary)[:\-–]\s*", "", text, flags=re.I)
        text = textwrap.dedent(text)
    return text.replace("```", "").replace("<pad>", "").strip()

# ────── traditional metric helpers ──────────────────────────────────────────
def exact_rate(preds, refs):
    """Calculates the percentage of exact matches."""
    return 100.0 * sum(p == r[0] for p, r in zip(preds, refs)) / len(preds) if preds else 0.0

def _minify(s: str) -> str:
    """Removes all whitespace from a string."""
    return _WS.sub("", s or "")

def hybrid_exact(preds, refs):
    """Averages exact match and containment match on minified strings."""
    if not preds: return 0.0
    p_min = [_minify(p) for p in preds]
    r_min = [_minify(r[0]) for r in refs]
    exact = sum(p == r for p, r in zip(p_min, r_min))
    contained = sum(1 for p, r in zip(p_min, r_min) if p and r and (p in r or r in p))
    return 0.5 * (exact + contained) / len(preds) * 100.0

def codebleu(preds, refs, lang):
    """Calculates the CodeBLEU score."""
    if not preds: return 0.0
    if not CODEBLEU_AVAILABLE:
        print("Warning: codebleu library not available. Skipping CodeBLEU metric.", file=sys.stderr)
        return 0.0
    
    try:
        # Filter out empty predictions and references
        valid_indices = [i for i, (p, r) in enumerate(zip(preds, refs)) if p and r[0]]
        if not valid_indices:
            return 0.0
            
        valid_preds = [preds[i] for i in valid_indices]
        valid_refs = [refs[i][0] for i in valid_indices]
        
        # Ensure we have valid code samples (not empty or just whitespace)
        valid_preds = [p.strip() for p in valid_preds if p.strip()]
        valid_refs = [r.strip() for r in valid_refs if r.strip()]
        
        if not valid_preds or not valid_refs:
            return 0.0
            
        # CodeBLEU expects references as a list of lists for each prediction
        # Since we have one reference per prediction, we need to format it correctly
        formatted_refs = [[ref] for ref in valid_refs]
        
        # Map language to CodeBLEU supported format
        lang_map = {
            "java": "java",
            "python": "python", 
            "cpp": "cpp",
            "c": "c",
            "javascript": "javascript",
            "go": "go",
            "php": "php",
            "ruby": "ruby",
            "rust": "rust"
        }
        
        codebleu_lang = lang_map.get(lang.lower(), "java")  # Default to java
        
        # Debug info
        print(f"Debug: Calculating CodeBLEU for {len(valid_preds)} samples, lang={lang} -> {codebleu_lang}")
        print(f"Debug: First pred: {valid_preds[0][:100]}...")
        print(f"Debug: First ref: {valid_refs[0][:100]}...")
        
        # Ensure we have the same number of predictions and references
        min_len = min(len(valid_preds), len(valid_refs))
        if min_len == 0:
            return 0.0
            
        valid_preds = valid_preds[:min_len]
        formatted_refs = formatted_refs[:min_len]
        
        result = calc_codebleu(formatted_refs, valid_preds, lang=codebleu_lang)
        return result["codebleu"] * 100
    except Exception as e:
        print(f"Warning: Error calculating CodeBLEU: {e}. Skipping metric.", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 0.0

def crystalbleu(preds, refs):
    """Calculates the CrystalBLEU (chrF) score."""
    if not preds: return 0.0
    if not SACREBLEU_AVAILABLE:
        print("Warning: sacrebleu not available. Skipping CrystalBLEU (chrF) metric.", file=sys.stderr)
        return 0.0
    # sacrebleu expects a list of predictions and a list of lists of references,
    # where each inner list is a *full set of references for the corpus*.
    # We have one reference per prediction, so we format it as [[ref1, ref2, ...]]
    single_ref_list = [r[0] for r in refs]
    return sacrebleu.corpus_chrf(preds, [single_ref_list], word_order=2).score

def nls(preds, refs):
    """Calculates Normalized Levenshtein Similarity."""
    if not preds: return 0.0
    if not LEVENSHTEIN_AVAILABLE:
        print("Warning: python-Levenshtein not available. Skipping NLS metric.", file=sys.stderr)
        return 0.0
    sims = [1 - Levenshtein.distance(p, r[0]) / max(1, len(r[0]), len(p)) for p, r in zip(preds, refs)]
    return np.mean(sims) * 100

def rouge1_l(preds, refs):
    """Calculates ROUGE-1 and ROUGE-L F1-scores."""
    if not preds: return {"ROUGE1": 0.0, "ROUGEL": 0.0}
    if not ROUGE_AVAILABLE:
        print("Warning: rouge-score not available. Skipping ROUGE metrics.", file=sys.stderr)
        return {"ROUGE1": 0.0, "ROUGEL": 0.0}
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [sc.score(r[0], p) for p, r in zip(preds, refs)]
    return {
        "ROUGE1": np.mean([s["rouge1"].fmeasure for s in scores]) * 100,
        "ROUGEL": np.mean([s["rougeL"].fmeasure for s in scores]) * 100
    }

def meteor_avg(preds, refs):
    """Calculates the average METEOR score with tokenized inputs."""
    if not preds: return 0.0
    tokenized_refs = [[ref.split() for ref in r_list] for r_list in refs]
    tokenized_preds = [p.split() for p in preds]
    vals = [meteor_score(ref, pred) if pred and ref[0] else 0 for pred, ref in zip(tokenized_preds, tokenized_refs)]
    return np.mean(vals) * 100

def bscore(preds, refs, model="bert-base-uncased"):
    """Calculates the BERTScore F1."""
    p_clean = [p for p in preds if p]
    if not BERTSCORE_AVAILABLE:
        print("Warning: bert-score not available. Skipping BERTScore metric.", file=sys.stderr)
        return 0.0
    # The reference list needs to be flattened for bert_score with single references
    r_clean = [r[0] for p, r in zip(preds, refs) if p]
    if not p_clean: return 0.0
    # Note: bert_score expects a list of strings for refs when there's one reference per prediction
    _, _, F1 = bert_score(p_clean, r_clean, model_type=model, lang="en", rescale_with_baseline=True)
    return F1.mean().item() * 100

# ────── embedding metric helper (IN-MEMORY) ────────────────────────────────
EMBEDDING_MODEL = None
if SentenceTransformer:
    try:
        EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Could not load SentenceTransformer model. Please ensure pytorch is installed. Error: {e}", file=sys.stderr)

def embedding_similarity(preds, refs, model, batch_size=32):
    """Calculates cosine similarity from predictions and references in memory."""
    if model is None:
        print("Error: Embedding model not loaded. Skipping similarity calculation.", file=sys.stderr)
        return 0.0
    if not preds: return 0.0

    # Flatten references for a single pass
    flat_refs = [r[0] for r in refs]

    ref_embeds = model.encode(flat_refs, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    pred_embeds = model.encode(preds, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    
    # Dot product of normalized vectors is the cosine similarity
    similarity = np.sum(ref_embeds * pred_embeds, axis=1)
    
    return np.mean(similarity) * 100

# ────── task configuration ─────────────────────────────────────────────────


TASKS = {
    "completion": dict(
        base="results_example/generated_results_baseline_completion.json",
        adv ="deepseek_results/generated_results_advanced_final_completion.json",
        lang="java",
        metrics=("HybridExact", "CrystalBLEU", "NLS", "EmbeddingSim","CodeBLEU")),
    # "summarization": dict(
    #     base="results_example/generated_results_baseline_summarization.json",
    #     adv ="llama_results/generated_results_advanced_final_summarization.json",
    #     lang="java",
    #     metrics=("ROUGE1", "ROUGEL", "METEOR", "EmbeddingSim","BERTScore")),
}
# ────── load / evaluate helpers ─────────────────────────────────────────────
def load_json(path: str, task: str, do_clean: bool):
    """Loads and optionally cleans data from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    preds = [clean(d.get("generated_output", ""), task) if do_clean else d.get("generated_output", "") for d in data]
    refs  = [[clean(d.get("ground_truth", ""), task) if do_clean else d.get("ground_truth", "")] for d in data]

    return preds, refs

def evaluate(preds, refs, lang, metrics, use_embeddings):
    """Runs a suite of evaluation metrics."""
    o={}
    if "HybridExact" in metrics: o["HybridExact"] = hybrid_exact(preds, refs)
    if "CodeBLEU"    in metrics: o["CodeBLEU"]    = codebleu(preds, refs, lang)
    if "CrystalBLEU" in metrics: o["CrystalBLEU"] = crystalbleu(preds, refs)
    if "NLS"         in metrics: o["NLS"]         = nls(preds, refs)
    if "ROUGE1" in metrics or "ROUGEL" in metrics: o.update(rouge1_l(preds, refs))
    if "METEOR"      in metrics: o["METEOR"]      = meteor_avg(preds, refs)
    if "BERTScore"   in metrics: o["BERTScore"]   = bscore(preds, refs)
    if "EmbeddingSim" in metrics and use_embeddings: 
        o["EmbeddingSim"] = embedding_similarity(preds, refs, EMBEDDING_MODEL)
    return o

def print_tables(res):
    """Prints results in a formatted table."""
    for task, r in res.items():
        print(f"\n=== {task.upper()} ===")
        print(f"{'Metric':<15} | {'Baseline':>12} | {'Advanced':>12}")
        print("-" * 45)
        metric_order = sorted(list({k for d in r.values() for k in d}))
        
        for m in metric_order:
            b, a = r["Baseline"].get(m, float("nan")), r["Advanced"].get(m, float("nan"))
            fmt = lambda x: f"{x:9.2f}" if not math.isnan(x) else "  n/a "
            b_s, a_s = fmt(b), fmt(a)
            if not math.isnan(b) and not math.isnan(a):
                if a > b: a_s = f"🏆 {a_s}"
                elif b > a: b_s = f"🏆 {b_s}"
            print(f"{m:<15} | {b_s:>12} | {a_s:>12}")
        print("-" * 45)

# ────── main execution logic ────────────────────────────────────────────────
def run_evaluation(config):
    """Main evaluation function, adapted for notebooks."""
    if config['use_embeddings'] and EMBEDDING_MODEL is None:
        print("ERROR: 'sentence-transformers' is required for embeddings but could not be loaded.", file=sys.stderr)
        print("Please run: pip install sentence-transformers torch", file=sys.stderr)
        # Disable embeddings if the model failed to load
        config['use_embeddings'] = False

    out = defaultdict(dict)
    for task, cfg in tqdm(TASKS.items(), desc="Evaluating Tasks"):
        for tag, path in (("Baseline", cfg["base"]), ("Advanced", cfg["adv"])):
            if not os.path.exists(path):
                print(f"Warning: File not found at '{path}'. Skipping.", file=sys.stderr)
                continue
            
            preds, refs = load_json(path, task, config['clean'])
            out[task][tag] = evaluate(preds, refs, cfg["lang"], cfg["metrics"], config['use_embeddings'])
            
    print_tables(out)

# ======================================================================
# 🚀 RUN THE SCRIPT HERE
# ======================================================================

# --- Set your configuration options ---
Config = {
    "clean": True,            # Set to True to clean code/text, False otherwise.
    "use_embeddings": True,   # Set to True to enable the embedding similarity metric.
}

# --- Run the evaluation ---
# This executes the entire evaluation based on the configuration above.
# The first time you run with use_embeddings=True, it may download the
# 'all-MiniLM-L6-v2' model, which can take a minute.
if __name__ == '__main__':
    run_evaluation(Config)