import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import os
import re 
import matplotlib.colors as mcolors
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

# --- Custom Model Definition & Utility Functions ---
class LlamaForCausalLMWithWeightedLoss(LlamaForCausalLM):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

def load_model_and_tokenizer(model_path, is_advanced_model):
    logging.info(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The directory was not found: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if is_advanced_model:
        model = LlamaForCausalLMWithWeightedLoss.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def clean_token(token):
    """Final cleaning function: removes special tokens and shortens very long ones."""
    token = re.sub(r'<[^>]+>', '', token)
    token = token.replace('Ġ', ' ')
    if len(token) > 18:
        token = token[:15] + '...'
    token = re.sub(r'([;})\]])C$', r'\1', token)
    token = token.replace('.ĠĠ', ';')
    return token.strip()

def get_attention_matrix(model, tokenizer, prompt, device):
    """Generates text and extracts the attention matrix for the final layer."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=60, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, output_attentions=True,
            return_dict_in_generate=True, do_sample=False
        )
    generated_ids = outputs.sequences[0, input_ids.shape[-1]:]
    input_tokens = [clean_token(tok) for tok in tokenizer.convert_ids_to_tokens(input_ids[0])]
    generated_tokens = [clean_token(tok) for tok in tokenizer.convert_ids_to_tokens(generated_ids)]
    
    num_generated_tokens, num_input_tokens = len(generated_tokens), len(input_tokens)
    attention_matrix = torch.zeros(num_generated_tokens, num_input_tokens)
    for i in range(num_generated_tokens):
        step_attentions = outputs.attentions[i]
        last_layer_attention = step_attentions[-1]
        avg_head_attention = last_layer_attention.mean(dim=1).squeeze(0)
        if avg_head_attention.ndim == 2:
            attention_vector = avg_head_attention[-1, :]
        else:
            attention_vector = avg_head_attention
        if len(generated_tokens) > i and len(input_tokens) > 0:
            attention_matrix[i, :] = attention_vector[:num_input_tokens].cpu()
    return input_tokens, generated_tokens, attention_matrix

def plot_attention_heatmap(ax, matrix, x_labels, y_labels):
    """Plots a single heatmap with final, polished settings."""
    if matrix.min() <= 0: vmin = 1e-5
    else: vmin = matrix.min()
    norm = mcolors.LogNorm(vmin=vmin, vmax=matrix.max())
    sns.heatmap(
        matrix, xticklabels=x_labels, yticklabels=y_labels, cmap="magma", 
        norm=norm, ax=ax, cbar=True, cbar_kws={"shrink": 0.8}
    )
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)

# --- Main Execution Block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    tasks = [ "summarization", "completion"]
    DEVICE = "cuda:1"

    code_snippets = {
        "translation": "public static List<String> readLines(File file) throws IOException {\n    List<String> lines = new ArrayList<String>();\n    BufferedReader reader = new BufferedReader(new FileReader(file));\n    String line;\n    while ((line = reader.readLine()) != null) {\n        lines.add(line);\n    }\n    reader.close();\n    return lines;\n}",
        "summarization": "public static long factorial(int n) {\n    if (n < 0) {\n        throw new IllegalArgumentException(\"Factorial not defined for negative numbers.\");\n    }\n    long result = 1;\n    for (int i = 1; i <= n; i++) {\n        result *= i;\n    }\n    return result;\n}",
        "completion": "public static String getFileExtension(String fullName) {\n    if (fullName == null || fullName.isEmpty()) {\n        return \"\";\n    }\n    String fileName = new File(fullName).getName();\n    int dotIndex = fileName.lastIndexOf('.');\n"
    }
    
    plot_data = {}

    # Pre-computation Step
    for task in tasks:
        logging.info(f"--- Pre-computing data for task: {task} ---")
        ADVANCED_MODEL_PATH = f"./llama_advanced_final_{task}"
        BASELINE_MODEL_PATH = f"./llama_baseline_{task}"
        code_to_visualize = code_snippets[task].strip()
        prompts = {
            "translation": {"instruction": "Translate the following Java code to C#.", "input_header": "### Java Code:", "output_header": "### C# Code:"},
            "summarization": {"instruction": "Summarize the following Java code.", "input_header": "### Java Code:", "output_header": "### Summary:"},
            "completion": {"instruction": "Complete the following Java code.", "input_header": "### Java Code:", "output_header": "### Completion:"}
        }
        prompt_config = prompts[task]
        prompt = (f"### Instruction:\n{prompt_config['instruction']}\n\n"
                  f"{prompt_config['input_header']}\n{code_to_visualize}\n\n"
                  f"{prompt_config['output_header']}\n")
        
        try:
            adv_model, adv_tokenizer = load_model_and_tokenizer(ADVANCED_MODEL_PATH, is_advanced_model=True)
            base_model, base_tokenizer = load_model_and_tokenizer(BASELINE_MODEL_PATH, is_advanced_model=False)
            adv_model.to(DEVICE); base_model.to(DEVICE)
            adv_inputs, adv_outputs, adv_attn = get_attention_matrix(adv_model, adv_tokenizer, prompt, DEVICE)
            base_inputs, base_outputs, base_attn = get_attention_matrix(base_model, base_tokenizer, prompt, DEVICE)
            plot_data[task] = {'adv': (adv_attn, adv_inputs, adv_outputs), 'base': (base_attn, base_inputs, base_outputs)}
        except Exception as e:
            logging.error(f"Could not process task {task}: {e}")

    # Generate the final master figure
    if plot_data:
        logging.info("--- Generating final master figure ---")
        fig, axes = plt.subplots(2, 3, figsize=(26, 15))
        
        for i, task in enumerate(tasks):
            if task in plot_data:
                axes[0, i].set_title(f"Code {task.capitalize()}", fontsize=22, pad=15)
        
        fig.text(0.04, 0.70, 'Baseline Model', ha='center', va='center', rotation='vertical', fontsize=26)
        fig.text(0.04, 0.30, 'Scancoder', ha='center', va='center', rotation='vertical', fontsize=26)
        
        for i, task in enumerate(tasks):
            if task in plot_data:
                adv_data = plot_data[task]['adv']
                base_data = plot_data[task]['base']
                plot_attention_heatmap(axes[0, i], base_data[0], base_data[1], base_data[2])
                plot_attention_heatmap(axes[1, i], adv_data[0], adv_data[1], adv_data[2])

        # ✅ Use tight_layout() for the most compact arrangement possible.
        fig.tight_layout(pad=1.0, w_pad=1.5, h_pad=2.0)
        # Adjust layout to bring row titles in
        plt.subplots_adjust(left=0.1)

        output_filename = "attention_matrix.pdf"
        plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
        logging.info(f"✅ Saved DEFINITIVE master figure to {output_filename}")
        plt.close(fig)