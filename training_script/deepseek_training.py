import os
import torch
import json
import logging
import numpy as np
import time
import math
import transformers
import gc
import sys
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    LlamaForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# --- Task Configuration ---
# Choose from completion, translation and summarization
TASK = "completion"

# --- Main Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"
MODEL_NAME = "deepseek"

DATA_FOLDER = "./"
OUTPUT_DIR = f"./{MODEL_NAME}_advanced_final_{TASK}"
TRAIN_FILE = os.path.join(DATA_FOLDER, f"{TASK}_train_final.jsonl")
VALID_FILE = os.path.join(DATA_FOLDER, f"{TASK}_valid_final.jsonl")
MAX_LENGTH = 1024

# --- Global Variables for Loaded Data ---
# Removed n-gram and beta distribution variables as they're no longer used


# --- Custom Model with Corrected Loss Calculation ---
class LlamaForCausalLMWithWeightedLoss(LlamaForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        weights: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        # 1. Get raw logits from the base model by passing `labels=None`.
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_attentions=False,
            output_hidden_states=False,
            **kwargs,
        )
        logits = outputs.logits

        # 2. Compute our custom loss only if labels are provided.
        final_loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens and calculate per-token loss
            loss_fct = CrossEntropyLoss(reduction="none")
            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_labels = shift_labels.view(-1)
            loss_per_token = loss_fct(flat_logits, flat_labels)

            # Filter for active (non-padded) tokens
            active_loss_mask = flat_labels != -100
            active_losses = loss_per_token[active_loss_mask]

            # If weights are provided, apply them to create a scaled loss
            if weights is not None:
                shift_weights = weights[..., 1:].contiguous().view(-1)
                active_weights = shift_weights[active_loss_mask]
                
                # The final loss is the mean of the losses scaled by the weights
                scaled_loss = (active_losses * active_weights).sum()
                num_active_tokens = active_loss_mask.sum()
                final_loss = scaled_loss / (num_active_tokens + 1e-9)
            else:
                # Fallback to standard mean loss if no weights are passed
                final_loss = active_losses.mean()

        # Return a new output object with our custom loss
        return CausalLMOutputWithPast(
            loss=final_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

# --- Custom Data Collator ---
@dataclass
class CustomDataCollator:
    tokenizer: AutoTokenizer
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        max_len = MAX_LENGTH
        for key in features[0].keys():
            # Truncate sequences to max_len before padding
            for i in range(len(features)):
                if len(features[i][key]) > max_len:
                    features[i][key] = features[i][key][:max_len]

            if key == "weights": pad_value, dtype = 1.0, torch.float
            elif key == "labels": pad_value, dtype = -100, torch.long
            else: pad_value, dtype = self.tokenizer.pad_token_id, torch.long

            padded_list = [f[key] + [pad_value] * (max_len - len(f[key])) for f in features]
            batch[key] = torch.tensor(padded_list, dtype=dtype)
        return batch

def calculate_weight(mask, order, complexity):
    if mask == 0:
        return 1.0  # No special emphasis for non-masked tokens
#    norm_complexity = (complexity + 1e-3)
#    norm_order = (order + 1e-3)
    base_weight = 3.0
#    complexity_bonus = math.log(norm_complexity + 1)
    order_bonus = 1.0 / order
    print(base_weight, order_bonus)  # Commented out for cleaner output
    final_weight = base_weight + order_bonus
    return final_weight


# --- Data Loading and Preprocessing ---
def load_and_validate_data(file_path: str) -> List[Dict]:
    data = []
    required_keys = ['code_tokens', 'mask', 'order_sequence', 'code', 'content']
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line)
                    if not all(k in record and record[k] is not None for k in required_keys): continue
                    token_len = len(record['code_tokens'])
                    if not all(len(record[key]) == token_len for key in required_keys if key not in ['code', 'content']): continue
                    data.append(record)
                except Exception as e:
                    logging.warning(f"Skipping malformed line {i+1} in {file_path}: {e}")
                    continue
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        return []
    # Minimize data for debugging: only keep first 2 samples
#    data = data[1:2]
#    print(f"Loaded {len(data)} samples from {file_path} (showing first sample):\n{json.dumps(data[0], indent=2) if data else 'No data'}")
    return data

def create_and_process_data(tokenizer):
    logging.info(f"Loading training data from: {TRAIN_FILE}")
    logging.info(f"Loading validation data from: {VALID_FILE}")
    train_data = load_and_validate_data(TRAIN_FILE)
    valid_data = load_and_validate_data(VALID_FILE)

    if not train_data or not valid_data:
        logging.error("Training or validation data is empty. Please check file paths and content integrity.")
        sys.exit(1)

    # Debug: Print first sample for inspection
    print(f"Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
    if train_data:
        print("First training sample keys:", list(train_data[0].keys()))
        print("First training sample structure preview:")
        for key, value in train_data[0].items():
            if isinstance(value, list):
                print(f"  {key}: list of length {len(value)}")
            else:
                print(f"  {key}: {type(value).__name__}")

    raw_datasets = DatasetDict({"train": Dataset.from_list(train_data), "validation": Dataset.from_list(valid_data)})

    task_prompts = {
        "translation": "### Instruction:\nTranslate the following Java code to C#.\n\n",
        "summarization": "### Instruction:\nSummarize the following code.\n\n",
        "completion": "### Instruction:\nComplete the following code snippet.\n\n"
    }

    def preprocess_and_weigh(example):
        code_str, content_str = example['code'], example['content']
        instruction_prompt = task_prompts.get(TASK, "### Instruction:\nProcess the following code.\n\n")
        input_header = "### Input Code:\n"
        output_header = f"\n\n### Output:\n"
        
        weights_per_code_token = [
            calculate_weight(
                example['mask'][j],
                example['order_sequence'][j],
                example['complexity_sequence'][j] if 'complexity_sequence' in example else 1.0
            ) for j in range(len(example['code_tokens']))
        ]

        full_prompt = f"{instruction_prompt}{input_header}{code_str}{output_header}"
        full_text = f"{full_prompt}{content_str}{tokenizer.eos_token}"

        tokenized_full = tokenizer(full_text, max_length=MAX_LENGTH, truncation=True)
        tokenized_prompt = tokenizer(full_prompt, max_length=MAX_LENGTH, truncation=True)
        
        input_ids = tokenized_full['input_ids']
        attention_mask = [1] * len(input_ids)
        labels = list(input_ids)
        labels[:len(tokenized_prompt['input_ids'])] = [-100] * len(tokenized_prompt['input_ids'])
        
        # Align code tokens to output tokens (after prompt)
        # Tokenize code tokens individually to get their subword lengths
        code_token_weights = []
        for code_token, weight in zip(example['code_tokens'], weights_per_code_token):
            tokenized = tokenizer(code_token, add_special_tokens=False)
            code_token_weights.extend([weight] * len(tokenized['input_ids']))
        # The output tokens are those after the prompt
        n_prompt = len(tokenized_prompt['input_ids'])
        n_output = len(input_ids) - n_prompt
        # If code_token_weights is too short, pad; if too long, truncate
        if len(code_token_weights) < n_output:
            code_token_weights.extend([1.0] * (n_output - len(code_token_weights)))
        elif len(code_token_weights) > n_output:
            code_token_weights = code_token_weights[:n_output]
        weights = [1.0] * n_prompt + code_token_weights
        final_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        for i, (token_id, token, label, weight) in enumerate(zip(input_ids, final_tokens, labels, weights)):
            label_str = f"LABEL:{label}" if label != -100 else "IGNORE"
 #           print(f"  {i}: {token_id} -> '{token}' | {label_str} | weight:{weight}")
            
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "weights": weights}

    logging.info("Applying data-driven preprocessing with MAX weight aggregation...")
    
    # Debug: Print columns present in the dataset before mapping
    print("Columns in train dataset before map:", raw_datasets["train"].column_names)
    candidate_remove = [
        'code_content_hash', 'content', 'flag', 'code', 'ngrams', 'code_tokens', 'code_occurrence',
        'line_numbers', 'mask', 'order_sequence'
    ]
    # Defensive: ensure only existing columns are removed
    current_columns = set(raw_datasets["train"].column_names)
    columns_to_remove = [col for col in candidate_remove if col in current_columns]
    print(f"Current columns: {current_columns}")
    print(f"Removing columns: {columns_to_remove}")
    logging.info(f"Removing columns: {columns_to_remove}")
    
    return raw_datasets.map(preprocess_and_weigh, num_proc=1, remove_columns=columns_to_remove)

def main():
    # --- Set random seeds for reproducible training ---
    import random
    import numpy as np
    import torch
    
    # Set seeds for all random number generators
    seed = 42  # Default seed for reproducible training
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"Set random seed to {seed} for reproducible training")
    
    # Removed global variables for n-gram and beta distribution data
    
    logging.info("Starting training with scanpath-based weight calculation...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=AUTH_TOKEN, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id; tokenizer.padding_side = "right"

    model = LlamaForCausalLMWithWeightedLoss.from_pretrained(
        MODEL_ID, use_auth_token=AUTH_TOKEN, trust_remote_code=True)
    
    tokenized_datasets = create_and_process_data(tokenizer)
    
    # Debug: Print first tokenized sample for inspection
    print('tokenized_datasets["train"][0]:')
    print(tokenized_datasets["train"][0])
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-5, num_train_epochs=3, report_to="tensorboard",
        logging_steps=10, eval_strategy="epoch",
        # Removed save_strategy and load_best_model_at_end to only save at the end
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        dataloader_num_workers=0,
        seed=seed,  # Set seed for training reproducibility
    )
    
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=CustomDataCollator(tokenizer=tokenizer)
    )

    logging.info(f"Starting data-driven advanced training for task '{TASK}'...")
    trainer.train()
    
    logging.info("***** Training Complete *****")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info(f"Final model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()