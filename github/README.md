# ScanCoder: Leveraging Human Attention Pattern to Enhance LLM for Coding

This repository contains the implementation and artifact code for the paper **"ScanCoder: Leveraging Human Attention Pattern to Enhance LLM for Coding"** submitted to FSE 2026.

## Overview

ScanCoder is a comprehensive framework that systematically integrates cognitive simulation with LLM enhancement for software engineering tasks. The framework addresses critical limitations in human-LLM collaboration by generating human-like attention patterns at scale using minimal eye-tracking data through cognitive simulation based on the ACT-R architecture.

### Key Contributions

1. **Novel Cognitive Simulation Framework**: First comprehensive framework combining cognitive simulation through ACT-R architecture with LLM enhancement for software engineering
2. **Cross-Language Generalizability**: Demonstrates that C++-derived cognitive patterns can effectively enhance Java programming task performance
3. **Significant Performance Improvements**: Achieves up to 39% improvement on CrystalBLEU completion metrics and 22% improvement on BERTScore summarization
4. **Mechanistic Insights**: Reveals how cognitive guidance fundamentally reshapes model attention in task-dependent ways

## Repository Structure

```
github/
├── data_preprocessing/           # Data preprocessing and scanpath generation
│   ├── generate_scanpath_training_file.py
│   └── generate_scanpath_training_file_summarization.py
├── training_script/             # Model training scripts
│   ├── llama_training_completion.py
│   ├── llama_training_summerization.py
│   └── deepseek_training.py
├── evaluation_script/           # Model evaluation scripts
│   ├── eval_llama_scancoder.py
│   └── eval_deepseek_scancoder.py
├── ablation_study/             # Ablation study experiments
│   ├── llama_non_order.py      # Order-only component ablation
│   └── llama_non_cover.py      # Coverage-only component ablation
├── analysis/                   # Analysis and visualization tools
│   ├── attention_distribution.py
│   └── cross_task_evaluation.py
└── README.md                   # This file
```

## Methodology

### 1. Cognitive Simulation (CodeACT-R)

ScanCoder leverages **CodeACT-R**, a cognitive simulation model based on ACT-R architecture that generates human-like attention patterns (scanpaths) during code reading. The simulation operates through:

- **Declarative Memory**: Stores structured knowledge extracted from human scanpaths, encoding statements with semantic properties and temporal positions
- **Production Rules**: Pattern matching between declarative memory and current code buffer
- **Cross-Language Adaptation**: Successfully transfers C++-trained patterns to Java tasks

### 2. Scanpath-to-Subword Projection

The framework addresses the granularity mismatch between human attention (semantic tokens) and LLM processing (subwords) through:

- Uniform weight propagation from semantic tokens to subword tokens
- Preservation of cognitive signal alignment with LLM tokenization
- Integration with instruction-output templates for supervised learning

### 3. Cognitively-Guided Fine-Tuning

A weighted cross-entropy loss function emphasizes tokens according to their:
- **Coverage Information**: Whether tokens received human attention
- **Order Information**: Temporal sequence of attention flow
- **Complexity Factors**: Semantic complexity and cognitive load

The weight calculation follows:
```python
weight = base_weight + (1.0 / attention_order)
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.21+
- NumPy, pandas
- javalang (for Java AST parsing)

### Installation

```bash
pip install torch transformers datasets numpy pandas javalang matplotlib seaborn
```

### Data Preprocessing

Generate scanpath-augmented training data:

```bash
# For completion task
python data_preprocessing/generate_scanpath_training_file.py \
    --start_idx 0 \
    --max_samples 1000 \
    --seed 42

# For summarization task  
python data_preprocessing/generate_scanpath_training_file_summarization.py \
    --start_idx 0 \
    --max_samples 1000 \
    --seed 42
```

### Model Training

Train models with cognitive guidance:

```bash
# Llama completion training
python training_script/llama_training_completion.py --seed 42

# Llama summarization training
python training_script/llama_training_summerization.py --seed 42

# DeepSeek training
python training_script/deepseek_training.py --task completion --seed 42
```

### Model Evaluation

Evaluate trained models:

```bash
# Evaluate Llama models
python evaluation_script/eval_llama_scancoder.py --task completion --device cuda:0

# Evaluate DeepSeek models  
python evaluation_script/eval_deepseek_scancoder.py --task completion --device cuda:0
```

### Ablation Studies

Run ablation experiments to understand component contributions:

```bash
# Order-only ablation (removes coverage information)
python ablation_study/llama_non_order.py --seed 42

# Coverage-only ablation (removes order information)  
python ablation_study/llama_non_cover.py --seed 42
```

## Key Features

### Cognitive Simulation Engine

- **AST-based Semantic Analysis**: Extracts semantic types using Java Abstract Syntax Tree parsing
- **Complexity Calculation**: Multi-dimensional scoring considering arithmetic operations, logical operations, function calls, and control structures
- **Scanpath Generation**: Two-step process with length estimation and pattern matching strategies

### Advanced Training Framework

- **Custom Loss Function**: Weighted cross-entropy that emphasizes cognitively important tokens
- **Task-Specific Adaptation**: Different attention strategies for completion vs summarization tasks
- **Reproducible Training**: Comprehensive seed management for consistent results

### Comprehensive Evaluation

- **Multiple Metrics**: CodeBLEU, CrystalBLEU, H-Exact for completion; ROUGE-L, METEOR, BERTScore for summarization
- **Attention Analysis**: Generation Confidence Score, Recency Focus Score, Average Focus Score, Attention Entropy
- **Cross-Architecture**: Evaluated on both Llama-3.2 and DeepSeek-Coder models

## Experimental Results

### Performance Improvements

| Model | Task | Metric | Baseline | ScanCoder | Improvement |
|-------|------|--------|----------|-----------|-------------|
| Llama-3.2 | Completion | CrystalBLEU | 26.65 | 65.90 | +39.25 |
| Llama-3.2 | Completion | CodeBLEU | 19.47 | 60.48 | +41.01 |
| Llama-3.2 | Summarization | BERTScore | 33.41 | 50.30 | +16.89 |
| DeepSeek | Completion | CrystalBLEU | 18.98 | 54.61 | +35.63 |
| DeepSeek | Summarization | BERTScore | 28.64 | 50.57 | +21.93 |

### Attention Mechanism Analysis

ScanCoder produces task-dependent attention reshaping:

- **Code Completion**: Enhanced sequential processing with higher recency focus
- **Code Summarization**: Transformed global understanding with 2.5× higher focus on semantically critical tokens

## Analysis and Visualization

Generate attention visualizations:

```bash
python analysis/attention_distribution.py
```

This creates attention heatmaps comparing baseline and ScanCoder models, saved as `attention_matrix.pdf`.

## Reproducibility

All experiments use fixed random seeds (default: 42) for reproducibility:

```bash
# Set seed for all training scripts
python training_script/llama_training_completion.py --seed 42
```

The framework ensures deterministic results through:
- PyTorch seed management
- NumPy random state control
- CUDA deterministic operations

## Limitations and Future Work

### Current Limitations

1. **Limited Scanpaths**: Initial eye-tracking data from 48 participants, reduced to 15-30 scanpaths per stimulus after quality assurance
2. **Code Complexity**: Evaluation on relatively short code snippets suitable for controlled experiments
3. **Individual Differences**: Learns average attention patterns, potentially missing personalized cognitive strategies

### Future Directions

1. **Extended Code Complexity**: Adaptation to longer functions and complex class hierarchies
2. **Personalized Guidance**: Individual programmer profile-based cognitive guidance
3. **Additional SE Tasks**: Extension to bug detection, refactoring, and code review tasks

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{scancoder2025,
    title={ScanCoder: Leveraging Human Attention Pattern to Enhance LLM for Coding},
    author={Anonymous Author(s)},
    booktitle={Proceedings of FSE 2026},
    year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the implementation or paper, please contact the authors or open an issue in this repository.

## Acknowledgments

This work builds upon the ACT-R cognitive architecture and leverages insights from eye-tracking research in software engineering. We thank the participants in our eye-tracking studies and the open-source community for providing the foundation models and tools that made this research possible.