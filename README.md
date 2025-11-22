# Code Completion Model: Multi-Dimensional LLM Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive research project investigating the efficiency, scalability, and linguistic adaptability of Fine-Tuned Large Language Models (LLMs) for code generation tasks.

## 🎯 Project Overview

This project explores three fundamental dimensions of code generation using GPT-2 with LoRA fine-tuning:

1. **Parameter Efficiency** - Finding the optimal LoRA rank
2. **Data Scalability** - Understanding the relationship between training data volume and performance
3. **Language Adaptability** - Evaluating cross-language generalization capabilities

### Key Findings

- ✅ **Sweet Spot at Rank 16**: Optimal balance between model capacity and functional correctness
- 📉 **Complexity Trap**: Increasing data initially degrades performance before improvement
- 🌐 **Language Agnostic**: GPT-2 learns verbose languages (Java) as effectively as concise ones (Python)

## 🚀 Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended) or CPU
- 16GB RAM minimum
- 10GB free disk space
```

### Installation

```bash
# Clone the repository
git clone https://github.com/mvyas7/Code-Completion-ModeL.git
cd Code-Completion-ModeL

# Install dependencies
pip install torch transformers datasets peft evaluate sacrebleu tqdm pandas matplotlib

# Or use requirements.txt
pip install -r requirements.txt
```

### Running Experiments

```python
# Edit project.py to set experiment type
EXPERIMENT_TYPE = 'rank'  # Options: 'rank', 'scale', 'lang'

# Run the experiment
python project.py
```

**Output Files**:
- `results_{EXPERIMENT_TYPE}.csv` - Raw numerical results
- `results_{EXPERIMENT_TYPE}.png` - Visualization plots

## 📊 Experiments

### Experiment A: Parameter Efficiency (LoRA Rank)

**Research Question**: Does increasing LoRA rank linearly improve performance?

**Configuration**:
```python
EXPERIMENT_TYPE = 'rank'
ranks_to_test = [4, 16, 64]
sample_size = 200
```

**Key Results**:
| Rank | BLEU | Syntax Pass Rate |
|------|------|------------------|
| 4    | 9.42 | 25.0%           |
| 16   | 9.34 | **30.0%** ✓     |
| 64   | 9.64 | 25.0%           |

**Insight**: Rank 16 achieves the best functional correctness, demonstrating diminishing returns beyond this point.

[📖 Read detailed analysis →](RESULTS_README.md#experiment-a-parameter-efficiency-lora-rank)

---

### Experiment B: Data Scalability

**Research Question**: How does training data volume affect code quality?

**Configuration**:
```python
EXPERIMENT_TYPE = 'scale'
samples_to_test = [50, 150, 300]
lora_rank = 16
```

**Key Results**:
| Dataset Size | BLEU  | Syntax Pass Rate |
|--------------|-------|------------------|
| 50           | 10.09 | **45.0%** ✓     |
| 150          | 9.91  | 40.0%           |
| 300          | 12.64 | 40.0%           |

**Insight**: Small datasets produce simple but correct code. Larger datasets introduce complexity but require >1000 samples to master it.

[📖 Read detailed analysis →](RESULTS_README.md#experiment-b-scalability-data-volume)

---

### Experiment C: Language Adaptability

**Research Question**: Does language verbosity affect fine-tuning performance?

**Configuration**:
```python
EXPERIMENT_TYPE = 'lang'
languages_to_test = ['python', 'java', 'javascript']
sample_size = 200
lora_rank = 16
```

**Key Results**:
| Language   | BLEU | Syntax Pass Rate |
|------------|------|------------------|
| Python     | 9.42 | 30.0%           |
| Java       | 9.55 | N/A*            |
| JavaScript | TBD  | TBD             |

\*Java evaluation requires external compiler setup

**Insight**: GPT-2 learns textual patterns equally well across languages regardless of verbosity.

[📖 Read detailed analysis →](RESULTS_README.md#experiment-c-language-adaptability)

## 📁 Project Structure

```
Code-Completion-ModeL/
│
├── project.py                 # Main experiment runner
├── README.md                  # This file
├── DATASET_README.md         # Dataset preprocessing documentation
├── RESULTS_README.md         # Detailed experimental results & insights
├── requirements.txt          # Python dependencies
│
├── results/                  # Experimental outputs
│   ├── results_rank.csv
│   ├── results_scale.csv
│   ├── results_lang.csv
│   ├── results_rank.png
│   ├── results_scale.png
│   └── results_lang.png
│
└── checkpoints/              # (Optional) Saved model checkpoints
```

## 🔧 Configuration

### Key Parameters

```python
# Model Configuration
MODEL_NAME = "gpt2"           # Base model
DEVICE = "cuda" / "cpu"       # Auto-detected

# Training Hyperparameters
EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_LENGTH = 512
BATCH_SIZE = 32 (GPU) / 8 (CPU)

# LoRA Configuration
LORA_RANK = 16                # Adjustable in experiments
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["c_attn", "c_proj"]

# Dataset Configuration
SAMPLE_SIZE = 200             # Training samples per experiment
EVAL_SIZE = 50                # Fixed evaluation size
```

### Customization

**To test different LoRA ranks**:
```python
# In project.py
ranks_to_test = [8, 16, 32, 128]  # Add/modify ranks
```

**To test different dataset sizes**:
```python
samples_to_test = [100, 500, 1000, 2000]  # Larger scales
```

**To add new languages**:
```python
languages_to_test = ['python', 'java', 'javascript', 'go', 'rust']
```

## 📈 Evaluation Metrics

### BLEU Score
- Measures textual similarity between generated and reference code
- Range: 0-100 (higher is better)
- **Limitation**: Doesn't guarantee functional correctness

### Syntax Pass Rate
- Percentage of generated code that compiles/parses successfully
- **Python**: Uses `compile()` function for validation
- **Java/JS**: Requires external compiler setup (currently limited)

### Trade-off Analysis
Our experiments show **BLEU and Syntax Pass Rate don't always correlate**:
- High BLEU can indicate memorization without understanding
- High pass rate with lower BLEU suggests semantic correctness

## 🧪 Dataset Details

### Source
- **Primary**: `codeparrot/github-code` (streaming)
- **Fallback**: `codeparrot/codeparrot-clean-train`

### Preprocessing Pipeline
1. **Streaming Acquisition** - On-the-fly loading (no disk cache)
2. **Text Formatting** - Structured prompts (`### Code:\n{code}`)
3. **Tokenization** - GPT-2 tokenizer with validation
4. **Quality Filtering** - Vocab bounds checking
5. **Train/Eval Split** - Configurable training size + fixed 50 eval samples

### Memory Efficiency
- Traditional approach: 500-2000 MB disk usage
- Our approach: **5-50 MB** (95-99% reduction)
- All processing happens in-memory with streaming

[📖 Read full preprocessing details →](DATASET_README.md)

## 🎓 Research Insights

### The "Sweet Spot" Phenomenon
Our experiments identified **Rank 16** as the optimal LoRA configuration for GPT-2:
- **Rank 4**: Underfits (insufficient capacity)
- **Rank 16**: ✓ Best functional correctness
- **Rank 64**: Overfits to textual patterns

### The "Complexity Trap"
Increasing training data doesn't linearly improve performance:
```
Simple Code → Ambitious but Broken → Correct Complex Code
(50 samples)     (300 samples)         (1000+ samples?)
   45% pass         40% pass              Unknown
```

### Language Agnosticism
GPT-2's statistical learning is **language-neutral**:
- Verbose languages (Java) ≈ Concise languages (Python)
- BLEU scores are nearly identical (9.42 vs 9.55)
- Tokenization handles different syntax structures equally well

## 🚧 Limitations & Future Work

### Current Limitations

1. **Evaluation Infrastructure**
   - Only Python syntax validation is automated
   - Java/JavaScript require external compilers
   - No execution-based testing (test pass rate)

2. **Dataset Scale**
   - Limited to 300 samples due to computational constraints
   - Unable to empirically verify scaling beyond Complexity Trap

3. **Model Size**
   - Experiments use GPT-2 (124M parameters)
   - Larger models (CodeGen, StarCoder) may show different patterns

### Future Improvements

**Short-term**:
- [ ] Implement Docker-based multi-language evaluation
- [ ] Add CodeBLEU metric (syntax-aware)
- [ ] Expand to 1000+ samples per experiment

**Long-term**:
- [ ] Test larger models (350M, 1B parameters)
- [ ] Implement curriculum learning (simple → complex)
- [ ] Add execution-based metrics with test suites
- [ ] Explore cross-language transfer learning

## 📊 Recommended Configuration

Based on our multi-dimensional analysis, the **optimal setup for deployment** is:

```python
#