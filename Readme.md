# SML Multi-Dimensional Code Generation Analysis

This repository contains a research platform for investigating the efficiency, scalability, and multi-language capabilities of Large Language Models (LLMs) in code generation tasks.

The project is designed to move beyond simple implementation by conducting multi-dimensional experiments:

- **Efficiency**: Impact of LoRA Rank on performance vs. parameter count
- **Scalability**: Relationship between training data volume and syntax correctness
- **Complexity**: Comparative performance of fine-tuning on Python vs. Java

## 🚀 Hardware Requirements

This codebase is optimized for high-performance computing environments, specifically targeting the NVIDIA H200 (Hopper Architecture).

- **GPU**: NVIDIA H200 (or H100/A100 with adjustments)
- **VRAM**: >80GB Recommended (Script uses batch_size=64 and BF16 precision)
- **CUDA**: 12.1 or newer

## 🛠️ Installation

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies

It is recommended to install PyTorch with CUDA support first to ensure the correct version is used.

```bash
# Install PyTorch for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements
pip install -r requirements.txt
```

## 🧪 Running Experiments

The main script `Research_Experiments.py` is a unified runner. You control the specific experiment by modifying the configuration section at the top of the file.

### Configuration

Open `Research_Experiments.py` and locate:

```python
# ==========================================
# 1. CONFIGURATION SECTION (CHANGE THIS!)
# ==========================================

# Choose which dimension to investigate:
# Options: 'rank', 'scale', 'lang'
EXPERIMENT_TYPE = 'rank'
```

### Experiment Types

| Type    | Description                      | Research Question                                                        |
|---------|----------------------------------|--------------------------------------------------------------------------|
| `rank`  | Tests LoRA Ranks [4, 16, 64]     | Does increasing the rank of update matrices yield diminishing returns?   |
| `scale` | Tests Data Sizes [1k, 5k, 15k]   | What is the minimum data volume required for valid syntax?               |
| `lang`  | Tests Languages [Python, Java]   | Does model architecture bias performance towards specific languages?     |

### Execution

Run the script directly:

```bash
python Research_Experiments.py
```

## 📊 Evaluation & Metrics

The script performs robust evaluation beyond standard loss metrics:

- **BLEU Score**: Measures textual similarity to reference code
- **Syntax Pass Rate**: Compiles the generated code (Python only) to check for syntax errors without executing potentially unsafe logic
- **LLM Comparison**: Includes a stub to compare your fine-tuned model's output against external APIs (Gemini/GPT) for a baseline

### Output

- **Console**: Real-time training logs and final summary tables
- **Plots**: Matplotlib graphs visualizing the trade-offs (e.g., Rank vs. BLEU) are displayed or saved
- **Checkpoints**: Trained models are saved in `./results/`

## ⚠️ External LLM Benchmarking

To enable the comparison with Gemini or GPT models, locate Section 7 in the script and configure your API keys. By default, it uses a mock function to demonstrate the pipeline structure.