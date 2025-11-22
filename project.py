# ==========================================
# 0. CUDA RESET & ENVIRONMENT SETUP
# ==========================================

import torch
import os

# Clear CUDA cache and reset if there was a previous error
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    try:
        torch.cuda.synchronize()
    except:
        print("⚠ CUDA context was corrupted. Restarting kernel is recommended.")
        print("However, attempting to continue...")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("✓ CUDA environment reset complete")

# ==========================================
# SML MULTI-DIMENSIONAL RESEARCH EXPERIMENTS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    logging as hf_logging,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
from tqdm.auto import tqdm

hf_logging.set_verbosity_error()
os.environ["WANDB_DISABLED"] = "true"

# ==========================================
# 1. GPU DETECTION & CONFIGURATION
# ==========================================

def setup_device():
    """
    Automatically detect and configure the best available device (GPU/CPU)
    """
    if torch.cuda.is_available():
        device = "cuda"
        num_gpus = torch.cuda.device_count()
        
        print("="*60)
        print("🚀 GPU DETECTED!")
        print("="*60)
        print(f"Number of GPUs available: {num_gpus}")
        
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name}")
            print(f"    Memory: {gpu_memory:.2f} GB")
        
        # Use first GPU by default
        torch.cuda.set_device(0)
        print(f"\n✓ Using GPU 0: {torch.cuda.get_device_name(0)}")
        
        # Enable optimizations for modern GPUs (Ampere/Hopper)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        print("✓ TF32 optimizations enabled")
        print("✓ cuDNN benchmark mode enabled")
        print("="*60 + "\n")
        
        return device, True
    else:
        print("="*60)
        print("⚠️  NO GPU DETECTED - Using CPU")
        print("="*60)
        print("Note: Training will be significantly slower on CPU")
        print("="*60 + "\n")
        return "cpu", False

# Setup device
DEVICE, HAS_GPU = setup_device()

def print_gpu_memory(stage=""):
    """Print current GPU memory usage"""
    if HAS_GPU:
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Configuration
EXPERIMENT_TYPE = 'rank'  # Options: 'rank', 'scale', 'lang'
MODEL_NAME = "gpt2"
EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_LENGTH = 512

print(f"Running Experiment Type: {EXPERIMENT_TYPE}")
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")

# ==========================================
# 2. DATA & UTILS (FIXED)
# ==========================================

def get_dataset(lang="python", sample_size=5000):
    """
    Loads dataset with proper token ID validation.
    """
    print(f"Loading {lang} dataset...")
    try:
        ds = load_dataset("codeparrot/github-code", streaming=False, split="train", languages=[lang])
        ds = ds.take(sample_size * 3)  # Take more for filtering
    except:
        print("Falling back to CodeParrot...")
        ds = load_dataset("codeparrot/codeparrot-clean-train", split="train", streaming=False)
        ds = ds.take(sample_size * 3)
    
    # Initialize tokenizer FIRST
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Get model to check actual vocab size
    temp_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    actual_vocab_size = temp_model.config.vocab_size
    del temp_model
    
    # Clear GPU memory if available
    if HAS_GPU:
        torch.cuda.empty_cache()
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model vocab size: {actual_vocab_size}")
    
    # Format dataset
    def format_prompt(example):
        code = example.get("code", "") or example.get("content", "")
        if len(code) > 2000:  # Limit code length
            code = code[:2000]
        return {"text": f"### Code:\n{code}"}
    
    # Process dataset
    processed_ds = ds.map(format_prompt)
    
    def tokenize_and_validate(examples):
        """Tokenize with strict validation"""
        outputs = tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None
        )
        
        # CRITICAL: Ensure all token IDs are within valid range
        valid_input_ids = []
        valid_attention_mask = []
        
        for ids, mask in zip(outputs["input_ids"], outputs["attention_mask"]):
            # Check if any token is out of bounds
            if all(token_id < actual_vocab_size for token_id in ids):
                valid_input_ids.append(ids)
                valid_attention_mask.append(mask)
        
        if not valid_input_ids:
            # Return empty batch if nothing valid
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        return {
            "input_ids": valid_input_ids,
            "attention_mask": valid_attention_mask,
            "labels": [ids.copy() for ids in valid_input_ids]
        }
    
    tokenized_ds = processed_ds.map(
        tokenize_and_validate,
        batched=True,
        batch_size=100,
        remove_columns=processed_ds.column_names
    )
    
    # Filter out empty entries
    tokenized_ds = tokenized_ds.filter(lambda x: len(x['input_ids']) > 0)
    
    # Create train/eval split
    if len(tokenized_ds) < sample_size:
        print(f"⚠ Only {len(tokenized_ds)} valid samples available")
        sample_size = len(tokenized_ds) - 50
    
    train_ds = tokenized_ds.select(range(min(sample_size, len(tokenized_ds) - 50)))
    eval_ds = tokenized_ds.select(range(len(tokenized_ds) - 50, len(tokenized_ds)))
    
    print(f"✓ Dataset loaded: {len(train_ds)} train, {len(eval_ds)} eval samples")
    
    return train_ds, eval_ds, tokenizer

# ==========================================
# 3. EVALUATION (FIXED)
# ==========================================

def check_syntax(code_str, lang="python"):
    """Returns True if code compiles"""
    if lang != "python":
        return True
    try:
        compile(code_str, '<string>', 'exec')
        return True
    except:
        return False

def evaluate_model(model, tokenizer, eval_ds, lang="python"):
    """
    Evaluate with proper error handling and vocab bounds checking
    """
    # Clear GPU memory if available
    if HAS_GPU:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Ensure model is on correct device
    model = model.to(DEVICE)
    model.eval()
    bleu = evaluate.load("sacrebleu")
    
    preds, refs = [], []
    syntax_passes = 0
    vocab_size = model.config.vocab_size
    
    print(f"Evaluating with vocab_size={vocab_size}")
    
    num_samples = min(20, len(eval_ds))  # Reduce evaluation samples
    
    for i in tqdm(range(num_samples), desc="Generating"):
        try:
            ex = eval_ds[i]
            
            # Validate input tokens
            input_ids_list = ex['input_ids']
            if max(input_ids_list) >= vocab_size:
                print(f"⚠ Skipping sample {i}: max token {max(input_ids_list)} >= {vocab_size}")
                continue
            
            # Truncate input for generation
            input_ids = torch.tensor([input_ids_list[:100]], dtype=torch.long, device=DEVICE)
            
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    top_p=None
                )
            
            # Check generated tokens are valid
            if (gen_ids >= vocab_size).any():
                print(f"⚠ Generated invalid tokens in sample {i}")
                continue
            
            decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            gen_code = decoded.split("### Code:")[-1].strip() if "### Code:" in decoded else decoded
            
            ref_text = tokenizer.decode(ex['labels'], skip_special_tokens=True)
            ref_code = ref_text.split("### Code:")[-1].strip() if "### Code:" in ref_text else ref_text
            
            preds.append(gen_code)
            refs.append([ref_code])
            
            if check_syntax(gen_code, lang):
                syntax_passes += 1
                
        except RuntimeError as e:
            if "CUDA" in str(e) or "device" in str(e).lower():
                print(f"⚠ Device error on sample {i}: {str(e)[:100]}")
                if HAS_GPU:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                break  # Stop evaluation on device errors
            else:
                print(f"⚠ Error on sample {i}: {str(e)[:100]}")
                continue
    
    # Compute metrics
    if len(preds) > 0:
        bleu_score = bleu.compute(predictions=preds, references=refs)['score']
        syntax_rate = (syntax_passes / len(preds)) * 100
    else:
        bleu_score = 0.0
        syntax_rate = 0.0
    
    print(f"Evaluated {len(preds)} samples successfully")
    return {"BLEU": bleu_score, "Syntax_Pass_Rate": syntax_rate}

# ==========================================
# 4. TRAINING (FIXED)
# ==========================================

def run_training(train_ds, eval_ds, tokenizer, lora_rank=8, output_name="run"):
    """
    Training with proper model initialization and automatic device placement
    """
    print(f"Loading model {MODEL_NAME}...")
    
    # Determine dtype based on device
    if HAS_GPU:
        dtype = torch.bfloat16
        print("Using bfloat16 precision on GPU")
    else:
        dtype = torch.float32
        print("Using float32 precision on CPU")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype
    )
    
    # CRITICAL: Ensure model vocab matches tokenizer
    if model.config.vocab_size != tokenizer.vocab_size:
        print(f"⚠ Vocab size mismatch! Model: {model.config.vocab_size}, Tokenizer: {tokenizer.vocab_size}")
        print(f"Using model vocab size: {model.config.vocab_size}")
    
    # Move model to device
    print(f"Moving model to {DEVICE}...")
    model = model.to(DEVICE)
    print_gpu_memory("After model load")
    
    # LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]  # Specific to GPT-2
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training Arguments - adapt based on device
    if HAS_GPU:
        training_args = TrainingArguments(
            output_dir=f"./results/{output_name}",
            per_device_train_batch_size=32,
            gradient_accumulation_steps=2,
            num_train_epochs=EPOCHS,
            logging_steps=20,
            learning_rate=LEARNING_RATE,
            save_strategy="no",
            report_to="none",
            bf16=True,  # Use bfloat16 on GPU
            optim="adamw_torch",
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
    else:
        # CPU-optimized settings
        training_args = TrainingArguments(
            output_dir=f"./results/{output_name}",
            per_device_train_batch_size=8,  # Smaller batch for CPU
            gradient_accumulation_steps=4,  # More accumulation
            num_train_epochs=EPOCHS,
            logging_steps=20,
            learning_rate=LEARNING_RATE,
            save_strategy="no",
            report_to="none",
            fp16=False,  # No fp16 on CPU
            optim="adamw_torch",
            dataloader_num_workers=2,  # Fewer workers on CPU
            remove_unused_columns=False,
        )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print(f"Training {output_name}...")
    trainer.train()
    print_gpu_memory("After training")
    
    return model

# ==========================================
# 5. EXPERIMENT RUNNERS
# ==========================================

results_log = []

if EXPERIMENT_TYPE == 'rank':
    ranks_to_test = [4, 16, 64]
    train_ds, eval_ds, tokenizer = get_dataset(lang="python", sample_size=5000)
    
    for r in ranks_to_test:
        print(f"\n{'='*50}")
        print(f"Testing LoRA Rank: {r}")
        print('='*50)
        
        # Clear GPU memory if available
        if HAS_GPU:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        try:
            model = run_training(train_ds, eval_ds, tokenizer, lora_rank=r, output_name=f"rank_{r}")
            
            # Clear GPU memory if available
            if HAS_GPU:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            metrics = evaluate_model(model, tokenizer, eval_ds)
            metrics['Configuration'] = f"Rank {r}"
            results_log.append(metrics)
            print(f"✓ Rank {r} Results: BLEU={metrics['BLEU']:.2f}, Syntax={metrics['Syntax_Pass_Rate']:.1f}%")
            
            del model
            if HAS_GPU:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error in rank {r} experiment: {str(e)}")
            import traceback
            traceback.print_exc()
            if HAS_GPU:
                torch.cuda.empty_cache()
            continue

elif EXPERIMENT_TYPE == 'scale':
    samples_to_test = [1000, 5000, 10000]
    full_train_ds, eval_ds, tokenizer = get_dataset(lang="python", sample_size=15000)
    
    for s in samples_to_test:
        print(f"\n{'='*50}")
        print(f"Testing Dataset Size: {s}")
        print('='*50)
        
        # Clear GPU memory if available
        if HAS_GPU:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        try:
            subset_train = full_train_ds.select(range(min(s, len(full_train_ds))))
            model = run_training(subset_train, eval_ds, tokenizer, lora_rank=16, output_name=f"size_{s}")
            
            # Clear GPU memory if available
            if HAS_GPU:
                torch.cuda.empty_cache()
            
            metrics = evaluate_model(model, tokenizer, eval_ds)
            metrics['Configuration'] = f"Size {s}"
            results_log.append(metrics)
            print(f"✓ Size {s} Results: BLEU={metrics['BLEU']:.2f}, Syntax={metrics['Syntax_Pass_Rate']:.1f}%")
            
            del model
            if HAS_GPU:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error in size {s} experiment: {str(e)}")
            if HAS_GPU:
                torch.cuda.empty_cache()
            continue

# ==========================================
# 6. RESULTS
# ==========================================

if results_log:
    df = pd.DataFrame(results_log)
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(df)
    
    # Save results
    df.to_csv(f'results_{EXPERIMENT_TYPE}.csv', index=False)
    print(f"\n✓ Results saved to results_{EXPERIMENT_TYPE}.csv")
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = range(len(df))
    
    color = 'tab:blue'
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('BLEU Score', color=color, fontsize=12)
    ax1.bar(x, df['BLEU'], color=color, alpha=0.6, label='BLEU')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Configuration'], rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Syntax Pass Rate (%)', color=color, fontsize=12)
    ax2.plot(x, df['Syntax_Pass_Rate'], color=color, marker='o', linewidth=2, markersize=8, label='Syntax Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Experiment Results: {EXPERIMENT_TYPE.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results_{EXPERIMENT_TYPE}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to results_{EXPERIMENT_TYPE}.png")
    plt.show()

# Print final GPU statistics if available
if HAS_GPU:
    print("\n" + "="*60)
    print("GPU MEMORY SUMMARY")
    print("="*60)
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"GPU {i}:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Total: {total:.2f} GB")
    print("="*60)

print("\n✓ Experiment complete!")