#!/usr/bin/env python3
import os
import time
import math
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None else default


def detect_text_column(example: Dict[str, Any]) -> str:
    candidates = [
        'text', 'content', 'raw', 'body', 'code', 'message', 'prompt', 'document'
    ]
    for k in candidates:
        if k in example and isinstance(example[k], str):
            return k
    # Fallback: first string field
    for k, v in example.items():
        if isinstance(v, str):
            return k
    # Last resort
    return 'text'


class LRFloorCallback(TrainerCallback):
    def __init__(self, min_lr: float):
        self.min_lr = float(min_lr)

    def on_step_end(self, args, state, control, **kwargs):
        opt = kwargs.get('optimizer')
        if not opt:
            return
        for g in opt.param_groups:
            if g.get('lr', 0) < self.min_lr:
                g['lr'] = self.min_lr


class TokensPerSecondCallback(TrainerCallback):
    def __init__(self):
        self.last_t = None
        self.last_tokens = 0
        self.total_tokens = 0

    def on_step_end(self, args, state, control, **kwargs):
        # We compute tokens from inputs via attention_mask if available
        inputs = kwargs.get('inputs')
        if not inputs:
            return
        attn = inputs.get('attention_mask')
        input_ids = inputs.get('input_ids')
        if attn is not None:
            tokens = int(attn.sum().item())
        elif input_ids is not None:
            tokens = int(input_ids.numel())
        else:
            tokens = 0
        self.total_tokens += tokens
        now = time.perf_counter()
        if self.last_t is None:
            self.last_t = now
            self.last_tokens = self.total_tokens
            return
        if state.global_step % max(1, args.logging_steps) == 0:
            dt = max(1e-6, now - self.last_t)
            dtk = self.total_tokens - self.last_tokens
            tps = dtk / dt
            gpu_mem = 0.0
            if torch.cuda.is_available():
                try:
                    gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
                except Exception:
                    gpu_mem = 0.0
            logger.info(
                f"⚙️ step={state.global_step} tokens/s={tps:.1f} tokens_total={self.total_tokens} gpu_mem={gpu_mem:.0f}MB lr={[g['lr'] for g in kwargs.get('optimizer').param_groups][0]:.2e}"
            )
            self.last_t = now
            self.last_tokens = self.total_tokens

class EarlyStopWithMinEpochs(TrainerCallback):
    def __init__(self, patience: int, min_epochs: float, metric: str = 'eval_loss', minimize: bool = True):
        self.patience = int(patience)
        self.min_epochs = float(min_epochs)
        self.metric = metric
        self.minimize = minimize
        self.best = math.inf if minimize else -math.inf
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.metric not in metrics:
            return
        value = float(metrics[self.metric])
        improved = (value < self.best) if self.minimize else (value > self.best)
        if improved:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
        epoch = state.epoch or 0.0
        if epoch >= self.min_epochs and self.wait >= self.patience:
            logger.info(f"🛑 Early stopping at epoch {epoch:.2f} with metric {self.metric}={value:.6f} (best={self.best:.6f})")
            control.should_training_stop = True


def main():
    model_id = env('NEBULA_MODEL_ID', 'gpt2')
    dataset_id = env('NEBULA_DATASET_ID', 'wikitext')
    dataset_name = env('NEBULA_DATASET_NAME', 'wikitext-2-raw-v1')
    train_split = env('NEBULA_TRAIN_SPLIT', 'train')
    eval_split = env('NEBULA_EVAL_SPLIT', 'validation')
    text_col_env = env('NEBULA_TEXT_COLUMN')
    max_len = int(env('NEBULA_MAX_SEQ', '512'))
    bs = int(env('NEBULA_BATCH_SIZE', '2'))
    accum = int(env('NEBULA_ACCUM_STEPS', '8'))
    lr = float(env('NEBULA_LR', '5e-5'))
    min_lr = float(env('NEBULA_MIN_LR', '1e-6'))
    workers = int(env('NEBULA_NUM_WORKERS', '0' if os.name == 'nt' else '4'))
    out_dir = env('NEBULA_OUTPUT_DIR', 'runs/hf_trainer')
    logging_steps = int(env('NEBULA_LOGGING_STEPS', '50'))
    eval_steps = int(env('NEBULA_EVAL_STEPS', '500'))
    save_steps = int(env('NEBULA_SAVE_STEPS', '1000'))
    num_train_epochs = float(env('NEBULA_EPOCHS', '1'))
    estop_patience = int(env('NEBULA_PATIENCE', '50'))
    estop_min_epochs = float(env('NEBULA_MIN_EPOCHS', '50'))
    fp16 = env('NEBULA_FP16', '1') == '1'
    grad_ckpt = env('NEBULA_GRAD_CKPT', '0') == '1'

    logger.info(f"📦 Loading dataset: {dataset_id}/{dataset_name}")
    ds = load_dataset(dataset_id, dataset_name)
    if train_split not in ds:
        # Best effort fallbacks
        train_split = 'train' if 'train' in ds else list(ds.keys())[0]
    eval_available = eval_split in ds

    trust = env('NEBULA_TRUST_REMOTE_CODE', '0') == '1'
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    sample = ds[train_split][0]
    text_col = text_col_env or detect_text_column(sample)
    logger.info(f"📝 Using text column: {text_col}")

    def tokenize_fn(batch):
        texts = batch[text_col]
        return tok(texts, truncation=True, max_length=max_len)

    logger.info("🔧 Tokenizing dataset ...")
    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds[train_split].column_names if c != text_col])
    train_ds = ds_tok[train_split]
    eval_ds = ds_tok[eval_split] if eval_available else None

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    logger.info(f"🧠 Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=trust)
    if grad_ckpt:
        model.gradient_checkpointing_enable()

    if torch.cuda.is_available():
        logger.info(f"🎮 CUDA available: {torch.cuda.get_device_name(0)}")

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=max(1, bs),
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        evaluation_strategy='steps' if eval_available else 'no',
        eval_steps=eval_steps if eval_available else None,
        save_steps=save_steps,
        save_total_limit=2,
        dataloader_num_workers=workers,
        fp16=fp16,
        report_to=[],
        run_name='nebula_hf_trainer',
        gradient_checkpointing=grad_ckpt,
        load_best_model_at_end=eval_available,
        metric_for_best_model='eval_loss' if eval_available else None,
        logging_first_step=True,
    )

    class TokenCountingTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._epoch_tokens = 0
            self._t0 = time.perf_counter()

        def training_step(self, model, inputs):
            attn = inputs.get('attention_mask')
            input_ids = inputs.get('input_ids')
            if attn is not None:
                self._epoch_tokens += int(attn.sum().item())
            elif input_ids is not None:
                self._epoch_tokens += int(input_ids.numel())
            return super().training_step(model, inputs)

        def log(self, logs):
            now = time.perf_counter()
            elapsed = max(1e-6, now - self._t0)
            tps = self._epoch_tokens / elapsed
            logs = dict(logs)
            logs['tokens_per_second'] = tps
            if torch.cuda.is_available():
                try:
                    logs['gpu_mem_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                except Exception:
                    logs['gpu_mem_mb'] = 0.0
            return super().log(logs)

    trainer = TokenCountingTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=collator,
        callbacks=[
            LRFloorCallback(min_lr),
            TokensPerSecondCallback(),
            EarlyStopWithMinEpochs(estop_patience, estop_min_epochs, metric='eval_loss', minimize=True) if eval_available else None,
        ],
    )

    logger.info("🚀 Starting HF training ...")
    trainer.train()
    logger.info("✅ Training finished")


if __name__ == '__main__':
    main()
