#!/usr/bin/env python3
"""
Multi-Teacher Knowledge Distillation - Fixed Version

Key improvements:
- Pre-computed teacher outputs (no blocking HTTP in __getitem__)
- Dynamic loss weidef list_data_files(data_dir: str, limit: int) -> List[str]:
    files = glob.glob(os.path.join(data_dir, \"*.json\"))
    # Sort by size (smallest first) to avoid huge files early
    files = sorted(files, key=lambda p: os.path.getsize(p))
    logger.info(f\"Found {len(files)} JSON files, processing smallest first\")
    return files[:limit]g to balance CE and distillation losses
- Proper checkpoint versioning
- Tokenizer passed explicitly (no global leaks)
- Better error handling and caching
"""

from __future__ import annotations

import os
import json
import time
import math
import hashlib
import logging
import glob
from dataclasses import dataclass
import sys
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("distill")

# Set conservative env flags early to avoid heavy optional backends
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Keep CPU thread usage modest to avoid RAM spikes
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None else default


@dataclass
class DistillationConfig:
    data_dir: str = env("NEBULA_DATA_DIR", "D:/01_1/Nebula/training/ComprehensiveTrainingData") or "D:/01_1/Nebula/training/ComprehensiveTrainingData"
    teachers: List[str] = None
    student_id: str = env("NEBULA_STUDENT_ID", "nebula") or "nebula"
    student_tokenizer_id: str = env("NEBULA_STUDENT_TOKENIZER", "gpt2") or "gpt2"
    batch_size: int = int(env("DISTILL_BATCH_SIZE", "4") or "4")
    seq_len: int = int(env("DISTILL_SEQ_LEN", "256") or "256")
    base_alpha: float = float(env("DISTILL_ALPHA", "0.7") or "0.7")
    dynamic_alpha: bool = env("DISTILL_DYNAMIC_ALPHA", "1") == "1"
    lr: float = float(env("DISTILL_LR", "5e-5") or "5e-5")
    epochs: int = int(float(env("DISTILL_EPOCHS", "10") or "10"))
    max_files: int = int(env("DISTILL_MAX_FILES", "1000") or "1000")
    max_samples: int = int(env("DISTILL_MAX_SAMPLES", "100000") or "100000")
    cache_dir: str = env("DISTILL_CACHE_DIR", "training/teacher_cache") or "training/teacher_cache"
    precomputed_cache: str = env("DISTILL_PRECOMPUTED", "training/precomputed_teachers.jsonl") or "training/precomputed_teachers.jsonl"
    # Checkpoint saving options
    save_every_epoch: bool = env("DISTILL_SAVE_EVERY_EPOCH", "0") == "1"
    save_final_only: bool = env("DISTILL_SAVE_FINAL_ONLY", "1") == "1"
    checkpoint_dir: str = env("DISTILL_CHECKPOINT_DIR", "runs/distill_student") or "runs/distill_student"
    # If set, rebuild precomputed cache even if a file exists
    ignore_existing_precomputed: bool = env("DISTILL_IGNORE_PRECOMPUTED", "0") == "1"
    temperature: float = float(env("DISTILL_TEMPERATURE", "0.7") or "0.7")
    top_p: float = float(env("DISTILL_TOP_P", "0.9") or "0.9")
    max_new_tokens: int = int(env("DISTILL_MAX_NEW_TOKENS", "128") or "128")
    prompt_template: str = env("DISTILL_PROMPT", "Summarize or answer concisely based on the following content:\n\n{content}\n\nAnswer:") or "Summarize or answer concisely based on the following content:\n\n{content}\n\nAnswer:"
    # Nebula student sizing (env-overridable)
    nebula_vocab_size: int = int(env("NEBULA_STUDENT_VOCAB", "50257") or "50257")
    nebula_d_model: int = int(env("NEBULA_STUDENT_D_MODEL", "512") or "512")
    nebula_n_heads: int = int(env("NEBULA_STUDENT_N_HEADS", "8") or "8")
    nebula_n_layers: int = int(env("NEBULA_STUDENT_N_LAYERS", "8") or "8")
    nebula_mlp_ratio: int = int(env("NEBULA_STUDENT_MLP_RATIO", "4") or "4")
    nebula_dropout: float = float(env("NEBULA_STUDENT_DROPOUT", "0.1") or "0.1")
    # Resume/init options
    resume_from: Optional[str] = env("DISTILL_RESUME_FROM")
    # Performance features
    fp16: bool = env("DISTILL_FP16", "0") == "1"
    grad_accum: int = int(env("DISTILL_GRAD_ACC", "1") or "1")
    lr_scheduler: str = env("DISTILL_LR_SCHEDULER", "linear_with_warmup") or "linear_with_warmup"
    warmup_steps: int = int(env("DISTILL_WARMUP_STEPS", "500") or "500")
    logging_steps: int = int(env("DISTILL_LOGGING_STEPS", "50") or "50")
    dataloader_workers: int = int(env("DISTILL_DATALOADER_WORKERS", "0" if os.name == 'nt' else "4") or ("0" if os.name == 'nt' else "4"))
    prefetch_factor: int = int(env("DISTILL_PREFETCH_FACTOR", "2") or "2")
    save_best: bool = env("DISTILL_SAVE_BEST", "0") == "1"
    metric_for_best: str = env("DISTILL_METRIC_FOR_BEST", "loss") or "loss"
    # Novelty hooks
    novel_manifold_proj: bool = env("NOVEL_MANIFOLD_PROJ", "0") == "1"
    novel_manifold_dim: int = int(env("NOVEL_MANIFOLD_DIM", "128") or "128")
    novel_manifold_reorth_every: int = int(env("NOVEL_MANIFOLD_REORTH_EVERY", "1000") or "1000")
    # STARFORGED Σ hooks
    stwpi_enabled: bool = env("STARFORGED_STWPI", "0") == "1"
    stwpi_theta: float = float(env("STARFORGED_STWPI_THETA", "0.5") or "0.5")
    stwpi_sigma: float = float(env("STARFORGED_STWPI_SIGMA", "0.1") or "0.1")
    stwpi_min: float = float(env("STARFORGED_STWPI_MIN", "0.1") or "0.1")
    stwpi_max: float = float(env("STARFORGED_STWPI_MAX", "10.0") or "10.0")
    csgl_enabled: bool = env("STARFORGED_CSGL", "0") == "1"
    csgl_interval: int = int(env("STARFORGED_CSGL_INTERVAL", "200") or "200")
    csgl_lambda: float = float(env("STARFORGED_CSGL_LAMBDA", "1e-3") or "1e-3")
    csgl_layers: str = env("STARFORGED_CSGL_LAYERS", "last") or "last"
    # PoUW integration
    pouw_enabled: bool = env("POUW_ENABLED", "0") == "1"
    pouw_receipt_interval: int = int(env("POUW_RECEIPT_INTERVAL", "100") or "100")
    pouw_worker_id: str = env("POUW_WORKER_ID", "default_worker") or "default_worker"
    pouw_local_receipts: str = env("POUW_LOCAL_RECEIPTS", "training/pouw_receipts") or "training/pouw_receipts"

    def __post_init__(self):
        teachers_env = env("DISTILL_TEACHERS", "gpt-oss:20b,llava-phi3:latest") or "gpt-oss:20b,llava-phi3:latest"
        self.teachers = [t.strip() for t in teachers_env.split(",") if t.strip()]
        os.makedirs(self.cache_dir, exist_ok=True)


cfg = DistillationConfig()
_overrides_applied = False

def _apply_config_overrides_from_file(c: DistillationConfig) -> DistillationConfig:
    path = env("DISTILL_CONFIG")
    if not path:
        return c
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load DISTILL_CONFIG '{path}': {e}")
        return c

    def set_if(k, cast=None):
        if k in data:
            v = data[k]
            try:
                if cast is not None:
                    v = cast(v)
            except Exception:
                pass
            setattr(c, k, v)

    # Apply known keys with appropriate casting
    set_if('data_dir', str)
    set_if('teachers')
    set_if('student_id', str)
    set_if('student_tokenizer_id', str)
    set_if('batch_size', int)
    set_if('seq_len', int)
    set_if('base_alpha', float)
    set_if('dynamic_alpha', bool)
    set_if('lr', float)
    set_if('epochs', int)
    set_if('max_files', int)
    set_if('max_samples', int)
    set_if('cache_dir', str)
    set_if('precomputed_cache', str)
    set_if('save_every_epoch', bool)
    set_if('save_final_only', bool)
    set_if('checkpoint_dir', str)
    set_if('ignore_existing_precomputed', bool)
    set_if('temperature', float)
    set_if('top_p', float)
    set_if('max_new_tokens', int)
    set_if('prompt_template', str)
    set_if('nebula_vocab_size', int)
    set_if('nebula_d_model', int)
    set_if('nebula_n_heads', int)
    set_if('nebula_n_layers', int)
    set_if('nebula_mlp_ratio', int)
    set_if('nebula_dropout', float)
    set_if('resume_from', str)
    set_if('fp16', bool)
    set_if('grad_accum', int)
    set_if('lr_scheduler', str)
    set_if('warmup_steps', int)
    set_if('logging_steps', int)
    set_if('dataloader_workers', int)
    set_if('prefetch_factor', int)
    set_if('save_best', bool)
    set_if('metric_for_best', str)
    # Novelty
    set_if('novel_manifold_proj', bool)
    set_if('novel_manifold_dim', int)
    set_if('novel_manifold_reorth_every', int)
    # STARFORGED
    set_if('stwpi_enabled', bool)
    set_if('stwpi_theta', float)
    set_if('stwpi_sigma', float)
    set_if('stwpi_min', float)
    set_if('stwpi_max', float)
    set_if('csgl_enabled', bool)
    set_if('csgl_interval', int)
    set_if('csgl_lambda', float)
    set_if('csgl_layers', str)

    # If teachers is provided as comma-separated string, split
    if isinstance(c.teachers, str):
        c.teachers = [t.strip() for t in c.teachers.split(',') if t.strip()]
    return c

def get_cfg() -> DistillationConfig:
    """Accessor that applies config-file overrides once if DISTILL_CONFIG is set."""
    global _overrides_applied, cfg
    if not _overrides_applied:
        cfg = _apply_config_overrides_from_file(cfg)
        _overrides_applied = True
    return cfg


class TeacherPrecomputer:
    """Precompute all teacher outputs before training to avoid HTTP blocking"""
    
    def __init__(self):
        self.client = OllamaClient()
    
    def precompute_all(self, samples: List[Dict[str, str]], output_path: str):
        """Precompute teacher responses for all samples. Writes JSONL for streaming."""
        logger.info(f"🔄 Precomputing teacher outputs for {len(samples)} samples...")
        if self.client.offline:
            logger.info("🛟 Offline mode: generating empty teacher outputs (no HTTP calls)")
        # Normalize to JSONL extension
        if not output_path.endswith('.jsonl'):
            output_path = os.path.splitext(output_path)[0] + '.jsonl'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, sample in enumerate(samples):
                content = sample["content"]
                prompt = build_prompt(content)
                teacher_outputs = {}
                if self.client.offline:
                    teacher_outputs = {t: "" for t in cfg.teachers}
                else:
                    for teacher in cfg.teachers:
                        try:
                            teacher_outputs[teacher] = self.client.generate(teacher, prompt)
                        except Exception as e:
                            logger.warning(f"Failed to get response from {teacher}: {e}")
                            teacher_outputs[teacher] = ""
                rec = {"content": content, "prompt": prompt, "teacher_outputs": teacher_outputs}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (i + 1) % 100 == 0 or i < 10:
                    logger.info(f"Progress: {i + 1}/{len(samples)}")
        logger.info(f"✅ Saved precomputed teacher outputs to {output_path}")
        return output_path


class OllamaClient:
    def __init__(self, host: str | None = None):
        self.host = (host or env("OLLAMA_HOST", "http://localhost:11434") or "http://localhost:11434").rstrip("/")
        self.session = requests.Session()
        self.timeout = float(env("OLLAMA_TIMEOUT", "120") or "120")  # Increased timeout
        self.offline = (env("DISTILL_OFFLINE", "0") == "1")

    def _cache_path(self, model: str, prompt: str) -> str:
        h = hashlib.sha256((model + "\n" + prompt).encode("utf-8", errors="ignore")).hexdigest()
        return os.path.join(cfg.cache_dir, f"{model.replace(':','_')}__{h}.json")

    def generate(self, model: str, prompt: str) -> str:
        if self.offline:
            return ""
            
        path = self._cache_path(model, prompt)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                return cached.get("response", "")
            except Exception:
                pass

        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "num_predict": cfg.max_new_tokens,
            },
        }
        
        try:
            logger.debug(f"Requesting {model} for prompt length {len(prompt)}")
            r = self.session.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            text = data.get("response", "")
        except Exception as e:
            logger.warning(f"Ollama generation failed for {model}: {e}")
            text = ""

        # Cache result
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"model": model, "prompt": prompt, "response": text}, f, ensure_ascii=False)
        except Exception:
            pass
            
        return text


def list_data_files(data_dir: str, limit: int) -> List[str]:
    files = glob.glob(os.path.join(data_dir, "*.json"))
    # Sort to prioritize files with actual content (avoid 0MB files first, then by size)
    files = sorted(files, key=lambda p: (os.path.getsize(p) == 0, os.path.getsize(p)))
    logger.info(f"Found {len(files)} JSON files, prioritizing files with content")
    return files[:limit]


def load_samples(files: List[str], max_samples: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for fp in files:
        # Skip huge files that will cause memory issues, but allow comprehensive training data
        try:
            file_size_mb = os.path.getsize(fp) / (1024 * 1024)
            if file_size_mb > 100:  # Skip files over 100MB (avoid 556MB monsters)
                logger.warning(f"Skipping large file {os.path.basename(fp)} ({file_size_mb:.1f}MB)")
                continue
            if file_size_mb > 0.1:  # Log files with actual content
                logger.info(f"Loading {os.path.basename(fp)} ({file_size_mb:.1f}MB)...")
        except Exception:
            continue
            
        try:
            with open(fp, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {os.path.basename(fp)}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Failed to open {os.path.basename(fp)}: {e}")
            continue

        rows: List[Dict[str, object]]
        if isinstance(data, dict):
            if isinstance(data.get("samples"), list):
                rows = data["samples"]
            else:
                rows = list(data.values())
        elif isinstance(data, list):
            rows = data
        else:
            rows = [data]

        samples_from_file = 0
        for row in rows:
            if len(out) >= max_samples:
                logger.info(f"🎯 Hit target {max_samples:,} samples, stopping")
                return out
                
            try:
                if isinstance(row, dict):
                    content = row.get("content")
                    if content is None:
                        for k in ("text", "body", "code", "raw", "message", "prompt", "data"):
                            if k in row:
                                content = row[k]
                                break
                    if content is None:
                        continue
                else:
                    content = str(row)
                    
                content = str(content)
                if len(content.strip()) < 20:  # Require meaningful content
                    continue
                    
                out.append({"content": content})
                samples_from_file += 1
                
                # Progress reporting for large datasets
                if len(out) % 5000 == 0:
                    logger.info(f"📈 Progress: {len(out):,}/{max_samples:,} samples loaded")
                    
            except Exception:
                continue
        
        if samples_from_file > 0:
            logger.info(f"✅ Added {samples_from_file:,} samples from {os.path.basename(fp)}")
    
    # If no samples found, create fallback test samples
    if not out:
        logger.warning("No valid samples found in data files, creating test samples")
        test_samples = [
            {"content": "The quick brown fox jumps over the lazy dog. This is a simple test sentence for training."},
            {"content": "Machine learning is a subset of artificial intelligence that focuses on algorithms."},
            {"content": "Python is a high-level programming language known for its simplicity and readability."},
            {"content": "Deep learning uses neural networks with multiple layers to process complex data patterns."},
            {"content": "Natural language processing helps computers understand and generate human language."},
        ]
        out.extend(test_samples[:max_samples])
    
    return out


def build_prompt(content: str) -> str:
    return cfg.prompt_template.format(content=content[:5000])


def aggregate_teacher_outputs(teacher_outputs: Dict[str, str]) -> str:
    """Aggregate outputs from multiple teachers"""
    texts = [t for t in teacher_outputs.values() if t and t.strip()]
    if not texts:
        return ""
    
    # For now, just use the first non-empty response
    # Could implement more sophisticated aggregation (voting, averaging, etc.)
    return texts[0]


class PrecomputedDistillDataset(torch.utils.data.Dataset):
    """Dataset that uses precomputed teacher outputs"""
    
    def __init__(self, precomputed_data: List[Dict], tokenizer):
        self.data = precomputed_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        content = item["content"]
        prompt = item["prompt"]
        teacher_outputs = item["teacher_outputs"]
        
        # Aggregate teacher responses
        teacher_answer = aggregate_teacher_outputs(teacher_outputs)
        # In offline mode, use truncated content as fallback if no teacher answer
        if not teacher_answer.strip():
            teacher_answer = content[:200] + "..." if len(content) > 200 else content
        
        # Tokenize for distillation (prompt + teacher answer)
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=cfg.seq_len, add_special_tokens=True)["input_ids"]
        ans_ids = self.tokenizer(teacher_answer or "", truncation=True, max_length=max(1, cfg.seq_len - len(prompt_ids)), add_special_tokens=False)["input_ids"]
        
        input_ids = (prompt_ids + ans_ids)[:cfg.seq_len]
        
        # Labels: -100 for prompt positions, actual tokens for answer
        distill_labels = [-100] * min(len(prompt_ids), len(input_ids))
        tail_len = len(input_ids) - len(distill_labels)
        if tail_len > 0:
            distill_labels += input_ids[len(distill_labels):]
        
        # LM target on original content
        lm_ids = self.tokenizer(content, truncation=True, max_length=cfg.seq_len, add_special_tokens=True)["input_ids"]
        lm_labels = lm_ids[1:] + [self.tokenizer.eos_token_id]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "distill_labels": torch.tensor(distill_labels, dtype=torch.long),
            "lm_input_ids": torch.tensor(lm_ids, dtype=torch.long),
            "lm_labels": torch.tensor(lm_labels, dtype=torch.long),
        }


class JsonlDistillDataset(torch.utils.data.Dataset):
    """Memory-light dataset that streams from a JSONL precomputed file via byte offsets."""
    def __init__(self, jsonl_path: str, tokenizer, max_items: Optional[int] = None):
        self.path = jsonl_path
        self.tokenizer = tokenizer
        self.offsets: List[int] = []
        self._build_index(max_items=max_items)

    def _build_index(self, max_items: Optional[int] = None):
        self.offsets.clear()
        off = 0
        count = 0
        # Build offsets using binary length; avoids loading JSON into RAM
        with open(self.path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.offsets.append(off)
                off += len(line)
                count += 1
                if max_items is not None and count >= max_items:
                    break

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline().decode("utf-8").strip()
        
        if not line:
            # Handle empty lines gracefully
            logger.warning(f"Empty line at index {idx}, using fallback content")
            return self._get_fallback_item()
            
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error at index {idx}: {e}, using fallback")
            return self._get_fallback_item()
            
        content = item.get("content", "")
        if not isinstance(content, str) or not content.strip():
            return self._get_fallback_item()
        prompt = item.get("prompt") or build_prompt(content)
        teacher_outputs = item.get("teacher_outputs", {})
        teacher_answer = aggregate_teacher_outputs(teacher_outputs)
        # In offline mode, use truncated content as fallback if no teacher answer
        if not teacher_answer.strip():
            teacher_answer = content[:200] + "..." if len(content) > 200 else content

        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=cfg.seq_len, add_special_tokens=True)["input_ids"]
        ans_ids = self.tokenizer(teacher_answer or "", truncation=True, max_length=max(1, cfg.seq_len - len(prompt_ids)), add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + ans_ids)[:cfg.seq_len]

        distill_labels = [-100] * min(len(prompt_ids), len(input_ids))
        tail_len = len(input_ids) - len(distill_labels)
        if tail_len > 0:
            distill_labels += input_ids[len(distill_labels):]

        lm_ids = self.tokenizer(content, truncation=True, max_length=cfg.seq_len, add_special_tokens=True)["input_ids"]
        lm_labels = lm_ids[1:] + [self.tokenizer.eos_token_id]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "distill_labels": torch.tensor(distill_labels, dtype=torch.long),
            "lm_input_ids": torch.tensor(lm_ids, dtype=torch.long),
            "lm_labels": torch.tensor(lm_labels, dtype=torch.long),
        }
    
    def _get_fallback_item(self):
        """Generate a fallback item when JSONL parsing fails"""
        content = "This is a fallback training sample for error recovery."
        prompt = "Summarize: " + content
        teacher_answer = "This is a fallback response."
        
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=cfg.seq_len, add_special_tokens=True)["input_ids"]
        ans_ids = self.tokenizer(teacher_answer, truncation=True, max_length=max(1, cfg.seq_len - len(prompt_ids)), add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + ans_ids)[:cfg.seq_len]

        distill_labels = [-100] * min(len(prompt_ids), len(input_ids))
        tail_len = len(input_ids) - len(distill_labels)
        if tail_len > 0:
            distill_labels += input_ids[len(distill_labels):]

        lm_ids = self.tokenizer(content, truncation=True, max_length=cfg.seq_len, add_special_tokens=True)["input_ids"]
        lm_labels = lm_ids[1:] + [self.tokenizer.eos_token_id]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "distill_labels": torch.tensor(distill_labels, dtype=torch.long),
            "lm_input_ids": torch.tensor(lm_ids, dtype=torch.long),
            "lm_labels": torch.tensor(lm_labels, dtype=torch.long),
        }


def create_collate_fn(tokenizer):
    """Factory function to create collate_fn with tokenizer closure"""
    def collate_fn(batch: List[Dict[str, torch.Tensor]]):
        def pad(seqs: List[torch.Tensor], pad_val: int):
            if not seqs:
                return torch.tensor([], dtype=torch.long)
            maxlen = max(s.size(0) for s in seqs)
            out = torch.full((len(seqs), maxlen), pad_val, dtype=seqs[0].dtype)
            for i, s in enumerate(seqs):
                out[i, : s.size(0)] = s
            return out

        input_ids = pad([b["input_ids"] for b in batch], pad_val=tokenizer.pad_token_id)
        distill_labels = pad([b["distill_labels"] for b in batch], pad_val=-100)
        lm_input_ids = pad([b["lm_input_ids"] for b in batch], pad_val=tokenizer.pad_token_id)
        lm_labels = pad([b["lm_labels"] for b in batch], pad_val=-100)
        
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        lm_attention_mask = (lm_input_ids != tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "distill_labels": distill_labels,
            "lm_input_ids": lm_input_ids,
            "lm_attention_mask": lm_attention_mask,
            "lm_labels": lm_labels,
        }
    return collate_fn


class DynamicAlphaScheduler:
    """Dynamically adjust alpha based on loss magnitudes"""
    
    def __init__(self, base_alpha: float = 0.7, window_size: int = 10):
        self.base_alpha = base_alpha
        self.window_size = window_size
        self.distill_losses = []
        self.ce_losses = []
    
    def update(self, distill_loss: float, ce_loss: float) -> float:
        self.distill_losses.append(distill_loss)
        self.ce_losses.append(ce_loss)
        
        # Keep only recent losses
        if len(self.distill_losses) > self.window_size:
            self.distill_losses.pop(0)
            self.ce_losses.pop(0)
        
        if len(self.distill_losses) < 3:  # Need some history
            return self.base_alpha
        
        # Calculate ratio of average losses
        avg_ce = sum(self.ce_losses) / len(self.ce_losses)
        avg_distill = sum(self.distill_losses) / len(self.distill_losses)
        
        if avg_distill < 1e-8:  # Avoid division by zero
            return self.base_alpha
        
        # If CE loss is much larger, increase distill weight
        ratio = avg_ce / avg_distill
        adaptive_alpha = min(0.9, max(0.1, 1.0 / (1.0 + math.log(max(1.0, ratio)))))
        
        return adaptive_alpha


def train_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    except Exception:
        pass
    
    # Load data
    files = list_data_files(cfg.data_dir, cfg.max_files)
    samples = load_samples(files, cfg.max_samples)
    
    # Fallback to test samples if no data found
    if not samples:
        logger.warning("No samples found in data files; using fallback test samples")
        samples = [
            {"content": "The quick brown fox jumps over the lazy dog. This is a test sentence for language modeling."},
            {"content": "Machine learning is a subset of artificial intelligence that focuses on algorithms."},
            {"content": "Python is a high-level programming language known for its simplicity and readability."},
            {"content": "Deep learning uses neural networks with multiple layers to learn complex patterns."},
        ]
        samples = samples[:cfg.max_samples]
    
    if not samples:
        raise RuntimeError("No training samples found for distillation.")
    
    logger.info(f"📂 Loaded {len(files)} files, {len(samples)} samples")
    
    # Early progress indication before expensive import
    logger.info("📦 Loading transformers library (this may take 30-60 seconds)...")
    
    # Lazy import tokenizer only (avoid heavy model imports)
    try:
        from transformers import AutoTokenizer
        logger.info("✅ Transformers library loaded")
    except ImportError:
        raise SystemExit("Missing dependencies. Install with: pip install transformers torch")
    
    # Initialize tokenizer and model
    logger.info("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_tokenizer_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"✅ Tokenizer loaded: {cfg.student_tokenizer_id}")

    # Choose student implementation
    logger.info(f"🧠 Creating student model: {cfg.student_id}...")
    model_is_hf = False
    is_nebula_student = False
    if cfg.student_id.lower() in ("nebula", "nebula-small", "nebula_student"):
        # Lightweight Nebula student
        logger.info("🌌 Loading Nebula model architecture...")
        from nebula_1b_model import Nebula1BModel
        # If resuming from a Nebula checkpoint, adopt its config to ensure shape match
        resume_conf = None
        resume_path = cfg.resume_from
        if resume_path and os.path.isdir(resume_path):
            try:
                with open(os.path.join(resume_path, "config.json"), "r", encoding="utf-8") as rf:
                    resume_conf = json.load(rf)
                logger.info(f"🔄 Resume requested from {resume_path}; adopting checkpoint config if compatible")
            except Exception as e:
                logger.warning(f"Could not read resume config: {e}")

        nebula_config = {
            'vocab_size': cfg.nebula_vocab_size,
            'd_model': resume_conf.get('d_model') if (resume_conf and 'd_model' in resume_conf) else cfg.nebula_d_model,
            'n_heads': resume_conf.get('n_heads') if (resume_conf and 'n_heads' in resume_conf) else cfg.nebula_n_heads,
            'n_layers': resume_conf.get('n_layers') if (resume_conf and 'n_layers' in resume_conf) else cfg.nebula_n_layers,
            'max_seq_len': max(cfg.seq_len, int(resume_conf.get('max_seq_len', 256) if resume_conf else 256)),
            'mlp_ratio': resume_conf.get('mlp_ratio') if (resume_conf and 'mlp_ratio' in resume_conf) else cfg.nebula_mlp_ratio,
            'dropout': resume_conf.get('dropout') if (resume_conf and 'dropout' in resume_conf) else cfg.nebula_dropout,
        }
        logger.info(f"📐 Nebula config: {nebula_config['n_layers']}L, {nebula_config['d_model']}D, {nebula_config['n_heads']}H")
        model = Nebula1BModel(nebula_config)
        # Enable checkpointing for memory
        model.gradient_checkpointing_enable = True
        is_nebula_student = True
    else:
        # Fallback to a HF model id if explicitly requested
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(cfg.student_id)
        model_is_hf = True
    
    # Setup device
    if torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            if visible:
                logger.info(f"🎯 CUDA_VISIBLE_DEVICES={visible}")
        except Exception:
            pass
        model = model.cuda()
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            names = []
            try:
                names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            except Exception:
                pass
            logger.info(f"🎮 Using {gpu_count} GPUs via DataParallel (no NCCL){' -> ' + ', '.join(names) if names else ''}")
            model = torch.nn.DataParallel(model)
        else:
            try:
                logger.info(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
            except Exception:
                logger.info("🎮 Using single CUDA device")
    else:
        model = model.to(device)
        logger.info("🖥️  Using CPU")

    # Optional: resume weights from checkpoint (supports Nebula student)
    if is_nebula_student and cfg.resume_from and os.path.isdir(cfg.resume_from):
        pt_path = os.path.join(cfg.resume_from, "pytorch_model.bin")
        if os.path.exists(pt_path):
            try:
                # Load to underlying module when DataParallel is used
                target = model.module if isinstance(model, torch.nn.DataParallel) else model
                sd = torch.load(pt_path, map_location=device)
                missing, unexpected = target.load_state_dict(sd, strict=False)
                logger.info(f"🔁 Loaded resume weights from {pt_path} (missing={len(missing)}, unexpected={len(unexpected)})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load resume weights: {e}")
    
    # Check for precomputed teacher outputs (prefer JSONL streaming)
    precomputed_path = cfg.precomputed_cache
    if cfg.ignore_existing_precomputed or not os.path.exists(precomputed_path):
        logger.info("🔄 Building JSONL precomputed cache (streaming-friendly)...")
        precomputed_path = TeacherPrecomputer().precompute_all(samples, precomputed_path)
    else:
        # Check if existing file is actually valid/non-empty
        try:
            file_size = os.path.getsize(precomputed_path)
            if file_size == 0:
                logger.warning("⚠️ Existing precomputed file is empty, rebuilding...")
                precomputed_path = TeacherPrecomputer().precompute_all(samples, precomputed_path)
            else:
                logger.info(f"📋 Found existing precomputed cache: {precomputed_path} ({file_size/1024/1024:.1f}MB)")
        except Exception:
            logger.warning("⚠️ Could not read existing precomputed file, rebuilding...")
            precomputed_path = TeacherPrecomputer().precompute_all(samples, precomputed_path)
    # Create dataset and dataloader
    if precomputed_path.endswith('.jsonl'):
        logger.info(f"📋 Using streaming dataset from {precomputed_path}")
        dataset = JsonlDistillDataset(precomputed_path, tokenizer, max_items=cfg.max_samples)
    else:
        logger.info(f"📋 Loading precomputed JSON array from {precomputed_path}")
        with open(precomputed_path, "r", encoding="utf-8") as f:
            precomputed_data = json.load(f)
        if len(precomputed_data) > cfg.max_samples:
            precomputed_data = precomputed_data[:cfg.max_samples]
        dataset = PrecomputedDistillDataset(precomputed_data, tokenizer)
    collate_fn = create_collate_fn(tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=0,  # Windows-safe
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    logger.info("🧰 Dataloader ready; starting training...")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    # Scheduler: linear with warmup by default
    steps_per_epoch = len(dataloader)
    optim_steps_per_epoch = max(1, math.ceil(steps_per_epoch / max(1, cfg.grad_accum)))
    total_optim_steps = max(1, optim_steps_per_epoch * max(1, cfg.epochs))
    def lr_lambda(current_step: int):
        ws = max(1, cfg.warmup_steps)
        if current_step < ws:
            return float(current_step) / float(ws)
        # linear decay to 0
        return max(0.0, float(total_optim_steps - current_step) / float(max(1, total_optim_steps - ws)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Dynamic alpha scheduler if enabled
    alpha_scheduler = DynamicAlphaScheduler(cfg.base_alpha) if cfg.dynamic_alpha else None
    
    total_params = sum(p.numel() for p in (model.module.parameters() if isinstance(model, torch.nn.DataParallel) else model.parameters()))
    logger.info(f"🧠 Student: {cfg.student_id} ({'HF' if model_is_hf else 'Nebula'}), {total_params:,} parameters")
    logger.info(f"📚 Training: {cfg.epochs} epochs, {steps_per_epoch} steps/epoch")
    
    model.train()
    global_step = 0
    
    # GradScaler: prefer torch.amp.GradScaler with device_type, fallback to cuda.amp for older torch
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(torch.cuda.is_available() and cfg.fp16))
    except Exception:
        from torch.cuda.amp import GradScaler as _OldGradScaler
        scaler = _OldGradScaler(enabled=(torch.cuda.is_available() and cfg.fp16))

    # STARFORGED: STW-PI OU process
    class _OUProcess:
        def __init__(self, theta: float, mu: float = 1.0, nu: float = 0.2):
            self.theta = theta
            self.mu = mu
            self.nu = nu
            self.eta = torch.tensor(mu, dtype=torch.float32)
        @torch.no_grad()
        def step(self, dt: float = 1.0) -> float:
            dx = torch.randn((), dtype=self.eta.dtype) * (dt ** 0.5)
            self.eta = self.eta + self.theta * (self.mu - self.eta) * dt + self.nu * dx
            return float(torch.clamp(self.eta, min=cfg.stwpi_min, max=cfg.stwpi_max).item())

    ou_proc = _OUProcess(theta=cfg.stwpi_theta, mu=1.0, nu=cfg.stwpi_sigma) if cfg.stwpi_enabled else None

    @torch.no_grad()
    def _lap1d(n: int, device, dtype):
        if n <= 1:
            return torch.zeros((1, 1), device=device, dtype=dtype)
        L = torch.zeros((n, n), device=device, dtype=dtype)
        for i in range(n):
            if i > 0:
                L[i, i-1] = -1
            L[i, i] = 2
            if i < n-1:
                L[i, i+1] = -1
        # Neumann-ish ends: reduce boundary strength
        L[0,0] = 1; L[-1,-1] = 1
        return L

    @torch.no_grad()
    def _apply_csgl_smoothing(target_model: nn.Module):
        lam = float(cfg.csgl_lambda)
        applied = 0
        modules = []
        for name, m in target_model.named_modules():
            if isinstance(m, nn.Linear):
                modules.append((name, m))
        if not modules:
            return 0
        if cfg.csgl_layers == 'last':
            modules = modules[-3:]  # last few linear layers as proxy
        for _, m in modules:
            W = m.weight
            dev, dt = W.device, W.dtype
            out_dim, in_dim = W.shape
            Lr = _lap1d(out_dim, dev, dt)
            Lc = _lap1d(in_dim, dev, dt)
            # W <- W - lam * (Lr @ W + W @ Lc^T)
            W.data = W.data - lam * (Lr @ W.data + (W.data @ Lc.transpose(0,1)))
            applied += 1
        return applied

    # Novelty Engine (MAGP)
    novelty = None
    if cfg.novel_manifold_proj:
        try:
            from novel_strategies import NoveltyEngine, NoveltyConfig as _NoveltyConfig
            ncfg = _NoveltyConfig(
                enable_manifold_proj=True,
                manifold_dim=cfg.novel_manifold_dim,
                manifold_reorth_every=cfg.novel_manifold_reorth_every,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            novelty = NoveltyEngine(model.module if isinstance(model, torch.nn.DataParallel) else model, ncfg)
            logger.info(f"🧪 Novelty: Manifold gradient projection enabled (d={cfg.novel_manifold_dim})")
        except Exception as e:
            logger.warning(f"NoveltyEngine not available: {e}")

    # PoUW Receipt Generation Setup
    pouw_receipts = []
    pouw_current_shard_data = []
    if cfg.pouw_enabled:
        os.makedirs(cfg.pouw_local_receipts, exist_ok=True)
        logger.info(f"🔐 PoUW: Receipt generation enabled (interval={cfg.pouw_receipt_interval})")

    best_metric = float('inf')
    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        total_loss_sum = 0.0
        distill_loss_sum = 0.0
        ce_loss_sum = 0.0
        steps_run = 0
        
        accum = max(1, cfg.grad_accum)
        for step, batch in enumerate(dataloader):
            global_step += 1
            
            # Move to device
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            
            # Forward passes and manual CausalLM losses (shifted)
            def compute_causal_loss(forward_input_ids, forward_attention_mask, labels_tensor):
                outputs = model(
                    input_ids=forward_input_ids,
                    attention_mask=(
                        # Nebula expects a (B, 1, T, T) or broadcastable attn mask; build causal+padding
                        (torch.tril(torch.ones((forward_input_ids.size(1), forward_input_ids.size(1)), device=forward_input_ids.device))
                         .unsqueeze(0).unsqueeze(0)
                         .to(dtype=torch.bool)
                         & forward_attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.bool))
                        if is_nebula_student else forward_attention_mask
                    ),
                )
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels_tensor[:, 1:].contiguous()
                if shift_logits.numel() == 0 or shift_labels.numel() == 0:
                    return torch.zeros((), device=forward_input_ids.device, dtype=logits.dtype, requires_grad=True)
                # Ensure at least one non-ignored label exists to avoid NaN
                flat_labels = shift_labels.view(-1)
                valid_mask = flat_labels != -100
                if not torch.any(valid_mask):
                    return torch.zeros((), device=forward_input_ids.device, dtype=logits.dtype, requires_grad=True)
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                loss = nn.functional.cross_entropy(
                    flat_logits[valid_mask],
                    flat_labels[valid_mask],
                    ignore_index=-100,
                )
                return loss

            use_amp = (torch.cuda.is_available() and cfg.fp16)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    distill_loss = compute_causal_loss(
                        batch["input_ids"], batch["attention_mask"], batch["distill_labels"]
                    )
                    ce_loss = compute_causal_loss(
                        batch["lm_input_ids"], batch["lm_attention_mask"], batch["lm_labels"]
                    )
                    if alpha_scheduler:
                        alpha = alpha_scheduler.update(float(distill_loss.item()), float(ce_loss.item()))
                    else:
                        alpha = cfg.base_alpha
                    total_loss = alpha * distill_loss + (1.0 - alpha) * ce_loss
                if not torch.isfinite(total_loss):
                    logger.warning("Non-finite total_loss encountered; skipping step")
                    continue
                scaled = total_loss / accum
                scaler.scale(scaled).backward()
                # Step per accumulation
                if (step + 1) % accum == 0:
                    # STARFORGED: STW-PI per-step LR warp
                    orig_lrs = None
                    if ou_proc is not None:
                        try:
                            scale = ou_proc.step()
                            orig_lrs = [pg['lr'] for pg in optimizer.param_groups]
                            for pg in optimizer.param_groups:
                                pg['lr'] = pg['lr'] * scale
                        except Exception:
                            orig_lrs = None
                    scaler.unscale_(optimizer)
                    if novelty is not None:
                        novelty.project_current_gradients()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Restore LR after warp
                    if orig_lrs is not None:
                        for pg, lr0 in zip(optimizer.param_groups, orig_lrs):
                            pg['lr'] = lr0
            else:
                distill_loss = compute_causal_loss(
                    batch["input_ids"], batch["attention_mask"], batch["distill_labels"]
                )
                ce_loss = compute_causal_loss(
                    batch["lm_input_ids"], batch["lm_attention_mask"], batch["lm_labels"]
                )
                if alpha_scheduler:
                    alpha = alpha_scheduler.update(float(distill_loss.item()), float(ce_loss.item()))
                else:
                    alpha = cfg.base_alpha
                total_loss = alpha * distill_loss + (1.0 - alpha) * ce_loss
                if not torch.isfinite(total_loss):
                    logger.warning("Non-finite total_loss encountered; skipping step")
                    continue
                scaled = total_loss / accum
                scaled.backward()
                if (step + 1) % accum == 0:
                    orig_lrs = None
                    if ou_proc is not None:
                        try:
                            scale = ou_proc.step()
                            orig_lrs = [pg['lr'] for pg in optimizer.param_groups]
                            for pg in optimizer.param_groups:
                                pg['lr'] = pg['lr'] * scale
                        except Exception:
                            orig_lrs = None
                    if novelty is not None:
                        novelty.project_current_gradients()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if orig_lrs is not None:
                        for pg, lr0 in zip(optimizer.param_groups, orig_lrs):
                            pg['lr'] = lr0
            
            # Accumulate losses
            tl = float(total_loss.item())
            dl = float(distill_loss.item())
            cl = float(ce_loss.item())
            if math.isfinite(tl):
                total_loss_sum += tl
            if math.isfinite(dl):
                distill_loss_sum += dl
            if math.isfinite(cl):
                ce_loss_sum += cl
            steps_run += 1

            # PoUW: Collect data for receipt generation
            if cfg.pouw_enabled:
                # Store sample data for receipt (just metadata, not full content)
                sample_info = {
                    'step': global_step,
                    'epoch': epoch,
                    'batch_size': batch["input_ids"].size(0) if "input_ids" in batch else 0,
                    'seq_len': batch["input_ids"].size(1) if "input_ids" in batch else 0,
                    'loss': tl,
                    'distill_loss': dl,
                    'ce_loss': cl
                }
                pouw_current_shard_data.append(sample_info)

                # Generate PoUW receipt at intervals
                if global_step % cfg.pouw_receipt_interval == 0:
                    try:
                        import time as time_module
                        receipt_data = {
                            'shard_id': f"{cfg.pouw_worker_id}_epoch{epoch}_step{global_step}",
                            'worker_id': cfg.pouw_worker_id,
                            'start_time': epoch_start,
                            'end_time': time_module.time(),
                            'work_hash': hashlib.sha256(str(pouw_current_shard_data).encode()).hexdigest(),
                            'result_summary': {
                                'steps_completed': len(pouw_current_shard_data),
                                'avg_loss': sum(s['loss'] for s in pouw_current_shard_data) / max(1, len(pouw_current_shard_data)),
                                'avg_distill_loss': sum(s['distill_loss'] for s in pouw_current_shard_data) / max(1, len(pouw_current_shard_data)),
                                'avg_ce_loss': sum(s['ce_loss'] for s in pouw_current_shard_data) / max(1, len(pouw_current_shard_data)),
                                'total_samples': sum(s['batch_size'] for s in pouw_current_shard_data)
                            }
                        }
                        
                        # Save receipt locally
                        receipt_path = os.path.join(cfg.pouw_local_receipts, f"receipt_{receipt_data['shard_id']}.json")
                        with open(receipt_path, 'w') as f:
                            json.dump(receipt_data, f, indent=2)
                        
                        pouw_receipts.append(receipt_data)
                        pouw_current_shard_data = []  # Reset for next shard
                        
                        logger.info(f"🔐 PoUW: Generated receipt {receipt_data['shard_id']} ({receipt_data['result_summary']['steps_completed']} steps)")
                    
                    except Exception as e:
                        logger.warning(f"PoUW receipt generation failed: {e}")
            
            # Logging
            if step % max(1, cfg.logging_steps) == 0:
                mem_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
                try:
                    lr_cur = optimizer.param_groups[0]['lr']
                except Exception:
                    lr_cur = None
                logger.info(
                    f"E{epoch+1} S{step:3d}/{steps_per_epoch} | "
                    f"Loss: {float(total_loss.item()):.4f} (α={alpha:.3f}) | "
                    f"Distill: {float(distill_loss.item()):.4f} | CE: {float(ce_loss.item()):.4f} | "
                    f"GPU: {mem_mb:.0f}MB" + (f" LR={lr_cur:.2e}" if lr_cur is not None else "")
                )
        
            # STARFORGED: CSGL periodic smoothing
            if cfg.csgl_enabled and cfg.csgl_interval > 0 and (global_step % cfg.csgl_interval == 0):
                tgt = model.module if isinstance(model, torch.nn.DataParallel) else model
                try:
                    applied = _apply_csgl_smoothing(tgt)
                    if applied > 0 and step % max(1, cfg.logging_steps) == 0:
                        logger.info(f"Σ/CSGL smoothing applied on {applied} linear layers (λ={cfg.csgl_lambda})")
                except Exception as e:
                    logger.warning(f"CSGL smoothing failed: {e}")

        # End of epoch
        epoch_time = time.perf_counter() - epoch_start
        denom = max(1, steps_run)
        avg_total = total_loss_sum / denom
        avg_distill = distill_loss_sum / denom
        avg_ce = ce_loss_sum / denom
        
        logger.info(
            f"✅ Epoch {epoch+1} completed in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_total:.4f} | Distill: {avg_distill:.4f} | CE: {avg_ce:.4f}"
        )
        
        # Save checkpoint with configurable frequency / best
        should_save = (
            cfg.save_every_epoch or 
            (epoch + 1) == cfg.epochs or  # Always save final
            not cfg.save_final_only
        )
        if cfg.save_best and avg_total < best_metric:
            best_metric = avg_total
            best_dir = f"{cfg.checkpoint_dir}/best_model"
            os.makedirs(best_dir, exist_ok=True)
            to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            if model_is_hf and hasattr(to_save, "save_pretrained"):
                to_save.save_pretrained(best_dir)
            else:
                torch.save(to_save.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
                try:
                    conf = {
                        "vocab_size": getattr(to_save, "vocab_size", None),
                        "d_model": getattr(to_save, "d_model", None),
                        "n_heads": getattr(to_save, "n_heads", None),
                        "n_layers": getattr(to_save, "n_layers", None),
                        "max_seq_len": getattr(to_save, "max_seq_len", None),
                        "mlp_ratio": getattr(to_save, "mlp_ratio", None),
                        "dropout": getattr(to_save, "dropout", None),
                    }
                    with open(os.path.join(best_dir, "config.json"), "w", encoding="utf-8") as cf:
                        json.dump(conf, cf, indent=2)
                except Exception:
                    pass
            try:
                tokenizer.save_pretrained(best_dir)
            except Exception:
                pass
            with open(os.path.join(best_dir, "training_metadata.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "avg_total_loss": avg_total}, f, indent=2)
            logger.info(f"🏅 Saved best checkpoint to {best_dir}")
        
        if should_save:
            if cfg.save_final_only and (epoch + 1) == cfg.epochs:
                checkpoint_dir = f"{cfg.checkpoint_dir}/final_model"
            else:
                checkpoint_dir = f"{cfg.checkpoint_dir}/epoch_{epoch+1:02d}"
            
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Handle DataParallel wrapping
            to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            if model_is_hf and hasattr(to_save, "save_pretrained"):
                to_save.save_pretrained(checkpoint_dir)
            else:
                # Save raw PyTorch weights and Nebula config
                torch.save(to_save.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
                try:
                    # Persist minimal config
                    conf = {
                        "vocab_size": getattr(to_save, "vocab_size", None),
                        "d_model": getattr(to_save, "d_model", None),
                        "n_heads": getattr(to_save, "n_heads", None),
                        "n_layers": getattr(to_save, "n_layers", None),
                        "max_seq_len": getattr(to_save, "max_seq_len", None),
                        "mlp_ratio": getattr(to_save, "mlp_ratio", None),
                        "dropout": getattr(to_save, "dropout", None),
                    }
                    with open(os.path.join(checkpoint_dir, "config.json"), "w", encoding="utf-8") as cf:
                        json.dump(conf, cf, indent=2)
                except Exception:
                    pass
            # Always save tokenizer
            try:
                tokenizer.save_pretrained(checkpoint_dir)
            except Exception:
                pass

            # Save training metadata
            metadata = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "avg_total_loss": avg_total,
            "avg_distill_loss": avg_distill,
            "avg_ce_loss": avg_ce,
            "alpha": alpha if alpha_scheduler else cfg.base_alpha,
            "config": cfg.__dict__
        }

            with open(os.path.join(checkpoint_dir, "training_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"💾 Saved checkpoint to {checkpoint_dir}")
        else:
            logger.info(f"⏭️ Skipping checkpoint save for epoch {epoch+1} (save_every_epoch={cfg.save_every_epoch})")


def main():
    # Environment setup for Windows + CUDA
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    # TRANSFORMERS_NO_TF is set above to avoid TF import

    # Quick dry-run supports fast sanity checks
    if cfg.epochs <= 0:
        files = list_data_files(cfg.data_dir, cfg.max_files)
        samples = load_samples(files, min(cfg.max_samples, 5))
        logger.info(
            f"🧪 Dry run: teachers={len(cfg.teachers)}, files={len(files)}, samples={len(samples)}, offline={env('DISTILL_OFFLINE','0')}"
        )
        return 0

    train_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())