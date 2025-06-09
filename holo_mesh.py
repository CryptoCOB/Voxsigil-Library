from __future__ import annotations
"""HOLO-1.5 Recursive Cognition Mesh implementation.

This module provides a lightweight scaffolding for running a mesh of
quantized local LLM agents using symbolic triggers and recursive loops.
The design focuses on small hardware footprints by leveraging 4-bit
quantization and LoRA adapter fusion.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import asyncio
import logging

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    HAVE_TRANSFORMERS = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_TRANSFORMERS = False  # type: ignore
    torch = None

logger = logging.getLogger("holo_mesh")


@dataclass
class HOLOAgentConfig:
    """Configuration for a single HOLO agent."""

    model_name: str
    lora_adapters: List[str] = field(default_factory=list)
    max_tokens: int = 512
    device: str = "cuda"


class HOLOAgent:
    """A lightweight LLM agent with optional LoRA adapters."""

    def __init__(self, name: str, config: HOLOAgentConfig):
        if not HAVE_TRANSFORMERS:
            raise RuntimeError("transformers and peft required for HOLOAgent")
        self.name = name
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        for adapter in config.lora_adapters:
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter)
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to load LoRA adapter %s: %s", adapter, e)
        self.model.eval()

    @torch.inference_mode()
    async def generate(self, prompt: str) -> str:
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = await asyncio.to_thread(
            self.model.generate,
            **tokens,
            max_new_tokens=self.config.max_tokens,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


@dataclass
class HOLOMeshConfig:
    """Configuration for the HOLO mesh."""

    agents: Dict[str, HOLOAgentConfig]
    max_loaded: int = 2  # maximum simultaneously loaded models


class HOLOMesh:
    """Orchestrates a mesh of lightweight LLM agents."""

    def __init__(self, config: HOLOMeshConfig):
        self.config = config
        self.pool: Dict[str, HOLOAgent] = {}
        self.lock = asyncio.Lock()

    async def _ensure_loaded(self, name: str) -> HOLOAgent:
        async with self.lock:
            if name in self.pool:
                return self.pool[name]
            if len(self.pool) >= self.config.max_loaded:
                unload_name, unload_agent = next(iter(self.pool.items()))
                logger.debug("Unloading agent %s", unload_name)
                del self.pool[unload_name]
                del unload_agent
            agent_cfg = self.config.agents[name]
            logger.debug("Loading agent %s", name)
            self.pool[name] = HOLOAgent(name, agent_cfg)
            return self.pool[name]

    async def ask(self, name: str, prompt: str) -> str:
        agent = await self._ensure_loaded(name)
        return await agent.generate(prompt)

    async def conversation(self, chain: List[str], prompt: str) -> str:
        context = prompt
        for name in chain:
            context = await self.ask(name, context)
        return context


def demo() -> None:
    if not HAVE_TRANSFORMERS:
        print("transformers not installed")
        return
    config = HOLOMeshConfig(
        agents={
            "planner": HOLOAgentConfig(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            "critic": HOLOAgentConfig(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        }
    )
    mesh = HOLOMesh(config)
    result = asyncio.run(mesh.conversation(["planner", "critic"], "Hello"))
    print(result)


if __name__ == "__main__":
    demo()
