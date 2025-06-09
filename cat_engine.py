# File: vanta_cat_engine.py (Refactored Revision 3)
"""
C.A.T. Engine Module – VantaCore Adaptation
Encapsulates six core breakthroughs.
This version uses VantaCore for basic orchestration and manages its own
specialized cognitive components via dependency injection or internal defaults.
"""

# pylint: disable=import-error

import asyncio  # For async tasks
import json
import logging
import math
import random
import threading
import time
import uuid
from typing import Any, Protocol, runtime_checkable

import numpy as np

from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType
from Vanta.core.UnifiedVantaCore import (
    UnifiedVantaCore as VantaCore,  # Core orchestrator singleton
)

# --- Basic logging setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VantaCore.CATEngine")  # Logger for this specific engine


# --- Configuration Class ---
class CATEngineConfig:
    def __init__(
        self,
        interval_s: int = 300,
        log_level: str = "INFO",
        default_memory_braid_config: dict[str, Any] | None = None,
        default_echo_memory_config: dict[str, Any] | None = None,
    ):
        if not isinstance(interval_s, int) or interval_s <= 0:
            logger.warning(
                f"Invalid CATEngine interval_s: {interval_s}. Defaulting to 300."
            )
            self.interval_s: int = 300
        else:
            self.interval_s: int = interval_s

        self.log_level: str = log_level
        self.default_memory_braid_config = default_memory_braid_config or {}
        self.default_echo_memory_config = default_echo_memory_config or {}

        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)


# --- Component Interface Definitions (Protocols) ---
@runtime_checkable
class MemoryClusterInterface(Protocol):
    def get_recent_memories(self, limit: int = 10) -> list[dict[str, Any]]: ...
    def store(self, data: Any, metadata: dict[str, Any] | None = None) -> Any: ...
    def store_event(self, data: Any, event_type: str | None = None) -> Any: ...
    def search_by_modality(
        self, modality: str, limit: int = 10
    ) -> list[dict[str, Any]]: ...
    def search(
        self, query: str, metadata_filter: dict[str, Any] | None = None, limit: int = 10
    ) -> list[dict[str, Any]]: ...
    def embed_text(self, text: str) -> list[float]: ...
    def get_beliefs(self) -> list[dict[str, Any]]: ...


@runtime_checkable
class BeliefRegistryInterface(Protocol):
    def get_active_beliefs(self) -> list[dict[str, Any]]: ...
    def update_belief_confidence(
        self, belief_id: str, new_confidence: float
    ) -> None: ...
    def add_contradiction(self, contradiction_data: dict[str, Any]) -> None: ...
    def add_belief(
        self, statement: str, confidence: float, belief_id: str | None = None
    ) -> Any: ...
    def record_contradiction(self, id1: str, id2: str, type: str) -> None: ...


@runtime_checkable
class StateProviderInterface(
    Protocol
):  # Renamed from Omega3Interface for VantaCore clarity
    def get_current_state(self) -> dict[str, Any]: ...
    def get_data_by_modality(self, modality: str, limit: int = 5) -> list[Any]: ...


@runtime_checkable
class FocusManagerInterface(Protocol):  # Renamed from ArtControllerInterface
    def get_current_focus(self) -> str: ...
    def get_active_tasks(self) -> list[str]: ...


@runtime_checkable
class MetaLearnerInterface(Protocol):
    def get_heuristics(self) -> list[dict[str, Any]]: ...
    def update_heuristic(self, heuristic_id: str, updates: dict[str, Any]) -> None: ...
    def add_heuristic(self, heuristic_data: dict[str, Any]) -> Any: ...


@runtime_checkable
class ModelManagerInterface(Protocol):
    def get_embedding(self, text: str) -> list[float]: ...
    def run_simulation(self, scenario: dict[str, Any]) -> dict[str, Any]: ...
    def evaluate_causal_impact(self, perturbation: dict[str, Any]) -> float: ...
    def simulate(self, scenario: dict[str, Any]) -> dict[str, Any]: ...
    def evaluate_impact(self, perturbation: dict[str, Any]) -> float: ...


@runtime_checkable
class MemoryBraidInterface(Protocol):
    def store_mirrored_data(
        self,
        original_key: Any,
        mirrored_data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
    def retrieve_mirrored_data(self, original_key: Any) -> Any | None: ...
    def get_braid_stats(self) -> dict[str, Any]: ...
    def adapt_behavior(self, context_key: str) -> dict[str, Any]: ...


@runtime_checkable
class EchoMemoryInterface(Protocol):
    def record_cognitive_trace(
        self, component_name: str, action: str, details: dict[str, Any]
    ) -> None: ...
    def get_recent_traces(self, limit: int = 10) -> list[dict[str, Any]]: ...


# --- Default Internal Implementations for VantaCore Components ---
# These are basic, in-memory versions. VantaCore or user code can provide more sophisticated ones.
class DefaultVantaMemoryCluster(MemoryClusterInterface):
    def __init__(self, config: dict[str, Any] | None = None):
        self._mem: list[dict[str, Any]] = []
        logger.info("DefaultVantaMemoryCluster (in-memory) active.")

    def get_recent_memories(self, limit: int = 10):
        return self._mem[-limit:]

    def store(self, data: Any, meta: dict[str, Any] | None = None):
        self._mem.append({"d": data, "m": meta or {}})
        return f"mem_{len(self._mem) - 1}"

    def store_event(self, data: Any, et: str | None = None):
        self._mem.append({"evt": data, "et": et})
        return f"evt_{len(self._mem) - 1}"

    def search_by_modality(self, modality: str, limit: int = 10):
        results = [
            item
            for item in reversed(self._mem)
            if (
                (isinstance(item, dict) and item.get("et") == modality)
                or (
                    isinstance(item, dict)
                    and isinstance(item.get("m"), dict)
                    and item["m"].get("modality") == modality
                )
            )
        ]
        return results[:limit]

    def search(self, q: str, mf: dict[str, Any] | None = None, limit: int = 10):
        q_lower = q.lower()
        results = []
        for item in reversed(self._mem):
            text = ""
            if "d" in item:
                text = str(item["d"])
            elif "evt" in item:
                text = str(item["evt"])
            if q_lower in text.lower():
                if mf:
                    meta = item.get("m", {})
                    if all(meta.get(k) == v for k, v in mf.items()):
                        results.append(item)
                else:
                    results.append(item)
            if len(results) >= limit:
                break
        return results

    def embed_text(self, t: str):
        # Simple deterministic embedding based on character codes
        vec = [0.0] * 16
        for i, ch in enumerate(t.encode("utf-8")):
            vec[i % 16] += ch / 255.0
        return vec

    def get_beliefs(self):
        beliefs = [
            item["evt"]
            for item in self._mem
            if item.get("et") == "belief" and "evt" in item
        ]
        return beliefs


class DefaultVantaBeliefRegistry(BeliefRegistryInterface):
    def __init__(self, config: dict[str, Any] | None = None):
        self._b: dict = {}
        self._contradictions: list[dict[str, Any]] = []
        logger.info("DefaultVantaBeliefRegistry (in-memory) active.")

    def get_active_beliefs(self):
        return list(self._b.values())

    def update_belief_confidence(self, bid: str, nc: float):
        if bid in self._b:
            self._b[bid]["confidence"] = nc

    def add_contradiction(self, cd: dict[str, Any]):
        self._contradictions.append(cd)
        logger.debug(f"DefaultBelief: Contradiction added {cd}")

    def add_belief(self, s: str, c: float, bid: str | None = None):
        _id = bid or f"b_{len(self._b)}"
        self._b[_id] = {"id": _id, "statement": s, "confidence": c}
        return _id

    def record_contradiction(self, id1: str, id2: str, type_str: str):
        self.add_contradiction({"ids": [id1, id2], "type": type_str})

    def get_contradictions_for(self, belief_id: str) -> list[dict[str, Any]]:
        return [c for c in self._contradictions if belief_id in c.get("ids", [])]


class DefaultVantaStateProvider(StateProviderInterface):  # Was Omega3
    def __init__(self, config: dict[str, Any] | None = None):
        logger.info("DefaultVantaStateProvider active.")

    def get_current_state(self):
        return {"default_state_active": True, "val": random.random()}

    def get_data_by_modality(self, modality: str, limit: int = 5):
        return []


class DefaultVantaFocusManager(FocusManagerInterface):  # Was ArtController
    def __init__(self, config: dict[str, Any] | None = None):
        logger.info("DefaultVantaFocusManager active.")

    def get_current_focus(self):
        return "Default Focus"

    def get_active_tasks(self):
        return ["Default Task"]


class DefaultVantaMetaLearner(MetaLearnerInterface):
    def __init__(self, config: dict[str, Any] | None = None):
        self._h: list = []
        logger.info("DefaultVantaMetaLearner active.")

    def get_heuristics(self):
        return self._h

    def update_heuristic(self, hid: str, u: dict[str, Any]):
        for h in self._h:
            if isinstance(h, dict) and h.get("id") == hid:
                h.update(u)
                return
        logger.warning(f"Heuristic {hid} not found; adding new one")
        new_h = {"id": hid}
        new_h.update(u)
        self._h.append(new_h)

    def add_heuristic(self, hd: dict[str, Any]):
        self._h.append(hd)
        return f"h_{len(self._h) - 1}"


class DefaultVantaModelManager(ModelManagerInterface):
    def __init__(self, config: dict[str, Any] | None = None):
        logger.info("DefaultVantaModelManager active.")

    def get_embedding(self, t: str):
        return [random.random() for _ in range(128)]

    def run_simulation(self, s: dict[str, Any]):
        return {
            "outcome": "default_sim_outcome_model_mgr",
            "confidence": random.random(),
        }

    def evaluate_causal_impact(self, p: dict[str, Any]):
        return random.random()

    def simulate(self, s: dict[str, Any]):
        return self.run_simulation(s)

    def evaluate_impact(self, p: dict[str, Any]):
        return self.evaluate_causal_impact(p)


class DefaultVantaMemoryBraid(MemoryBraidInterface):
    def __init__(self, config: dict[str, Any] | None = None):
        self._storage: dict[Any, Any] = {}
        logger.info("DefaultVantaMemoryBraid (in-memory) active.")

    def store_mirrored_data(self, k: Any, v: Any, m: dict[str, Any] | None = None):
        self._storage[k] = v

    def retrieve_mirrored_data(self, k: Any):
        return self._storage.get(k)

    def get_braid_stats(self):
        return {"braid_size": len(self._storage)}

    def adapt_behavior(self, key: str):
        return {"adapted_action": "default_braid_adapt"}


class DefaultVantaEchoMemory(EchoMemoryInterface):
    def __init__(self, config: dict[str, Any] | None = None):
        self._log: list = []
        logger.info("DefaultVantaEchoMemory (in-memory list) active.")

    def record_cognitive_trace(self, c: str, a: str, d: dict[str, Any]):
        entry = {"ts": time.time(), "comp": c, "act": a, "det": d}
        self._log.append(entry)
        logger.debug(f"EchoTrace: {entry}")

    def get_recent_traces(self, limit: int = 10):
        return self._log[-limit:]


# --- C.A.T. Engine Implementation ---
class CATEngine:
    COMPONENT_NAME = "cat_engine"

    def __init(
        self,
        vanta_core: VantaCore,
        config: CATEngineConfig,
        memory_cluster: MemoryClusterInterface | None = None,
        belief_registry: BeliefRegistryInterface | None = None,
        state_provider: StateProviderInterface | None = None,
        focus_manager: FocusManagerInterface | None = None,
        meta_learner: MetaLearnerInterface | None = None,
        model_manager: ModelManagerInterface | None = None,
        memory_braid: MemoryBraidInterface | None = None,
        echo_memory: EchoMemoryInterface | None = None,
        rag_engine: Any | None = None,  # Generic RAG for now
    ):
        self.vanta_core = vanta_core
        self.config = config
        logger.info(
            f"CATEngine initializing. Interval: {self.config.interval_s}s. LogLevel: {self.config.log_level}"
        )

        self.memory: MemoryClusterInterface = (
            memory_cluster or DefaultVantaMemoryCluster()
        )
        self.beliefs: BeliefRegistryInterface = (
            belief_registry or DefaultVantaBeliefRegistry()
        )
        self.state_provider: StateProviderInterface = (
            state_provider or DefaultVantaStateProvider()
        )  # Renamed omega3
        self.focus_manager: FocusManagerInterface = (
            focus_manager or DefaultVantaFocusManager()
        )  # Renamed art
        self.learner: MetaLearnerInterface = meta_learner or DefaultVantaMetaLearner()
        self.model_mgr: ModelManagerInterface = (
            model_manager or DefaultVantaModelManager()
        )

        self.memory_braid_instance: MemoryBraidInterface = (
            memory_braid
            or DefaultVantaMemoryBraid(self.config.default_memory_braid_config)
        )

        self.echo_memory_instance: EchoMemoryInterface = (
            echo_memory
            or DefaultVantaEchoMemory(self.config.default_echo_memory_config)
        )

        self.rag_component = (
            rag_engine  # Assumed to be an optional component for some operations
        )

        self.running = False
        self.thread: threading.Thread | None = None
        self.current_phase: str | None = None
        self.last_error: str | None = None

        self.advanced_learner: MetaLearnerInterface | None = (
            self.vanta_core.get_component("advanced_meta_learner")
        )  # Example: an optional advanced learner

        # Register with VantaCore
        self.vanta_core.register_component(
            self.COMPONENT_NAME, self, metadata={"type": "cognitive_engine"}
        )
        logger.info("CATEngine instance registered with VantaCore.")

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.thread = threading.Thread(
                target=self._run_loop, daemon=True, name="CATEngineLoop"
            )
            self.thread.start()
            logger.info("C.A.T. Engine started its processing loop.")
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.started",
                {"interval_s": self.config.interval_s},
                source=self.COMPONENT_NAME,
            )
        else:
            logger.warning("C.A.T. Engine attempt to start when already running.")

    def stop(self) -> None:
        if self.running:
            logger.info("C.A.T. Engine stopping...")
            self.running = False
            if self.thread and self.thread.is_alive():
                try:
                    self.thread.join(
                        timeout=max(1.0, self.config.interval_s / 10)
                    )  # Ensure timeout is positive
                except Exception as e:
                    logger.error(f"Error joining C.A.T. Engine thread: {e}")
            if self.thread and self.thread.is_alive():
                logger.warning("C.A.T. Engine thread did not stop cleanly.")
            self.thread = None
            logger.info("C.A.T. Engine stopped.")
            self.vanta_core.publish_event(
                f"{self.COMPONENT_NAME}.stopped", {}, source=self.COMPONENT_NAME
            )
        else:
            logger.info("C.A.T. Engine already stopped.")

    def _run_loop(self) -> None:
        logger.info(f"{self.COMPONENT_NAME}._run_loop started.")
        while self.running:
            cycle_start_time = time.monotonic()
            try:
                logger.info(f"--- {self.COMPONENT_NAME}: Starting Cycle ---")
                self._categorize_phase()
                if not self.running:
                    break
                self._analyze_phase()
                if not self.running:
                    break
                self._test_phase()
                logger.info(f"--- {self.COMPONENT_NAME}: Completed Cycle ---")
            except Exception as e:
                self.last_error = str(e)
                logger.exception(
                    f"Critical error in C.A.T. cycle (phase: {self.current_phase}): {e}"
                )

            cycle_duration = time.monotonic() - cycle_start_time
            sleep_time = max(
                0.1, self.config.interval_s - cycle_duration
            )  # Ensure minimum sleep
            if self.running:
                logger.debug(f"{self.COMPONENT_NAME} sleeping for {sleep_time:.2f}s.")
                time.sleep(sleep_time)
        logger.info(f"{self.COMPONENT_NAME}._run_loop finished.")

    def _categorize_phase(self) -> None:
        self.current_phase = "Categorize"
        logger.info(f"Entering {self.current_phase} Phase.")
        try:
            self._hypermutable_heuristic_forest()
            self._cross_modal_analogy_bloom()
        except Exception as e:
            logger.exception(f"Error in {self.current_phase} Phase: {e}")
        finally:
            self.current_phase = None
            logger.info(
                f"Finished {self.current_phase if self.current_phase else 'Categorize'} Phase."
            )  # Ensure phase name if error

    def _hypermutable_heuristic_forest(self) -> None:
        logger.debug("Running Hypermutable Heuristic Forest...")
        try:
            heuristics = self.learner.get_heuristics()
            recent_memories = self.memory.get_recent_memories(limit=20)
            if not heuristics or not recent_memories:
                return logger.debug(
                    "Not enough heuristics/memories for Heuristic Forest."
                )

            updated_heuristics = 0
            for h_dict in heuristics:  # Make sure h_dict is a dict
                if not isinstance(h_dict, dict) or "id" not in h_dict:
                    continue
                matches = self._check_heuristic_matches(h_dict, recent_memories)
                new_novelty = self._calculate_novelty_score(h_dict, matches)
                if abs(new_novelty - h_dict.get("novelty_score", 0.5)) > 0.01:
                    self.learner.update_heuristic(
                        h_dict["id"],
                        {"novelty_score": new_novelty, "last_updated": time.time()},
                    )
                    updated_heuristics += 1

            if heuristics:
                # Sort ensuring items are dicts and have 'novelty_score'
                valid_heuristics_for_sort = [
                    h
                    for h in heuristics
                    if isinstance(h, dict) and "novelty_score" in h
                ]
                valid_heuristics_for_sort.sort(
                    key=lambda x: x["novelty_score"], reverse=True
                )
                if (
                    len(valid_heuristics_for_sort) > 5
                    and valid_heuristics_for_sort[-1]["novelty_score"] < 0.2
                ):
                    self._mutate_heuristic(valid_heuristics_for_sort[-1])
            logger.debug(f"Heuristic Forest: Updated {updated_heuristics} heuristics.")
        except Exception as e:
            logger.exception(f"Error in _hypermutable_heuristic_forest: {e}")

    def _cross_modal_analogy_bloom(self) -> None:
        logger.debug("Running Cross-Modal Analogy Bloom...")
        try:
            text_data_raw = self.state_provider.get_data_by_modality("text", limit=10)
            image_embeddings_raw = self.state_provider.get_data_by_modality(
                "image_embedding", limit=10
            )

            # Ensure text_data_raw contains strings for embedding
            text_data = [str(item) for item in text_data_raw if item is not None]
            # Ensure image_embeddings_raw contains lists of floats (or np.ndarrays)
            image_embeddings = [
                item
                for item in image_embeddings_raw
                if isinstance(item, (list, np.ndarray))
            ]

            if not text_data or not image_embeddings:
                return logger.debug("Not enough valid modal data for analogies.")

            # Convert numpy arrays to lists to ensure type compatibility
            image_embeddings_as_lists = [
                item.tolist() if isinstance(item, np.ndarray) else item
                for item in image_embeddings
            ]

            text_embeddings = [self.model_mgr.get_embedding(t) for t in text_data]
            analogies = self._find_analogies(
                text_data, text_embeddings, image_embeddings_as_lists
            )
            logger.debug(
                f"Analogy Bloom: Found {analogies} potential cross-modal analogies."
            )
        except Exception as e:
            logger.exception(f"Error in _cross_modal_analogy_bloom: {e}")

    def _analyze_phase(self) -> None:
        self.current_phase = "Analyze"
        logger.info(f"Entering {self.current_phase} Phase.")
        try:
            self._contradiction_topology_mapper()
            self._self_reflective_uncertainty_quantizer()
        except Exception as e:
            logger.exception(f"Error in {self.current_phase} Phase: {e}")
        finally:
            self.current_phase = None
            logger.info(
                f"Finished {self.current_phase if self.current_phase else 'Analyze'} Phase."
            )

    def _contradiction_topology_mapper(self) -> None:
        logger.debug("Running Contradiction Topology Mapper...")
        try:
            active_beliefs = self.beliefs.get_active_beliefs()
            if len(active_beliefs) < 2:
                return logger.debug("Not enough beliefs for contradiction check.")
            contradictions = self._check_belief_contradictions(active_beliefs)
            logger.debug(
                f"Contradiction Mapper: Found {contradictions} potential contradictions."
            )
        except Exception as e:
            logger.exception(f"Error in _contradiction_topology_mapper: {e}")

    def _self_reflective_uncertainty_quantizer(self) -> None:
        logger.debug("Running Self-Reflective Uncertainty Quantizer...")
        try:
            active_beliefs = self.beliefs.get_active_beliefs()
            if not active_beliefs:
                return logger.debug("No active beliefs to quantify uncertainty.")
            updates = self._update_belief_confidence_levels(active_beliefs)
            logger.debug(
                f"Uncertainty Quantizer: Updated confidence for {updates} beliefs."
            )
        except Exception as e:
            logger.exception(f"Error in _self_reflective_uncertainty_quantizer: {e}")

    def _test_phase(self) -> None:
        self.current_phase = "Test"
        logger.info(f"Entering {self.current_phase} Phase.")
        try:
            self._counterfactual_sandbox_loops()
            self._zero_shot_causal_perturbation_matrix()
        except Exception as e:
            logger.exception(f"Error in {self.current_phase} Phase: {e}")
        finally:
            self.current_phase = None
            logger.info(
                f"Finished {self.current_phase if self.current_phase else 'Test'} Phase."
            )

    def _counterfactual_sandbox_loops(self) -> None:
        logger.debug("Running Counterfactual Sandbox Loops...")
        try:
            active_beliefs = self.beliefs.get_active_beliefs()
            beliefs_to_test = self._select_beliefs_for_testing(active_beliefs)
            if not beliefs_to_test:
                return logger.debug("No beliefs for counterfactual testing.")
            sims_run = self._run_belief_simulations(beliefs_to_test)
            logger.debug(f"Counterfactual Sandbox: Ran {sims_run} simulations.")
        except Exception as e:
            logger.exception(f"Error in _counterfactual_sandbox_loops: {e}")

    def _zero_shot_causal_perturbation_matrix(self) -> None:
        logger.debug("Running Zero-Shot Causal Perturbation Matrix...")
        try:
            current_state = self.state_provider.get_current_state()
            perturb_vars = [
                k for k, v in current_state.items() if isinstance(v, (int, float))
            ]
            if not perturb_vars:
                return logger.debug("No numeric vars to perturb in current state.")
            tests_done = self._test_perturbations(perturb_vars, current_state)
            logger.debug(f"Causal Perturbation: Tested {tests_done} perturbations.")
        except Exception as e:
            logger.exception(f"Error in _zero_shot_causal_perturbation_matrix: {e}")

    # --- Handler for async-bus-driven classification and pattern analysis
    def handle_classification_request(self, message: AsyncMessage) -> None:
        """Handle classification requests from async bus and respond."""
        logger.info(f"CATEngine: Received classification request: {message.content}")
        result = {
            "request": message.content,
            "classification": "default",
            "timestamp": time.time(),
        }
        # Send result back via processing response
        asyncio.create_task(
            self.vanta_core.async_bus.publish(
                AsyncMessage(
                    MessageType.PROCESSING_RESPONSE,
                    self.COMPONENT_NAME,
                    result,
                    target_ids=[message.sender_id],
                )
            )
        )

    def handle_pattern_analysis(self, message: AsyncMessage) -> None:
        """Handle pattern analysis requests from async bus and respond."""
        logger.info(f"CATEngine: Received pattern analysis request: {message.content}")
        analysis = {"analysis": "default", "timestamp": time.time()}
        asyncio.create_task(
            self.vanta_core.async_bus.publish(
                AsyncMessage(
                    MessageType.PATTERN_ANALYSIS,
                    self.COMPONENT_NAME,
                    analysis,
                    target_ids=[message.sender_id],
                )
            )
        )

    # --- Helper Methods (using instance components) ---
    def _get_heuristics(self) -> list[dict[str, Any]]:
        return self.learner.get_heuristics()

    def _get_recent_memories(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.memory.get_recent_memories(limit=limit)

    def _check_heuristic_matches(
        self, h: dict[str, Any], mems: list[dict[str, Any]]
    ) -> int:
        # (Implementation from previous refactor is fine)
        matches = 0
        rule = h.get("rule", "").lower()
        if not rule:
            return 0
        for mem_item in mems:
            if not isinstance(mem_item, dict):
                continue
            content_to_check = str(
                mem_item.get("content", mem_item.get("text", mem_item.get("data", "")))
            ).lower()
            if rule in content_to_check:
                matches += 1
        return matches

    def _calculate_novelty_score(self, h: dict[str, Any], matches: int) -> float:
        # (Implementation from previous refactor is fine)
        score = h.get("novelty_score", 0.5)
        return min(1.0, score + 0.1) if matches > 0 else max(0.0, score - 0.05)

    def _update_heuristic(self, h_id: str, updates: dict[str, Any]):
        self.learner.update_heuristic(h_id, updates)

    def _mutate_heuristic(self, h: dict[str, Any]):
        # (Implementation from previous refactor is fine)
        if not isinstance(h, dict):
            return logger.warning("Cannot mutate non-dict heuristic.")
        mutated = h.copy()
        mutated["id"] = f"mutated_{h.get('id', uuid.uuid4().hex[:6])}"
        mutated["rule"] = (
            f"Mutated rule based on {h.get('id', 'original')}: {random.choice(['variation_A', 'variation_B'])}"
        )
        mutated["novelty_score"] = 0.5
        mutated["last_updated"] = time.time()
        self.learner.add_heuristic(mutated)
        logger.debug(f"Mutated heuristic, new ID: {mutated['id']}")

    def _get_modal_data(self, modality: str, limit: int = 10):
        return self.state_provider.get_data_by_modality(modality, limit)

    def _get_embedding(self, text: str) -> list[float]:
        return self.model_mgr.get_embedding(text)

    def _find_analogies(
        self,
        text_list: list[str],
        txt_embs: list[list[float]],
        img_embs: list[list[float]],
    ):
        # (Implementation from previous refactor is fine)
        count = 0
        threshold = 0.7
        for i, t_emb in enumerate(txt_embs):
            if i >= len(text_list):
                continue
            text_content = text_list[i]
            for j, i_emb in enumerate(img_embs):
                if len(t_emb) == len(i_emb) and len(t_emb) > 0:  # Add len > 0 check
                    sim = self._calculate_cosine_similarity(t_emb, i_emb)
                    if sim > threshold:
                        self._store_analogy(text_content, f"img_emb_ref_{j}", sim)
                        count += 1
        return count

    def _calculate_cosine_similarity(self, v1: list[float], v2: list[float]):
        # (Implementation from previous refactor is fine)
        if not v1 or not v2:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        nv1 = math.sqrt(sum(a * a for a in v1))
        nv2 = math.sqrt(sum(b * b for b in v2))
        return dot / (nv1 * nv2) if nv1 > 0 and nv2 > 0 else 0.0

    def _store_analogy(self, src_data: str, target_ref: str, sim: float):
        # (Implementation from previous refactor is fine)
        self.memory.store_event(
            {
                "type": "cross_modal_analogy",
                "src": src_data,
                "tgt": target_ref,
                "sim": sim,
            },
            "analogy",
        )
        self.log_memory_access(
            "store_analogy",
            {"source_len": len(src_data), "target_ref": target_ref, "similarity": sim},
        )

    def _get_active_beliefs(self) -> list[dict[str, Any]]:
        return self.beliefs.get_active_beliefs()

    def _check_belief_contradictions(self, belief_list: list[dict[str, Any]]) -> int:
        # (Implementation from previous refactor is fine)
        count = 0
        stmts = {
            b.get("id", f"bid_{i}"): b.get("statement", "").lower()
            for i, b in enumerate(belief_list)
            if isinstance(b, dict)
        }
        bids = list(stmts.keys())
        for i in range(len(bids)):
            for j in range(i + 1, len(bids)):
                id1, id2 = bids[i], bids[j]
                s1, s2 = stmts[id1], stmts[id2]
                if self._statements_contradict(s1, s2):
                    self._record_contradiction(id1, id2, s1, s2)
                    count += 1
        return count

    def _statements_contradict(self, s1: str, s2: str) -> bool:
        # (Implementation from previous refactor is fine)
        if not s1 or not s2:
            return False
        s1_lower, s2_lower = s1.lower(), s2.lower()  # ensure lowercase for comparison
        if f"not {s1_lower}" == s2_lower or f"not {s2_lower}" == s1_lower:
            return True
        if (
            s1_lower.startswith("it is true that ")
            and s2_lower.startswith("it is false that ")
            and s1_lower[15:] == s2_lower[16:]
        ):
            return True
        if (
            s2_lower.startswith("it is true that ")
            and s1_lower.startswith("it is false that ")
            and s2_lower[15:] == s1_lower[16:]
        ):
            return True
        return False

    def _record_contradiction(self, id1: str, id2: str, s1: str, s2: str):
        self.beliefs.record_contradiction(id1, id2, "direct_negation_heuristic")

    def _update_belief_confidence_levels(
        self, belief_list: list[dict[str, Any]]
    ) -> int:
        # (Implementation from previous refactor is fine)
        updates = 0
        for belief in belief_list:
            if not isinstance(belief, dict) or "id" not in belief:
                continue
            b_id = belief["id"]
            cur_conf = belief.get("confidence", 0.5)
            ev = belief.get("evidence_ids", [])
            chg = self._calculate_confidence_change(belief, ev)
            new_c = max(0.01, min(0.99, cur_conf + chg))
            if abs(new_c - cur_conf) > 0.01:
                self.beliefs.update_belief_confidence(b_id, new_c)
                updates += 1
        return updates

    def _calculate_confidence_change(self, b: dict[str, Any], e_ids: list) -> float:
        return (len(e_ids) * 0.05 if e_ids else -0.02) + (
            -0.2 if self._belief_is_contradicted(b) else 0.0
        )

    def _belief_is_contradicted(self, b: dict[str, Any]) -> bool:
        if not isinstance(b, dict) or "id" not in b:
            return False
        b_id = b["id"]
        contradictions = []
        if hasattr(self.beliefs, "get_contradictions_for"):
            contradictions = self.beliefs.get_contradictions_for(b_id)
        return len(contradictions) > 0

    def _update_belief_confidence(self, bid: str, nc: float):
        self.beliefs.update_belief_confidence(bid, nc)

    def _select_beliefs_for_testing(self, bl: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            [b for b in bl if isinstance(b, dict)],
            key=lambda x: x.get("confidence", 0.5),
        )[:3]

    def _run_belief_simulations(self, btt: list[dict[str, Any]]) -> int:
        # (Implementation from previous refactor is fine)
        run_count = 0
        for belief in btt:
            if not isinstance(belief, dict) or "id" not in belief:
                continue
            scenario = self._create_belief_scenario(belief)
            try:
                result = self.model_mgr.run_simulation(scenario)
                run_count += 1
                self._update_belief_from_simulation(belief, result)
            except Exception as e:
                logger.error(f"Error simulating belief {belief.get('id', 'N/A')}: {e}")
        return run_count

    def _create_belief_scenario(self, b: dict[str, Any]) -> dict[str, Any]:
        return {
            "base_state": self.state_provider.get_current_state(),
            "intervention": f"Assume '{b.get('statement', '')}' true",
            "belief_id": b.get("id"),
        }

    # _run_simulation -> model_mgr.run_simulation
    def _update_belief_from_simulation(self, b: dict[str, Any], r: dict[str, Any]):
        # (Implementation from previous refactor is fine)
        sim_c = r.get("confidence", 0.5)
        cur_c = b.get("confidence", 0.5)
        new_c = (cur_c + sim_c) / 2
        self.beliefs.update_belief_confidence(
            b.get("id", "unknown_id_in_update_belief"), new_c
        )

    def _get_current_state(self) -> dict[str, Any]:
        return self.state_provider.get_current_state()

    def _test_perturbations(self, p_vars: list[str], c_state: dict[str, Any]):
        # (Implementation from previous refactor is fine)
        td = 0
        for var_n in random.sample(p_vars, min(len(p_vars), 5)):
            try:
                ov = c_state[var_n]
                pa = ov * (random.random() * 0.1 - 0.05)
                pv = ov + pa
                pd = {"var": var_n, "orig_v": ov, "pert_v": pv}
                impact = self.model_mgr.evaluate_causal_impact(pd)
                td += 1
                if impact > 0.5:
                    self._record_causal_link(var_n, impact)
            except Exception as e:
                logger.error(f"Perturbation test error on {var_n}: {e}")
        return td

    # _evaluate_perturbation -> model_mgr.evaluate_causal_impact
    def _record_causal_link(self, v_name: str, imp: float):
        # (Implementation from previous refactor is fine)
        self.memory.store_event(
            {"type": "causal_link", "cause": v_name, "eff": "sys_out_sens", "str": imp},
            "causal_discovery",
        )
        self.log_memory_access("record_causal_link", {"cause": v_name, "impact": imp})
        self.beliefs.add_belief(
            f"Var '{v_name}' impacts system (strength {imp:.2f})", imp
        )

    def _build_test_prompt(self) -> str:  # Simplified using instance components
        # (Implementation from previous refactor is fine)
        state_str = json.dumps(
            self.state_provider.get_current_state(),
            default=str,
            indent=None,
            separators=(",", ":"),
        )
        mem_str = "\n".join(
            f"- {str(m.get('d', m)[:50])}"
            for m in self.memory.get_recent_memories(limit=3)
        )
        bel_str = "\n".join(
            f"- {b.get('statement', 'N/A')[:50]} ({b.get('confidence', 0):.2f})"
            for b in self.beliefs.get_active_beliefs()[:3]
        )
        return f"Analyze Context:\nFocus: {self.focus_manager.get_current_focus()}\nState: {state_str[:200]}\nMemories:\n{mem_str}\nBeliefs:\n{bel_str}\nAnalyze:"

    def _get_current_focus(self) -> str:
        return self.focus_manager.get_current_focus()

    def _get_active_tasks(self) -> list[str]:
        return self.focus_manager.get_active_tasks()

    def _parse_response(self, r_str: str) -> dict[str, Any]:
        # (Implementation from previous refactor is fine)
        try:
            js_s = r_str.find("{")
            js_e = r_str.rfind("}")
            return (
                json.loads(r_str[js_s : js_e + 1])
                if js_s != -1 and js_e != -1
                else {"raw": r_str}
            )
        except Exception:
            return {"raw": r_str, "error": "parse_failed"}

    def log_memory_access(
        self, action: str, details: dict[str, Any] | None = None
    ) -> None:
        """Logs memory access using the injected EchoMemory and VantaCore event bus."""
        final_details = details or {}
        self.echo_memory_instance.record_cognitive_trace(
            self.COMPONENT_NAME, action, final_details
        )
        self.vanta_core.publish_event(
            event_type=f"{self.COMPONENT_NAME}.memory_access",
            data={
                "action": action,
                "details": final_details,
                "phase": self.current_phase,
            },
            source=self.COMPONENT_NAME,
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "engine_name": self.COMPONENT_NAME,
            "running": self.running,
            "current_phase": self.current_phase,
            "interval_s": self.config.interval_s,
            "last_error": self.last_error,
            "thread_alive": self.thread.is_alive() if self.thread else False,
            "health_assessment": "healthy_operational"
            if not self.last_error
            else "state_error_detected",
        }

    def diagnose(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        self.current_phase = "Diagnose"
        self.log_memory_access(
            "diagnose_cycle_start", {"context_provided": context is not None}
        )

        health = self._calculate_health_metrics()
        issues = self._identify_issues(health)
        recs = self._generate_recommendations(issues)

        active_b_list = self.beliefs.get_active_beliefs()
        b_count = len(active_b_list)
        avg_c = (
            sum(b.get("confidence", 0.5) for b in active_b_list) / max(1, b_count)
            if active_b_list
            else 0.0
        )

        diag = {
            "timestamp": time.time(),
            "engine": self.COMPONENT_NAME,
            "overall_health_score": health.get("overall", 0.5),
            "health_categories": health,
            "current_phase": self.current_phase,
            "identified_issues": issues,
            "suggested_actions": recs,
            "is_running": self.running,
            "active_belief_count": b_count,
            "average_belief_confidence": avg_c,
        }
        self.log_memory_access(
            "diagnose_cycle_complete", {"overall_score": diag["overall_health_score"]}
        )
        self.current_phase = None
        self.vanta_core.publish_event(
            f"{self.COMPONENT_NAME}.diagnosis_results", diag, source=self.COMPONENT_NAME
        )
        logger.info(
            f"CAT Engine Diagnosis: Score={diag['overall_health_score']:.2f}, Issues={len(issues)}"
        )
        return diag

    def _calculate_health_metrics(
        self,
    ) -> dict[str, float]:  # Stub, VantaCore specific health calculation
        # This should potentially use metrics from VantaCore if available, or internal CATEngine state
        ph = 0.75  # phase health
        if (
            self.current_phase
            and "error" in self.current_phase.lower()
            or self.last_error
        ):
            ph = 0.25
        elif not self.running:
            ph = 0.1

        belief_health = 0.5
        try:
            active_b_list = self.beliefs.get_active_beliefs()
            if active_b_list:
                avg_conf = sum(b.get("confidence", 0.5) for b in active_b_list) / max(
                    1, len(active_b_list)
                )
                belief_health = avg_conf * (
                    min(len(active_b_list), 10) / 10
                )  # penalize very few beliefs
        except Exception as e:
            logger.warning(f"Error calculating belief health: {e}")
            pass  # Keep default if error

        return {
            "overall": (ph * 0.6 + belief_health * 0.4),
            "phase_execution_health": ph,
            "belief_system_health": belief_health,
            "resource_utilization": 0.7,
            "data_flow_integrity": 0.8,
        }  # Example metrics

    def _identify_issues(
        self, health: dict[str, float]
    ) -> list[str]:  # Stub for VantaCore
        issues = []
        if not self.running:
            issues.append("Engine is not running.")
        if self.last_error:
            issues.append(f"Last error recorded: {self.last_error}")
        if health.get("overall", 1.0) < 0.5:
            issues.append("Overall health score is low.")
        if health.get("belief_system_health", 1.0) < 0.4:
            issues.append("Belief system confidence or diversity is low.")
        return issues

    def _generate_recommendations(
        self, issues: list[str]
    ) -> list[str]:  # Stub for VantaCore
        recs = []
        if not issues:
            return ["Engine appears healthy. Continue monitoring."]
        if "Engine is not running." in issues:
            recs.append("Restart the CATEngine.")
        if any("Last error" in i for i in issues):
            recs.append("Review logs for specific error details and causes.")
        if any("Overall health" in i for i in issues):
            recs.append("Investigate sub-component health and data flow.")
        if any("Belief system" in i for i in issues):
            recs.append(
                "Consider injecting new information or running hypothesis generation cycles."
            )
        return recs if recs else ["Address identified issues based on logs."]


# --- Example Usage (Adapted for VantaCore) ---
if __name__ == "__main__":
    # This example assumes vanta_core.py is in the same directory or python path
    # and provides VantaCore() instance.

    # 1. Initialize VantaCore (singleton)
    vanta_system = VantaCore()
    main_logger = vanta_system.logger  # Use VantaCore's logger for example
    main_logger.setLevel(logging.DEBUG)  # Set VantaCore global logger level

    main_logger.info("--- Starting C.A.T. Engine VantaCore Example ---")

    # 2. Create CATEngineConfig
    cat_config = CATEngineConfig(
        interval_s=5, log_level="DEBUG"
    )  # Short interval for demo

    # 3. Instantiate CATEngine.
    #    It will use the vanta_system instance and its own default components if specific
    #    providers are not passed here and not found in vanta_system.registry.
    cat_engine_instance = CATEngine(vanta_core=vanta_system, config=cat_config)

    # Example: Register a custom belief registry with VantaCore if needed by other components
    # my_beliefs = DefaultVantaBeliefRegistry()
    # vanta_system.register_component("shared_belief_registry", my_beliefs)
    # cat_engine_instance_shared = CATEngine(vanta_core=vanta_system, config=cat_config, belief_registry_provider=my_beliefs)

    # 4. Start the engine
    cat_engine_instance.start()

    try:
        main_logger.info(
            "C.A.T. Engine running. Test interval is short. Ctrl+C to stop."
        )
        for i in range(3):  # Let it run for a few cycles
            time.sleep(
                cat_config.interval_s + 1
            )  # Sleep a bit longer than interval to ensure cycle completes
            status_report = cat_engine_instance.get_status()
            main_logger.info(
                f"STATUS Cycle ~{i + 1}: Phase='{status_report.get('current_phase')}', Health='{status_report.get('health_assessment')}'"
            )
            if not cat_engine_instance.running:
                main_logger.info("Engine stopped unexpectedly.")
                break

        diag_results = cat_engine_instance.diagnose(
            {"trigger_event": "manual_test_end_of_run"}
        )
        main_logger.info(
            f"Manual Diagnosis: Score={diag_results.get('overall_health_score', 0.0):.2f}, Issues: {diag_results.get('identified_issues')}"
        )

    except KeyboardInterrupt:
        main_logger.info("Keyboard interrupt received by main.")
    finally:
        main_logger.info("Attempting to stop CAT Engine and VantaCore.")
        cat_engine_instance.stop()
        # VantaCore shutdown might involve stopping other components or threads if it had them.
        # vanta_system.shutdown() # Add if VantaCore gets a shutdown method
        main_logger.info("--- C.A.T. Engine VantaCore Example Finished ---")
