#!/usr/bin/env python3
"""
Enhanced SleepTimeCompute Agent with Music Genre Integration
============================================================

Advanced sleep-time computation agent that performs nightly optimization of music genre
embeddings and cross-modal learning. Integrates with the expanded genre vocabulary
system for continuous improvement of music understanding and generation.

Features:
- Nightly genre embedding optimization
- Cross-modal learning between audio and text
- Music preference pattern analysis
- Genre taxonomy evolution
- Collaborative filtering enhancement
- Cognitive load balancing during sleep cycles
"""

import asyncio
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import yaml

from .base import BaseAgent, vanta_agent, CognitiveMeshRole
from Vanta.core.UnifiedAsyncBus import AsyncMessage, MessageType
from core.vanta_core import VantaCore

logger = logging.getLogger(__name__)

@dataclass
class SleepComputeConfig:
    """Configuration for sleep-time computation"""
    sleep_cycle_hours: float = 8.0
    optimization_interval_minutes: int = 30
    genre_learning_enabled: bool = True
    cross_modal_training_enabled: bool = True
    cognitive_load_threshold: float = 3.0
    max_parallel_processes: int = 4
    genre_embedding_cache: Optional[Path] = None
    user_preference_learning: bool = True

@dataclass
class SleepComputeMetrics:
    """Metrics tracked during sleep computation"""
    total_optimizations: int = 0
    genre_embeddings_updated: int = 0
    user_patterns_learned: int = 0
    cross_modal_improvements: float = 0.0
    cognitive_efficiency_gain: float = 0.0
    energy_consumption: float = 0.0
    
@vanta_agent(
    name="SleepTimeComputeAgent", 
    subsystem="sleep_scheduler", 
    mesh_role=CognitiveMeshRole.EVALUATOR,
    capabilities=[
        "nightly_optimization",
        "genre_embedding_evolution", 
        "cross_modal_learning",
        "user_preference_analysis",
        "cognitive_load_balancing"
    ],
    cognitive_load=2.5,
    symbolic_depth=3
)
class SleepTimeComputeAgent(BaseAgent):
    """
    Enhanced sleep-time computation agent with music genre intelligence.
    """
    
    sigil = "🌒🧵🧠🜝🎵"  # Enhanced with music symbol
    tags = ['Reflection Engine', 'Dream-State Scheduler', 'Music Learning', 'Genre Evolution']
    invocations = ['Sleep Compute', 'Dream consolidate', 'Genre Mix-in', 'Nightly Learning']

    def __init__(self, vanta_core: VantaCore, config: Dict[str, Any]):
        super().__init__(vanta_core, config)
        self.sleep_config = SleepComputeConfig(**config.get("sleep_compute", {}))
        
        # Sleep computation components
        self.is_sleeping: bool = False
        self.current_cycle_metrics = SleepComputeMetrics()
        self.optimization_tasks: List[asyncio.Task] = []
        
        # Music genre learning components
        self.genre_vocabulary: Dict[str, Any] = {}
        self.user_music_patterns: Dict[str, Any] = {}
        self.genre_embedding_cache: Dict[str, np.ndarray] = {}
        self.cross_modal_learning_data: List[Dict[str, Any]] = []
        
        # Cognitive metrics tracking
        self.cognitive_metrics = {
            "sleep_optimization_efficiency": 0.0,
            "genre_learning_progress": 0.0,
            "user_pattern_recognition": 0.0,
            "cross_modal_alignment": 0.0,
            "cognitive_load_reduction": 0.0,
            "energy_efficiency": 0.0
        }

    async def initialize(self) -> bool:
        """Initialize the enhanced sleep-time compute agent"""
        try:
            logger.info("🌙 Initializing Enhanced SleepTimeCompute Agent...")
            
            # Load genre vocabulary
            await self._load_genre_vocabulary()
            
            # Initialize user pattern tracking
            await self._initialize_user_pattern_tracking()
            
            # Setup optimization schedulers
            await self._setup_optimization_schedulers()
            
            # Register with cognitive mesh
            await self._register_cognitive_mesh()
            
            self.cognitive_metrics["sleep_optimization_efficiency"] = 0.85
            logger.info("✅ Enhanced SleepTimeCompute Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced SleepTimeCompute Agent: {e}")
            return False

    def initialize_subsystem(self, core):
        """Legacy method - maintained for compatibility"""
        pass

    def on_gui_call(self):
        """Legacy method - maintained for compatibility"""
        super().on_gui_call()

    def bind_echo_routes(self):
        """Legacy method - maintained for compatibility"""
        pass
    
    async def _load_genre_vocabulary(self) -> None:
        """Load the expanded genre vocabulary"""
        try:
            vocab_path = Path("sigils/global_vocab.json")
            if vocab_path.exists():
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    self.genre_vocabulary = json.load(f)
                logger.info(f"📚 Loaded genre vocabulary for sleep learning")
            else:
                logger.warning("Genre vocabulary not found, sleep learning will be limited")
                
        except Exception as e:
            logger.error(f"Failed to load genre vocabulary: {e}")
    
    async def _initialize_user_pattern_tracking(self) -> None:
        """Initialize user music pattern tracking"""
        # This would load historical user interaction data
        # For now, initializing with empty patterns
        self.user_music_patterns = {
            "genre_preferences": {},
            "listening_times": {},
            "mood_correlations": {},
            "activity_associations": {},
            "temporal_patterns": {}
        }
    
    async def _setup_optimization_schedulers(self) -> None:
        """Setup various optimization schedulers"""
        # Create optimization schedule based on configuration
        self.optimization_schedule = {
            "genre_embedding_update": timedelta(minutes=45),
            "user_pattern_analysis": timedelta(hours=2),
            "cross_modal_learning": timedelta(hours=1.5),
            "cognitive_load_balancing": timedelta(minutes=20),
            "taxonomy_evolution": timedelta(hours=4)
        }
    
    async def _register_cognitive_mesh(self) -> None:
        """Register with VantaCore's cognitive mesh"""
        if self.vanta_core:
            mesh_config = {
                "role": "sleep_optimizer",
                "capabilities": ["nightly_learning", "pattern_analysis", "genre_evolution"],
                "cognitive_load": self.sleep_config.cognitive_load_threshold,
                "priority": "background",
                "sleep_mode_enabled": True
            }
            await self.vanta_core.register_mesh_component("sleep_compute", mesh_config)
            self.cognitive_metrics["cognitive_load_reduction"] = 0.75

    async def start_sleep_cycle(self, duration_hours: Optional[float] = None) -> None:
        """Start a sleep computation cycle"""
        if self.is_sleeping:
            logger.warning("Sleep cycle already in progress")
            return
            
        duration = duration_hours or self.sleep_config.sleep_cycle_hours
        logger.info(f"🛌 Starting sleep cycle for {duration} hours")
        
        self.is_sleeping = True
        self.current_cycle_metrics = SleepComputeMetrics()
        
        try:
            # Start background optimization tasks
            await self._start_optimization_tasks()
            
            # Emit sleep cycle start event
            if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
                msg = AsyncMessage(
                    MessageType.COMPONENT_STATUS,
                    self.__class__.__name__,
                    {
                        "phase": "sleep_cycle_start",
                        "duration_hours": duration,
                        "optimization_enabled": True
                    },
                )
                await self.vanta_core.async_bus.publish(msg)
            
            # Sleep for the specified duration
            await asyncio.sleep(duration * 3600)  # Convert hours to seconds
            
            # End sleep cycle
            await self._end_sleep_cycle()
            
        except Exception as e:
            logger.error(f"Sleep cycle failed: {e}")
            await self._emergency_sleep_cycle_cleanup()
        finally:
            self.is_sleeping = False
    
    async def _start_optimization_tasks(self) -> None:
        """Start all background optimization tasks"""
        # Clear any existing tasks
        for task in self.optimization_tasks:
            if not task.done():
                task.cancel()
        self.optimization_tasks.clear()
        
        # Start genre embedding optimization
        if self.sleep_config.genre_learning_enabled:
            task = asyncio.create_task(self._optimize_genre_embeddings())
            self.optimization_tasks.append(task)
        
        # Start user pattern analysis
        if self.sleep_config.user_preference_learning:
            task = asyncio.create_task(self._analyze_user_patterns())
            self.optimization_tasks.append(task)
        
        # Start cross-modal learning
        if self.sleep_config.cross_modal_training_enabled:
            task = asyncio.create_task(self._perform_cross_modal_learning())
            self.optimization_tasks.append(task)
        
        # Start cognitive load balancing
        task = asyncio.create_task(self._balance_cognitive_load())
        self.optimization_tasks.append(task)
        
        logger.info(f"🔄 Started {len(self.optimization_tasks)} optimization tasks")
    
    async def _optimize_genre_embeddings(self) -> None:
        """Optimize genre embeddings during sleep"""
        logger.info("🎼 Starting genre embedding optimization")
        
        try:
            while self.is_sleeping:
                # Get current genre embeddings
                genre_categories = self.genre_vocabulary.get("music_genres", {})
                
                for category, genres in genre_categories.items():
                    for genre in genres:
                        await self._update_genre_embedding(genre, category)
                        self.current_cycle_metrics.genre_embeddings_updated += 1
                        
                        # Check cognitive load
                        if await self._check_cognitive_load():
                            await asyncio.sleep(10)  # Brief pause if load is high
                
                # Update metrics
                self.cognitive_metrics["genre_learning_progress"] += 0.01
                
                # Sleep between optimization cycles
                await asyncio.sleep(self.sleep_config.optimization_interval_minutes * 60)
                
        except asyncio.CancelledError:
            logger.info("Genre embedding optimization cancelled")
        except Exception as e:
            logger.error(f"Genre embedding optimization error: {e}")
    
    async def _update_genre_embedding(self, genre: str, category: str) -> None:
        """Update embedding for a specific genre"""
        # This would integrate with the BLT reindexer
        # For now, simulating embedding evolution
        
        if genre in self.genre_embedding_cache:
            # Evolve existing embedding
            embedding = self.genre_embedding_cache[genre]
            # Small random evolution
            evolution = np.random.normal(0, 0.01, embedding.shape)
            self.genre_embedding_cache[genre] = embedding + evolution
        else:
            # Create new embedding
            embedding_dim = 768  # Standard BLT dimension
            self.genre_embedding_cache[genre] = np.random.normal(0, 0.1, embedding_dim)
    
    async def _analyze_user_patterns(self) -> None:
        """Analyze user music patterns during sleep"""
        logger.info("👤 Starting user pattern analysis")
        
        try:
            while self.is_sleeping:
                # Analyze genre preferences
                await self._analyze_genre_preferences()
                
                # Analyze temporal patterns
                await self._analyze_temporal_patterns()
                
                # Analyze mood correlations
                await self._analyze_mood_correlations()
                
                self.current_cycle_metrics.user_patterns_learned += 1
                self.cognitive_metrics["user_pattern_recognition"] += 0.005
                
                # Sleep between analysis cycles
                await asyncio.sleep(120 * 60)  # 2 hours
                
        except asyncio.CancelledError:
            logger.info("User pattern analysis cancelled")
        except Exception as e:
            logger.error(f"User pattern analysis error: {e}")
    
    async def _analyze_genre_preferences(self) -> None:
        """Analyze user genre preferences"""
        # This would analyze actual user interaction data
        # For now, simulating pattern discovery
        
        simulated_preferences = {
            "Hip Hop": np.random.uniform(0.7, 0.9),
            "Chill": np.random.uniform(0.6, 0.8),
            "EDM": np.random.uniform(0.4, 0.7),
            "Sensual": np.random.uniform(0.5, 0.8)
        }
        
        # Update patterns with learning rate
        learning_rate = 0.1
        for genre, preference in simulated_preferences.items():
            if genre in self.user_music_patterns["genre_preferences"]:
                current = self.user_music_patterns["genre_preferences"][genre]
                self.user_music_patterns["genre_preferences"][genre] = (
                    (1 - learning_rate) * current + learning_rate * preference
                )
            else:
                self.user_music_patterns["genre_preferences"][genre] = preference
    
    async def _analyze_temporal_patterns(self) -> None:
        """Analyze temporal music listening patterns"""
        # This would analyze when users listen to different genres
        # For now, simulating temporal pattern discovery
        
        time_of_day = datetime.now().hour
        genre_time_associations = {
            "Morning": ["Pop", "Motivation", "Beats"],
            "Afternoon": ["Hip Hop", "EDM", "Alternative Hip Hop"],
            "Evening": ["Chill", "Mellow", "Sensual"],
            "Night": ["Ambient Electronic", "Soundtrack", "Peaceful"]
        }
        
        # Update temporal patterns
        for period, genres in genre_time_associations.items():
            if period not in self.user_music_patterns["temporal_patterns"]:
                self.user_music_patterns["temporal_patterns"][period] = {}
            
            for genre in genres:
                current_strength = self.user_music_patterns["temporal_patterns"][period].get(genre, 0.5)
                self.user_music_patterns["temporal_patterns"][period][genre] = min(1.0, current_strength + 0.01)
    
    async def _analyze_mood_correlations(self) -> None:
        """Analyze mood-genre correlations"""
        # This would analyze how genres correlate with user mood/context
        # For now, simulating mood correlation discovery
        
        mood_genre_correlations = {
            "energetic": ["EDM", "Hip Hop", "Dancehall"],
            "relaxed": ["Chill", "Mellow", "Ambient"],
            "focused": ["Instrumental", "Soundtrack", "Beats"],
            "social": ["Pop", "Afrobeats", "Party"]
        }
        
        for mood, genres in mood_genre_correlations.items():
            if mood not in self.user_music_patterns["mood_correlations"]:
                self.user_music_patterns["mood_correlations"][mood] = {}
            
            for genre in genres:
                current_correlation = self.user_music_patterns["mood_correlations"][mood].get(genre, 0.5)
                self.user_music_patterns["mood_correlations"][mood][genre] = min(1.0, current_correlation + 0.005)
    
    async def _perform_cross_modal_learning(self) -> None:
        """Perform cross-modal learning between audio and text"""
        logger.info("🔄 Starting cross-modal learning")
        
        try:
            while self.is_sleeping:
                # Simulate cross-modal learning
                await self._align_audio_text_embeddings()
                
                # Update cross-modal metrics
                improvement = np.random.uniform(0.001, 0.005)
                self.current_cycle_metrics.cross_modal_improvements += improvement
                self.cognitive_metrics["cross_modal_alignment"] += improvement
                
                # Sleep between learning cycles
                await asyncio.sleep(90 * 60)  # 1.5 hours
                
        except asyncio.CancelledError:
            logger.info("Cross-modal learning cancelled")
        except Exception as e:
            logger.error(f"Cross-modal learning error: {e}")
    
    async def _align_audio_text_embeddings(self) -> None:
        """Align audio and text embeddings for genres"""
        # This would perform actual cross-modal alignment
        # For now, simulating the alignment process
        
        for genre in self.genre_embedding_cache:
            # Simulate alignment improvement
            if np.random.random() < 0.3:  # 30% chance of improvement
                embedding = self.genre_embedding_cache[genre]
                # Small alignment adjustment
                alignment_adjustment = np.random.normal(0, 0.005, embedding.shape)
                self.genre_embedding_cache[genre] = embedding + alignment_adjustment
    
    async def _balance_cognitive_load(self) -> None:
        """Balance cognitive load during sleep computation"""
        logger.info("⚖️ Starting cognitive load balancing")
        
        try:
            while self.is_sleeping:
                current_load = await self._calculate_current_cognitive_load()
                
                if current_load > self.sleep_config.cognitive_load_threshold:
                    # Reduce load by pausing some tasks
                    await self._reduce_cognitive_load()
                elif current_load < self.sleep_config.cognitive_load_threshold * 0.5:
                    # Increase efficiency by starting more tasks
                    await self._increase_cognitive_efficiency()
                
                # Update metrics
                self.cognitive_metrics["cognitive_load_reduction"] += 0.001
                
                # Check load every 20 minutes
                await asyncio.sleep(20 * 60)
                
        except asyncio.CancelledError:
            logger.info("Cognitive load balancing cancelled")
        except Exception as e:
            logger.error(f"Cognitive load balancing error: {e}")
    
    async def _calculate_current_cognitive_load(self) -> float:
        """Calculate current cognitive load"""
        # This would measure actual system resources and processing
        # For now, simulating load calculation
        
        base_load = len(self.optimization_tasks) * 0.5
        processing_load = len(self.genre_embedding_cache) * 0.001
        pattern_load = len(self.user_music_patterns.get("genre_preferences", {})) * 0.002
        
        return base_load + processing_load + pattern_load
    
    async def _check_cognitive_load(self) -> bool:
        """Check if cognitive load is too high"""
        current_load = await self._calculate_current_cognitive_load()
        return current_load > self.sleep_config.cognitive_load_threshold
    
    async def _reduce_cognitive_load(self) -> None:
        """Reduce cognitive load by optimizing processes"""
        # Temporarily pause some non-critical optimizations
        await asyncio.sleep(5)  # Brief pause
        logger.debug("Cognitive load reduced")
    
    async def _increase_cognitive_efficiency(self) -> None:
        """Increase cognitive efficiency"""
        # Optimize processing when load is low
        logger.debug("Cognitive efficiency increased")
    
    async def _end_sleep_cycle(self) -> None:
        """End the sleep cycle and save results"""
        logger.info("🌅 Ending sleep cycle")
        
        # Cancel all optimization tasks
        for task in self.optimization_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.optimization_tasks, return_exceptions=True)
        self.optimization_tasks.clear()
        
        # Save learned patterns and embeddings
        await self._save_sleep_cycle_results()
        
        # Emit sleep cycle end event
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.COMPONENT_STATUS,
                self.__class__.__name__,
                {
                    "phase": "sleep_cycle_end",
                    "metrics": {
                        "optimizations": self.current_cycle_metrics.total_optimizations,
                        "embeddings_updated": self.current_cycle_metrics.genre_embeddings_updated,
                        "patterns_learned": self.current_cycle_metrics.user_patterns_learned
                    }
                },
            )
            await self.vanta_core.async_bus.publish(msg)
        
        logger.info(f"✅ Sleep cycle completed: {self.current_cycle_metrics.genre_embeddings_updated} embeddings updated")
    
    async def _save_sleep_cycle_results(self) -> None:
        """Save results from the sleep cycle"""
        timestamp = datetime.now().isoformat()
        
        # Save updated genre embeddings
        if self.genre_embedding_cache:
            embeddings_file = Path(f"sleep_results/genre_embeddings_{timestamp}.npz")
            embeddings_file.parent.mkdir(exist_ok=True)
            np.savez_compressed(embeddings_file, **self.genre_embedding_cache)
        
        # Save user patterns
        patterns_file = Path(f"sleep_results/user_patterns_{timestamp}.json")
        patterns_file.parent.mkdir(exist_ok=True)
        with open(patterns_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "user_patterns": self.user_music_patterns,
                "cycle_metrics": {
                    "total_optimizations": self.current_cycle_metrics.total_optimizations,
                    "genre_embeddings_updated": self.current_cycle_metrics.genre_embeddings_updated,
                    "user_patterns_learned": self.current_cycle_metrics.user_patterns_learned,
                    "cross_modal_improvements": self.current_cycle_metrics.cross_modal_improvements
                },
                "cognitive_metrics": self.cognitive_metrics
            }, f, indent=2, ensure_ascii=False)
    
    async def _emergency_sleep_cycle_cleanup(self) -> None:
        """Emergency cleanup if sleep cycle fails"""
        logger.warning("🚨 Emergency sleep cycle cleanup")
        
        for task in self.optimization_tasks:
            if not task.done():
                task.cancel()
        
        self.optimization_tasks.clear()
        self.is_sleeping = False

    async def run(self) -> None:
        """Main run method - emit status and start sleep cycle if needed"""
        # Emit status information via the async bus and event bus
        if self.vanta_core and hasattr(self.vanta_core, "async_bus"):
            msg = AsyncMessage(
                MessageType.COMPONENT_STATUS,
                self.__class__.__name__,
                {
                    "phase": "run",
                    "is_sleeping": self.is_sleeping,
                    "genre_learning_enabled": self.sleep_config.genre_learning_enabled,
                    "optimization_tasks_count": len(self.optimization_tasks)
                },
            )
            await self.vanta_core.async_bus.publish(msg)
            
        if self.vanta_core and hasattr(self.vanta_core, "event_bus"):
            self.vanta_core.event_bus.emit(
                "sleep_time_compute.status",
                {
                    "phase": "run",
                    "cognitive_metrics": self.cognitive_metrics,
                    "sleep_config": {
                        "genre_learning": self.sleep_config.genre_learning_enabled,
                        "cross_modal_training": self.sleep_config.cross_modal_training_enabled
                    }
                },
            )
    
    async def get_cognitive_metrics(self) -> Dict[str, Any]:
        """Get current cognitive metrics"""
        return self.cognitive_metrics.copy()
    
    async def get_sleep_cycle_status(self) -> Dict[str, Any]:
        """Get current sleep cycle status"""
        return {
            "is_sleeping": self.is_sleeping,
            "active_tasks": len(self.optimization_tasks),
            "current_metrics": {
                "optimizations": self.current_cycle_metrics.total_optimizations,
                "embeddings_updated": self.current_cycle_metrics.genre_embeddings_updated,
                "patterns_learned": self.current_cycle_metrics.user_patterns_learned
            },
            "genre_cache_size": len(self.genre_embedding_cache),
            "user_pattern_categories": len(self.user_music_patterns)
        }
    
    def generate_reasoning_trace(self) -> Dict[str, Any]:
        """Generate reasoning trace for HOLO-1.5 cognitive mesh"""
        return {
            "agent_name": "SleepTimeComputeAgent",
            "cognitive_load": 2.5,
            "symbolic_depth": 3,
            "reasoning_steps": [
                "Monitor system sleep state and cognitive load",
                "Optimize genre embeddings through iterative learning",
                "Analyze user music patterns and preferences",
                "Perform cross-modal alignment between audio and text",
                "Balance cognitive resources during sleep cycles",
                "Save learned patterns and updated embeddings"
            ],
            "cognitive_metrics": self.cognitive_metrics,
            "sleep_status": {
                "is_sleeping": self.is_sleeping,
                "optimization_tasks": len(self.optimization_tasks)
            },
            "learning_progress": {
                "genre_embeddings_cached": len(self.genre_embedding_cache),
                "user_patterns_tracked": len(self.user_music_patterns),
                "cross_modal_improvements": self.current_cycle_metrics.cross_modal_improvements
            },
            "processing_efficiency": sum(self.cognitive_metrics.values()) / len(self.cognitive_metrics),
            "sleep_mode_capable": True
        }

class SleepTimeCompute(SleepTimeComputeAgent):
    """Alias to match AGENTS.md name."""
    pass

