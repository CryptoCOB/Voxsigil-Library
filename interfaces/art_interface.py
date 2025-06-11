#!/usr/bin/env python
"""
ART Interface for Vanta Core Integration

This module provides the interface between ART (Adaptive Resonance Theory) components
and the Vanta orchestration system, enabling pattern recognition and adaptive learning
within the VoxSigil ecosystem.

Created: 2025-06-02
Purpose: Bridge ART Manager with Vanta Core production systems
"""

import logging
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger("Vanta.Interfaces.ART")


class BaseARTInterface(ABC):
    """Abstract base interface for ART integration with Vanta."""

    @abstractmethod
    def analyze_input(
        self, input_data: Union[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze input data for patterns and categories."""
        pass

    @abstractmethod
    def learn_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """Learn a new pattern or update existing pattern knowledge."""
        pass

    @abstractmethod
    def get_pattern_insights(self, query: str) -> List[Dict[str, Any]]:
        """Get insights about patterns related to the query."""
        pass

    @abstractmethod
    def get_categories(self) -> List[Dict[str, Any]]:
        """Get all known categories from ART analysis."""
        pass


class ProductionARTInterface(BaseARTInterface):
    """Production implementation of ART interface using real ARTManager."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the production ART interface.

        Args:
            config: Optional configuration for ART components
        """
        self.config = config or {}
        self.art_manager = None
        self._initialize_art_manager()

    def _initialize_art_manager(self):
        """Initialize the ARTManager with proper error handling."""
        try:
            from ART.art_manager import ARTManager

            self.art_manager = ARTManager(config=self.config)
            logger.info("✅ ProductionARTInterface initialized with real ARTManager")
        except ImportError as e:
            logger.error(f"❌ Failed to import ARTManager: {e}")
            raise ImportError(
                "ARTManager not available - check ART module installation"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize ARTManager: {e}")
            raise RuntimeError(f"ARTManager initialization failed: {e}")

    def analyze_input(
        self, input_data: Union[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze input data using ARTManager.

        Args:
            input_data: String or dictionary data to analyze

        Returns:
            Analysis results with categories, patterns, and metadata
        """
        if not self.art_manager:
            logger.warning("⚠️ ARTManager not available for analysis")
            return None

        try:
            # Convert string input to format expected by ARTManager
            if isinstance(input_data, str):
                analysis_input = {"content": input_data, "type": "text"}
            else:
                analysis_input = input_data

            result = self.art_manager.analyze_input(analysis_input)

            if result:
                logger.debug(
                    f"✅ ART analysis completed for input type: {type(input_data).__name__}"
                )
                return result
            else:
                logger.warning("⚠️ ART analysis returned no results")
                return None

        except Exception as e:
            logger.error(f"❌ Error during ART analysis: {e}")
            return None

    def learn_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """
        Learn a new pattern using ARTManager.

        Args:
            pattern_data: Dictionary containing pattern information

        Returns:
            True if pattern was learned successfully, False otherwise
        """
        if not self.art_manager:
            logger.warning("⚠️ ARTManager not available for learning")
            return False

        try:
            # Use ARTManager's learning capabilities
            if hasattr(self.art_manager, "learn_pattern"):
                result = self.art_manager.learn_pattern(pattern_data)
                logger.debug(f"✅ Pattern learning result: {result}")
                return bool(result)
            elif hasattr(self.art_manager, "train"):
                # Fallback to training method if available
                result = self.art_manager.train(pattern_data)
                logger.debug(f"✅ Pattern training result: {result}")
                return bool(result)
            else:
                logger.warning("⚠️ ARTManager has no learning method available")
                return False

        except Exception as e:
            logger.error(f"❌ Error during pattern learning: {e}")
            return False

    def get_pattern_insights(self, query: str) -> List[Dict[str, Any]]:
        """
        Get pattern insights related to the query.

        Args:
            query: Query string to find related patterns

        Returns:
            List of pattern insights and related information
        """
        if not self.art_manager:
            logger.warning("⚠️ ARTManager not available for pattern insights")
            return []

        try:
            insights = []

            # Try different methods to get pattern information
            if hasattr(self.art_manager, "get_insights"):
                insights = self.art_manager.get_insights(query)
            elif hasattr(self.art_manager, "query_patterns"):
                insights = self.art_manager.query_patterns(query)
            elif hasattr(self.art_manager, "analyze_input"):
                # Use analysis as fallback for insights
                analysis = self.art_manager.analyze_input(
                    {"content": query, "type": "query"}
                )
                if analysis:
                    insights = [analysis]

            logger.debug(f"✅ Retrieved {len(insights)} pattern insights for query")
            return (
                insights
                if isinstance(insights, list)
                else [insights]
                if insights
                else []
            )

        except Exception as e:
            logger.error(f"❌ Error retrieving pattern insights: {e}")
            return []

    def get_categories(self) -> List[Dict[str, Any]]:
        """
        Get all known categories from ART analysis.

        Returns:
            List of category information dictionaries
        """
        if not self.art_manager:
            logger.warning("⚠️ ARTManager not available for categories")
            return []

        try:
            categories = []

            # Try different methods to get category information
            if hasattr(self.art_manager, "get_categories"):
                categories = self.art_manager.get_categories()
            elif hasattr(self.art_manager, "list_categories"):
                categories = self.art_manager.list_categories()
            elif hasattr(self.art_manager, "controller") and hasattr(
                self.art_manager.controller, "get_categories"
            ):
                categories = self.art_manager.controller.get_categories()

            logger.debug(f"✅ Retrieved {len(categories)} ART categories")
            return categories if isinstance(categories, list) else []

        except Exception as e:
            logger.error(f"❌ Error retrieving ART categories: {e}")
            return []


class StubARTInterface(BaseARTInterface):
    """Stub implementation for development/testing when ART is not available."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize stub ART interface."""
        self.config = config or {}
        logger.warning("⚠️ Using StubARTInterface - ART functionality limited")

    def analyze_input(
        self, input_data: Union[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Stub analysis that returns basic categorization."""
        if isinstance(input_data, str):
            content = input_data
        else:
            content = input_data.get("content", str(input_data))

        # Simple stub categorization
        return {
            "category": {"id": "general", "name": "General"},
            "confidence": 0.5,
            "patterns": ["text_input"],
            "metadata": {"stub": True, "content_length": len(content)},
        }

    def learn_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """Stub learning that always returns True."""
        logger.debug("📝 Stub pattern learning - no actual learning performed")
        return True

    def get_pattern_insights(self, query: str) -> List[Dict[str, Any]]:
        """Stub insights that return basic information."""
        return [
            {
                "pattern": "stub_pattern",
                "relevance": 0.3,
                "description": f"Stub insight for query: {query[:50]}",
                "metadata": {"stub": True},
            }
        ]

    def get_categories(self) -> List[Dict[str, Any]]:
        """Stub categories that return basic category list."""
        return [
            {"id": "general", "name": "General", "count": 0},
            {"id": "text", "name": "Text", "count": 0},
            {"id": "query", "name": "Query", "count": 0},
        ]


def create_art_interface(
    config: Optional[Dict[str, Any]] = None, use_production: bool = True
) -> BaseARTInterface:
    """
    Factory function to create appropriate ART interface.

    Args:
        config: Optional configuration dictionary
        use_production: Whether to use production implementation

    Returns:
        ART interface instance (production or stub)
    """
    if use_production:
        try:
            return ProductionARTInterface(config)
        except (ImportError, RuntimeError) as e:
            logger.warning(f"⚠️ Production ART interface unavailable: {e}")
            logger.info("🔄 Falling back to stub ART interface")
            return StubARTInterface(config)
    else:
        logger.info("📝 Using stub ART interface (production disabled)")
        return StubARTInterface(config)
