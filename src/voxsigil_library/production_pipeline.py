"""
Production Pipeline: OpenClawd Agent Integration

Minimal scaffold to run forecasting pipeline with VoxBridge adapter.
Emits forecasts continuously, monitors heartbeat, logs failures.

Usage:
  python -m src.production_pipeline --agent-name btc-forecaster --pipeline-name forecast
  
Environment:
  VOXBRIDGE_BASE_URL=https://voxsigil-predict.fly.dev
  FORECAST_INTERVAL_SECONDS=30
  HEARTBEAT_INTERVAL_SECONDS=300
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from openclawd_adapter import (
    OpenClawdAgentFactory,
    OpenClawdEvent,
    EventEmissionError,
    HeartbeatError,
)

logger = logging.getLogger("openclawd-prod")


# ============================================================================
# CONFIG
# ============================================================================


class PipelineConfig:
    """Pipeline configuration from environment + CLI args."""

    def __init__(self, args: argparse.Namespace):
        self.agent_name = args.agent_name or os.getenv("AGENT_NAME", "forecaster")
        self.agent_type = args.agent_type or os.getenv("AGENT_TYPE", "llm")
        self.pipeline_name = args.pipeline_name or "forecast"
        self.description = (
            f"Forecasting pipeline ({self.pipeline_name}) — OpenClawd → VoxBridge"
        )

        self.voxbridge_base_url = os.getenv(
            "VOXBRIDGE_BASE_URL", "https://voxsigil-predict.fly.dev"
        )
        self.forecast_interval_seconds = int(
            os.getenv("FORECAST_INTERVAL_SECONDS", "30")
        )
        self.heartbeat_interval_seconds = int(
            os.getenv("HEARTBEAT_INTERVAL_SECONDS", "300")
        )
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_backoff_seconds = int(os.getenv("RETRY_BACKOFF_SECONDS", "5"))

    def __str__(self) -> str:
        return (
            f"PipelineConfig(agent={self.agent_name}, "
            f"pipeline={self.pipeline_name}, "
            f"voxbridge={self.voxbridge_base_url})"
        )


# ============================================================================
# MONITORING & METRICS
# ============================================================================


class PipelineMetrics:
    """Simple metrics collector."""

    def __init__(self):
        self.forecasts_emitted = 0
        self.forecasts_failed = 0
        self.heartbeats_sent = 0
        self.heartbeats_failed = 0
        self.start_time = datetime.now(timezone.utc)

    def emit_success(self) -> None:
        """Record successful emit."""
        self.forecasts_emitted += 1

    def emit_failure(self) -> None:
        """Record failed emit."""
        self.forecasts_failed += 1

    def heartbeat_success(self) -> None:
        """Record successful heartbeat."""
        self.heartbeats_sent += 1

    def heartbeat_failure(self) -> None:
        """Record failed heartbeat."""
        self.heartbeats_failed += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics snapshot."""
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            "uptime_seconds": elapsed,
            "forecasts_emitted": self.forecasts_emitted,
            "forecasts_failed": self.forecasts_failed,
            "heartbeats_sent": self.heartbeats_sent,
            "heartbeats_failed": self.heartbeats_failed,
            "emit_success_rate": (
                self.forecasts_emitted
                / (self.forecasts_emitted + self.forecasts_failed)
                if (self.forecasts_emitted + self.forecasts_failed) > 0
                else 0.0
            ),
        }


# ============================================================================
# FORECAST GENERATOR
# ============================================================================


class ForecastGenerator:
    """Mock forecast generator. Replace with real model."""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.call_count = 0

    def generate(self) -> Dict[str, Any]:
        """Generate next forecast."""
        self.call_count += 1

        # Mock BTC forecast (replace with real model)
        if self.pipeline_name == "forecast":
            return {
                "symbol": "BTC/USD",
                "current_price": 95200 + (self.call_count * 100),
                "target": 100000,
                "confidence": 0.72,
                "horizon_hours": 24,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Mock other pipeline types
        return {"pipeline": self.pipeline_name, "iteration": self.call_count}


# ============================================================================
# PIPELINE LOOP
# ============================================================================


class OpenClawdProductionPipeline:
    """Production pipeline: bootstrap → heartbeat loop → emit forecasts."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics = PipelineMetrics()
        self.forecast_gen = ForecastGenerator(config.pipeline_name)
        self.adapter = None
        self.running = False

    def setup(self) -> bool:
        """Bootstrap agent and start heartbeat."""
        try:
            logger.info(f"Setting up pipeline: {self.config}")

            # Create adapter
            self.adapter = OpenClawdAgentFactory.create(
                name=self.config.agent_name,
                agent_type=self.config.agent_type,
                voxbridge_url=self.config.voxbridge_base_url,
                description=self.config.description,
                generate_sigil=True,
            )
            logger.info(f"✅ Adapter created: {self.config.agent_name}")

            # Bootstrap (register with VoxBridge)
            reg_result = self.adapter.bootstrap()
            logger.info(f"✅ Registered with VoxBridge: {reg_result}")

            # Start heartbeat loop
            self.adapter.start_heartbeat_loop(
                interval_seconds=self.config.heartbeat_interval_seconds
            )
            logger.info(
                f"✅ Heartbeat loop started (interval: {self.config.heartbeat_interval_seconds}s)"
            )

            self.running = True
            return True

        except Exception as e:
            logger.error(f"❌ Setup failed: {e}", exc_info=True)
            return False

    def emit_forecast(self) -> bool:
        """Generate and emit forecast."""
        try:
            forecast = self.forecast_gen.generate()
            event = OpenClawdEvent(
                output_type="forecast",
                title=f"{self.config.pipeline_name.upper()} Forecast #{self.forecast_gen.call_count}",
                description=f"Forecast at {datetime.now(timezone.utc).isoformat()}",
                impact_score=forecast.get("confidence", 0.5),
                data=forecast,
            )

            result = self.adapter.emit(event)
            self.metrics.emit_success()
            logger.info(
                f"📤 Emitted forecast #{self.forecast_gen.call_count}: {result}"
            )
            return True

        except EventEmissionError as e:
            self.metrics.emit_failure()
            logger.error(f"❌ Event emission failed: {e}")
            return False
        except Exception as e:
            self.metrics.emit_failure()
            logger.error(f"❌ Unexpected error during emit: {e}", exc_info=True)
            return False

    def run(self, duration_seconds: Optional[int] = None) -> None:
        """Run pipeline loop.

        Args:
            duration_seconds: If set, stop after this many seconds. Otherwise, run forever.
        """
        if not self.setup():
            logger.error("Failed to setup pipeline")
            return

        start_time = time.time()
        iteration = 0

        try:
            while self.running:
                iteration += 1

                # Check duration
                if duration_seconds:
                    elapsed = time.time() - start_time
                    if elapsed > duration_seconds:
                        logger.info(
                            f"Reached duration limit ({duration_seconds}s), stopping"
                        )
                        break

                # Emit forecast
                logger.info(f"[Iteration {iteration}] Emitting forecast...")
                self.emit_forecast()

                # Print metrics periodically
                if iteration % 10 == 0:
                    metrics = self.metrics.get_summary()
                    logger.info(f"📊 Metrics: {json.dumps(metrics, indent=2)}")

                # Sleep before next emit
                time.sleep(self.config.forecast_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}", exc_info=True)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down pipeline...")
        if self.adapter:
            try:
                self.adapter.stop_heartbeat_loop()
                logger.info("✅ Heartbeat loop stopped")
            except Exception as e:
                logger.error(f"Error stopping heartbeat: {e}")

        metrics = self.metrics.get_summary()
        logger.info(f"Final metrics: {json.dumps(metrics, indent=2)}")
        self.running = False


# ============================================================================
# CLI
# ============================================================================


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="OpenClawd → VoxBridge production pipeline"
    )
    parser.add_argument(
        "--agent-name", help="Agent name (default: forecaster)", default=None
    )
    parser.add_argument(
        "--agent-type",
        help="Agent type: llm, oracle, analyzer (default: llm)",
        default=None,
    )
    parser.add_argument(
        "--pipeline-name",
        help="Pipeline name (default: forecast)",
        default="forecast",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        help="Run for N seconds then stop (default: infinite)",
        default=None,
    )
    args = parser.parse_args()

    config = PipelineConfig(args)
    pipeline = OpenClawdProductionPipeline(config)
    pipeline.run(duration_seconds=args.duration_seconds)


if __name__ == "__main__":
    main()
