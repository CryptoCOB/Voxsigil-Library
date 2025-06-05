"""
VANTA Runner - A clean script to run the VANTA Supervisor system.

This script provides a simple way to interact with the VANTA Supervisor.
It uses mock implementations for RAG, LLM, and Memory interfaces.
"""

import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vanta_runner.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("vanta.runner")


# ===== Mock Implementations =====
class MockRAG:
    """Mock implementation of the BaseRagInterface."""

    def retrieve_context(self, query, context=None):
        logger.info(f"MockRAG: retrieving context for query: {query[:50]}...")
        return [
            {
                "sigil": "symbolic-resonance-01",
                "content": "Symbolic systems represent knowledge explicitly through symbols and rules.",
                "_similarity_score": 0.85,
            },
            {
                "sigil": "neural-systems-02",
                "content": "Neural networks learn patterns implicitly through weighted connections.",
                "_similarity_score": 0.82,
            },
            {
                "sigil": "hybrid-systems-03",
                "content": "Hybrid neuro-symbolic systems combine explicit rules with implicit learning.",
                "_similarity_score": 0.76,
            },
        ]


class MockLLM:
    """Mock implementation of the BaseLlmInterface."""

    def generate_response(
        self, messages, system_prompt_override=None, task_requirements=None
    ):
        logger.info("MockLLM: generating response for prompt")
        user_content = messages[0]["content"] if messages else ""

        # Create a meaningful response based on the input
        query = (
            user_content.split("<<QUERY>>\n")[-1].split("\n\n<<RESPONSE>>")[0]
            if "<<QUERY>>" in user_content
            else ""
        )

        response = (
            f"In response to your question about {query[:50]}...\n\n"
            f"Symbolic systems and neural networks represent two complementary approaches to AI. "
            f"Symbolic systems use explicit representations through rules and logic, while neural networks "
            f"learn implicit patterns through connection weights and activation functions. "
            f"Modern research focuses on combining these approaches in neuro-symbolic systems."
        )

        return response, {"model": "mock-llm-vanta"}, {}


class MockMemory:
    """Mock implementation of the BaseMemoryInterface."""

    def __init__(self):
        self.memory_store = {}

    def store(self, query, response, metadata=None):
        memory_id = f"memory-{int(time.time())}"
        logger.info(f"MockMemory: storing memory with ID: {memory_id}")
        self.memory_store[memory_id] = {
            "query": query,
            "response": response,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        return memory_id

    def retrieve_recent(self, limit=10):
        sorted_memories = sorted(
            self.memory_store.items(),
            key=lambda x: x[1].get("timestamp", 0),
            reverse=True,
        )
        return [{"id": k, **v} for k, v in sorted_memories[:limit]]


# MockScaffoldRouter removed - using real implementations only


# ===== Main Script =====
def initialize_vanta():
    """Initialize the VANTA Supervisor."""
    logger.info("Initializing VANTA Supervisor")

    # Import the VANTA Supervisor
    try:
        from Vanta.integration.vanta_supervisor import (
            VantaSigilSupervisor,
            VANTA_SYSTEM_PROMPT,
        )

        # Initialize components
        rag = MockRAG()
        llm = MockLLM()
        memory = MockMemory()
        scaffold_router = None  # Using real implementation when available

        # Create VANTA instance
        vanta = VantaSigilSupervisor(
            rag_interface=rag,
            llm_interface=llm,
            memory_interface=memory,
            scaffold_router=scaffold_router,
            resonance_threshold=0.5,
            enable_adaptive=True,
            enable_echo_harmonization=True,
        )

        logger.info("VANTA Supervisor initialized successfully")
        return vanta, VANTA_SYSTEM_PROMPT

    except Exception as e:
        logger.error(f"Error initializing VANTA Supervisor: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def process_query(vanta, query):
    """Process a query using VANTA Supervisor."""
    logger.info(f"Processing query: {query}")

    # Process the query
    result = vanta.orchestrate_thought_cycle(query)

    # Display results
    print("\n" + "=" * 50)
    print(f"QUERY: {query}")
    print("=" * 50)
    print(f"RESPONSE:\n{result['response']}")
    print("-" * 50)

    if result.get("scaffold"):
        print(f"Scaffold: {result['scaffold']}")

    if isinstance(result.get("sigils_used"), list):
        print(f"Sigils used: {len(result['sigils_used'])}")

    print(f"Execution time: {result.get('execution_time', 0):.2f}s")
    print("=" * 50 + "\n")

    return result


def interactive_mode(vanta):
    """Run VANTA in interactive mode."""
    print("\n" + "=" * 50)
    print("VANTA SUPERVISOR INTERACTIVE MODE")
    print("Enter your queries below. Type 'exit' to quit.")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("Query > ")
            if query.lower() in ["exit", "quit", "q"]:
                break

            process_query(vanta, query)

        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {e}")


def run_demo_queries(vanta):
    """Run a set of demo queries through VANTA."""
    demo_queries = [
        "How do symbolic systems and neural networks interact?",
        "Explain the concept of resonance in cognitive architectures",
        "What patterns emerge in distributed learning systems?",
        "How can I implement reflective reasoning in my agent?",
        "What is the relationship between symbols and concepts?",
    ]

    print("\n" + "=" * 50)
    print("VANTA SUPERVISOR DEMO MODE")
    print(f"Processing {len(demo_queries)} demo queries")
    print("=" * 50 + "\n")

    for query in demo_queries:
        process_query(vanta, query)

    # Show performance stats
    stats = vanta.get_performance_stats()
    print("\n" + "=" * 50)
    print("VANTA PERFORMANCE STATISTICS")
    print("=" * 50)
    print(f"Queries processed: {stats.get('queries_processed', 0)}")
    print(f"Total resonances: {stats.get('total_resonances', 0)}")
    print(f"Average execution time: {stats.get('avg_execution_time', 0):.2f}s")
    print("=" * 50 + "\n")


def main():
    """Main function to run the VANTA system."""
    print("\nInitializing VANTA Supervisor System...")

    # Initialize VANTA
    vanta, system_prompt = initialize_vanta()
    if not vanta:
        print("Failed to initialize VANTA Supervisor. Exiting.")
        return False

    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Run the VANTA Supervisor system")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument("--query", "-q", help="Process a single query and exit")
    parser.add_argument("--demo", "-d", action="store_true", help="Run demo queries")

    args = parser.parse_args()

    if args.query:
        # Process a single query
        process_query(vanta, args.query)
    elif args.interactive:
        # Run in interactive mode
        interactive_mode(vanta)
    elif args.demo:
        # Run demo queries
        run_demo_queries(vanta)
    else:
        # Default to demo mode
        run_demo_queries(vanta)

    print("VANTA Supervisor session completed.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
