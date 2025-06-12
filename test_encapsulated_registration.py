#!/usr/bin/env python3
"""
Test Enhanced Encapsulated Registration Pattern with HOLO-1.5
Validates the self-registration system and HOLO-1.5 mesh capabilities
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import (
    BaseAgent, 
    vanta_agent, 
    CognitiveMeshRole,
    set_vanta_instance,
    get_registered_agents,
    register_all_agents_auto,
    create_holo_mesh_network,
    execute_mesh_task
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockVantaCore:
    """Mock Vanta Core for testing without full system."""
    
    def __init__(self):
        self.registered_agents = {}
        self.components = {}
        
    def register_agent(self, name: str, agent: BaseAgent, metadata: dict = None):
        """Register an agent."""
        self.registered_agents[name] = {
            'agent': agent,
            'metadata': metadata or {}
        }
        logger.info(f"✅ Registered {name} with metadata: {metadata}")
        return True
    
    def get_agent(self, name: str):
        """Get a registered agent."""
        return self.registered_agents.get(name, {}).get('agent')
    
    def get_component(self, name: str):
        """Get a mock component."""
        return f"mock_{name}_component"


@vanta_agent(name="TestPlanner", subsystem="test_planner", mesh_role=CognitiveMeshRole.PLANNER)
class TestPlannerAgent(BaseAgent):
    """Test agent for PLANNER role."""
    sigil = "🧪📋✨🎯"
    tags = ['Test Planner', 'Strategic Test']
    invocations = ['Plan test', 'Create strategy']


@vanta_agent(name="TestGenerator", subsystem="test_generator", mesh_role=CognitiveMeshRole.GENERATOR)
class TestGeneratorAgent(BaseAgent):
    """Test agent for GENERATOR role."""
    sigil = "🧪⚡🎨🔧"
    tags = ['Test Generator', 'Creative Test']
    invocations = ['Generate test', 'Create content']


@vanta_agent(name="TestCritic", subsystem="test_critic", mesh_role=CognitiveMeshRole.CRITIC)
class TestCriticAgent(BaseAgent):
    """Test agent for CRITIC role."""
    sigil = "🧪🔍⚖️🛡️"
    tags = ['Test Critic', 'Analytical Test']
    invocations = ['Critique test', 'Analyze quality']


@vanta_agent(name="TestEvaluator", subsystem="test_evaluator", mesh_role=CognitiveMeshRole.EVALUATOR)
class TestEvaluatorAgent(BaseAgent):
    """Test agent for EVALUATOR role."""
    sigil = "🧪📊🎖️✅"
    tags = ['Test Evaluator', 'Assessment Test']
    invocations = ['Evaluate test', 'Final assessment']


async def test_basic_registration():
    """Test basic self-registration functionality."""
    logger.info("🧪 Testing basic self-registration...")
    
    # Create mock Vanta core
    mock_vanta = MockVantaCore()
    set_vanta_instance(mock_vanta)
    
    # Create test agents (should auto-register)
    planner = TestPlannerAgent()
    generator = TestGeneratorAgent()
    critic = TestCriticAgent()
    evaluator = TestEvaluatorAgent()
    
    # Verify registration
    assert len(mock_vanta.registered_agents) == 4
    assert "TestPlanner" in mock_vanta.registered_agents
    assert "TestGenerator" in mock_vanta.registered_agents
    assert "TestCritic" in mock_vanta.registered_agents
    assert "TestEvaluator" in mock_vanta.registered_agents
    
    logger.info("✅ Basic self-registration test passed")
    return [planner, generator, critic, evaluator]


async def test_holo_mesh_roles():
    """Test HOLO-1.5 mesh role detection and assignment."""
    logger.info("🧪 Testing HOLO-1.5 mesh roles...")
    
    mock_vanta = MockVantaCore()
    agents = [
        TestPlannerAgent(mock_vanta),
        TestGeneratorAgent(mock_vanta),
        TestCriticAgent(mock_vanta),
        TestEvaluatorAgent(mock_vanta)
    ]
    
    # Verify mesh roles
    assert agents[0]._mesh_role == CognitiveMeshRole.PLANNER
    assert agents[1]._mesh_role == CognitiveMeshRole.GENERATOR
    assert agents[2]._mesh_role == CognitiveMeshRole.CRITIC
    assert agents[3]._mesh_role == CognitiveMeshRole.EVALUATOR
    
    logger.info("✅ HOLO-1.5 mesh roles test passed")
    return agents


async def test_symbolic_compression():
    """Test symbolic compression and expansion."""
    logger.info("🧪 Testing symbolic compression...")
    
    mock_vanta = MockVantaCore()
    agent = TestGeneratorAgent(mock_vanta)
    
    # Test data compression
    test_data = {
        "complex_structure": {
            "nested": ["data", "with", "many", "values"],
            "numbers": [1, 2, 3, 4, 5]
        },
        "metadata": {"type": "test", "version": "1.0"}
    }
    
    # Compress and expand
    symbol_ref = agent.compress_to_symbol(test_data, "test_complex_data")
    expanded_data = agent.expand_symbol(symbol_ref)
    
    # Verify compression format
    assert symbol_ref.startswith('⧈') and symbol_ref.endswith('⧈')
    assert symbol_ref == "⧈test_complex_data⧈"
    assert expanded_data is not None
    
    logger.info(f"🔄 Compressed: {test_data} -> {symbol_ref}")
    logger.info(f"🔄 Expanded: {symbol_ref} -> {expanded_data}")
    logger.info("✅ Symbolic compression test passed")


async def test_cognitive_chains():
    """Test cognitive chain creation and reasoning."""
    logger.info("🧪 Testing cognitive chains...")
    
    mock_vanta = MockVantaCore()
    agent = TestPlannerAgent(mock_vanta)
    
    # Test chain of thought
    chain_id = agent.create_cognitive_chain("Test complex problem solving", "chain_of_thought")
    assert chain_id.startswith("chain_of_thought_")
    
    # Add reasoning steps
    agent.add_reasoning_step(chain_id, "Analyze the problem", "Break down complexity")
    agent.add_reasoning_step(chain_id, "Generate solutions", "Create multiple approaches")
    agent.add_reasoning_step(chain_id, "Evaluate options", "Compare effectiveness")
    
    # Find the chain
    test_chain = None
    for chain in agent._cognitive_chains:
        if chain['id'] == chain_id:
            test_chain = chain
            break
    
    assert test_chain is not None
    assert len(test_chain['steps']) == 3
    assert test_chain['type'] == 'chain_of_thought'
    
    # Test tree of thought
    tree_id = agent.create_cognitive_chain("Test branching decisions", "tree_of_thought")
    agent.add_reasoning_step(tree_id, "Branch A: Direct approach", "Fast but risky")
    agent.add_reasoning_step(tree_id, "Branch B: Careful approach", "Slow but safe")
    
    logger.info(f"🧠 Created chains: {[c['id'] for c in agent._cognitive_chains]}")
    logger.info("✅ Cognitive chains test passed")


async def test_mesh_network_creation():
    """Test HOLO-1.5 mesh network creation."""
    logger.info("🧪 Testing mesh network creation...")
    
    mock_vanta = MockVantaCore()
    agents = [
        TestPlannerAgent(mock_vanta),
        TestGeneratorAgent(mock_vanta),
        TestCriticAgent(mock_vanta),
        TestEvaluatorAgent(mock_vanta),
        TestGeneratorAgent(mock_vanta),  # Second generator
    ]
    
    # Create mesh network
    mesh_network = create_holo_mesh_network(agents)
    
    # Verify structure
    assert CognitiveMeshRole.PLANNER in mesh_network
    assert CognitiveMeshRole.GENERATOR in mesh_network
    assert CognitiveMeshRole.CRITIC in mesh_network
    assert CognitiveMeshRole.EVALUATOR in mesh_network
    
    # Verify agent counts
    assert len(mesh_network[CognitiveMeshRole.PLANNER]) == 1
    assert len(mesh_network[CognitiveMeshRole.GENERATOR]) == 2  # Two generators
    assert len(mesh_network[CognitiveMeshRole.CRITIC]) == 1
    assert len(mesh_network[CognitiveMeshRole.EVALUATOR]) == 1
    
    logger.info("✅ Mesh network creation test passed")
    return mesh_network, agents


async def test_mesh_collaboration():
    """Test HOLO-1.5 mesh collaboration."""
    logger.info("🧪 Testing mesh collaboration...")
    
    mesh_network, agents = await test_mesh_network_creation()
    
    # Execute mesh task
    result = execute_mesh_task(mesh_network, "Develop a comprehensive testing strategy")
    
    # Verify result structure
    assert 'task' in result
    assert 'mesh_flow' in result
    assert 'final_output' in result
    assert 'participants' in result
    
    # Verify all roles participated
    assert len(result['mesh_flow']) == 4  # All 4 roles
    assert CognitiveMeshRole.PLANNER in result['participants']
    assert CognitiveMeshRole.GENERATOR in result['participants']
    assert CognitiveMeshRole.CRITIC in result['participants']
    assert CognitiveMeshRole.EVALUATOR in result['participants']
    
    logger.info(f"🤝 Mesh collaboration result: {result['final_output']}")
    logger.info("✅ Mesh collaboration test passed")


async def test_symbolic_triggers():
    """Test symbolic trigger responses."""
    logger.info("🧪 Testing symbolic triggers...")
    
    mock_vanta = MockVantaCore()
    agent = TestPlannerAgent(mock_vanta)
    
    # Test trigger responses
    context = {"priority": "high", "complexity": "medium"}
    response = agent.trigger_symbolic_response("plan_test", context)
    
    # Verify response format
    assert "📋 Planning for plan_test" in response
    assert "⧈ctx_plan_test⧈" in response
    
    # Test different mesh roles
    generator = TestGeneratorAgent(mock_vanta)
    gen_response = generator.trigger_symbolic_response("generate_test", context)
    assert "⚡ Generating for generate_test" in gen_response
    
    critic = TestCriticAgent(mock_vanta)
    crit_response = critic.trigger_symbolic_response("critique_test", context)
    assert "🔍 Analyzing critique_test" in crit_response
    
    evaluator = TestEvaluatorAgent(mock_vanta)
    eval_response = evaluator.trigger_symbolic_response("evaluate_test", context)
    assert "⚖️ Evaluating evaluate_test" in eval_response
    
    logger.info("✅ Symbolic triggers test passed")


async def test_registry_functionality():
    """Test the global agent registry."""
    logger.info("🧪 Testing registry functionality...")
    
    # Get registered agents from decorators
    registered = get_registered_agents()
    
    # Should include our test agents
    assert "TestPlanner" in registered
    assert "TestGenerator" in registered
    assert "TestCritic" in registered
    assert "TestEvaluator" in registered
    
    logger.info(f"📋 Registry contains: {list(registered.keys())}")
    logger.info("✅ Registry functionality test passed")


async def run_comprehensive_test():
    """Run comprehensive test suite."""
    logger.info("🚀 Starting comprehensive encapsulated registration test suite...")
    
    try:
        # Run all tests
        await test_basic_registration()
        await test_holo_mesh_roles()
        await test_symbolic_compression()
        await test_cognitive_chains()
        await test_mesh_collaboration()
        await test_symbolic_triggers()
        await test_registry_functionality()
        
        logger.info("🎯 All tests passed! Encapsulated registration working correctly.")
        
        # Performance summary
        registered = get_registered_agents()
        logger.info(f"📊 Test Summary:")
        logger.info(f"  - Agents in registry: {len(registered)}")
        logger.info(f"  - HOLO-1.5 mesh roles: 4 (Planner, Generator, Critic, Evaluator)")
        logger.info(f"  - Symbolic compression: ✅ Working")
        logger.info(f"  - Cognitive chains: ✅ Working")
        logger.info(f"  - Mesh collaboration: ✅ Working")
        logger.info(f"  - Auto-registration: ✅ Working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)
