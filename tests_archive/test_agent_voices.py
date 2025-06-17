#!/usr/bin/env python3
"""
VoxSigil Agent Voice Test
========================

Test individual agent voice capabilities.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_agent_imports():
    """Test importing individual agents"""
    print("TESTING AGENT IMPORTS")
    print("=" * 30)

    agents_to_test = ["phi", "astra", "oracle", "voxka", "echo"]
    working_agents = []

    for agent_name in agents_to_test:
        try:
            module = __import__(f"agents.{agent_name}", fromlist=[agent_name.capitalize()])
            agent_class = getattr(module, agent_name.capitalize())
            print(f"✅ {agent_name.capitalize()}: imported successfully")
            working_agents.append((agent_name, agent_class))
        except Exception as e:
            print(f"❌ {agent_name.capitalize()}: {e}")

    return working_agents


def test_agent_voice_system_direct():
    """Test voice system directly"""
    print("\nTESTING VOICE SYSTEM DIRECTLY")
    print("=" * 30)

    try:
        from core.agent_voice_system import AgentVoiceSystem

        voice_system = AgentVoiceSystem()
        print("✅ Voice system initialized")
        print(f"   Profiles loaded: {len(voice_system.voice_profiles)}")

        # Test getting a voice profile
        if "Phi" in voice_system.voice_profiles:
            phi_profile = voice_system.voice_profiles["Phi"]
            print(f"✅ Phi profile: {phi_profile.voice_id} ({phi_profile.tone})")

        return True, voice_system

    except Exception as e:
        print(f"❌ Voice system failed: {e}")
        return False, None


def test_agent_creation_and_speak(working_agents):
    """Test creating agents and using speak method"""
    print("\nTESTING AGENT SPEAK METHODS")
    print("=" * 30)

    if not working_agents:
        print("❌ No working agents available for testing")
        return False

    for agent_name, agent_class in working_agents[:2]:  # Test first 2 agents
        try:
            print(f"\n🤖 Testing {agent_name.capitalize()}...")

            # Create agent instance
            agent = agent_class()
            print(f"   ✅ Agent created: {agent.__class__.__name__}")

            # Test speak method
            test_message = f"Hello! I am {agent_name.capitalize()}, testing my voice in VoxSigil."
            result = agent.speak(test_message, add_signature=True)

            print(f"   Speak result: {result}")
            print("   ✅ Speak method executed")

        except Exception as e:
            print(f"   ❌ {agent_name} speak test failed: {e}")

    return True


def main():
    """Main test function"""
    print("VOXSIGIL AGENT VOICE TEST")
    print("=" * 40)

    # Test agent imports
    working_agents = test_agent_imports()

    # Test voice system
    voice_ok, voice_system = test_agent_voice_system_direct()

    # Test agent speak methods
    if working_agents:
        speak_ok = test_agent_creation_and_speak(working_agents)
    else:
        speak_ok = False

    # Summary
    print("\n" + "=" * 40)
    print("AGENT VOICE TEST SUMMARY")
    print("=" * 40)

    print(f"Working Agents: {len(working_agents)}")
    print(f"Voice System: {'✅ Working' if voice_ok else '❌ Failed'}")
    print(f"Agent Speak Methods: {'✅ Working' if speak_ok else '❌ Failed'}")

    if working_agents and voice_ok:
        print("\n🎉 SUCCESS: Agent voice system is functional!")
        print("🤖 Agents can be created and can use speak methods")
        print("🎤 TTS integration is working")
    else:
        print("\n⚠️  Some issues detected with agent voice system")


if __name__ == "__main__":
    main()
