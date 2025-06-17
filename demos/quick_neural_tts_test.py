"""
Quick Neural TTS Test
"""

from core.production_neural_tts import ProductionNeuralTTS


def quick_test():
    print("🎙️  Quick Neural TTS Test")
    print("=" * 40)

    # Initialize TTS
    tts = ProductionNeuralTTS()

    print(f"Available engines: {tts.get_available_engines()}")
    print(f"Available voices: {tts.list_available_voices()}")

    # Test one voice
    print("\n🎤 Testing Nova voice...")
    success = tts.speak_text("Hello! I am Nova, your AI assistant.", "Nova", blocking=True)

    if success:
        print("✅ Voice test successful!")
    else:
        print("❌ Voice test failed")

    print("\n🎉 Test complete!")


if __name__ == "__main__":
    quick_test()
