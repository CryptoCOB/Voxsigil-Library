"""
Test script for Production Neural TTS
"""

import time

from core.production_neural_tts import ProductionNeuralTTS


def main():
    print("Initializing Production Neural TTS...")
    tts = ProductionNeuralTTS()

    print(f"Available engines: {tts.get_available_engines()}")
    print(f"Available voices: {tts.list_available_voices()}")

    print("\nTesting voice profiles...")
    for voice_name in ["Nova", "Aria", "Kai", "Echo", "Sage"]:
        print(f"\nTesting {voice_name}:")
        voice_info = tts.get_voice_info(voice_name)
        print(f"  Style: {voice_info['speaking_style']}, Gender: {voice_info['gender']}")
        success = tts.test_voice(voice_name)
        if success:
            print(f"  ✅ {voice_name} voice test successful")
        else:
            print(f"  ❌ {voice_name} voice test failed")
        time.sleep(0.5)

    print("\n✅ Production TTS system ready!")


if __name__ == "__main__":
    main()
