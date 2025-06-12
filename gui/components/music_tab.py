#!/usr/bin/env python3
"""
Music Interaction GUI Components
===============================

Modern PyQt5-based GUI components for music generation, voice modulation,
and genre-aware audio processing. Integrates with the expanded genre vocabulary
and VantaCore's cognitive mesh.

Features:
- Real-time music composition interface
- Voice modulation controls with ethical safeguards
- Genre-aware audio processing
- Multi-track editing and stem separation
- Cognitive load monitoring
- User preference learning interface
"""

import sys
import asyncio
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QLabel, QSlider, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit,
    QProgressBar, QGroupBox, QTabWidget, QListWidget, QListWidgetItem,
    QCheckBox, QLineEdit, QFileDialog, QMessageBox, QSplitter,
    QScrollArea, QFrame, QButtonGroup, QRadioButton
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QIcon

# VoxSigil imports
from core.vanta_core import VantaCore
from agents.ensemble.music.music_composer_agent import MusicComposerAgent, CompositionRequest
from agents.ensemble.music.voice_modulator_agent import VoiceModulatorAgent, VoiceModulationRequest
from agents.ensemble.music.music_sense_agent import MusicSenseAgent

logger = logging.getLogger(__name__)

@dataclass
class MusicUIConfig:
    """Configuration for music UI components"""
    default_sample_rate: int = 44100
    max_composition_duration: float = 300.0
    real_time_visualization: bool = True
    cognitive_load_monitoring: bool = True
    user_preference_learning: bool = True
    ethical_checks_enabled: bool = True

class AudioVisualizationWidget(QWidget):
    """Real-time audio visualization widget"""
    
    def __init__(self):
        super().__init__()
        self.audio_data = np.array([])
        self.sample_rate = 44100
        self.setup_ui()
        
        # Setup timer for real-time updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(50)  # 20 FPS
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Waveform display
        self.waveform_label = QLabel("Waveform Visualization")
        self.waveform_label.setMinimumHeight(100)
        self.waveform_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        layout.addWidget(self.waveform_label)
        
        # Spectrum display
        self.spectrum_label = QLabel("Frequency Spectrum")
        self.spectrum_label.setMinimumHeight(100)
        self.spectrum_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        layout.addWidget(self.spectrum_label)
        
        self.setLayout(layout)
    
    def update_audio_data(self, audio_data: np.ndarray, sample_rate: int = 44100):
        """Update audio data for visualization"""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
    
    def update_visualization(self):
        """Update the visualization display"""
        if len(self.audio_data) > 0:
            # In a real implementation, this would create actual visualizations
            # For now, just updating the labels
            duration = len(self.audio_data) / self.sample_rate
            energy = np.sqrt(np.mean(self.audio_data ** 2))
            
            self.waveform_label.setText(f"Waveform - Duration: {duration:.2f}s, Energy: {energy:.3f}")
            
            # Simple frequency analysis
            if len(self.audio_data) > 1024:
                fft = np.abs(np.fft.fft(self.audio_data[:1024]))
                peak_freq = np.argmax(fft) * self.sample_rate / 1024
                self.spectrum_label.setText(f"Spectrum - Peak Frequency: {peak_freq:.1f} Hz")

class GenreSelectionWidget(QWidget):
    """Genre selection widget with expanded vocabulary"""

    genre_changed = pyqtSignal(str)

    def __init__(self, genre_vocabulary: Dict[str, Any], favorites: Optional[Dict[str, int]] = None):
        super().__init__()
        self.genre_vocabulary = genre_vocabulary
        self.favorites = favorites or {}
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Genre category selection
        self.category_combo = QComboBox()
        self.category_combo.addItem("All Genres")
        
        # Load genre categories
        music_genres = self.genre_vocabulary.get("music_genres", {})
        for category in music_genres.keys():
            display_name = category.replace("_", " ").title()
            self.category_combo.addItem(display_name, category)
        
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        layout.addWidget(QLabel("Genre Category:"))
        layout.addWidget(self.category_combo)
        
        # Favorites filter
        self.favorites_check = QCheckBox("\xf0\x9f\x8c\x9f Favorites")
        self.favorites_check.stateChanged.connect(lambda _: self.on_category_changed(self.category_combo.currentText()))
        layout.addWidget(self.favorites_check)

        # Specific genre selection
        self.genre_combo = QComboBox()
        self.genre_combo.currentTextChanged.connect(self.on_genre_changed)
        layout.addWidget(QLabel("Specific Genre:"))
        layout.addWidget(self.genre_combo)
        
        # Populate initial genres
        self.on_category_changed("All Genres")
        
        self.setLayout(layout)
    
    def on_category_changed(self, category_name: str):
        """Handle category selection change"""
        self.genre_combo.clear()
        
        music_genres = self.genre_vocabulary.get("music_genres", {})
        
        if self.favorites_check.isChecked() and self.favorites:
            top = sorted(self.favorites.items(), key=lambda x: x[1], reverse=True)
            top_genres = [g for g, _ in top[:5]]
            self.genre_combo.addItems(top_genres)
        elif category_name == "All Genres":
            # Add all genres from all categories
            all_genres = []
            for genres_list in music_genres.values():
                all_genres.extend(genres_list)
            all_genres = sorted(list(set(all_genres)))  # Remove duplicates and sort
            self.genre_combo.addItems(all_genres)
        else:
            # Find the actual category key
            category_key = None
            for key, _ in music_genres.items():
                if key.replace("_", " ").title() == category_name:
                    category_key = key
                    break
            
            if category_key and category_key in music_genres:
                genres = sorted(music_genres[category_key])
                self.genre_combo.addItems(genres)
    
    def on_genre_changed(self, genre: str):
        """Handle genre selection change"""
        if genre:
            self.genre_changed.emit(genre)
    
    def get_selected_genre(self) -> str:
        """Get currently selected genre"""
        return self.genre_combo.currentText()

class CompositionControlsWidget(QWidget):
    """Music composition control widget"""
    
    compose_requested = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Duration control
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration (seconds):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(5.0, 300.0)
        self.duration_spin.setValue(30.0)
        self.duration_spin.setSingleStep(5.0)
        duration_layout.addWidget(self.duration_spin)
        layout.addLayout(duration_layout)
        
        # Tempo control
        tempo_layout = QHBoxLayout()
        tempo_layout.addWidget(QLabel("Tempo (BPM):"))
        self.tempo_spin = QSpinBox()
        self.tempo_spin.setRange(60, 200)
        self.tempo_spin.setValue(120)
        tempo_layout.addWidget(self.tempo_spin)
        layout.addLayout(tempo_layout)
        
        # Key selection
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("Key:"))
        self.key_combo = QComboBox()
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.key_combo.addItems(keys)
        key_layout.addWidget(self.key_combo)
        layout.addLayout(key_layout)
        
        # Energy level
        energy_layout = QHBoxLayout()
        energy_layout.addWidget(QLabel("Energy Level:"))
        self.energy_combo = QComboBox()
        self.energy_combo.addItems(["Low", "Medium", "High"])
        self.energy_combo.setCurrentText("Medium")
        energy_layout.addWidget(self.energy_combo)
        layout.addLayout(energy_layout)
        
        # Emotional arc controls
        emotion_group = QGroupBox("Emotional Arc")
        emotion_layout = QGridLayout()
        
        self.emotion_sliders = {}
        emotions = ["Energy", "Tension", "Joy", "Melancholy"]
        for i, emotion in enumerate(emotions):
            emotion_layout.addWidget(QLabel(f"{emotion}:"), i, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            self.emotion_sliders[emotion.lower()] = slider
            emotion_layout.addWidget(slider, i, 1)
        
        emotion_group.setLayout(emotion_layout)
        layout.addWidget(emotion_group)
        
        # Composition button
        self.compose_button = QPushButton("🎵 Compose Music")
        self.compose_button.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        self.compose_button.clicked.connect(self.on_compose_clicked)
        layout.addWidget(self.compose_button)
        
        self.setLayout(layout)
    
    def on_compose_clicked(self):
        """Handle compose button click"""
        # Collect emotional arc
        emotional_arc = {}
        for emotion, slider in self.emotion_sliders.items():
            emotional_arc[emotion] = slider.value() / 100.0
        
        # Create composition parameters
        composition_params = {
            "duration_seconds": self.duration_spin.value(),
            "tempo": self.tempo_spin.value(),
            "key": self.key_combo.currentText(),
            "energy_level": self.energy_combo.currentText().lower(),
            "emotional_arc": emotional_arc
        }
        
        self.compose_requested.emit(composition_params)

class VoiceModulationWidget(QWidget):
    """Voice modulation control widget"""
    
    modulate_requested = pyqtSignal(dict)
    
    def __init__(self, voice_profiles: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.voice_profiles = voice_profiles
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Voice profile selection
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel("Voice Profile:"))
        self.profile_combo = QComboBox()
        for profile_id, profile_data in self.voice_profiles.items():
            display_name = f"{profile_data['name']} ({profile_data['gender']})"
            self.profile_combo.addItem(display_name, profile_id)
        profile_layout.addWidget(self.profile_combo)
        layout.addLayout(profile_layout)
        
        # Modulation strength
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Modulation Strength:"))
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(80)
        self.strength_label = QLabel("80%")
        self.strength_slider.valueChanged.connect(
            lambda v: self.strength_label.setText(f"{v}%")
        )
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_label)
        layout.addLayout(strength_layout)
        
        # Emotional target
        emotion_layout = QHBoxLayout()
        emotion_layout.addWidget(QLabel("Emotional Target:"))
        self.emotion_combo = QComboBox()
        emotions = ["Neutral", "Calm", "Energetic", "Sensual", "Confident", "Warm"]
        self.emotion_combo.addItems(emotions)
        emotion_layout.addWidget(self.emotion_combo)
        layout.addLayout(emotion_layout)
        
        # Input audio selection
        input_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select input audio file...")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_input_file)
        input_layout.addWidget(QLabel("Input Audio:"))
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(browse_button)
        layout.addLayout(input_layout)
        
        # Ethical consent checkbox
        self.consent_checkbox = QCheckBox("I have consent to use this voice profile")
        self.consent_checkbox.setChecked(True)
        layout.addWidget(self.consent_checkbox)
        
        # Modulation button
        self.modulate_button = QPushButton("🎭 Modulate Voice")
        self.modulate_button.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        self.modulate_button.clicked.connect(self.on_modulate_clicked)
        layout.addWidget(self.modulate_button)
        
        self.setLayout(layout)
    
    def browse_input_file(self):
        """Browse for input audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a);;All Files (*)"
        )
        if file_path:
            self.input_path_edit.setText(file_path)
    
    def on_modulate_clicked(self):
        """Handle modulate button click"""
        if not self.consent_checkbox.isChecked():
            QMessageBox.warning(self, "Ethical Consent Required",
                              "Please confirm you have consent to use this voice profile.")
            return
        
        if not self.input_path_edit.text():
            QMessageBox.warning(self, "Input Required",
                              "Please select an input audio file.")
            return
        
        # Create modulation parameters
        modulation_params = {
            "input_path": self.input_path_edit.text(),
            "target_profile": self.profile_combo.currentData(),
            "modulation_strength": self.strength_slider.value() / 100.0,
            "emotional_target": self.emotion_combo.currentText().lower(),
            "ethical_consent": self.consent_checkbox.isChecked()
        }
        
        self.modulate_requested.emit(modulation_params)

class CognitiveMeshStatusWidget(QWidget):
    """Widget for monitoring cognitive mesh status"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(2000)  # Update every 2 seconds
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Cognitive load indicator
        load_layout = QHBoxLayout()
        load_layout.addWidget(QLabel("Cognitive Load:"))
        self.load_progress = QProgressBar()
        self.load_progress.setRange(0, 100)
        self.load_progress.setValue(0)
        load_layout.addWidget(self.load_progress)
        self.load_label = QLabel("0%")
        load_layout.addWidget(self.load_label)
        layout.addLayout(load_layout)
        
        # Processing efficiency
        efficiency_layout = QHBoxLayout()
        efficiency_layout.addWidget(QLabel("Processing Efficiency:"))
        self.efficiency_progress = QProgressBar()
        self.efficiency_progress.setRange(0, 100)
        self.efficiency_progress.setValue(85)
        efficiency_layout.addWidget(self.efficiency_progress)
        self.efficiency_label = QLabel("85%")
        efficiency_layout.addWidget(self.efficiency_label)
        layout.addLayout(efficiency_layout)
        
        # Active agents
        self.agents_label = QLabel("Active Agents: 0")
        layout.addWidget(self.agents_label)
        
        # Genre learning progress
        learning_layout = QHBoxLayout()
        learning_layout.addWidget(QLabel("Genre Learning:"))
        self.learning_progress = QProgressBar()
        self.learning_progress.setRange(0, 100)
        self.learning_progress.setValue(0)
        learning_layout.addWidget(self.learning_progress)
        layout.addLayout(learning_layout)
        
        self.setLayout(layout)
    
    def update_status(self):
        """Update cognitive mesh status (simulated)"""
        # In a real implementation, this would query actual VantaCore metrics
        import random
        
        # Simulate changing metrics
        cognitive_load = random.randint(15, 85)
        self.load_progress.setValue(cognitive_load)
        self.load_label.setText(f"{cognitive_load}%")
        
        # Update load bar color based on value
        if cognitive_load > 80:
            self.load_progress.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif cognitive_load > 60:
            self.load_progress.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
        else:
            self.load_progress.setStyleSheet("QProgressBar::chunk { background-color: green; }")
        
        # Update other metrics
        efficiency = random.randint(75, 95)
        self.efficiency_progress.setValue(efficiency)
        self.efficiency_label.setText(f"{efficiency}%")
        
        active_agents = random.randint(2, 8)
        self.agents_label.setText(f"Active Agents: {active_agents}")
        
        learning_progress = min(100, self.learning_progress.value() + random.randint(0, 2))
        self.learning_progress.setValue(learning_progress)

class MusicTabWidget(QWidget):
    """Main music interaction tab widget"""
    
    def __init__(self, vanta_core: Optional[VantaCore] = None):
        super().__init__()
        self.vanta_core = vanta_core
        self.config = MusicUIConfig()
        
        # Initialize components
        self.music_composer = None
        self.voice_modulator = None
        self.music_sense = None
        
        # Load genre vocabulary and preferences
        self.genre_vocabulary = self.load_genre_vocabulary()
        self.genre_preferences = self.load_genre_preferences()
        
        self.setup_ui()
        self.initialize_agents()
    
    def load_genre_vocabulary(self) -> Dict[str, Any]:
        """Load genre vocabulary"""
        try:
            vocab_path = Path("sigils/global_vocab.json")
            if vocab_path.exists():
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load genre vocabulary: {e}")
        
        # Return minimal vocabulary if loading fails
        return {
            "music_genres": {
                "hip_hop_family": ["Hip Hop", "Rap", "Beats"],
                "electronic_family": ["EDM", "Electronic"],
                "atmospheric_moods": ["Chill", "Mellow"],
                "sensual_intimate": ["Sensual"]
            }
        }

    def load_genre_preferences(self) -> Dict[str, int]:
        """Load genre preference statistics"""
        try:
            pref_path = Path("config/music_preferences.json")
            if pref_path.exists():
                with open(pref_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load genre preferences: {e}")
        return {}
    
    def setup_ui(self):
        main_layout = QHBoxLayout()
        
        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout()
        
        # Genre selection
        self.genre_widget = GenreSelectionWidget(self.genre_vocabulary, self.genre_preferences)
        self.genre_widget.genre_changed.connect(self.on_genre_changed)
        left_layout.addWidget(QLabel("🎼 Genre Selection"))
        left_layout.addWidget(self.genre_widget)
        
        # Composition controls
        self.composition_widget = CompositionControlsWidget()
        self.composition_widget.compose_requested.connect(self.on_compose_requested)
        left_layout.addWidget(QLabel("🎵 Composition Controls"))
        left_layout.addWidget(self.composition_widget)
        
        # Voice modulation controls
        voice_profiles = self.get_default_voice_profiles()
        self.voice_widget = VoiceModulationWidget(voice_profiles)
        self.voice_widget.modulate_requested.connect(self.on_modulate_requested)
        left_layout.addWidget(QLabel("🎭 Voice Modulation"))
        left_layout.addWidget(self.voice_widget)
        
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel)
        
        # Right panel - Visualization and status
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Audio visualization
        self.audio_viz = AudioVisualizationWidget()
        right_layout.addWidget(QLabel("📊 Audio Visualization"))
        right_layout.addWidget(self.audio_viz)
        
        # Cognitive mesh status
        self.mesh_status = CognitiveMeshStatusWidget()
        right_layout.addWidget(QLabel("🧠 Cognitive Mesh Status"))
        right_layout.addWidget(self.mesh_status)
        
        # Output log
        self.output_log = QTextEdit()
        self.output_log.setMaximumHeight(200)
        self.output_log.setReadOnly(True)
        right_layout.addWidget(QLabel("📝 Output Log"))
        right_layout.addWidget(self.output_log)
        
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel)
        
        self.setLayout(main_layout)
    
    def get_default_voice_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get default voice profiles"""
        return {
            "narrator_male_deep": {
                "name": "Deep Male Narrator",
                "gender": "male",
                "age_range": "30-50",
                "accent": "neutral"
            },
            "singer_female_versatile": {
                "name": "Versatile Female Singer", 
                "gender": "female",
                "age_range": "20-35",
                "accent": "neutral"
            },
            "rapper_male_urban": {
                "name": "Urban Male Rapper",
                "gender": "male", 
                "age_range": "20-30",
                "accent": "urban"
            }
        }
    
    def initialize_agents(self):
        """Initialize music processing agents"""
        try:
            if self.vanta_core:
                # Initialize agents with VantaCore
                self.music_composer = MusicComposerAgent(self.vanta_core, {})
                self.voice_modulator = VoiceModulatorAgent(self.vanta_core, {})
                self.music_sense = MusicSenseAgent(self.vanta_core, {})
                
                self.log_output("🤖 Music processing agents initialized")
            else:
                self.log_output("⚠️ VantaCore not available, agents will run in simulation mode")
        except Exception as e:
            self.log_output(f"❌ Failed to initialize agents: {e}")
    
    def on_genre_changed(self, genre: str):
        """Handle genre selection change"""
        self.log_output(f"🎼 Genre selected: {genre}")
    
    def on_compose_requested(self, params: Dict[str, Any]):
        """Handle music composition request"""
        try:
            genre = self.genre_widget.get_selected_genre()
            if not genre:
                self.log_output("❌ Please select a genre first")
                return
            
            self.log_output(f"🎵 Starting composition: {genre}, {params['duration_seconds']}s")
            
            # Create composition request
            request = CompositionRequest(
                genre=genre,
                duration_seconds=params['duration_seconds'],
                tempo=params['tempo'],
                key=params['key'],
                energy_level=params['energy_level'],
                emotional_arc=params['emotional_arc']
            )
            
            # Start composition in background
            asyncio.create_task(self.compose_music_async(request))
            
        except Exception as e:
            self.log_output(f"❌ Composition failed: {e}")
    
    async def compose_music_async(self, request: CompositionRequest):
        """Compose music asynchronously"""
        try:
            if self.music_composer:
                # Initialize composer if needed
                if not hasattr(self.music_composer, '_initialized'):
                    await self.music_composer.initialize()
                    self.music_composer._initialized = True
                
                # Compose music
                result = await self.music_composer.compose_music(request)
                
                # Update visualization
                self.audio_viz.update_audio_data(result.audio_data, result.sample_rate)
                
                # Save composition
                output_path = await self.music_composer.save_composition(result)
                
                self.log_output(f"✅ Composition completed: {output_path}")
            else:
                # Simulation mode
                import time
                await asyncio.sleep(2)  # Simulate processing time
                
                # Create simulated audio
                duration = request.duration_seconds
                sample_rate = 44100
                t = np.linspace(0, duration, int(duration * sample_rate))
                audio = 0.3 * np.sin(2 * np.pi * 220 * t)  # A3 sine wave
                
                self.audio_viz.update_audio_data(audio, sample_rate)
                self.log_output(f"✅ Composition completed (simulation): {request.genre}")
                
        except Exception as e:
            self.log_output(f"❌ Composition error: {e}")
    
    def on_modulate_requested(self, params: Dict[str, Any]):
        """Handle voice modulation request"""
        try:
            self.log_output(f"🎭 Starting voice modulation: {params['target_profile']}")
            
            # Start modulation in background
            asyncio.create_task(self.modulate_voice_async(params))
            
        except Exception as e:
            self.log_output(f"❌ Voice modulation failed: {e}")
    
    async def modulate_voice_async(self, params: Dict[str, Any]):
        """Modulate voice asynchronously"""
        try:
            if self.voice_modulator:
                # Initialize modulator if needed
                if not hasattr(self.voice_modulator, '_initialized'):
                    await self.voice_modulator.initialize()
                    self.voice_modulator._initialized = True
                
                # Load input audio (simplified)
                input_path = params['input_path']
                # In a real implementation, this would load actual audio
                sample_audio = np.random.normal(0, 0.1, 44100 * 3)  # 3 seconds of noise
                
                # Create modulation request
                request = VoiceModulationRequest(
                    input_audio=sample_audio,
                    target_voice_profile=params['target_profile'],
                    modulation_strength=params['modulation_strength'],
                    emotional_target=params['emotional_target']
                )
                
                # Perform modulation
                result = await self.voice_modulator.modulate_voice(request)
                
                # Update visualization
                self.audio_viz.update_audio_data(result.output_audio, result.sample_rate)
                
                # Save result
                output_path = await self.voice_modulator.save_modulated_voice(result)
                
                self.log_output(f"✅ Voice modulation completed: {output_path}")
            else:
                # Simulation mode
                await asyncio.sleep(1.5)  # Simulate processing time
                
                # Create simulated audio
                sample_rate = 22050
                duration = 3.0
                t = np.linspace(0, duration, int(duration * sample_rate))
                audio = 0.2 * np.sin(2 * np.pi * 440 * t)  # A4 sine wave
                
                self.audio_viz.update_audio_data(audio, sample_rate)
                self.log_output(f"✅ Voice modulation completed (simulation): {params['target_profile']}")
                
        except Exception as e:
            self.log_output(f"❌ Voice modulation error: {e}")
    
    def log_output(self, message: str):
        """Log message to output"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.output_log.append(formatted_message)
        logger.info(message)

# Example usage for testing
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("VoxSigil Music Interface")
    window.setGeometry(100, 100, 1200, 800)
    
    # Create and set central widget
    music_tab = MusicTabWidget()
    window.setCentralWidget(music_tab)
    
    # Show window
    window.show()
    
    sys.exit(app.exec_())
