"""
LibrasLive Text-to-Speech Module
Handles audio generation from translated text using gTTS and pyttsx3
"""

import os
import time
import logging
import hashlib
from io import BytesIO
import threading

# TTS Libraries
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

logger = logging.getLogger(__name__)

class TTSManager:
    """
    Text-to-Speech manager supporting both online (gTTS) and offline (pyttsx3) synthesis
    """
    
    def __init__(self, temp_dir="temp_audio", default_engine="gtts", language="pt"):
        self.temp_dir = temp_dir
        self.default_engine = default_engine
        self.language = language
        
        # Create temp directory for audio files
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize engines
        self.gtts_available = GTTS_AVAILABLE
        self.pyttsx3_available = PYTTSX3_AVAILABLE
        self.pyttsx3_engine = None
        
        # Audio cache to avoid regenerating same text
        self.audio_cache = {}
        self.cache_max_size = 50
        
        # Lock for thread safety
        self.tts_lock = threading.Lock()
        
        self.initialize_engines()
        
        # Phrase translations for better Portuguese TTS
        self.phrase_translations = {
            "OLA": "Olá",
            "OBRIGADO": "Obrigado",
            "POR_FAVOR": "Por favor",
            "DESCULPA": "Desculpa",
            "SIM": "Sim",
            "NAO": "Não",
            "BOM_DIA": "Bom dia",
            "BOA_TARDE": "Boa tarde",
            "BOA_NOITE": "Boa noite",
            "TUDO_BEM": "Tudo bem",
            "EU_TE_AMO": "Eu te amo",
            "FAMILIA": "Família",
            "CASA": "Casa",
            "TRABALHO": "Trabalho",
            "ESCOLA": "Escola",
            "AGUA": "Água",
            "COMIDA": "Comida",
            "AJUDA": "Ajuda",
            "TCHAU": "Tchau",
            "ATE_LOGO": "Até logo"
        }
    
    def initialize_engines(self):
        """Initialize TTS engines"""
        if self.gtts_available:
            logger.info("gTTS is available for online TTS")
        else:
            logger.warning("gTTS not available - install with: pip install gtts")
        
        if self.pyttsx3_available:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                
                # Configure pyttsx3 properties
                voices = self.pyttsx3_engine.getProperty('voices')
                
                # Try to find a Portuguese voice
                for voice in voices:
                    if 'portuguese' in voice.name.lower() or 'brasil' in voice.name.lower():
                        self.pyttsx3_engine.setProperty('voice', voice.id)
                        break
                
                # Set speech rate and volume
                self.pyttsx3_engine.setProperty('rate', 150)  # Speed
                self.pyttsx3_engine.setProperty('volume', 0.9)  # Volume
                
                logger.info("pyttsx3 is available for offline TTS")
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")
                self.pyttsx3_available = False
                self.pyttsx3_engine = None
        else:
            logger.warning("pyttsx3 not available - install with: pip install pyttsx3")
    
    def translate_text(self, text):
        """Translate text for better Portuguese pronunciation"""
        # Convert to uppercase for lookup
        text_upper = text.upper()
        
        # Check if it's a known phrase
        if text_upper in self.phrase_translations:
            return self.phrase_translations[text_upper]
        
        # For single letters, say "letra" + letter name
        if len(text) == 1 and text.isalpha():
            return f"Letra {text.upper()}"
        
        return text
    
    def generate_cache_key(self, text, engine):
        """Generate cache key for audio files"""
        content = f"{text}_{engine}_{self.language}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def generate_speech_gtts(self, text):
        """Generate speech using gTTS (online)"""
        try:
            # Translate text for better pronunciation
            translated_text = self.translate_text(text)
            
            # Create gTTS object
            tts = gTTS(text=translated_text, lang=self.language, slow=False)
            
            # Generate filename
            timestamp = int(time.time() * 1000)
            filename = f"gtts_{timestamp}.mp3"
            filepath = os.path.join(self.temp_dir, filename)
            
            # Save audio file
            tts.save(filepath)
            
            logger.info(f"Generated gTTS audio: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"gTTS generation failed: {e}")
            return None
    
    def generate_speech_pyttsx3(self, text):
        """Generate speech using pyttsx3 (offline)"""
        try:
            if self.pyttsx3_engine is None:
                return None
            
            # Translate text for better pronunciation
            translated_text = self.translate_text(text)
            
            # Generate filename
            timestamp = int(time.time() * 1000)
            filename = f"pyttsx3_{timestamp}.wav"
            filepath = os.path.join(self.temp_dir, filename)
            
            # Save to file
            self.pyttsx3_engine.save_to_file(translated_text, filepath)
            self.pyttsx3_engine.runAndWait()
            
            logger.info(f"Generated pyttsx3 audio: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"pyttsx3 generation failed: {e}")
            return None
    
    def generate_speech(self, text, engine=None):
        """
        Generate speech from text using specified engine
        Returns: filepath to generated audio file or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        
        # Use default engine if not specified
        if engine is None:
            engine = self.default_engine
        
        # Check cache first
        cache_key = self.generate_cache_key(text, engine)
        
        with self.tts_lock:
            if cache_key in self.audio_cache:
                cached_file = self.audio_cache[cache_key]
                if os.path.exists(cached_file):
                    logger.info(f"Using cached audio: {cached_file}")
                    return cached_file
                else:
                    # Remove from cache if file no longer exists
                    del self.audio_cache[cache_key]
        
        # Generate new audio
        audio_file = None
        
        # Try specified engine first
        if engine == "gtts" and self.gtts_available:
            audio_file = self.generate_speech_gtts(text)
        elif engine == "pyttsx3" and self.pyttsx3_available:
            audio_file = self.generate_speech_pyttsx3(text)
        
        # Fallback to available engines
        if audio_file is None:
            if engine != "gtts" and self.gtts_available:
                logger.info("Falling back to gTTS")
                audio_file = self.generate_speech_gtts(text)
            elif engine != "pyttsx3" and self.pyttsx3_available:
                logger.info("Falling back to pyttsx3")
                audio_file = self.generate_speech_pyttsx3(text)
        
        # Cache the result if successful
        if audio_file:
            with self.tts_lock:
                # Manage cache size
                if len(self.audio_cache) >= self.cache_max_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.audio_cache))
                    old_file = self.audio_cache.pop(oldest_key)
                    if os.path.exists(old_file):
                        try:
                            os.remove(old_file)
                        except:
                            pass
                
                # Add new entry to cache
                self.audio_cache[cache_key] = audio_file
        
        return audio_file
    
    def speak_text(self, text, engine=None):
        """Generate and return audio file path for text"""
        return self.generate_speech(text, engine)
    
    def clear_cache(self):
        """Clear audio cache and delete cached files"""
        with self.tts_lock:
            for filepath in self.audio_cache.values():
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        logger.error(f"Failed to delete cached file {filepath}: {e}")
            
            self.audio_cache.clear()
            logger.info("Audio cache cleared")
    
    def cleanup_old_files(self, max_age_hours=24):
        """Remove old audio files from temp directory"""
        if not os.path.exists(self.temp_dir):
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        deleted_count = 0
        for filename in os.listdir(self.temp_dir):
            filepath = os.path.join(self.temp_dir, filename)
            
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getctime(filepath)
                
                if file_age > max_age_seconds:
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete old file {filepath}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old audio files")
    
    def get_available_engines(self):
        """Return list of available TTS engines"""
        engines = []
        if self.gtts_available:
            engines.append("gtts")
        if self.pyttsx3_available:
            engines.append("pyttsx3")
        return engines
    
    def get_status(self):
        """Return TTS manager status"""
        return {
            "available_engines": self.get_available_engines(),
            "default_engine": self.default_engine,
            "language": self.language,
            "cache_size": len(self.audio_cache),
            "temp_dir": self.temp_dir,
            "gtts_available": self.gtts_available,
            "pyttsx3_available": self.pyttsx3_available
        }

# Convenience functions for quick TTS generation
def quick_tts(text, engine="gtts", language="pt"):
    """Quick TTS generation without managing a TTSManager instance"""
    tts_manager = TTSManager(default_engine=engine, language=language)
    return tts_manager.generate_speech(text)

def test_tts():
    """Test TTS functionality"""
    print("Testing TTS Manager...")
    
    tts = TTSManager()
    print(f"Status: {tts.get_status()}")
    
    # Test with different texts
    test_texts = ["A", "OBRIGADO", "OLA", "BOM_DIA"]
    
    for text in test_texts:
        print(f"Testing: {text}")
        audio_file = tts.generate_speech(text)
        if audio_file:
            print(f"  Generated: {audio_file}")
        else:
            print(f"  Failed to generate audio for: {text}")
    
    print("TTS testing complete!")

if __name__ == "__main__":
    test_tts()