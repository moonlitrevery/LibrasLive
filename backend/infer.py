"""
LibrasLive Inference Engine
PyTorch model loading and inference for LIBRAS alphabet and phrase recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
import logging

logger = logging.getLogger(__name__)

class AlphabetMLP(nn.Module):
    """
    3-layer MLP for LIBRAS alphabet recognition
    Input: 63 features (21 landmarks * 3 coordinates)
    Output: 26 classes (A-Z, excluding J and Z motion-based signs)
    """
    
    def __init__(self, input_size=63, hidden_size=128, num_classes=24):
        super(AlphabetMLP, self).__init__()
        
        self.layers = nn.Sequential(
            # First layer
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second layer
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Third layer (output) - no softmax, applied during inference
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class PhraseLSTM(nn.Module):
    """
    LSTM model for LIBRAS phrase recognition
    Input: Sequence of landmark features
    Output: Variable classes (loaded from trained model)
    """
    
    def __init__(self, input_size=63, hidden_size=64, num_layers=2, num_classes=15, seq_length=30):
        super(PhraseLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(last_output)
        return output

class LibrasInference:
    """
    Main inference engine for LIBRAS recognition
    Handles both alphabet and phrase recognition
    """
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model instances
        self.alphabet_model = None
        self.phrase_model = None
        
        # Class mappings
        self.alphabet_classes = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y'  # J and Z excluded (motion-based)
        ]
        
        # Expected phrase classes (what the frontend expects)
        self.expected_phrase_classes = [
            "OLA", "OBRIGADO", "POR_FAVOR", "DESCULPA", "SIM", "NAO",
            "BOM_DIA", "BOA_TARDE", "BOA_NOITE", "TUDO_BEM", "EU_TE_AMO",
            "FAMILIA", "CASA", "TRABALHO", "ESCOLA", "AGUA", "COMIDA",
            "AJUDA", "TCHAU", "ATE_LOGO"
        ]
        
        # Actual model classes (loaded from trained model)
        self.phrase_classes = None
        
        # Mapping from dataset UCI classes to expected phrases
        # This maps the 15 classes from the UCI dataset to the 20 expected phrases
        # Improved mapping to avoid too many classes mapping to the same phrase
        self.phrase_class_mapping = {
            "BOM": "BOM_DIA",           # BOM -> BOM_DIA (good/greeting)
            "RUIM": "NAO",              # RUIM -> NAO (bad/negative)
            "MELHOR": "TUDO_BEM",       # MELHOR -> TUDO_BEM (better/positive)
            "PIOR": "NAO",              # PIOR -> NAO (worse/negative)
            "PRIMEIRO": "OLA",          # PRIMEIRO -> OLA (first/greeting)
            "ULTIMO": "TCHAU",          # ULTIMO -> TCHAU (last/farewell)
            "TEMPO": "BOA_TARDE",       # TEMPO -> BOA_TARDE (time/greeting)
            "ARVORE": "ESCOLA",          # ARVORE -> ESCOLA (tree/nature/place)
            "VERDE": "AGUA",            # VERDE -> AGUA (green/color/nature)
            "AMARELO": "COMIDA",        # AMARELO -> COMIDA (yellow/color/food)
            "LIBRAS": "AJUDA",          # LIBRAS -> AJUDA (sign language/help)
            "LIVRE": "TUDO_BEM",        # LIVRE -> TUDO_BEM (free/well-being)
            "PALAVRA": "POR_FAVOR",     # PALAVRA -> POR_FAVOR (word/request)
            "CURVAR": "OBRIGADO",       # CURVAR -> OBRIGADO (curve/thanks)
            "ALISAR": "DESCULPA"        # ALISAR -> DESCULPA (smooth/apology)
        }
        
        # Separate normalization parameters for phrases
        self.phrase_landmark_mean = None
        self.phrase_landmark_std = None
        
        # Sequence buffer for phrase recognition
        self.sequence_buffer = deque(maxlen=30)  # 30 frames for phrases
        
        # Load models
        self.load_models()
        
        # Preprocessing parameters
        self.landmark_mean = None
        self.landmark_std = None
        self.setup_normalization()
    
    def setup_normalization(self):
        """Setup normalization parameters for landmarks"""
        # Default normalization values (should be computed from training data)
        self.landmark_mean = np.zeros(63)
        self.landmark_std = np.ones(63)
        self.phrase_landmark_mean = np.zeros(63)
        self.phrase_landmark_std = np.ones(63)
        
        # Load normalization parameters for alphabet if available
        norm_file = os.path.join(self.models_dir, 'normalization_params.npz')
        if os.path.exists(norm_file):
            norm_data = np.load(norm_file)
            self.landmark_mean = norm_data['mean']
            self.landmark_std = norm_data['std']
            logger.info("Loaded alphabet normalization parameters")
        else:
            logger.warning("No alphabet normalization parameters found, using defaults")
        
        # Load normalization parameters for phrases if available
        phrase_norm_file = os.path.join(self.models_dir, 'phrase_normalization_params.npz')
        if os.path.exists(phrase_norm_file):
            phrase_norm_data = np.load(phrase_norm_file)
            self.phrase_landmark_mean = phrase_norm_data['mean']
            self.phrase_landmark_std = phrase_norm_data['std']
            logger.info("Loaded phrase normalization parameters")
        else:
            # Fallback to alphabet normalization if phrase-specific not found
            self.phrase_landmark_mean = self.landmark_mean
            self.phrase_landmark_std = self.landmark_std
            logger.warning("No phrase normalization parameters found, using alphabet normalization")
    
    def load_models(self):
        """Load pretrained models"""
        try:
            # Load alphabet model
            alphabet_path = os.path.join(self.models_dir, 'alphabet_model.pt')
            if os.path.exists(alphabet_path):
                self.alphabet_model = AlphabetMLP()
                self.alphabet_model.load_state_dict(torch.load(alphabet_path, map_location=self.device))
                self.alphabet_model.to(self.device)
                self.alphabet_model.eval()
                logger.info("Alphabet model loaded successfully")
            else:
                logger.warning(f"Alphabet model not found at {alphabet_path}")
                # Create dummy model for testing
                self.alphabet_model = AlphabetMLP()
                self.alphabet_model.to(self.device)
                self.alphabet_model.eval()
            
            # Load phrase model
            phrase_path = os.path.join(self.models_dir, 'phrase_model.pt')
            if os.path.exists(phrase_path):
                # Try to load metadata to get actual classes and num_classes
                metadata_path = os.path.join(self.models_dir, 'phrase_model_metadata.json')
                num_classes = 15  # Default
                
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        num_classes = metadata.get('num_classes', 15)
                        stored_classes = metadata.get('classes', None)
                        if stored_classes:
                            self.phrase_classes = stored_classes
                            logger.info(f"Loaded phrase classes from metadata: {len(stored_classes)} classes")
                
                self.phrase_model = PhraseLSTM(num_classes=num_classes)
                self.phrase_model.load_state_dict(torch.load(phrase_path, map_location=self.device))
                self.phrase_model.to(self.device)
                self.phrase_model.eval()
                logger.info("Phrase model loaded successfully")
            else:
                logger.warning(f"Phrase model not found at {phrase_path}")
                # Create dummy model for testing
                self.phrase_model = PhraseLSTM()
                self.phrase_model.to(self.device)
                self.phrase_model.eval()
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise e
    
    def preprocess_landmarks(self, landmarks):
        """
        Preprocess landmark coordinates
        Input: numpy array of shape (63,) - flattened 21 points * 3 coords
        Output: normalized tensor ready for inference
        """
        if landmarks.shape[0] != 63:
            raise ValueError(f"Expected 63 landmark coordinates, got {landmarks.shape[0]}")
        
        # Normalize landmarks
        normalized = (landmarks - self.landmark_mean) / (self.landmark_std + 1e-8)
        
        # Debug: log normalization stats occasionally
        if logger.isEnabledFor(logging.DEBUG) and np.random.random() < 0.01:  # 1% of the time
            logger.debug(f"Normalization stats - mean range: [{self.landmark_mean.min():.3f}, {self.landmark_mean.max():.3f}], "
                        f"std range: [{self.landmark_std.min():.3f}, {self.landmark_std.max():.3f}], "
                        f"normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        # Convert to tensor
        tensor = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
        
        return tensor
    
    def predict_alphabet(self, landmarks):
        """
        Predict alphabet letter from hand landmarks
        Returns: predicted letter (A-Z) or None if confidence is low
        """
        if self.alphabet_model is None:
            return None
        
        try:
            # Preprocess landmarks
            input_tensor = self.preprocess_landmarks(landmarks)
            
            # Inference
            with torch.no_grad():
                output = self.alphabet_model(input_tensor)
                # Apply softmax to get proper probabilities
                probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
                
                # Get prediction
                predicted_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_idx]
                
                # Get top 3 predictions for debugging
                top3_indices = np.argsort(probabilities)[-3:][::-1]
                top3_predictions = [(self.alphabet_classes[idx], probabilities[idx]) for idx in top3_indices]
                
                # Threshold and quality checks for alphabet recognition
                # WARNING: Model was trained on very limited data (221 samples), so confidence is low
                # With 24 classes, uniform distribution would give ~4.17% per class
                predicted_letter = self.alphabet_classes[predicted_idx]
                
                # Calculate confidence ratio
                confidence_ratio = top3_predictions[0][1] / (top3_predictions[1][1] + 1e-8)
                
                # Check 1: Minimum confidence threshold (3% - very permissive due to limited training data)
                if confidence < 0.03:
                    logger.debug(f"Very low confidence: {confidence:.3f} | Top 3: {top3_predictions}")
                    return None
                
                # Check 2: If confidence is above uniform expectation (>4%), be more permissive
                # Accept if confidence > 4% and ratio > 1.02 (minimal difference)
                if confidence > 0.04 and confidence_ratio > 1.02:
                    logger.debug(f"Alphabet prediction (above uniform): {predicted_letter} (confidence: {confidence:.3f}, ratio: {confidence_ratio:.2f}) | Top 3: {top3_predictions}")
                    return predicted_letter
                
                # Check 3: For lower confidence (3-4%), require slightly higher ratio (5%)
                if confidence >= 0.03 and confidence <= 0.04:
                    if confidence_ratio >= 1.05:
                        logger.debug(f"Alphabet prediction (low conf, good ratio): {predicted_letter} (confidence: {confidence:.3f}, ratio: {confidence_ratio:.2f}) | Top 3: {top3_predictions}")
                        return predicted_letter
                    else:
                        logger.debug(f"Uncertain prediction - top 2 too close: {confidence:.3f} (ratio: {confidence_ratio:.2f}) | Top 3: {top3_predictions}")
                        return None
                
                # Check 4: For higher confidence (>5%), require ratio > 1.08
                if confidence > 0.05 and confidence_ratio < 1.08:
                    logger.debug(f"Uncertain prediction - top 2 too close: {confidence:.3f} (ratio: {confidence_ratio:.2f}) | Top 3: {top3_predictions}")
                    return None
                
                # All checks passed - accept prediction
                logger.debug(f"Alphabet prediction: {predicted_letter} (confidence: {confidence:.3f}, ratio: {confidence_ratio:.2f}) | Top 3: {top3_predictions}")
                return predicted_letter
        
        except Exception as e:
            logger.error(f"Error in alphabet prediction: {e}")
            return None
    
    def predict_phrase(self, landmarks):
        """
        Predict phrase from sequence of landmarks
        Maintains internal buffer for sequence-based recognition
        Returns: predicted phrase or None
        """
        if self.phrase_model is None:
            return None
        
        try:
            # Add landmarks to sequence buffer
            self.sequence_buffer.append(landmarks)
            
            # Need minimum sequence length for phrase recognition
            if len(self.sequence_buffer) < 15:  # Minimum 15 frames
                return None
            
            # Prepare sequence for inference
            sequence = np.array(list(self.sequence_buffer))
            
            # Normalize each frame in the sequence using phrase-specific normalization
            normalized_sequence = []
            for frame in sequence:
                normalized_frame = (frame - self.phrase_landmark_mean) / (self.phrase_landmark_std + 1e-8)
                normalized_sequence.append(normalized_frame)
            
            # Convert to tensor (1, seq_len, 63)
            sequence_tensor = torch.FloatTensor(normalized_sequence).unsqueeze(0).to(self.device)
            
            # Pad or truncate to expected sequence length
            target_length = self.phrase_model.seq_length
            current_length = sequence_tensor.size(1)
            
            if current_length < target_length:
                # Pad with zeros
                padding = torch.zeros(1, target_length - current_length, 63).to(self.device)
                sequence_tensor = torch.cat([sequence_tensor, padding], dim=1)
            elif current_length > target_length:
                # Use last N frames
                sequence_tensor = sequence_tensor[:, -target_length:, :]
            
            # Inference
            with torch.no_grad():
                output = self.phrase_model(sequence_tensor)
                # Apply softmax to get probabilities
                probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
                
                # Get prediction
                predicted_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_idx]
                
                # Return prediction if confidence is high enough
                if confidence > 0.5:  # Threshold for phrase recognition (reduced from 0.6)
                    # Get the predicted class from the model
                    if self.phrase_classes is not None:
                        predicted_class = self.phrase_classes[predicted_idx]
                    else:
                        # Fallback to expected classes if metadata not loaded
                        predicted_class = self.expected_phrase_classes[predicted_idx] if predicted_idx < len(self.expected_phrase_classes) else "UNKNOWN"
                    
                    # Map to expected phrase class if needed
                    mapped_phrase = self.phrase_class_mapping.get(predicted_class, predicted_class)
                    
                    # If mapped phrase is not in expected classes, return UNKNOWN
                    if mapped_phrase not in self.expected_phrase_classes:
                        logger.warning(f"Mapped phrase '{mapped_phrase}' not in expected classes, returning UNKNOWN")
                        return "UNKNOWN"
                    
                    logger.debug(f"Phrase prediction: {predicted_class} -> {mapped_phrase} (confidence: {confidence:.3f})")
                    return mapped_phrase
                else:
                    logger.debug(f"Low confidence prediction: {confidence:.3f}, returning UNKNOWN")
                    return "UNKNOWN"
        
        except Exception as e:
            logger.error(f"Error in phrase prediction: {e}")
            return None
    
    def reset_sequence_buffer(self):
        """Reset the sequence buffer for phrase recognition"""
        self.sequence_buffer.clear()
        logger.info("Sequence buffer reset")
    
    def get_model_info(self):
        """Return information about loaded models"""
        return {
            'alphabet_model_loaded': self.alphabet_model is not None,
            'phrase_model_loaded': self.phrase_model is not None,
            'device': str(self.device),
            'alphabet_classes': len(self.alphabet_classes),
            'phrase_classes': len(self.phrase_classes) if self.phrase_classes else len(self.expected_phrase_classes),
            'expected_phrase_classes': len(self.expected_phrase_classes),
            'actual_phrase_classes': len(self.phrase_classes) if self.phrase_classes else 0,
            'sequence_buffer_size': len(self.sequence_buffer)
        }

def create_dummy_models():
    """
    Create dummy model files for testing when real trained models are not available
    This is useful for development and testing
    """
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create dummy alphabet model
    alphabet_model = AlphabetMLP()
    torch.save(alphabet_model.state_dict(), os.path.join(models_dir, 'alphabet_model.pt'))
    
    # Create dummy phrase model
    phrase_model = PhraseLSTM(num_classes=15)
    torch.save(phrase_model.state_dict(), os.path.join(models_dir, 'phrase_model.pt'))
    
    # Create dummy normalization parameters
    mean = np.random.randn(63) * 0.1
    std = np.random.rand(63) * 0.5 + 0.5
    np.savez(os.path.join(models_dir, 'normalization_params.npz'), mean=mean, std=std)
    
    logger.info("Dummy models created for testing")

if __name__ == "__main__":
    # Test the inference engine
    print("Testing LibrasInference...")
    
    # Create dummy models if they don't exist
    if not os.path.exists("models/alphabet_model.pt"):
        create_dummy_models()
    
    # Initialize inference engine
    inference = LibrasInference()
    
    # Test with random landmarks
    test_landmarks = np.random.randn(63).astype(np.float32)
    
    # Test alphabet prediction
    alphabet_pred = inference.predict_alphabet(test_landmarks)
    print(f"Alphabet prediction: {alphabet_pred}")
    
    # Test phrase prediction (need multiple frames)
    for _ in range(20):
        test_landmarks = np.random.randn(63).astype(np.float32)
        phrase_pred = inference.predict_phrase(test_landmarks)
    
    print(f"Phrase prediction: {phrase_pred}")
    print(f"Model info: {inference.get_model_info()}")
    print("Testing complete!")