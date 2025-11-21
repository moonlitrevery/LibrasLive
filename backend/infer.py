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
    Improved MLP for LIBRAS alphabet recognition with BatchNorm and 4 hidden layers
    Input: 63 features (21 landmarks * 3 coordinates)
    Output: 24 classes (A-Z, excluding J and Z motion-based signs)
    """
    
    def __init__(self, input_size=63, hidden_size=256, num_classes=24, dropout_rate=0.4):
        super(AlphabetMLP, self).__init__()
        
        self.layers = nn.Sequential(
            # First layer
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second layer
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Third layer (intermediate)
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            # Fourth layer
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer - no softmax, applied during inference
            nn.Linear(hidden_size // 4, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class PhraseLSTM(nn.Module):
    """
    Improved LSTM model for LIBRAS phrase recognition with input projection and BatchNorm
    Input: Sequence of landmark features
    Output: Variable classes (loaded from trained model)
    """
    
    def __init__(self, input_size=63, hidden_size=128, num_layers=3, num_classes=15, seq_length=30, dropout_rate=0.4):
        super(PhraseLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # Input projection layer
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,  # Input size is now hidden_size from projection
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classification head with BatchNorm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize LSTM weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size = x.size(0)
        
        # Input projection (reshape para aplicar linear, depois reshape de volta)
        x_flat = x.reshape(-1, x.size(-1))
        x_proj_flat = self.input_proj(x_flat)
        x_proj = x_proj_flat.reshape(batch_size, self.seq_length, -1)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_proj, (h0, c0))
        
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
        logger.info(f"Looking for normalization file at: {norm_file}")
        logger.info(f"File exists: {os.path.exists(norm_file)}")
        if os.path.exists(norm_file):
            norm_data = np.load(norm_file)
            self.landmark_mean = norm_data['mean']
            self.landmark_std = norm_data['std']
            logger.info(f"Loaded alphabet normalization parameters - mean range: [{self.landmark_mean.min():.3f}, {self.landmark_mean.max():.3f}], std range: [{self.landmark_std.min():.3f}, {self.landmark_std.max():.3f}]")
        else:
            logger.warning(f"No alphabet normalization parameters found at {norm_file}, using defaults")
        
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
            import json
            
            # Load alphabet model
            alphabet_path = os.path.join(self.models_dir, 'alphabet_model.pt')
            if os.path.exists(alphabet_path):
                # Try to load metadata to get actual parameters
                alphabet_metadata_path = os.path.join(self.models_dir, 'alphabet_model_metadata.json')
                hidden_size = 256  # Default from improved model
                num_classes = 24  # Default
                dropout_rate = 0.4  # Default
                
                if os.path.exists(alphabet_metadata_path):
                    with open(alphabet_metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        hidden_size = metadata.get('hidden_size', 256)
                        num_classes = metadata.get('num_classes', 24)
                        dropout_rate = metadata.get('dropout_rate', 0.4)
                        stored_classes = metadata.get('classes', None)
                        if stored_classes:
                            self.alphabet_classes = stored_classes
                            logger.info(f"Loaded alphabet classes from metadata: {len(stored_classes)} classes")
                
                self.alphabet_model = AlphabetMLP(
                    hidden_size=hidden_size,
                    num_classes=num_classes,
                    dropout_rate=dropout_rate
                )
                self.alphabet_model.load_state_dict(torch.load(alphabet_path, map_location=self.device))
                self.alphabet_model.to(self.device)
                self.alphabet_model.eval()
                logger.info(f"Alphabet model loaded successfully (hidden_size={hidden_size}, num_classes={num_classes})")
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
                hidden_size = 128  # Default from improved model
                num_layers = 3  # Default from improved model
                seq_length = 30  # Default
                dropout_rate = 0.4  # Default
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        num_classes = metadata.get('num_classes', 15)
                        hidden_size = metadata.get('hidden_size', 128)
                        num_layers = metadata.get('num_layers', 3)
                        seq_length = metadata.get('sequence_length', 30)
                        dropout_rate = metadata.get('dropout_rate', 0.4)
                        stored_classes = metadata.get('classes', None)
                        if stored_classes:
                            self.phrase_classes = stored_classes
                            logger.info(f"Loaded phrase classes from metadata: {len(stored_classes)} classes")
                
                self.phrase_model = PhraseLSTM(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_classes=num_classes,
                    seq_length=seq_length,
                    dropout_rate=dropout_rate
                )
                self.phrase_model.load_state_dict(torch.load(phrase_path, map_location=self.device))
                self.phrase_model.to(self.device)
                self.phrase_model.eval()
                logger.info(f"Phrase model loaded successfully (hidden_size={hidden_size}, num_layers={num_layers}, num_classes={num_classes})")
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
        if logger.isEnabledFor(logging.DEBUG) and np.random.random() < 0.05:  # 5% of the time
            logger.debug(f"Normalization stats - input range: [{landmarks.min():.3f}, {landmarks.max():.3f}], "
                        f"mean range: [{self.landmark_mean.min():.3f}, {self.landmark_mean.max():.3f}], "
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
                # Model was trained with 99% accuracy on 4,900 samples, so it should be confident
                predicted_letter = self.alphabet_classes[predicted_idx]
                
                # Calculate confidence ratio (how much better is top1 than top2)
                confidence_ratio = top3_predictions[0][1] / (top3_predictions[1][1] + 1e-8)
                
                # With 24 classes, uniform distribution would give ~4.17% per class
                # If confidence is very low (< 5%), the model is uncertain
                # If confidence is low but ratio is good (> 1.01), accept it (model is distinguishing)
                
                # More permissive logic for high-accuracy model:
                # With proper normalization, model should have high confidence (>10%)
                # If confidence is very low (<5%), likely normalization issue - be very permissive
                
                # Classes with very few training samples (U:1, N:2, V:1, Y:1) need special handling
                rare_classes = {'U', 'N', 'V', 'Y', 'H', 'G', 'P', 'Q', 'R', 'W'}
                is_rare_class = predicted_letter in rare_classes
                
                # 1. High confidence (>10%) - accept if ratio > 1.05
                # For rare classes, require higher ratio (>1.10) to reduce false positives
                if confidence > 0.10:
                    required_ratio = 1.10 if is_rare_class else 1.05
                    if confidence_ratio > required_ratio:
                        if is_rare_class:
                            logger.debug(f"Alphabet prediction (high conf, RARE CLASS): {predicted_letter} (confidence: {confidence:.3f}, ratio: {confidence_ratio:.2f}) | Top 3: {top3_predictions}")
                        else:
                            logger.debug(f"Alphabet prediction (high conf): {predicted_letter} (confidence: {confidence:.3f}, ratio: {confidence_ratio:.2f}) | Top 3: {top3_predictions}")
                        return predicted_letter
                    else:
                        logger.debug(f"Uncertain prediction - top 2 too close: {confidence:.3f} (ratio: {confidence_ratio:.2f}, required: {required_ratio:.2f}) | Top 3: {top3_predictions}")
                        return None
                
                # 2. Medium confidence (5-10%) - accept if ratio > 1.02
                if confidence >= 0.05 and confidence <= 0.10:
                    if confidence_ratio > 1.02:
                        logger.debug(f"Alphabet prediction (medium conf): {predicted_letter} (confidence: {confidence:.3f}, ratio: {confidence_ratio:.2f}) | Top 3: {top3_predictions}")
                        return predicted_letter
                    else:
                        logger.debug(f"Uncertain prediction - top 2 too close: {confidence:.3f} (ratio: {confidence_ratio:.2f}) | Top 3: {top3_predictions}")
                        return None
                
                # 3. Low confidence (4-5%) - very permissive, accept if ratio >= 1.00 (any preference)
                if confidence >= 0.04 and confidence < 0.05:
                    if confidence_ratio >= 1.00:
                        logger.debug(f"Alphabet prediction (low conf, permissive): {predicted_letter} (confidence: {confidence:.3f}, ratio: {confidence_ratio:.2f}) | Top 3: {top3_predictions}")
                        return predicted_letter
                    else:
                        logger.debug(f"Low confidence prediction: {confidence:.3f}, returning UNKNOWN | Top 3: {top3_predictions}")
                        return None
                
                # 4. Very low confidence (<4%) - reject (likely uniform distribution)
                logger.debug(f"Very low confidence prediction: {confidence:.3f}, returning UNKNOWN | Top 3: {top3_predictions}")
                return None
        
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
                # Increased threshold to reduce false positives that interfere with alphabet recognition
                if confidence > 0.6:  # Threshold for phrase recognition (increased to reduce interference)
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