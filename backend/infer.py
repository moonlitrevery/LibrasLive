"""
LibrasLive Inference Engine
PyTorch model loading and inference for LIBRAS alphabet and phrase recognition
"""

import torch
import torch.nn as nn
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
            
            # Third layer (output)
            nn.Linear(hidden_size // 2, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.layers(x)

class PhraseLSTM(nn.Module):
    """
    LSTM model for LIBRAS phrase recognition
    Input: Sequence of landmark features
    Output: 20 classes (common phrases)
    """
    
    def __init__(self, input_size=63, hidden_size=64, num_layers=2, num_classes=20, seq_length=30):
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
            nn.Linear(hidden_size // 2, num_classes),
            nn.Softmax(dim=1)
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
        
        self.phrase_classes = [
            "OLA", "OBRIGADO", "POR_FAVOR", "DESCULPA", "SIM", "NAO",
            "BOM_DIA", "BOA_TARDE", "BOA_NOITE", "TUDO_BEM", "EU_TE_AMO",
            "FAMILIA", "CASA", "TRABALHO", "ESCOLA", "AGUA", "COMIDA",
            "AJUDA", "TCHAU", "ATE_LOGO"
        ]
        
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
        
        # Load normalization parameters if available
        norm_file = os.path.join(self.models_dir, 'normalization_params.npz')
        if os.path.exists(norm_file):
            norm_data = np.load(norm_file)
            self.landmark_mean = norm_data['mean']
            self.landmark_std = norm_data['std']
            logger.info("Loaded normalization parameters")
        else:
            logger.warning("No normalization parameters found, using defaults")
    
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
                self.phrase_model = PhraseLSTM()
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
                probabilities = output.cpu().numpy()[0]
                
                # Get prediction
                predicted_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_idx]
                
                # Return prediction if confidence is high enough
                if confidence > 0.3:  # Threshold for alphabet recognition
                    return self.alphabet_classes[predicted_idx]
                else:
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
            
            # Normalize each frame in the sequence
            normalized_sequence = []
            for frame in sequence:
                normalized_frame = (frame - self.landmark_mean) / (self.landmark_std + 1e-8)
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
                probabilities = output.cpu().numpy()[0]
                
                # Get prediction
                predicted_idx = np.argmax(probabilities)
                confidence = probabilities[predicted_idx]
                
                # Return prediction if confidence is high enough
                if confidence > 0.4:  # Higher threshold for phrases
                    return self.phrase_classes[predicted_idx]
                else:
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
            'phrase_classes': len(self.phrase_classes),
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
    phrase_model = PhraseLSTM()
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