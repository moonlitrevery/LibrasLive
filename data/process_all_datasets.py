#!/usr/bin/env python
"""
Script para processar TODOS os datasets disponíveis e combinar.

Este script processa:
1. Sign Language MNIST (já processado)
2. Leap Gesture Recognition (leapGestRecog)
3. Sign Language Digits (X.npy, Y.npy)
4. Combina tudo em um dataset maior

Execute: python data/process_all_datasets.py
"""

import cv2
import numpy as np
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Tuple, Optional
from collections import Counter

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("MediaPipe é necessário. Instale com: pip install mediapipe")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Constantes (mesmas do prepare_datasets.py)
ALPHABET_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y",
]
LANDMARK_DIM = 63
RANDOM_STATE = 42
SIGN_TRAIN_CSV = RAW_DIR / "sign_language_mnist" / "sign_mnist_train.csv"
SIGN_TEST_CSV = RAW_DIR / "sign_language_mnist" / "sign_mnist_test.csv"

# Funções auxiliares
def maybe_stratify(labels, split_name, num_classes):
    """Retorna labels para estratificação se possível."""
    from collections import Counter
    counts = Counter(labels)
    min_count = min(counts.values())
    if min_count < 2:
        logger.warning(f"⚠ Estratificação não possível para {split_name}: classe menos frequente tem {min_count} amostras")
        return None
    return labels

def load_sign_language_mnist(csv_path):
    """Carrega Sign Language MNIST do CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].astype(np.int64).to_numpy()
    pixels = df.iloc[:, 1:].to_numpy(dtype=np.uint8)
    images = pixels.reshape(-1, 28, 28)
    return images, labels

def remap_sign_mnist_labels(labels):
    """Mapeia labels do Sign Language MNIST para índices do alfabeto."""
    label_map = {}
    current_idx = 0
    for ascii_code in range(26):
        letter = chr(ord("A") + ascii_code)
        if letter in ("J", "Z"):
            continue
        label_map[ascii_code] = current_idx
        current_idx += 1
    return np.array([label_map[label] for label in labels], dtype=np.int64)

def extract_landmarks_from_images(images, labels, detector):
    """Extrai landmarks usando MediaPipe."""
    features = []
    kept_labels = []
    skipped = 0
    
    for image, label in tqdm(zip(images, labels), total=len(images), desc="MediaPipe landmarks"):
        # Converter imagem 28x28 para RGB 224x224
        image_resized = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
        
        results = detector.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            skipped += 1
            continue
        
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        
        if len(coords) != LANDMARK_DIM:
            skipped += 1
            continue
        
        features.append(np.array(coords, dtype=np.float32))
        kept_labels.append(label)
    
    logger.info(f"Extraídos {len(features)}/{len(images)} amostras (puladas {skipped} sem landmarks)")
    return np.stack(features), np.array(kept_labels, dtype=np.int64)

mp_hands = mp.solutions.hands

# Mapeamento de gestos do leapGestRecog para letras do alfabeto
GESTURE_TO_LETTER = {
    "02_l": "L",
    "09_c": "C",
    "01_palm": "B",  # Palm pode ser B
    "08_palm_moved": "B",
    "03_fist": "A",  # Fist pode ser A
    "04_fist_moved": "A",
    "05_thumb": "T",  # Thumb pode ser T
    "06_index": "I",  # Index pode ser I
    "07_ok": "O",  # OK pode ser O
    "10_down": "D",  # Down pode ser D
}

def extract_landmarks_from_image(image_path: Path, hands: mp_hands.Hands) -> Optional[np.ndarray]:
    """Extrai landmarks da mão de uma imagem."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(landmarks, dtype=np.float32)

def process_leap_gesture_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Processa o dataset leapGestRecog."""
    logger.info("=" * 70)
    logger.info("PROCESSANDO LEAP GESTURE RECOGNITION")
    logger.info("=" * 70)
    
    leap_dir = RAW_DIR / "leapGestRecog"
    
    if not leap_dir.exists():
        logger.warning("⚠ Diretório leapGestRecog não encontrado")
        return np.array([]), np.array([])
    
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    features = []
    labels = []
    
    # Processar todas as pastas (00-09)
    for folder_num in range(10):
        folder_path = leap_dir / f"{folder_num:02d}"
        
        if not folder_path.exists():
            continue
        
        logger.info(f"Processando pasta {folder_num:02d}...")
        
        for gesture_name, letter in GESTURE_TO_LETTER.items():
            gesture_dir = folder_path / gesture_name
            
            if not gesture_dir.exists():
                continue
            
            # Buscar imagens
            image_files = list(gesture_dir.glob("*.png"))
            
            if not image_files:
                continue
            
            # Mapear para índice da classe
            if letter not in ALPHABET_CLASSES:
                continue
            
            class_idx = ALPHABET_CLASSES.index(letter)
            
            logger.info(f"  {gesture_name} -> {letter}: {len(image_files)} imagens")
            
            for image_path in tqdm(image_files, desc=f"    {gesture_name}", leave=False):
                landmarks = extract_landmarks_from_image(image_path, hands)
                if landmarks is not None:
                    features.append(landmarks)
                    labels.append(class_idx)
    
    hands.close()
    
    if len(features) == 0:
        logger.warning("⚠ Nenhum landmark extraído do leapGestRecog")
        return np.array([]), np.array([])
    
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    logger.info(f"✓ Leap Gesture processado: {len(features)} amostras")
    return features, labels

def process_sign_language_digits() -> Tuple[np.ndarray, np.ndarray]:
    """Processa o dataset Sign Language Digits."""
    logger.info("=" * 70)
    logger.info("PROCESSANDO SIGN LANGUAGE DIGITS")
    logger.info("=" * 70)
    
    digits_dir = RAW_DIR / "Sign-language-digits-dataset"
    X_path = digits_dir / "X.npy"
    Y_path = digits_dir / "Y.npy"
    
    if not X_path.exists() or not Y_path.exists():
        logger.warning("⚠ Arquivos X.npy ou Y.npy não encontrados")
        return np.array([]), np.array([])
    
    try:
        X = np.load(X_path)
        Y = np.load(Y_path)
        
        logger.info(f"Dataset carregado: X.shape={X.shape}, Y.shape={Y.shape}")
        
        # X provavelmente são imagens (n_samples, height, width)
        # Y são labels (n_samples,)
        
        # Converter para formato de imagem para MediaPipe
        if len(X.shape) == 3:
            # (n_samples, height, width) -> imagens
            images = X
        elif len(X.shape) == 4:
            # (n_samples, height, width, channels) -> pegar primeiro canal
            images = X[:, :, :, 0] if X.shape[3] > 1 else X.squeeze()
        else:
            logger.warning("⚠ Formato de X não reconhecido")
            return np.array([]), np.array([])
        
        # Normalizar para 0-255
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)
        
        # Mapear dígitos para letras (0-9 -> algumas letras)
        # 0 -> O, 1 -> I, 2 -> Z (não temos), 3 -> E, etc.
        digit_to_letter = {
            0: "O", 1: "I", 3: "E", 4: "H", 5: "S",
            # Adicionar mais mapeamentos se necessário
        }
        
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        features = []
        labels = []
        
        for idx, (image, digit) in enumerate(zip(images, Y)):
            if digit not in digit_to_letter:
                continue
            
            letter = digit_to_letter[digit]
            if letter not in ALPHABET_CLASSES:
                continue
            
            # Converter imagem para RGB
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image
            
            # Redimensionar se necessário
            if image_rgb.shape[0] < 224 or image_rgb.shape[1] < 224:
                image_rgb = cv2.resize(image_rgb, (224, 224))
            
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_coords = []
                for lm in hand_landmarks.landmark:
                    landmark_coords.extend([lm.x, lm.y, lm.z])
                
                if len(landmark_coords) == LANDMARK_DIM:
                    features.append(np.array(landmark_coords, dtype=np.float32))
                    labels.append(ALPHABET_CLASSES.index(letter))
        
        hands.close()
        
        if len(features) == 0:
            logger.warning("⚠ Nenhum landmark extraído do Sign Language Digits")
            return np.array([]), np.array([])
        
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        logger.info(f"✓ Sign Language Digits processado: {len(features)} amostras")
        return features, labels
        
    except Exception as e:
        logger.error(f"✗ Erro ao processar Sign Language Digits: {e}")
        return np.array([]), np.array([])

def load_existing_sign_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """Carrega e processa Sign Language MNIST."""
    logger.info("=" * 70)
    logger.info("PROCESSANDO SIGN LANGUAGE MNIST")
    logger.info("=" * 70)
    
    if not SIGN_TRAIN_CSV.exists() or not SIGN_TEST_CSV.exists():
        logger.warning("⚠ Sign Language MNIST não encontrado")
        return np.array([]), np.array([])
    
    train_images, train_labels = load_sign_language_mnist(SIGN_TRAIN_CSV)
    test_images, test_labels = load_sign_language_mnist(SIGN_TEST_CSV)
    
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    
    X_train_raw, y_train_raw = extract_landmarks_from_images(train_images, train_labels, hands)
    X_test_raw, y_test_raw = extract_landmarks_from_images(test_images, test_labels, hands)
    hands.close()
    
    y_train = remap_sign_mnist_labels(y_train_raw)
    y_test = remap_sign_mnist_labels(y_test_raw)
    
    # Combinar train e test
    X_all = np.vstack([X_train_raw, X_test_raw])
    y_all = np.concatenate([y_train, y_test])
    
    logger.info(f"✓ Sign Language MNIST processado: {len(X_all)} amostras")
    return X_all, y_all

def main():
    """Função principal."""
    logger.info("=" * 70)
    logger.info("PROCESSAMENTO COMPLETO DE TODOS OS DATASETS")
    logger.info("=" * 70)
    logger.info("")
    
    all_features = []
    all_labels = []
    
    # 1. Sign Language MNIST
    X_smnist, y_smnist = load_existing_sign_mnist()
    if len(X_smnist) > 0:
        all_features.append(X_smnist)
        all_labels.append(y_smnist)
    
    # 2. Leap Gesture Recognition
    X_leap, y_leap = process_leap_gesture_dataset()
    if len(X_leap) > 0:
        all_features.append(X_leap)
        all_labels.append(y_leap)
    
    # 3. Sign Language Digits
    X_digits, y_digits = process_sign_language_digits()
    if len(X_digits) > 0:
        all_features.append(X_digits)
        all_labels.append(y_digits)
    
    if not all_features:
        logger.error("✗ Nenhum dataset processado!")
        return
    
    # Combinar todos
    logger.info("")
    logger.info("Combinando todos os datasets...")
    X_combined = np.vstack(all_features)
    y_combined = np.concatenate(all_labels)
    
    logger.info(f"✓ Total combinado: {len(X_combined)} amostras")
    
    # Distribuição de classes
    class_counts = Counter(y_combined)
    logger.info("")
    logger.info("Distribuição de classes:")
    for class_idx, class_name in enumerate(ALPHABET_CLASSES):
        count = class_counts.get(class_idx, 0)
        if count > 0:
            logger.info(f"  {class_name}: {count} amostras")
    
    # Split train/val/test (60/20/20)
    stratify_labels = maybe_stratify(y_combined, "train/temp", len(ALPHABET_CLASSES))
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_combined,
        y_combined,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=stratify_labels
    )
    
    stratify_temp = maybe_stratify(y_temp, "val/test", len(ALPHABET_CLASSES))
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=stratify_temp
    )
    
    logger.info(f"✓ Split: {len(X_train)} treino, {len(X_val)} val, {len(X_test)} teste")
    
    # Normalizar
    logger.info("Normalizando features...")
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    # Salvar
    scaler_path = PROCESSED_DIR / "alphabet_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    np.save(PROCESSED_DIR / "alphabet_landmarks_train.npy", X_train_norm)
    np.save(PROCESSED_DIR / "alphabet_landmarks_val.npy", X_val_norm)
    np.save(PROCESSED_DIR / "alphabet_landmarks_test.npy", X_test_norm)
    
    np.savez(
        PROCESSED_DIR / "alphabet_dataset.npz",
        X_train=X_train_norm,
        y_train=y_train,
        X_val=X_val_norm,
        y_val=y_val,
        X_test=X_test_norm,
        y_test=y_test,
        mean=scaler.mean_,
        std=scaler.scale_,
        classes=np.array(ALPHABET_CLASSES),
    )
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ PROCESSAMENTO CONCLUÍDO!")
    logger.info("=" * 70)
    logger.info(f"Total: {len(X_combined)} amostras")
    logger.info(f"Treino: {len(X_train_norm)} | Val: {len(X_val_norm)} | Teste: {len(X_test_norm)}")
    logger.info("")
    logger.info("Próximo passo: python notebooks/train_alphabet_standalone.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("\n✗ Processamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Erro fatal no processamento: {e}", exc_info=True)
        sys.exit(1)

