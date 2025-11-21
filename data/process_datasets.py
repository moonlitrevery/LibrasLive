#!/usr/bin/env python
"""
Script para processar datasets de imagens de LIBRAS.

Este script:
1. Processa imagens organizadas por classe (A/, B/, C/, etc.)
2. Extrai landmarks usando MediaPipe
3. Combina com dataset existente
4. Gera dataset final para treinamento

Execute: python data/process_datasets.py
"""

import cv2
import numpy as np
import pickle
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

# MediaPipe setup
mp_hands = mp.solutions.hands

# Classes do alfabeto
ALPHABET_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y",
]

LANDMARK_DIM = 63  # 21 pontos * 3 coordenadas
RANDOM_STATE = 42

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

def process_image_dataset(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processa um dataset de imagens organizado por classe.
    
    Estrutura esperada:
    dataset_dir/
        A/
            image1.png
            image2.png
        B/
            ...
    """
    if not dataset_dir.exists():
        return np.array([]), np.array([])
    
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    features = []
    labels = []
    
    for class_idx, class_name in enumerate(ALPHABET_CLASSES):
        class_dir = dataset_dir / class_name
        
        if not class_dir.exists():
            continue
        
        # Buscar imagens
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(class_dir.glob(f"*{ext}")))
        
        if not image_files:
            continue
        
        logger.info(f"Processando {class_name}: {len(image_files)} imagens")
        
        for image_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
            landmarks = extract_landmarks_from_image(image_path, hands)
            if landmarks is not None:
                features.append(landmarks)
                labels.append(class_idx)
    
    hands.close()
    
    if len(features) == 0:
        return np.array([]), np.array([])
    
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)

def load_existing_dataset() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Carrega o dataset existente."""
    existing_path = PROCESSED_DIR / "alphabet_dataset.npz"
    
    if not existing_path.exists():
        return None
    
    logger.info("Carregando dataset existente...")
    with np.load(existing_path) as data:
        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]
    
    X_existing = np.vstack([X_train, X_val, X_test])
    y_existing = np.concatenate([y_train, y_val, y_test])
    
    logger.info(f"✓ Dataset existente: {len(X_existing)} amostras")
    return X_existing, y_existing

def maybe_stratify(y, min_samples: int = 2):
    """Retorna y se houver amostras suficientes para estratificação."""
    counts = Counter(y)
    min_count = min(counts.values())
    if min_count < min_samples:
        logger.warning(f"⚠ Estratificação não possível: classe menos frequente tem {min_count} amostras")
        return None
    return y

def main():
    """Função principal."""
    logger.info("=" * 70)
    logger.info("PROCESSAMENTO DE DATASETS DE LIBRAS")
    logger.info("=" * 70)
    logger.info("")
    
    # Processar datasets na pasta raw
    datasets_to_process = []
    
    # 1. Verificar pasta custom_images
    custom_dir = RAW_DIR / "custom_images"
    if custom_dir.exists():
        logger.info("Processando custom_images...")
        X, y = process_image_dataset(custom_dir)
        if len(X) > 0:
            datasets_to_process.append((X, y, "custom_images"))
            logger.info(f"✓ custom_images: {len(X)} amostras")
    
    # 2. Verificar outras pastas comuns
    common_names = ["lasic", "dataset", "images", "libras"]
    for name in common_names:
        dataset_dir = RAW_DIR / name
        if dataset_dir.exists() and dataset_dir.is_dir():
            logger.info(f"Processando {name}...")
            X, y = process_image_dataset(dataset_dir)
            if len(X) > 0:
                datasets_to_process.append((X, y, name))
                logger.info(f"✓ {name}: {len(X)} amostras")
    
    # 3. Carregar dataset existente
    existing_data = load_existing_dataset()
    if existing_data is not None:
        datasets_to_process.insert(0, (*existing_data, "existing"))
    
    if not datasets_to_process:
        logger.error("✗ Nenhum dataset encontrado para processar!")
        logger.info("")
        logger.info("💡 COMO USAR:")
        logger.info("1. Organize suas imagens assim:")
        logger.info("   data/raw/custom_images/")
        logger.info("       A/")
        logger.info("           image1.png")
        logger.info("           image2.png")
        logger.info("       B/")
        logger.info("           ...")
        logger.info("2. Execute este script novamente")
        return
    
    # Combinar todos os datasets
    logger.info("")
    logger.info("Combinando datasets...")
    all_features = np.vstack([X for X, _, _ in datasets_to_process])
    all_labels = np.concatenate([y for _, y, _ in datasets_to_process])
    
    logger.info(f"✓ Total combinado: {len(all_features)} amostras")
    
    # Split train/val/test (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_features,
        all_labels,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=maybe_stratify(all_labels, 2)
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=maybe_stratify(y_temp, 2)
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
    logger.info(f"Total: {len(all_features)} amostras")
    logger.info(f"Treino: {len(X_train_norm)} | Val: {len(X_val_norm)} | Teste: {len(X_test_norm)}")
    logger.info("")
    logger.info("Próximo passo: python notebooks/train_alphabet_standalone.py")

if __name__ == "__main__":
    main()

