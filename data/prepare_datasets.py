#!/usr/bin/env python
"""
Data preparation pipeline for LibrasLive.

This script converts the public datasets listed in data/datasets_publicos.txt
into the normalized landmark tensors consumed by the training notebooks and
backend models.

Steps:
1. Sign Language MNIST (Kaggle) → MediaPipe landmarks → StandardScaler.
2. Libras Movement (UCI) → temporal sequences (30 × 63) → StandardScaler.
3. Saves intermediate .npy/.npz artifacts in data/processed/.

Run:
    python data/prepare_datasets.py
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - dependency is required at runtime
    raise ImportError(
        "Mediapipe is required to run this script. "
        "Install it with `pip install mediapipe`."
    ) from exc

# --------------------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

SIGN_DATASET_DIR = RAW_DIR / "sign_language_mnist"
SIGN_TRAIN_CSV = SIGN_DATASET_DIR / "sign_mnist_train.csv"
SIGN_TEST_CSV = SIGN_DATASET_DIR / "sign_mnist_test.csv"

PHRASE_DATASET_DIR = RAW_DIR / "libras_movement"
PHRASE_DATA_FILE_CANDIDATES = [
    PHRASE_DATASET_DIR / "movement_libras.data",
    PHRASE_DATASET_DIR / "libras_movement.data",
]

ALPHABET_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K",
    "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y",
]

# The UCI dataset ships with 15 classes (1-indexed). Extend/mapping later if needed.
PHRASE_CLASS_NAMES = [
    "CURVAR", "ALISAR", "LIBRAS", "LIVRE", "PALAVRA",
    "ARVORE", "VERDE", "AMARELO", "TEMPO", "BOM",
    "RUIM", "MELHOR", "PIOR", "PRIMEIRO", "ULTIMO",
]

SEQUENCE_LENGTH = 30
LANDMARK_DIM = 63  # 21 landmarks × (x, y, z)
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_datasets")


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------
def ensure_directories() -> None:
    """Ensure required directories exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def maybe_stratify(labels: np.ndarray, split_name: str, num_classes: int) -> np.ndarray | None:
    """
    Return labels for stratification only if every present class has >=2 samples.
    Otherwise log a warning and return None to disable stratification.
    """
    counts = np.bincount(labels, minlength=num_classes)
    positive_counts = counts[counts > 0]
    if positive_counts.size == 0:
        raise ValueError(f"Nenhuma amostra disponível para {split_name}.")
    if positive_counts.min() < 2:
        logger.warning(
            "Atenção: classes com menos de 2 amostras detectadas (%s). "
            "Divisão será feita sem estratificação.",
            split_name,
        )
        return None
    return labels


def resolve_phrase_data_file() -> Path:
    """
    Identify which .data file should be used for the Libras Movement dataset.
    Prefers known filenames, otherwise falls back to the largest .data file.
    """
    for candidate in PHRASE_DATA_FILE_CANDIDATES:
        if candidate.exists():
            return candidate

    data_files = sorted(
        PHRASE_DATASET_DIR.glob("*.data"),
        key=lambda p: p.stat().st_size if p.exists() else 0,
        reverse=True,
    )

    if data_files:
        logger.warning(
            "Arquivo %s selecionado automaticamente como dataset principal.",
            data_files[0].name,
        )
        return data_files[0]

    raise FileNotFoundError(
        "Nenhum arquivo *.data encontrado em data/raw/libras_movement. "
        "Verifique se o zip original foi extraído."
    )


def validate_paths() -> Path:
    """Check that raw dataset files exist before processing."""
    phrase_data_file = resolve_phrase_data_file()
    missing_files = [
        path for path in [SIGN_TRAIN_CSV, SIGN_TEST_CSV, phrase_data_file]
        if not path.exists()
    ]
    if missing_files:
        missing_str = "\n - ".join(str(p) for p in missing_files)
        raise FileNotFoundError(
            "The following dataset files are missing. "
            "Download them before running this script:\n - " + missing_str
        )
    return phrase_data_file


# --------------------------------------------------------------------------------------
# Sign Language MNIST → MediaPipe landmarks
# --------------------------------------------------------------------------------------
def load_sign_language_mnist(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the Kaggle CSV into numpy arrays (images, labels)."""
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].astype(np.int64).to_numpy()
    pixels = df.iloc[:, 1:].to_numpy(dtype=np.uint8)
    images = pixels.reshape(-1, 28, 28)
    return images, labels


def preprocess_image_for_mediapipe(image_28x28: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Upscale grayscale image to the size expected by MediaPipe (RGB)."""
    image_resized = cv2.resize(image_28x28, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    # MediaPipe expects uint8 RGB image
    return image_rgb


def extract_landmarks_from_images(
    images: np.ndarray,
    labels: np.ndarray,
    detector: mp.solutions.hands.Hands,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run MediaPipe Hands over every image and output 63-d landmarks."""
    features: List[np.ndarray] = []
    kept_labels: List[int] = []
    skipped = 0

    for image, label in tqdm(zip(images, labels), total=len(images), desc="MediaPipe landmarks"):
        rgb_image = preprocess_image_for_mediapipe(image)
        results = detector.process(rgb_image)

        if not results.multi_hand_landmarks:
            skipped += 1
            continue

        # Use only the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        coords: List[float] = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])

        if len(coords) != LANDMARK_DIM:
            skipped += 1
            continue

        features.append(np.array(coords, dtype=np.float32))
        kept_labels.append(label)

    logger.info("Extracted %d/%d samples (skipped %d without landmarks).", len(features), len(images), skipped)
    return np.stack(features), np.array(kept_labels, dtype=np.int64)


def remap_sign_mnist_labels(labels: np.ndarray) -> np.ndarray:
    """
    Sign Language MNIST uses 0-25 (A-Z) without J/Z.
    The backend expects indices aligned with ALPHABET_CLASSES.
    """
    label_map: Dict[int, int] = {}
    current_idx = 0
    for ascii_code in range(26):
        letter = chr(ord("A") + ascii_code)
        if letter in ("J", "Z"):
            continue
        label_map[ascii_code] = current_idx
        current_idx += 1

    return np.array([label_map[label] for label in labels], dtype=np.int64)


def build_alphabet_dataset() -> None:
    """Generate normalized landmark dataset for alphabet recognition."""
    logger.info("Processing Sign Language MNIST...")
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

    stratify_labels = maybe_stratify(y_train, "treino/validação (alfabeto)", len(ALPHABET_CLASSES))

    # Build validation split out of train portion
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test_raw)

    np.save(PROCESSED_DIR / "alphabet_landmarks_train.npy", X_train_norm)
    np.save(PROCESSED_DIR / "alphabet_landmarks_val.npy", X_val_norm)
    np.save(PROCESSED_DIR / "alphabet_landmarks_test.npy", X_test_norm)

    with open(PROCESSED_DIR / "alphabet_scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)

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

    logger.info("Alphabet dataset saved to %s", PROCESSED_DIR / "alphabet_dataset.npz")


# --------------------------------------------------------------------------------------
# Libras Movement → sequence tensors
# --------------------------------------------------------------------------------------
def load_libras_movement(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the UCI dataset. Each row has 90 numeric attributes + class label (1-15).
    Returns:
        features: (n_samples, 90)
        labels: (n_samples,)
    """
    df = pd.read_csv(data_path, header=None)
    features = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    labels = df.iloc[:, -1].astype(int).to_numpy() - 1  # zero-indexed
    return features, labels


def features_to_sequences(features: np.ndarray) -> np.ndarray:
    """
    Convert 90 features (45 frames × 2 coordinates) to (frames, 21, 3) representation.
    Currently replicates the single trajectory across 21 landmarks and sets z=0.
    """
    seq_45x2 = features.reshape(-1, 45, 2)
    seq_45x21x2 = np.repeat(seq_45x2[:, :, np.newaxis, :], repeats=21, axis=2)
    zeros = np.zeros_like(seq_45x21x2[..., :1])
    seq_45x21x3 = np.concatenate([seq_45x21x2, zeros], axis=-1)
    seq_flat = seq_45x21x3.reshape(seq_45x21x3.shape[0], 45, LANDMARK_DIM)
    return seq_flat


def temporal_resample(sequence: np.ndarray, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Linearly resample a (frames, features) sequence to the desired length."""
    orig_len = sequence.shape[0]
    if orig_len == target_len:
        return sequence

    original_idx = np.linspace(0, orig_len - 1, num=orig_len)
    target_idx = np.linspace(0, orig_len - 1, num=target_len)
    resampled = np.empty((target_len, sequence.shape[1]), dtype=np.float32)
    for dim in range(sequence.shape[1]):
        resampled[:, dim] = np.interp(target_idx, original_idx, sequence[:, dim])
    return resampled


def resample_sequences(sequences: np.ndarray) -> np.ndarray:
    """Apply temporal resampling to every sequence."""
    return np.stack([temporal_resample(seq) for seq in sequences])


def normalize_sequences(
    sequences: np.ndarray,
    scaler: StandardScaler | None = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """Normalize sequences frame-wise with a StandardScaler fitted on flattened frames."""
    n_samples, seq_len, feature_dim = sequences.shape
    flattened = sequences.reshape(n_samples * seq_len, feature_dim)

    scaler = scaler or StandardScaler()
    flattened_norm = scaler.fit_transform(flattened)
    sequences_norm = flattened_norm.reshape(n_samples, seq_len, feature_dim)
    return sequences_norm, scaler


def build_phrase_dataset(phrase_data_file: Path) -> None:
    """Generate normalized sequence dataset for phrase recognition."""
    logger.info("Processing Libras Movement dataset...")
    features, labels = load_libras_movement(phrase_data_file)
    sequences = features_to_sequences(features)
    sequences = resample_sequences(sequences)

    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences,
        labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    stratify_temp = maybe_stratify(y_temp, "validação/teste (alfabeto)", len(ALPHABET_CLASSES))

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=stratify_temp,
    )

    X_train_norm, seq_scaler = normalize_sequences(X_train)
    X_val_norm = seq_scaler.transform(X_val.reshape(-1, LANDMARK_DIM)).reshape(X_val.shape)
    X_test_norm = seq_scaler.transform(X_test.reshape(-1, LANDMARK_DIM)).reshape(X_test.shape)

    np.save(PROCESSED_DIR / "phrase_sequences_train.npy", X_train_norm)
    np.save(PROCESSED_DIR / "phrase_sequences_val.npy", X_val_norm)
    np.save(PROCESSED_DIR / "phrase_sequences_test.npy", X_test_norm)

    with open(PROCESSED_DIR / "phrase_scaler.pkl", "wb") as fh:
        pickle.dump(seq_scaler, fh)

    np.savez(
        PROCESSED_DIR / "phrase_dataset.npz",
        X_train=X_train_norm,
        y_train=y_train,
        X_val=X_val_norm,
        y_val=y_val,
        X_test=X_test_norm,
        y_test=y_test,
        mean=seq_scaler.mean_,
        std=seq_scaler.scale_,
        classes=np.array(PHRASE_CLASS_NAMES),
        sequence_length=SEQUENCE_LENGTH,
    )

    logger.info("Phrase dataset saved to %s", PROCESSED_DIR / "phrase_dataset.npz")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def main() -> None:
    ensure_directories()
    phrase_data_file = validate_paths()
    build_alphabet_dataset()
    build_phrase_dataset(phrase_data_file)
    logger.info("Datasets processed successfully! 🎉")


if __name__ == "__main__":
    main()

