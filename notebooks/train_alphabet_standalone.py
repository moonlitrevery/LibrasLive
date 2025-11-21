#!/usr/bin/env python
"""
Script standalone melhorado para treinar o modelo de alfabeto LIBRAS.

Melhorias:
- Arquitetura mais profunda com BatchNorm
- Data augmentation durante treinamento
- Learning rate schedule melhor
- Regularização aprimorada
- Early stopping inteligente

Execute: python notebooks/train_alphabet_standalone.py
"""

import os
import sys
import json
import random

# Configurações de compatibilidade ANTES de importar torch
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TREINAMENTO MELHORADO DO MODELO DE ALFABETO LIBRAS")
print("=" * 70)
print()

# Configurações
ALPHABET_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
    'V', 'W', 'X', 'Y'
]
NUM_CLASSES = len(ALPHABET_CLASSES)
FEATURE_SIZE = 63
BATCH_SIZE = 64  # Aumentado para melhor estabilidade
LEARNING_RATE = 0.001
NUM_EPOCHS = 200  # Mais épocas
PATIENCE = 25  # Mais paciência
WEIGHT_DECAY = 1e-4  # Regularização L2

# Carregar dataset
print("Carregando dataset...")
dataset_path = "data/processed/alphabet_dataset.npz"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"Arquivo {dataset_path} não encontrado. "
        "Execute `python data/process_all_datasets.py` antes de treinar."
    )

with np.load(dataset_path) as data:
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    mean_params = data["mean"]
    std_params = data["std"]
    stored_classes = data.get("classes")

if stored_classes is not None:
    ALPHABET_CLASSES = stored_classes.tolist()
    NUM_CLASSES = len(ALPHABET_CLASSES)

print(f"✓ Dataset carregado: {X_train.shape[0]} treino, {X_val.shape[0]} val, {X_test.shape[0]} teste")

# Data augmentation durante treinamento
def augment_landmarks(landmarks, noise_factor=0.02):
    """Adiciona ruído gaussiano aos landmarks para data augmentation."""
    noise = np.random.normal(0, noise_factor, landmarks.shape).astype(np.float32)
    return landmarks + noise

# Dataset class com augmentation
class LibrasAlphabetDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].clone()
        
        if self.augment and random.random() > 0.5:
            # Aplicar augmentation
            x_np = x.numpy()
            x_aug = augment_landmarks(x_np, noise_factor=0.02)
            x = torch.FloatTensor(x_aug)
        
        return x, self.y[idx]

# Criar datasets
train_dataset = LibrasAlphabetDataset(X_train, y_train, augment=True)
val_dataset = LibrasAlphabetDataset(X_val, y_val, augment=False)
test_dataset = LibrasAlphabetDataset(X_test, y_test, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Modelo melhorado com BatchNorm e mais camadas
class AlphabetMLP(nn.Module):
    def __init__(self, input_size=63, hidden_size=256, num_classes=24, dropout_rate=0.4):
        super(AlphabetMLP, self).__init__()
        self.layers = nn.Sequential(
            # Primeira camada
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Segunda camada
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Terceira camada (intermediária)
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            # Quarta camada
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            # Camada de saída
            nn.Linear(hidden_size // 4, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Usando dispositivo: {device}")

model = AlphabetMLP(
    input_size=FEATURE_SIZE,
    hidden_size=256,  # Aumentado
    num_classes=NUM_CLASSES,
    dropout_rate=0.4  # Aumentado para melhor regularização
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Modelo criado: {total_params:,} parâmetros")
print()

# Treinamento melhorado
criterion = nn.CrossEntropyLoss()
params = list(model.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler melhorado
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping para estabilidade
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    return running_loss / len(train_loader), 100.0 * correct / total

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return running_loss / len(val_loader), 100.0 * correct / total

print("Iniciando treinamento...")
print()

best_val_accuracy = 0.0
best_model_state = None
epochs_without_improvement = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    scheduler.step(val_loss)
    
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        epochs_without_improvement = 0
        best_model_state = model.state_dict().copy()
    else:
        epochs_without_improvement += 1
    
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Época [{epoch+1:3d}/{NUM_EPOCHS}] | "
              f"Treino: Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
              f"Val: Loss={val_loss:.4f} Acc={val_acc:.2f}% | "
              f"LR={current_lr:.6f}")
    
    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping após {epoch + 1} épocas")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

print(f"\n✓ Melhor acurácia de validação: {best_val_accuracy:.2f}%")
print()

# Avaliação
print("Avaliando no conjunto de teste...")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

test_accuracy = accuracy_score(all_targets, all_preds)
print(f"✓ Acurácia no teste: {test_accuracy*100:.2f}%")
print()

# Classification report (usar apenas classes presentes)
unique_classes = sorted(set(all_targets))
present_class_names = [ALPHABET_CLASSES[i] if i < len(ALPHABET_CLASSES) else f"Class_{i}" for i in unique_classes]

print("Relatório de classificação por classe:")
try:
    print(classification_report(all_targets, all_preds, labels=unique_classes, target_names=present_class_names))
except Exception as e:
    print(f"⚠ Erro ao gerar relatório detalhado: {e}")
    print(f"Classes presentes: {len(unique_classes)}")
print()

# Salvar modelo
print("Salvando modelo...")
try:
    os.makedirs('backend/models', exist_ok=True)

    model_path = 'backend/models/alphabet_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"✓ Modelo salvo em: {model_path}")

    norm_path = 'backend/models/normalization_params.npz'
    np.savez(norm_path, mean=mean_params, std=std_params)
    print(f"✓ Parâmetros de normalização salvos em: {norm_path}")

    # Salvar metadata
    metadata = {
        'model_type': 'AlphabetMLP',
        'num_classes': NUM_CLASSES,
        'feature_size': FEATURE_SIZE,
        'hidden_size': 256,
        'dropout_rate': 0.4,
        'classes': ALPHABET_CLASSES,
        'test_accuracy': float(test_accuracy),
        'best_val_accuracy': float(best_val_accuracy),
        'total_parameters': int(total_params),
        'training_samples': int(len(X_train)),
        'architecture': 'MLP with BatchNorm and 4 hidden layers'
    }

    metadata_path = 'backend/models/alphabet_model_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✓ Metadata do modelo salvo em: {metadata_path}")
except Exception as e:
    print(f"✗ ERRO ao salvar modelo: {e}")
    raise

print()
print("=" * 70)
print("✓ TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("=" * 70)
