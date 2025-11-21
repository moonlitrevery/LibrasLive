#!/usr/bin/env python
"""
Script standalone para treinar o modelo de frases LIBRAS.
Use este script se o Jupyter notebook estiver com problemas de DLL.
Execute: python notebooks/train_phrase_standalone.py
"""

import os
import sys
import json

# Configurações de compatibilidade ANTES de importar torch
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# Adiciona o diretório raiz ao path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TREINAMENTO DO MODELO DE FRASES LIBRAS")
print("=" * 70)
print()

# Configurações padrão
PHRASE_CLASSES = [
    "OLA", "OBRIGADO", "POR_FAVOR", "DESCULPA", "SIM", "NAO",
    "BOM_DIA", "BOA_TARDE", "BOA_NOITE", "TUDO_BEM", "EU_TE_AMO",
    "FAMILIA", "CASA", "TRABALHO", "ESCOLA", "AGUA", "COMIDA",
    "AJUDA", "TCHAU", "ATE_LOGO"
]
NUM_CLASSES = len(PHRASE_CLASSES)
FEATURE_SIZE = 63
SEQUENCE_LENGTH = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 150
PATIENCE = 20
GRAD_CLIP = 1.0

# Carregar dataset
print("Carregando dataset de frases...")
dataset_path = os.path.join(ROOT_DIR, "data/processed/phrase_dataset.npz")
if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        f"Arquivo {dataset_path} não encontrado. "
        "Execute `python data/prepare_datasets.py` antes de treinar."
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
    stored_seq_length = int(data.get("sequence_length", SEQUENCE_LENGTH))

if stored_classes is not None:
    PHRASE_CLASSES = stored_classes.tolist()
    NUM_CLASSES = len(PHRASE_CLASSES)

SEQUENCE_LENGTH = stored_seq_length

print(f"✓ Dataset carregado: {X_train.shape[0]} sequências de treino")
print(f"  Validação: {X_val.shape[0]}, Teste: {X_test.shape[0]}")
print(f"  Formato: (batch, {SEQUENCE_LENGTH} frames, {FEATURE_SIZE} features)")
print()

# Dataset class
class LibrasPhraseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Criar datasets
train_dataset = LibrasPhraseDataset(X_train, y_train)
val_dataset = LibrasPhraseDataset(X_val, y_val)
test_dataset = LibrasPhraseDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✓ Data loaders criados: {len(train_loader)} batches de treino")
print()

# Modelo LSTM
class PhraseLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=64, num_layers=2, 
                 num_classes=20, seq_length=30, dropout_rate=0.3):
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
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]
        
        # Classification
        output = self.classifier(last_output)
        
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Usando dispositivo: {device}")

model = PhraseLSTM(
    input_size=FEATURE_SIZE,
    hidden_size=64,
    num_layers=2,
    num_classes=NUM_CLASSES,
    seq_length=SEQUENCE_LENGTH,
    dropout_rate=0.3
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Modelo LSTM criado: {total_params:,} parâmetros")
print()

# Treinamento
criterion = nn.CrossEntropyLoss()
params = list(model.parameters())
optimizer = optim.Adam(params, lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)

def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=None):
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
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
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

print("Iniciando treinamento do modelo de frases...")
print()

best_val_accuracy = 0.0
best_model_state = None
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device, GRAD_CLIP
    )
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    scheduler.step(val_loss)
    
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        epochs_without_improvement = 0
        best_model_state = model.state_dict().copy()
        print(f"✓ Nova melhor acurácia de validação: {val_acc:.2f}%")
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

# Classification report
print("Relatório de classificação:")
print(classification_report(all_targets, all_preds, target_names=PHRASE_CLASSES))
print()

# Salvar modelo
print("Salvando modelo...")
models_dir = os.path.join(ROOT_DIR, 'backend/models')
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'phrase_model.pt')
torch.save(model.state_dict(), model_path)
print(f"✓ Modelo salvo em: {model_path}")

norm_path = os.path.join(models_dir, 'phrase_normalization_params.npz')
np.savez(norm_path, mean=mean_params, std=std_params)
print(f"✓ Parâmetros de normalização salvos em: {norm_path}")

# Salvar metadata
metadata = {
    'model_type': 'PhraseLSTM',
    'num_classes': NUM_CLASSES,
    'feature_size': FEATURE_SIZE,
    'sequence_length': SEQUENCE_LENGTH,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout_rate': 0.3,
    'classes': PHRASE_CLASSES,
    'test_accuracy': float(test_accuracy),
    'best_val_accuracy': float(best_val_accuracy),
    'total_parameters': int(total_params),
    'training_sequences': int(len(X_train)),
    'sequence_format': '(batch, seq_length, features)'
}

metadata_path = os.path.join(models_dir, 'phrase_model_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✓ Metadata do modelo salva em: {metadata_path}")

print()
print("=" * 70)
print("✓ TREINAMENTO DO MODELO DE FRASES CONCLUÍDO COM SUCESSO!")
print("=" * 70)

