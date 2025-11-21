#!/usr/bin/env python
"""
Script para baixar datasets públicos de LIBRAS/Sign Language.

Este script baixa datasets públicos disponíveis em:
- Kaggle (via API)
- GitHub (links diretos)
- Outras fontes públicas

Execute: python data/download_public_datasets.py
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest_path: Path, description: str = "Downloading"):
    """Baixa um arquivo com barra de progresso."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        logger.info(f"✓ {description} já existe: {dest_path.name}")
        return True
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"✓ {description} baixado com sucesso!")
        return True
    except Exception as e:
        logger.error(f"✗ Erro ao baixar {description}: {e}")
        return False

def extract_zip(zip_path: Path, extract_to: Path):
    """Extrai um arquivo ZIP."""
    if not zip_path.exists():
        return False
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"✓ Arquivo extraído: {zip_path.name}")
        return True
    except Exception as e:
        logger.error(f"✗ Erro ao extrair {zip_path.name}: {e}")
        return False

def download_kaggle_dataset(dataset_name: str, dest_dir: Path):
    """
    Baixa dataset do Kaggle usando a API.
    
    Requer: pip install kaggle
    Configure: ~/.kaggle/kaggle.json com suas credenciais
    """
    try:
        import kaggle
    except ImportError:
        logger.warning("⚠ Biblioteca 'kaggle' não instalada.")
        logger.info("💡 Instale com: pip install kaggle")
        logger.info("💡 Configure credenciais em: ~/.kaggle/kaggle.json")
        logger.info("💡 Obtenha credenciais em: https://www.kaggle.com/settings")
        return False
    
    try:
        logger.info(f"Baixando dataset do Kaggle: {dataset_name}")
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(dest_dir),
            unzip=True
        )
        logger.info(f"✓ Dataset {dataset_name} baixado com sucesso!")
        return True
    except Exception as e:
        logger.error(f"✗ Erro ao baixar do Kaggle: {e}")
        return False

def download_sign_language_mnist():
    """Baixa Sign Language MNIST (já deve estar no projeto, mas verifica)."""
    logger.info("=" * 70)
    logger.info("SIGN LANGUAGE MNIST")
    logger.info("=" * 70)
    
    sign_dir = RAW_DIR / "sign_language_mnist"
    
    if sign_dir.exists() and len(list(sign_dir.glob("*.csv"))) > 0:
        logger.info("✓ Sign Language MNIST já está no projeto")
        return True
    
    # URLs públicas do Sign Language MNIST
    urls = {
        "train": "https://www.kaggle.com/datasets/datamunge/sign-language-mnist/download?datasetVersionNumber=2",
        # Nota: Kaggle requer autenticação, então vamos usar alternativa
    }
    
    logger.info("ℹ Sign Language MNIST requer download manual do Kaggle")
    logger.info("🔗 Link: https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
    logger.info("💡 Ou use: kaggle datasets download -d datamunge/sign-language-mnist")
    
    return False

def download_hand_gesture_datasets():
    """Baixa datasets públicos de gestos de mão."""
    logger.info("=" * 70)
    logger.info("HAND GESTURE DATASETS")
    logger.info("=" * 70)
    
    # Dataset público de gestos de mão (exemplo genérico)
    # Você pode adicionar mais URLs aqui
    
    logger.info("💡 Datasets públicos de gestos de mão:")
    logger.info("   1. Kaggle: 'hand-gesture-recognition'")
    logger.info("   2. Kaggle: 'asl-sign-language'")
    logger.info("   3. GitHub: Buscar por 'sign-language-dataset'")
    
    return False

def download_from_github(url: str, dest_path: Path):
    """Baixa arquivo do GitHub (raw content)."""
    try:
        # Converter URL do GitHub para raw
        if "github.com" in url and "/blob/" in url:
            url = url.replace("/blob/", "/raw/")
        
        return download_file(url, dest_path, f"GitHub: {dest_path.name}")
    except Exception as e:
        logger.error(f"✗ Erro ao baixar do GitHub: {e}")
        return False

def main():
    """Função principal."""
    logger.info("=" * 70)
    logger.info("DOWNLOAD DE DATASETS PÚBLICOS DE LIBRAS/SIGN LANGUAGE")
    logger.info("=" * 70)
    logger.info("")
    
    # Verificar Sign Language MNIST
    download_sign_language_mnist()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("DATASETS DISPONÍVEIS NO KAGGLE")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Para baixar datasets do Kaggle, você precisa:")
    logger.info("1. Criar conta no Kaggle: https://www.kaggle.com/")
    logger.info("2. Obter API token: https://www.kaggle.com/settings")
    logger.info("3. Instalar: pip install kaggle")
    logger.info("4. Configurar: ~/.kaggle/kaggle.json")
    logger.info("")
    logger.info("Datasets recomendados:")
    logger.info("  - datamunge/sign-language-mnist")
    logger.info("  - kuchhbhi/hand-gesture-recognition-dataset")
    logger.info("  - ardamavi/sign-language-digits-dataset")
    logger.info("")
    logger.info("=" * 70)
    logger.info("PRÓXIMOS PASSOS")
    logger.info("=" * 70)
    logger.info("1. Configure a API do Kaggle (se quiser usar)")
    logger.info("2. Ou baixe datasets manualmente do Kaggle")
    logger.info("3. Coloque em: data/raw/")
    logger.info("4. Execute: python data/process_datasets.py")
    logger.info("")

if __name__ == "__main__":
    main()

