#!/usr/bin/env python
"""
Script master simples para configurar datasets de LIBRAS.

Este script processa qualquer dataset de imagens que você colocar em:
    data/raw/custom_images/[LETRA]/

Execute: python data/setup.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

def main():
    print("=" * 70)
    print("SETUP DE DATASETS LIBRAS")
    print("=" * 70)
    print()
    print("Este script processa imagens organizadas por classe.")
    print()
    print("Estrutura esperada:")
    print("  data/raw/custom_images/")
    print("      A/")
    print("          image1.png")
    print("          image2.png")
    print("      B/")
    print("          ...")
    print()
    
    input("Pressione ENTER para continuar...")
    print()
    
    # Executar processamento
    import subprocess
    script_path = ROOT_DIR / "data" / "process_datasets.py"
    
    print("Processando datasets...")
    print()
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT_DIR
    )
    
    if result.returncode == 0:
        print()
        print("=" * 70)
        print("✓ SETUP CONCLUÍDO!")
        print("=" * 70)
        print()
        print("Próximo passo: Treinar o modelo")
        print("  python notebooks/train_alphabet_standalone.py")
    else:
        print()
        print("✗ Erro no processamento. Verifique os logs acima.")

if __name__ == "__main__":
    main()

