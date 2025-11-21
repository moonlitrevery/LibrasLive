#!/usr/bin/env python
"""
Script master para processar datasets e treinar modelos.

Este script:
1. Processa todos os datasets disponíveis
2. Treina o modelo de alfabeto
3. Treina o modelo de frases

Execute: python train_all.py
"""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def run_script(script_path, description):
    """Executa um script Python."""
    print("=" * 70)
    print(description)
    print("=" * 70)
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT_DIR,
            capture_output=False,  # Mostrar saída em tempo real
            text=True,
            check=False  # Não lançar exceção, apenas verificar returncode
        )
        
        if result.returncode != 0:
            print(f"\n✗ Erro ao executar: {description}")
            print(f"   Código de saída: {result.returncode}")
            return False
        
        print(f"\n✓ {description} concluído!")
        print()
        return True
    except FileNotFoundError:
        print(f"\n✗ Script não encontrado: {script_path}")
        return False
    except Exception as e:
        print(f"\n✗ Erro inesperado ao executar {description}: {e}")
        return False

def main():
    print("=" * 70)
    print("TREINAMENTO COMPLETO - LIBRAS LIVE")
    print("=" * 70)
    print()
    print("Este script irá:")
    print("1. Processar todos os datasets disponíveis")
    print("2. Treinar modelo de alfabeto")
    print("3. Treinar modelo de frases")
    print()
    
    input("Pressione ENTER para continuar...")
    print()
    
    # 1. Processar datasets
    if not run_script(
        ROOT_DIR / "data" / "process_all_datasets.py",
        "PROCESSANDO DATASETS"
    ):
        print("⚠ Continuando mesmo com erro no processamento...")
        print()
    
    # 2. Treinar alfabeto
    if not run_script(
        ROOT_DIR / "notebooks" / "train_alphabet_standalone.py",
        "TREINANDO MODELO DE ALFABETO"
    ):
        print("✗ Falha no treinamento do alfabeto!")
        return
    
    # 3. Treinar frases
    if not run_script(
        ROOT_DIR / "notebooks" / "train_phrase_standalone.py",
        "TREINANDO MODELO DE FRASES"
    ):
        print("✗ Falha no treinamento de frases!")
        return
    
    print("=" * 70)
    print("✓ TODOS OS TREINAMENTOS CONCLUÍDOS COM SUCESSO!")
    print("=" * 70)
    print()
    print("Modelos salvos em: backend/models/")
    print("Próximo passo: Reinicie o backend e teste!")

if __name__ == "__main__":
    main()

