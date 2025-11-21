# LibrasLive 🤟

**Real-time LIBRAS Sign Language Translator**

LibrasLive é um sistema de tradução em tempo real para LIBRAS (Língua Brasileira de Sinais) que converte sinais das mãos capturados pela webcam em texto e áudio. O sistema reconhece o alfabeto LIBRAS (A-Y, excluindo J e Z) e 20 frases mais comuns.

![LibrasLive Demo](https://img.shields.io/badge/Status-Active-green) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![JavaScript](https://img.shields.io/badge/JavaScript-ES6-yellow) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)

## 🌟 Funcionalidades

- ✅ **Reconhecimento de Alfabeto**: 24 letras do alfabeto LIBRAS (A-Y, excluindo J e Z)
- ✅ **Reconhecimento de Frases**: 20 frases mais comuns em LIBRAS
- ✅ **Tempo Real**: Latência < 1 segundo para interação fluida
- ✅ **Interface Web**: Interface moderna e responsiva
- ✅ **Síntese de Voz**: TTS (Text-to-Speech) com gTTS e pyttsx3
- ✅ **Botão Repetir**: Reproduz novamente o áudio da última tradução
- ✅ **Visualização de Landmarks**: Exibe pontos de referência da mão (opcional)
- ✅ **Métricas em Tempo Real**: FPS, latência e predições por minuto
- ✅ **Histórico de Traduções**: Mantém registro das últimas traduções

## 📋 Requisitos do Sistema

### Hardware
- **Webcam**: Câmera com resolução mínima de 640x480
- **Processador**: CPU multi-core (recomendado: 4+ cores)
- **Memória**: 4GB RAM mínimo, 8GB recomendado
- **GPU**: Opcional (CUDA para acelerar inferência)

### Software
- **Python**: 3.8 ou superior
- **Node.js**: 14.0+ (opcional, para desenvolvimento frontend)
- **Navegador**: Chrome 80+, Firefox 75+, Safari 13+, Edge 80+

## 🚀 Instalação Rápida

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/libras-live.git
cd libras-live
```

### 2. Instale Dependências Python
"
```bash
cd backend
pip install -r requirements.txt
```

### 3. Execute o Backend

```bash
python app.py
```

### 4. Abra o Frontend

Navegue até `frontend/index.html` no seu navegador ou sirva via servidor web:

```bash
# Opção 1: Abrir diretamente
# Abrir frontend/index.html no navegador

# Opção 2: Servidor Python simples
cd frontend
python -m http.server 8000
# Acesse http://localhost:8000
```

## 📁 Estrutura do Projeto

```
libras-live/
├── backend/                    # Backend Python (Flask + SocketIO)
│   ├── app.py                 # Servidor principal
│   ├── infer.py               # Motor de inferência dos modelos
│   ├── tts.py                 # Módulo de síntese de voz
│   ├── requirements.txt       # Dependências Python
│   └── models/                # Modelos treinados
│       ├── alphabet_model.pt  # Modelo do alfabeto
│       └── phrase_model.pt    # Modelo de frases
├── frontend/                  # Frontend Web (HTML/CSS/JS)
│   ├── index.html            # Interface principal
│   ├── main.js               # Lógica JavaScript + MediaPipe
│   └── styles.css            # Estilos CSS
├── data/                     # Datasets e documentação
│   ├── datasets_publicos.txt # Fontes dos datasets
│   └── landmarks/            # Landmarks extraídos (opcional)
├── notebooks/                # Notebooks de treinamento
│   ├── train_alphabet.ipynb  # Treinamento do modelo alfabeto
│   └── train_phrase.ipynb    # Treinamento do modelo frases
├── Dockerfile               # Containerização (opcional)
└── README.md               # Este arquivo
```

## 🎯 Uso do Sistema

### Iniciando o Sistema

1. **Inicie o Backend**:
   ```bash
   cd backend
   python app.py
   ```
   O servidor estará disponível em `http://localhost:5000`

2. **Abra o Frontend**:
   - Navegue até `frontend/index.html` no navegador
   - Ou use um servidor web local

### Usando a Interface

1. **Permissões da Câmera**: Autorize o acesso à webcam quando solicitado
2. **Iniciar Captura**: Clique em "Iniciar Câmera"
3. **Fazer Sinais**: Posicione a mão na frente da câmera e faça sinais LIBRAS
4. **Ver Tradução**: O texto traduzido aparecerá em tempo real
5. **Ouvir Áudio**: O sistema reproduzirá automaticamente o áudio da tradução
6. **Repetir Som**: Use o botão "Repetir Som" para ouvir novamente

### Sinais Suportados

#### Alfabeto LIBRAS (24 letras)
```
A B C D E F G H I K L M N O P Q R S T U V W X Y
```
*Nota: J e Z não são suportadas pois requerem movimento*

#### Frases Comuns (20 frases)
- **Cumprimentos**: Olá, Tchau, Até logo
- **Cortesia**: Obrigado, Por favor, Desculpa
- **Respostas**: Sim, Não, Tudo bem
- **Saudações**: Bom dia, Boa tarde, Boa noite
- **Expressões**: Eu te amo
- **Substantivos**: Família, Casa, Trabalho, Escola, Água, Comida
- **Pedidos**: Ajuda

## ⚙️ Configuração Avançada

### Variáveis de Ambiente

Crie um arquivo `.env` no diretório `backend/`:

```env
# Configurações do Servidor
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False

# Configurações TTS
TTS_ENGINE=gtts  # ou pyttsx3
TTS_LANGUAGE=pt
TTS_CACHE_SIZE=50

# Configurações do Modelo
MODEL_CONFIDENCE_THRESHOLD=0.3
PHRASE_CONFIDENCE_THRESHOLD=0.4
TEMPORAL_SMOOTHING=True
PREDICTION_COOLDOWN=1.0
```

### Parâmetros de Inferência

Edite `backend/app.py` para ajustar:

```python
# Suavização temporal
STABILITY_THRESHOLD = 0.6  # 60% das predições devem ser iguais
MIN_PREDICTIONS = 5        # Mínimo de predições antes da decisão
PREDICTION_COOLDOWN = 1.0  # Segundos entre predições
```

### Configurações MediaPipe

Edite `frontend/main.js`:

```javascript
hands.setOptions({
    maxNumHands: 1,              // Máximo de mãos
    modelComplexity: 1,          // Complexidade do modelo (0-2)
    minDetectionConfidence: 0.7, // Confiança mínima detecção
    minTrackingConfidence: 0.5   // Confiança mínima rastreamento
});
```

## 🧠 Treinamento dos Modelos

### Preparar Ambiente

```bash
cd notebooks
pip install jupyter notebook
jupyter notebook
```

### Treinar Modelo do Alfabeto

1. Abra `train_alphabet.ipynb`
2. Execute todas as células em ordem
3. O modelo será salvo em `backend/models/alphabet_model.pt`

### Treinar Modelo de Frases

1. Abra `train_phrase.ipynb`
2. Execute todas as células em ordem
3. O modelo será salvo em `backend/models/phrase_model.pt`

### Datasets

Os notebooks utilizam datasets sintéticos por padrão. Para usar datasets reais:

1. Baixe os datasets públicos documentados em `data/datasets_publicos.txt`
2. Modifique as seções de carregamento de dados nos notebooks
3. Re-execute o treinamento

## 🐳 Docker (Opcional)

### Construir Imagem

```bash
docker build -t libras-live .
```

### Executar Container

```bash
docker run -p 5000:5000 -p 8000:8000 libras-live
```

### Docker Compose

```yaml
version: '3.8'
services:
  libras-live:
    build: .
    ports:
      - "5000:5000"
      - "8000:8000"
    volumes:
      - ./backend/models:/app/backend/models
```

## 🔧 Solução de Problemas

### Problemas Comuns

#### 1. Câmera não funciona
```bash
# Verificar permissões do navegador
# Chrome: chrome://settings/content/camera
# Firefox: about:preferences#privacy
```

#### 2. Modelos não carregam
```bash
# Verificar se os arquivos existem
ls -la backend/models/
# Executar notebooks de treinamento se necessário
```

#### 3. SocketIO não conecta
```bash
# Verificar se o backend está rodando
curl http://localhost:5000
# Verificar firewall/proxy
```

#### 4. TTS não funciona
```bash
# Instalar dependências TTS
pip install gtts pyttsx3
# Verificar conexão internet (para gTTS)
```

### Logs e Debug

```bash
# Backend logs
cd backend
python app.py --debug

# Frontend logs
# Abrir DevTools no navegador (F12)
# Verificar Console e Network tabs
```

### Performance

```bash
# Verificar uso de GPU
python -c "import torch; print(torch.cuda.is_available())"

# Monitor recursos
htop  # Linux/Mac
# Task Manager no Windows
```

## 📊 Métricas de Performance

### Benchmarks Esperados

| Métrica | Valor Esperado |
|---------|---------------|
| Latência Total | < 1000ms |
| FPS MediaPipe | 15-30 fps |
| Acurácia Alfabeto | > 85% |
| Acurácia Frases | > 75% |
| Tempo Inferência | < 100ms |

### Monitoramento

A interface exibe métricas em tempo real:
- **Latência**: Tempo total de processamento
- **FPS**: Quadros por segundo do MediaPipe
- **Predições/min**: Taxa de reconhecimento

## 🤝 Contribuição

Contribuições são bem-vindas! Veja como ajudar:

### 1. Reportar Problemas

- Use GitHub Issues para bugs e sugestões
- Inclua logs, screenshots e informações do sistema
- Descreva passos para reproduzir o problema

### 2. Melhorar Código

```bash
# Fork do repositório
git fork https://github.com/seu-usuario/libras-live

# Criar branch para feature
git checkout -b minha-feature

# Fazer alterações e commits
git add .
git commit -m "Adiciona nova funcionalidade"

# Abrir Pull Request
git push origin minha-feature
```

### 3. Adicionar Dados

- Contribua com novos sinais LIBRAS
- Valide sinais existentes com especialistas
- Melhore datasets sintéticos

### 4. Documentação

- Corrija erros na documentação
- Adicione tutoriais e exemplos
- Traduza para outros idiomas

## 📚 Recursos Adicionais

### Aprender LIBRAS
- [Instituto Nacional de Educação de Surdos (INES)](https://www.ines.gov.br/)
- [Dicionário de Libras](https://www.dicionariolibras.com.br/)
- [Curso de Libras Online](https://www.cursoslibras.com.br/)

### Tecnologias Utilizadas
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [PyTorch](https://pytorch.org/)
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/)
- [gTTS - Google Text-to-Speech](https://pypi.org/project/gTTS/)

### Datasets
- [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist)
- [Libras Movement Dataset](https://archive.ics.uci.edu/ml/datasets/libras+movement)

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🏆 Reconhecimentos

- **MediaPipe Team** por fornecer a solução de detecção de mãos
- **Comunidade LIBRAS** por recursos educacionais
- **Datasets Públicos** mencionados em `data/datasets_publicos.txt`

## 📞 Suporte

- **GitHub Issues**: Para bugs e solicitações de recursos
- **Discussions**: Para perguntas gerais e discussões
- **Email**: joao.bruschi@outlook.com.br
- **LinkedIn**: [Via mensagem direta](https://www.linkedin.com/in/joaobruschi/)
  
---

# 📚 Guia Rápido - Datasets LIBRAS

## 🚀 Como Usar (Super Simples)

### 1. Organize suas imagens

Coloque suas imagens assim:

```
data/raw/custom_images/
├── A/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── B/
│   ├── image1.png
│   └── ...
└── ...
```

### 2. Execute o script

```bash
python data/setup.py
```

Pronto! O script vai:
- ✅ Processar todas as imagens
- ✅ Extrair landmarks com MediaPipe
- ✅ Combinar com dataset existente
- ✅ Gerar dataset final para treinamento

### 3. Treine o modelo

```bash
python notebooks/train_alphabet_standalone.py
```

---

## 📥 Onde Conseguir Datasets Públicos?

### Kaggle (Recomendado)

1. **Sign Language MNIST**
   - Link: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
   - Já está no projeto!

2. **Hand Gesture Recognition**
   - Link: https://www.kaggle.com/datasets/kuchhbhi/hand-gesture-recognition-dataset
   - Baixe e organize por classe

3. **Sign Language Digits**
   - Link: https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset

### Como Baixar do Kaggle

**Opção 1: Manual**
1. Acesse o dataset no Kaggle
2. Clique em "Download"
3. Extraia o ZIP
4. Organize as imagens em `data/raw/custom_images/[LETRA]/`

**Opção 2: API do Kaggle**
```bash
# Instalar
pip install kaggle

# Configurar (obter token em kaggle.com/settings)
# Colocar em: ~/.kaggle/kaggle.json

# Baixar
kaggle datasets download -d datamunge/sign-language-mnist
```

---

## 📁 Estrutura de Pastas

```
data/
├── raw/                    # Dados brutos
│   └── custom_images/      # Suas imagens aqui
│       ├── A/
│       ├── B/
│       └── ...
├── processed/              # Dados processados (gerado automaticamente)
│   ├── alphabet_dataset.npz
│   └── ...
└── setup.py                # Script principal
```

---

## ⚙️ Scripts Disponíveis

### `setup.py` (Recomendado)
Script master que faz tudo:
```bash
python data/setup.py
```

### `process_datasets.py`
Apenas processa datasets:
```bash
python data/process_datasets.py
```

### `download_public_datasets.py`
Informações sobre datasets públicos:
```bash
python data/download_public_datasets.py
```

---

## 💡 Dicas

1. **Mais dados = Melhor acurácia**
   - Tente ter pelo menos 50-100 imagens por classe

2. **Qualidade importa**
   - Imagens claras com mãos bem visíveis funcionam melhor

3. **Organização**
   - Mantenha as imagens organizadas por classe (A/, B/, C/, etc.)

4. **Formatos suportados**
   - PNG, JPG, JPEG, BMP

---

## ❓ Problemas Comuns

**"Nenhum dataset encontrado"**
- Verifique se as imagens estão em `data/raw/custom_images/[LETRA]/`
- Verifique se os nomes das pastas são letras (A, B, C, etc.)

**"MediaPipe não detecta mãos"**
- Verifique qualidade das imagens
- Certifique-se de que as mãos estão bem visíveis
- Tente imagens com fundo simples

**"Dataset muito pequeno"**
- Baixe mais datasets do Kaggle
- Use data augmentation (rotação, escala, etc.)

---

## 📊 Resultados Esperados

Com datasets adicionais, você deve ver:
- ✅ Mais de 1000 amostras (idealmente 2000+)
- ✅ Acurácia > 85% no teste
- ✅ Melhor distinção entre sinais similares

---