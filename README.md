<div align="center">

# 🤟 LibrasLive

**Sistema de Tradução em Tempo Real para LIBRAS (Língua Brasileira de Sinais)**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://github.com)

*Traduza sinais LIBRAS em tempo real para texto e áudio usando inteligência artificial*

[Características](#-características) • [Instalação](#-instalação) • [Uso](#-como-usar) • [Documentação](#-documentação) • [Contribuir](#-contribuindo)

</div>

---

## 📖 Sobre o Projeto

**LibrasLive** é uma aplicação web que utiliza aprendizado de máquina para reconhecer e traduzir sinais da Língua Brasileira de Sinais (LIBRAS) em tempo real. O sistema captura gestos das mãos através da webcam, processa os dados usando MediaPipe e modelos de deep learning treinados, e converte os sinais em texto e áudio.

### 🎯 Objetivo

Facilitar a comunicação entre pessoas surdas e ouvintes, promovendo inclusão e acessibilidade através de tecnologia de ponta.

### ✨ Destaques

- 🚀 **Tempo Real**: Latência inferior a 1 segundo
- 🎯 **Alta Precisão**: Acurácia de 99.63% no reconhecimento do alfabeto
- 🧠 **Deep Learning**: Modelos PyTorch com arquitetura avançada
- 📱 **Web-Based**: Funciona diretamente no navegador, sem instalação
- 🔊 **TTS Integrado**: Síntese de voz automática
- 📊 **Métricas em Tempo Real**: FPS, latência e estatísticas

---

## 🌟 Características

### Reconhecimento de Alfabeto
- ✅ **24 letras** do alfabeto LIBRAS (A-Z)
- ✅ **Acurácia de 99.63%** no conjunto de teste
- ✅ **4.900 amostras** de treinamento por base, podendo ser escalável
- ✅ **Arquitetura profunda**: 4 camadas com BatchNorm e regularização

### Reconhecimento de Frases
- ✅ **20 frases comuns** em LIBRAS
- ✅ **Acurácia de 84.44%** no conjunto de teste
- ✅ **Modelo LSTM** para sequências temporais
- ✅ **Mapeamento inteligente** de classes

### Interface e Experiência
- ✅ **Interface moderna e responsiva**
- ✅ **Visualização de landmarks** em tempo real
- ✅ **Histórico de traduções**
- ✅ **Botão de repetir áudio**
- ✅ **Métricas de performance** (FPS, latência)
- ✅ **Feedback visual** durante reconhecimento

### Tecnologias
- ✅ **MediaPipe Hands** para detecção de mãos
- ✅ **PyTorch** para inferência de modelos
- ✅ **Flask + SocketIO** para comunicação em tempo real
- ✅ **gTTS** para síntese de voz
- ✅ **WebSocket** para baixa latência

---

## 📋 Requisitos

### Hardware
- **Webcam**: Resolução mínima 640x480
- **Processador**: CPU multi-core (4+ cores recomendado)
- **Memória**: 4GB RAM mínimo, 8GB recomendado
- **GPU**: Opcional (CUDA para acelerar inferência)

### Software
- **Python**: 3.8 ou superior
- **Navegador**: Chrome 80+, Firefox 75+, Safari 13+, Edge 80+
- **Sistema Operacional**: Windows, Linux ou macOS

---

## 🚀 Instalação

### Pré-requisitos

Certifique-se de ter o Python 3.8+ instalado:

```bash
python --version
# Python 3.8.0 ou superior
```

### Passo 1: Clonar o Repositório

```bash
git clone https://github.com/moonlitrevery/libras-live.git
cd libras-live
```

### Passo 2: Instalar Dependências

```bash
cd backend
pip install -r requirements.txt
```

**Nota**: Em sistemas Linux/Mac, use `pip3` em vez de `pip`.

### Passo 3: Verificar Modelos

Os modelos pré-treinados devem estar em `backend/models/`:
- `alphabet_model.pt`
- `phrase_model.pt`
- `normalization_params.npz`
- `phrase_normalization_params.npz`

Se os modelos não existirem, veja a seção [Treinamento](#-treinamento-dos-modelos).

### Passo 4: Executar o Backend

```bash
python app.py
```

O servidor estará disponível em `http://localhost:5000`

### Passo 5: Abrir o Frontend

**Opção 1: Abrir diretamente**
- Navegue até `frontend/index.html` e abra no navegador

**Opção 2: Servidor local (recomendado)**
```bash
cd frontend
python -m http.server 8000
# Acesse http://localhost:8000
```

---

## 🎮 Como Usar

### 1. Iniciar o Sistema

1. **Inicie o backend** (terminal 1):
   ```bash
   cd backend
   python app.py
   ```

2. **Abra o frontend** no navegador:
   - Acesse `http://localhost:8000` (se usando servidor)
   - Ou abra `frontend/index.html` diretamente

### 2. Usar a Interface

1. **Permitir acesso à câmera**: Autorize quando o navegador solicitar
2. **Clicar em "Iniciar Câmera"**: A webcam será ativada
3. **Posicionar a mão**: Mantenha a mão visível na frente da câmera
4. **Fazer sinais LIBRAS**: O sistema reconhecerá automaticamente
5. **Ver tradução**: O texto aparecerá em tempo real
6. **Ouvir áudio**: O sistema reproduzirá automaticamente

### 3. Sinais Suportados

#### Alfabeto LIBRAS (24 letras)
```
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
```

**Nota**: J e Z podem ser suportados, dependendo da escalabilidade dos datasets treinados pelo projeto.

#### Frases Comuns (20 frases)

**Cumprimentos:**
- Olá, Tchau, Até logo

**Cortesia:**
- Obrigado, Por favor, Desculpa

**Respostas:**
- Sim, Não, Tudo bem

**Saudações:**
- Bom dia, Boa tarde, Boa noite

**Expressões:**
- Eu te amo

**Substantivos:**
- Família, Casa, Trabalho, Escola, Água, Comida

**Pedidos:**
- Ajuda

---

## 🏗️ Arquitetura do Projeto

```
libras-live/
├── backend/                    # Backend Python (Flask + SocketIO)
│   ├── app.py                 # Servidor principal Flask
│   ├── infer.py               # Motor de inferência (MLP + LSTM)
│   ├── tts.py                 # Módulo de síntese de voz
│   ├── requirements.txt        # Dependências Python
│   ├── models/                # Modelos treinados
│   │   ├── alphabet_model.pt
│   │   ├── phrase_model.pt
│   │   └── *_normalization_params.npz
│   └── temp_audio/            # Cache de áudio TTS
│
├── frontend/                  # Frontend Web (HTML/CSS/JS)
│   ├── index.html            # Interface principal
│   ├── main.js               # Lógica JavaScript + MediaPipe
│   └── styles.css            # Estilos CSS
│
├── data/                     # Datasets e processamento
│   ├── raw/                  # Datasets brutos
│   │   ├── sign_language_mnist/
│   │   ├── leapGestRecog/
│   │   ├── Sign-language-digits-dataset/
│   │   └── libras_movement/
│   ├── processed/            # Datasets processados
│   ├── process_all_datasets.py
│   ├── prepare_datasets.py
│   └── download_public_datasets.py
│
├── notebooks/                # Notebooks e scripts de treinamento
│   ├── train_alphabet_standalone.py
│   ├── train_phrase_standalone.py
│   ├── train_alphabet.ipynb
│   └── train_phrase.ipynb
│
├── train_all.py             # Script master de treinamento
├── Dockerfile               # Containerização
├── docker-compose.yml       # Orquestração Docker
└── README.md               # Este arquivo
```

### Fluxo de Dados

```
Webcam → MediaPipe → Landmarks → Normalização → Modelo ML → Predição → TTS → Áudio
```

1. **Captura**: Webcam captura frames de vídeo
2. **Detecção**: MediaPipe detecta mãos e extrai 21 landmarks
3. **Normalização**: Landmarks são normalizados usando parâmetros de treinamento
4. **Inferência**: Modelo PyTorch faz predição (alfabeto ou frase)
5. **Suavização**: Predições são suavizadas temporalmente
6. **TTS**: Texto é convertido em áudio
7. **Exibição**: Resultado é exibido na interface

---

## 🧠 Treinamento dos Modelos

### Treinamento Completo (Recomendado)

Execute o script master que processa datasets e treina ambos os modelos:

```bash
python train_all.py
```

Este script:
1. ✅ Processa todos os datasets disponíveis
2. ✅ Treina modelo de alfabeto (MLP)
3. ✅ Treina modelo de frases (LSTM)

**Tempo estimado**: 30-60 minutos (dependendo do hardware)

### Treinamento Individual

#### 1. Processar Datasets

```bash
python data/process_all_datasets.py
```

Processa e combina:
- Sign Language MNIST
- Leap Gesture Recognition
- Sign Language Digits
- UCI Libras Movement (frases)

#### 2. Treinar Modelo de Alfabeto

```bash
python notebooks/train_alphabet_standalone.py
```

**Especificações:**
- Arquitetura: MLP com 4 camadas ocultas (256 neurônios)
- BatchNorm após cada camada
- Data augmentation (ruído gaussiano)
- Dropout: 0.4
- Épocas: 200 (com early stopping)
- Acurácia esperada: >99%

#### 3. Treinar Modelo de Frases

```bash
python notebooks/train_phrase_standalone.py
```

**Especificações:**
- Arquitetura: LSTM com 3 camadas (128 neurônios)
- Input projection layer
- BatchNorm no classificador
- Dropout: 0.4
- Épocas: 200 (com early stopping)
- Acurácia esperada: >84%

### Datasets Utilizados

| Dataset | Tipo | Amostras | Fonte |
|---------|------|----------|-------|
| Sign Language MNIST | Imagens | ~27,000 | [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) |
| Leap Gesture Recognition | Imagens | ~20,000 | [Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) |
| Sign Language Digits | Landmarks | ~2,000 | [Kaggle](https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset) |
| UCI Libras Movement | Sequências | ~360 | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/libras+movement) |

---

## ⚙️ Configuração Avançada

### Variáveis de Ambiente

Crie um arquivo `.env` em `backend/`:

```env
# Servidor
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False

# TTS
TTS_ENGINE=gtts
TTS_LANGUAGE=pt
TTS_CACHE_SIZE=50

# Modelos
MODEL_CONFIDENCE_THRESHOLD=0.3
PHRASE_CONFIDENCE_THRESHOLD=0.6
TEMPORAL_SMOOTHING=True
PREDICTION_COOLDOWN=0.5
```

### Parâmetros de Inferência

Edite `backend/app.py`:

```python
# Suavização temporal
STABILITY_THRESHOLD = 0.4      # 40% das predições devem ser iguais
MIN_PREDICTIONS = 3           # Mínimo de predições antes da decisão
PREDICTION_COOLDOWN = 0.5     # Segundos entre predições
```

### Configurações MediaPipe

Edite `frontend/main.js`:

```javascript
hands.setOptions({
    maxNumHands: 1,                    // Máximo de mãos
    modelComplexity: 1,                 // Complexidade (0-2)
    minDetectionConfidence: 0.7,       // Confiança mínima detecção
    minTrackingConfidence: 0.5          // Confiança mínima rastreamento
});
```

---

## 📊 Performance

### Métricas de Acurácia

| Modelo | Acurácia Teste | Acurácia Validação | Dataset Treino |
|--------|----------------|-------------------|----------------|
| Alfabeto (MLP) | **99.63%** | 99.33% | 4,900 amostras |
| Frases (LSTM) | **84.44%** | ~85% | ~300 sequências |

### Benchmarks de Latência

| Métrica | Valor Esperado | Observado |
|---------|---------------|-----------|
| Latência Total | < 1000ms | ~500-800ms |
| FPS MediaPipe | 15-30 fps | 20-25 fps |
| Tempo Inferência | < 100ms | ~50-80ms |
| Tempo TTS | < 500ms | ~200-400ms |

### Monitoramento em Tempo Real

A interface exibe:
- **FPS**: Quadros por segundo do MediaPipe
- **Latência**: Tempo total de processamento
- **Predições/min**: Taxa de reconhecimento
- **Confiança**: Nível de confiança da predição

---

## 🐳 Docker

### Construir Imagem

```bash
docker build -t libras-live .
```

### Executar Container

```bash
docker run -p 5000:5000 -p 8000:8000 libras-live
```

### Docker Compose

```bash
docker-compose up -d
```

O arquivo `docker-compose.yml` já está configurado.

---

## 🔧 Solução de Problemas

### Problemas Comuns

#### 1. Câmera não funciona

**Solução:**
- Verifique permissões do navegador
- Chrome: `chrome://settings/content/camera`
- Firefox: `about:preferences#privacy`
- Certifique-se de usar HTTPS ou localhost

#### 2. Modelos não carregam

**Solução:**
```bash
# Verificar se os arquivos existem
ls -la backend/models/

# Se não existirem, treinar modelos
python train_all.py
```

#### 3. SocketIO não conecta

**Solução:**
```bash
# Verificar se backend está rodando
curl http://localhost:5000

# Verificar firewall/proxy
# Verificar CORS no app.py
```

#### 4. TTS não funciona

**Solução:**
```bash
# Instalar dependências TTS
pip install gtts pyttsx3

# Verificar conexão internet (gTTS requer internet)
# Verificar permissões de escrita em temp_audio/
```

#### 5. Baixa acurácia

**Solução:**
- Verificar iluminação e posicionamento da mão
- Certificar-se de que a mão está completamente visível
- Verificar se os modelos foram treinados corretamente
- Ajustar thresholds de confiança em `backend/infer.py`

### Logs e Debug

**Backend:**
```bash
cd backend
python app.py --debug
```

**Frontend:**
- Abrir DevTools (F12)
- Verificar Console e Network tabs
- Verificar erros de WebSocket

### Performance

**Verificar GPU:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Monitorar recursos:**
- Linux/Mac: `htop`
- Windows: Task Manager

---

## 🤝 Contribuindo

Contribuições são muito bem-vindas! Veja como ajudar:

### 1. Reportar Problemas

- Use [GitHub Issues](https://github.com/moonlitrevery/libras-live/issues)
- Inclua logs, screenshots e informações do sistema
- Descreva passos para reproduzir o problema

### 2. Melhorar Código

```bash
# Fork do repositório
git clone https://github.com/moonlitrevery/libras-live.git

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
- Adicione mais amostras para classes raras (U, N, V, Y)

### 4. Documentação

- Corrija erros na documentação
- Adicione tutoriais e exemplos
- Traduza para outros idiomas
- Melhore este README

### 5. Testes

- Adicione testes unitários
- Teste em diferentes navegadores
- Valide com usuários reais

---

## 📚 Tecnologias Utilizadas

### Backend
- **Flask 3.0.0**: Framework web
- **Flask-SocketIO 5.3.6**: WebSocket para tempo real
- **PyTorch 2.1.0**: Deep learning
- **MediaPipe 0.10.7**: Detecção de mãos
- **gTTS 2.4.0**: Síntese de voz
- **NumPy 1.24.3**: Computação numérica
- **scikit-learn 1.3.0**: Machine learning

### Frontend
- **MediaPipe Hands**: Detecção de mãos no navegador
- **Socket.IO Client**: Comunicação WebSocket
- **HTML5/CSS3/JavaScript**: Interface web

### Datasets
- **Sign Language MNIST**: Alfabeto ASL
- **Leap Gesture Recognition**: Gestos de mão
- **Sign Language Digits**: Dígitos em sinais
- **UCI Libras Movement**: Movimentos LIBRAS

---

## 📄 Licença

Este projeto está licenciado sob a **Licença MIT** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 🏆 Reconhecimentos

- **MediaPipe Team** por fornecer a solução de detecção de mãos
- **Comunidade LIBRAS** por recursos educacionais
- **Datasets Públicos** mencionados na documentação
- **Contribuidores** que ajudaram a melhorar o projeto

---

## 📞 Suporte e Contato

- **GitHub Issues**: [Reportar bugs](https://github.com/moonlitrevery/libras-live/issues)
- **GitHub Discussions**: [Perguntas gerais](https://github.com/moonlitrevery/libras-live/discussions)
- **Email**: joao.bruschi@outlook.com.br
- **LinkedIn**: [João Bruschi](https://www.linkedin.com/in/joaobruschi/)

---

## 📖 Recursos Adicionais

### Aprender LIBRAS
- [Instituto Nacional de Educação de Surdos (INES)](https://www.ines.gov.br/)
- [Dicionário de Libras](https://www.dicionariolibras.com.br/)
- [Curso de Libras Online](https://www.cursoslibras.com.br/)

### Documentação Técnica
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/)

---

<div align="center">

⭐ **Se este projeto foi útil, considere dar uma estrela!** ⭐

[⬆ Voltar ao topo](#-libraslive)

</div>
