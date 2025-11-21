# Sistema de Reconhecimento de Sinais em LIBRAS

Sistema de reconhecimento de sinais da Língua Brasileira de Sinais (LIBRAS) utilizando redes neurais LSTM e MediaPipe para extração de características.

## Descrição do Projeto

Este projeto implementa um sistema completo de reconhecimento de gestos em LIBRAS, composto por módulos de coleta de dados, treinamento de modelo de aprendizado profundo e interface gráfica interativa para prática e avaliação em tempo real.

## Estrutura do Projeto

```
.
├── libras_trainer.py          # Aplicação principal com interface gráfica
├── treinamento.py              # Script de treinamento do modelo LSTM
├── coletando_videos.py         # Coleta de vídeos e extração de pontos MediaPipe
├── teste_acuracia.py           # Avaliação de acurácia do modelo
├── MP_Data_Novos/              # Dataset com pontos MediaPipe extraídos para testes
├── MP_Data/                    # Dataset com pontos MediaPipe extraídos para treinamento do modelo
├── checkpoints/                # Modelos treinados (.keras)
├── assets/                     # Vídeos didáticos das fases
└── requirements.txt            # Dependências do projeto
```

## Requisitos do Sistema

- **Python**: 3.9.13
- **Sistema Operacional**: Windows 10/11 (recomendado devido ao MediaPipe)
- **Hardware**: Webcam para captura em tempo real
- **GPU**: Opcional, mas recomendada para treinamento

## Instalação

### 1. Clonar o Repositório

```bash
git clone https://github.com/bberaldo/ensino-libras-lstm.git
cd ensino-libras-lstm
```

### 2. Criar Ambiente Virtual

```bash
python -m venv venv
```

### 3. Ativar o Ambiente Virtual

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Instalar Dependências

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` contém todas as bibliotecas necessárias com suas versões específicas para garantir compatibilidade.

## Utilização dos Módulos

### 1. Aplicação Interativa (`libras_trainer.py`)

Interface gráfica principal para prática e aprendizado de LIBRAS através de fases progressivas.

**Execução:**

```bash
python libras_trainer.py
```

**Funcionalidades:**

- 6 fases de aprendizado progressivo
- Reconhecimento em tempo real via webcam
- Vídeos didáticos demonstrativos
- Feedback visual instantâneo
- Sistema de validação por confidence threshold

**Sinais reconhecidos:**

- bom-bem
- dia
- oi
- joia
- eu
- amo
- você
- obrigado
- desculpa
- pessoa
- brasil

### 2. Coleta de Dados (`coletando_videos.py`)

Script responsável pela captura de vídeos via webcam e extração automática dos pontos de referência (landmarks) do MediaPipe Holistic.

**Execução:**

```bash
python coletando_videos.py
```

**Funcionalidades:**

- Captura de sequências de vídeo para cada classe de sinal
- Extração de 1662 pontos de referência por frame (pose, face, mãos)
- Armazenamento estruturado em `MP_Data_Novos/`

### 3. Treinamento do Modelo (`treinamento.py`)

Implementa e treina a rede neural LSTM para reconhecimento sequencial dos sinais.

**Execução:**

```bash
python treinamento.py
```

**Características do modelo:**

- Arquitetura: LSTM bidirecional
- Entrada: Sequências de 30 frames com 144 features reduzidas
- Saída: Classificação probabilística entre 11 classes de sinais

### 4. Avaliação de Acurácia (`teste_acuracia.py`)

Avalia o desempenho do modelo treinado utilizando os dados da pasta `MP_Data_Novos/`.

**Execução:**

```bash
python teste_acuracia.py
```

**Métricas calculadas:**

- Acurácia geral
- Matriz de confusão
- Precisão, recall e F1-score por classe

## Estrutura de Dados

### MP_Data_Novos/

Contém as sequências de pontos MediaPipe organizadas por classe:

```
MP_Data/
├── bom-bem/
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
├── dia/
├── oi/
└── ...
```

Cada arquivo `.npy` contém um array NumPy de shape `(num_frames, 1662)` com as coordenadas normalizadas dos landmarks.

### Checkpoints/

Armazena os modelos treinados no formato Keras:

```
checkpoints/
└── final_model-v2.keras
```

## Configuração e Parâmetros

## Solução de Problemas

### Erro ao abrir webcam

Verifique se a webcam está conectada e não está sendo utilizada por outro aplicativo. O sistema tenta múltiplos backends (MSMF, DSHOW) automaticamente.

### Modelo não encontrado

Certifique-se de que o arquivo `checkpoints/final_model-v2.keras` existe. Execute `treinamento.py` para gerar um novo modelo.

### Erros de dependências

Reinstale as dependências com versões exatas:

```bash
pip install -r requirements.txt --force-reinstall
```

## Referências

- MediaPipe Holistic: https://google.github.io/mediapipe/solutions/holistic
- TensorFlow/Keras: https://www.tensorflow.org/
- LIBRAS: Língua Brasileira de Sinais

## Trabalho de Conclusão de Curso

Este projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC) para Ciências da Computação, sob orientação de Prof. Me. Amaury Bosso André e Prof. Me. Sergio Eduardo Nunes.
