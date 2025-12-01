import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os, numpy as np, tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report, confusion_matrix

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Rosto
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Desenha pose 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Mão esquerda
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Mão direita

def draw_styled_landmarks(image, results):
    # Rosto
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          ) 
    # Desenha pose 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Mão esquerda
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Mão direita  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

# extraindo os pontos
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def apply_keep_idx(X, keep_idx):
    # se já está reduzido (mesma largura de keep_idx), devolve como está
    if X.shape[2] == keep_idx.size:
        return X
    # segurança: garante que os índices cabem no D atual
    assert np.max(keep_idx) < X.shape[2], \
        f"max(keep_idx)={np.max(keep_idx)} >= D={X.shape[2]}"
    return X[:, :, keep_idx]

def fix_time_leak(X, D0):
    """
    Conserta casos onde o tempo (T) foi concatenado na última dimensão por frame.
    Garante saída (N, T, D0) real, com cada frame contendo exatamente D0 features.
    """
    N, T, Dtot = X.shape

    # Caso 1: Dtot == T*D0 (clássico "vazou tudo" para o último eixo)
    if Dtot == T * D0 and Dtot != D0:
        X_fixed = np.stack([X[:, t, t*D0:(t+1)*D0] for t in range(T)], axis=1)
        return X_fixed

    # Caso 2: shape parece ok (Dtot == D0), mas o buffer tem T vezes mais elementos
    # (acontece quando uma view/broadcast criou algo "enganado")
    if Dtot == D0:
        # Verifica se alguma sequência tem número de elementos diferente do esperado
        exp_elems = T * D0
        need_fix = False
        for i in range(min(N, 10)):  # checa algumas amostras
            if X[i].size != exp_elems:
                need_fix = True
                break
        if need_fix:
            X_fixed = np.empty((N, T, D0), dtype=X.dtype)
            for i in range(N):
                seq = X[i]
                # se essa sequência já tem o número certo de elementos, só copia
                if seq.size == exp_elems:
                    X_fixed[i] = np.ascontiguousarray(seq)
                else:
                    # reconstrói por frame: pega o bloco correto do frame t
                    for t in range(T):
                        X_fixed[i, t, :] = seq[t, t*D0:(t+1)*D0]
            return X_fixed

    # Caso 3: já está ok
    return np.ascontiguousarray(X)

# setup dos diretórios
DATA_PATH = os.path.join('MP_Data') # Pasta para arquivos exportados

# Gestos
actions = np.array(['bom-bem', 'dia', 'oi', 'joia', 'eu', 'amo', 'voce', 'obrigado', 'desculpa', 'pessoa', 'brasil'])

# 100 vídeos
no_sequences = 100

# Vídeos com 30 frames
sequence_length = 30

# Por onde começa - para novos vídeos
start_folder = 0

# Suffix do modelo e checkpoints
modelSuffix = '-v3'

# pré processamento
label_map = {label:num for num, label in enumerate(actions)}
print("label_map", label_map)

sequences, labels = [], []

for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        print(action, sequence)
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

print("quantidade de vídeos, frames de cada vídeo, quantidade de pontos em cada frame", np.array(sequences).shape)

X_full = np.array(sequences)
Y = np.array(labels)

pose_offset  = 0
face_offset  = 33 * 4
handL_offset = face_offset + 468 * 3
handR_offset = handL_offset + 21 * 3

# índices da POSE (MediaPipe) — ombro, cotovelo, punho
pose_keep = np.array([11, 12, 13, 14, 15, 16])  # esquerdo e direito
# pular o canal "visibility" (só pegar x, y, z)
idx_pose_sup = (pose_keep[:, None] * 4 + np.array([0, 1, 2])).reshape(-1)

# índices das mãos (já são 3 coords cada)
idx_handL = np.arange(handL_offset, handL_offset + 21 * 3)
idx_handR = np.arange(handR_offset, handR_offset + 21 * 3)

# junta tudo
keep_idx = np.concatenate([idx_pose_sup, idx_handL, idx_handR])
print("keep_idx size:", keep_idx.size)
print("max idx:", keep_idx.max())

X = apply_keep_idx(X_full, keep_idx)
# 1661 => 144
print("X reduzido:", X.shape)

# 1) Sanity checks + "desembaralhador" automático (se necessário)
N, T, Dtot = X.shape
D0 = keep_idx.size  # deve ser 144
assert T == 30, f"T inesperado: {T}"
assert D0 == 144, f"D0 inesperado: {D0}"

# Se, por algum motivo, o tempo tiver vazado pra última dimensão (ex.: 4320 = 30*144),
# este patch reconstrói (N, T, D0) pegando o bloco correto de cada frame t.
if Dtot == T * D0 and Dtot != D0:
    X = np.stack([X[:, t, t*D0:(t+1)*D0] for t in range(T)], axis=1)
    print("X reconstituído (tempo voltou ao eixo 1):", X.shape)

assert X.shape[2] == D0 and X.shape[1] == T, f"Shape inesperado pós reconstituir: {X.shape}"

assert X_full.shape[2] > 144
assert keep_idx.max() < X_full.shape[2]

D0 = keep_idx.size  # 144
X = fix_time_leak(X, D0)

X = np.ascontiguousarray(X).astype(np.float32)

num_classes = int(np.max(labels)) + 1
labels = np.array(labels)

y = to_categorical(labels, num_classes=num_classes).astype("float32")

# sanidade
assert labels.min() == 0, "rótulos devem começar em 0"
assert labels.max() == num_classes - 1, "num_classes inconsistente"

cls_all, cnt_all = np.unique(labels, return_counts=True)
print("Distribuição total:", dict(zip(cls_all, cnt_all)))

# ========================
# LSTM
# ========================
# Divisão estratificada: mantém proporção de classes em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=labels, random_state=42
)
y_train_cls = np.argmax(y_train, axis=1)
y_test_cls  = np.argmax(y_test, axis=1)

def dist(lbls): 
    u,c = np.unique(lbls, return_counts=True); 
    return dict(zip(u, c))
print("Train dist:", dist(y_train_cls))
print("Test  dist:", dist(y_test_cls))

print("X_train:", X_train.shape, "y_train:", y_train.shape)  # (880, 30, 144), (880, 11)

X_train = X_train.astype("float32")
X_test  = X_test.astype("float32")
y_train = y_train.astype("float32")
y_test  = y_test.astype("float32")

# Arquitetura do modelo
model = Sequential()

# Camada de entrada: define o formato dos dados (30 timesteps, 144 features)
model.add(Input(shape=(30, X.shape[-1]))) # (T=30, D=144)

# Primeira camada LSTM
model.add(LSTM(
    64,                            # units: 64 neurônios LSTM (captura padrões temporais básicos)
    return_sequences=True,         # True: retorna saída para CADA timestep (necessário para empilhar LSTMs)
    activation='tanh',             # tanh: função de ativação padrão em LSTMs, intervalo [-1, 1]
    dropout=0.2,                   # 20% das conexões de entrada zeradas aleatoriamente (previne overfitting)
    recurrent_dropout=0.2,         # 20% das conexões recorrentes (entre estados) zeradas (regularização temporal)
    kernel_regularizer=regularizers.l2(1e-4)  # Penalização L2: adiciona 0.0001*Σ(peso²) à loss (força pesos menores)
))

# Segunda camada LSTM
model.add(LSTM(
    128,                           # 128 neurônios: DOBRO da primeira camada (captura padrões mais complexos)
    return_sequences=True,         # Ainda precisa retornar sequências para a próxima LSTM
    activation='tanh',             # Mantém tanh para consistência
    dropout=0.2,                   # Mesma taxa de dropout (20%) para regularização uniforme
    recurrent_dropout=0.2,         # Dropout recorrente mantido em 20%
    kernel_regularizer=regularizers.l2(1e-4)  # Mesma regularização L2
))

# Terceira camada LSTM
model.add(LSTM(
    64,                            # Volta para 64 neurônios (compressão antes das camadas densas)
    return_sequences=False,        # FALSE: retorna APENAS o último timestep (colapsa dimensão temporal)
                                   # Saída: (batch, 64) ao invés de (batch, 30, 64)
                                   # Necessário pois Dense não aceita sequências temporais
    activation='tanh',             # tanh mantida
    dropout=0.2,                   # Dropout 20%
    recurrent_dropout=0.2,         # Dropout recorrente 20%
    kernel_regularizer=regularizers.l2(1e-4)  # Regularização L2
))

# Camadas densas com L2 + Dropout

# Primeira camada densa: 64 neurônios totalmente conectados
model.add(Dense(
    64,                            # 64 neurônios (processa representação agregada da sequência)
    activation='tanh',             # tanh: consistência com LSTMs
    kernel_regularizer=regularizers.l2(1e-4)  # Regularização L2 nos pesos
))
model.add(Dropout(0.3))            # Dropout 30% (maior que LSTMs pois Dense tende a overfitting)

# Segunda camada densa: 32 neurônios (redução/gargalo)
model.add(Dense(
    32,                            # 32 neurônios (força representação compacta)
    activation='tanh',             # tanh mantida
    kernel_regularizer=regularizers.l2(1e-4)  # Regularização L2
))
model.add(Dropout(0.3))            # Dropout 30%

# Camada de saída
model.add(Dense(
    num_classes,                   # 11 neurônios (um por classe de gesto)
    activation='softmax'           # softmax: converte em probabilidades que somam 1.0
))                                 # Exemplo saída: [0.05, 0.82, 0.03, ...] → classe 1 = 82%

# AdamW: versão moderna do Adam com weight decay correto
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-3,            # 0.001: taxa de aprendizado (tamanho do passo na descida do gradiente)
    weight_decay=1e-4,             # 0.0001: decaimento de pesos (similar a L2 mas aplicado diretamente)
    clipnorm=1.0                   # Limita norma do gradiente a 1.0 (previne gradientes explosivos em LSTMs)
)

# Compilação do Modelo
model.compile(
    optimizer=optimizer,
    loss=CategoricalCrossentropy(label_smoothing=0.05),  # label_smoothing: suaviza rótulos [0,1,0] → [0.005,0.985,0.005]
                                                          # Previne overconfidence e melhora calibração
    metrics=['categorical_accuracy']  # Métrica: acurácia categórica (% de predições corretas)
)
model.summary()

# Callbacks de treinamento
os.makedirs("checkpoints", exist_ok=True)

# Callback 1: Salva pesos do modelo com MENOR validation loss
cp_best_val_loss = ModelCheckpoint(
    filepath=os.path.join("checkpoints", f"best_val_loss{modelSuffix}.weights.h5"),
    monitor="val_loss",            # monitora loss de validação
    mode="min",                    # salva quando diminui
    save_best_only=True,           # salva APENAS se melhorar
    save_weights_only=True,        # salva só pesos (não arquitetura)
    verbose=1                      # imprime quando salvar
)

# Callback 2: Salva pesos do modelo com MAIOR validation accuracy
cp_best_val_acc = ModelCheckpoint(
    filepath=os.path.join("checkpoints", f"best_val_acc{modelSuffix}.weights.h5"),
    monitor="val_categorical_accuracy",  # monitora acurácia de validação
    mode="max",                    # salva quando aumenta
    save_best_only=True,           # salva APENAS se melhorar
    save_weights_only=True,        # salva só pesos
    verbose=1
)

# Callback 3: Para treinamento se não houver melhoria
early_stopping = EarlyStopping(
    monitor="val_loss",            # monitora loss de validação
    patience=20,                   # espera 20 épocas SEM melhoria antes de parar
    restore_best_weights=True,     # ao parar, volta aos MELHORES pesos encontrados
    verbose=1
)

# Callback 4: Reduz learning rate quando estagnar
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",            # monitora loss de validação
    factor=0.5,                    # multiplica LR por 0.5 (reduz pela metade)
    patience=8,                    # espera 8 épocas sem melhoria antes de reduzir
    min_lr=1e-5,                   # LR mínimo: 0.00001 (não reduz abaixo disso)
    verbose=1
)

# Callback 5: TensorBoard para visualização de métricas
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Treinamento do modelo
history = model.fit(
    X_train, y_train,              # dados de treinamento
    validation_data=(X_test, y_test),  # dados de validação (avalia a cada época)
    epochs=1500,                   # máximo de 1500 épocas (EarlyStopping geralmente para antes)
    batch_size=16,                 # processa 16 amostras por vez antes de atualizar pesos
                                   # Menor batch = mais ruído, pode escapar de mínimos locais
                                   # Maior batch = mais estável mas menos exploração
    callbacks=[cp_best_val_loss, cp_best_val_acc, early_stopping, reduce_lr, tb_callback],
    shuffle=True                   # embaralha ordem dos dados a cada época (evita aprender ordem)
)

model.summary()

# Salva modelo completo (arquitetura + pesos) no formato Keras
model.save(f"checkpoints/final_model{modelSuffix}.keras")

# ==========================
# AVALIAÇÃO E MÉTRICAS
# ==========================

# Predições no conjunto de teste
y_pred = model.predict(X_test, verbose=0)
y_hat  = y_pred.argmax(1)  # converte probabilidades em classe predita

# Relatório de classificação: precision, recall, F1-score por classe
print(classification_report(y_test.argmax(1), y_hat, target_names=list(actions), digits=4))

# Matriz de confusão: linhas=real, colunas=predito
print("matriz de confusão:", confusion_matrix(y_test.argmax(1), y_hat))

# Teste de sanidade: treina com rótulos embaralhados (deve ter acurácia ~9% = aleatória)
y_shuf = np.random.permutation(y_train)
model_shuf = tf.keras.models.clone_model(model); model_shuf.build((None,30,X.shape[-1]))
model_shuf.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                   metrics=['categorical_accuracy'])
model_shuf.fit(X_train, y_shuf, epochs=3, batch_size=16, verbose=0)
print("model evaluate", model_shuf.evaluate(X_test, y_test, verbose=0))

# Converte predições para classes
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

y_pred = model.predict(X_test, verbose=0)
y_hat  = y_pred.argmax(1)

# Relatório detalhado de classificação
print("classification report", classification_report(y_test_cls, y_hat, target_names=list(actions), digits=4))

# Matriz de confusão
print("matriz de confusão", confusion_matrix(y_test_cls, y_hat))

# Acurácia geral
print('accuracy score', accuracy_score(ytrue, yhat))

# testando
sample = []
gesture = 'dia'
seq_id = '29'
for f in range(30):
    full = np.load(os.path.join(DATA_PATH, gesture, seq_id, f"{f}.npy"))  # carrega frame completo (1662 features)
    reduced = full[keep_idx].astype(np.float32)                            # reduz para 144 features
    sample.append(reduced)
arr = np.array(sample, dtype=np.float32)                                   # (30, 144)

x = np.expand_dims(arr, 0)                                                 # (1, 30, 144) - adiciona dimensão batch
res = model.predict(x, verbose=0)[0]                                       # prediz probabilidades
print("Esperado:", gesture, "| Predito:", actions[np.argmax(res)], "| Conf:", float(res.max()))
