import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

def apply_keep_idx(X, keep_idx):
    # Se os dados já têm exatamente o tamanho esperado, não faz nada
    if X.shape[2] == keep_idx.size:
        return X

    # Garante que nenhum índice extrapole o tamanho atual do eixo de features
    assert np.max(keep_idx) < X.shape[2], \
        f"max(keep_idx)={np.max(keep_idx)} >= D={X.shape[2]}"

    # Retorna apenas as colunas (features) selecionadas
    return X[:, :, keep_idx]

def fix_time_leak(X, D0):
    """
    Conserta casos onde os frames foram concatenados na última
    dimensão por engano. Garante que o formato de saída seja
    (N, T, D0), onde:
      N = número de vídeos
      T = número de frames
      D0 = número de features por frame
    """
    N, T, Dtot = X.shape

    # Caso 1: houve vazamento — todas as features de T frames foram para o último eixo
    if Dtot == T * D0 and Dtot != D0:
        X_fixed = np.stack([X[:, t, t*D0:(t+1)*D0] for t in range(T)], axis=1)
        return X_fixed

    # Caso 2: formato parece correto, mas os dados internos podem estar "esticados"
    if Dtot == D0:
        exp_elems = T * D0
        need_fix = False

        # Verifica algumas amostras para detectar inconsistência
        for i in range(min(N, 10)):
            if X[i].size != exp_elems:
                need_fix = True
                break

        # Se houver inconsistência, reconstrói amostra por amostra
        if need_fix:
            X_fixed = np.empty((N, T, D0), dtype=X.dtype)
            for i in range(N):
                seq = X[i]

                # Se estiver tudo certo, copia diretamente
                if seq.size == exp_elems:
                    X_fixed[i] = np.ascontiguousarray(seq)
                else:
                    # Reconstrói frame por frame
                    for t in range(T):
                        X_fixed[i, t, :] = seq[t, t*D0:(t+1)*D0]
            return X_fixed

    # Caso 3: os dados já estão corretos
    return np.ascontiguousarray(X)

# Definição das classes (gestos) usados no modelo
actions = np.array(['bom-bem', 'dia', 'oi', 'joia', 'eu', 'amo', 
                    'voce', 'obrigado', 'desculpa', 'pessoa', 'brasil'])

# Quantidade de frames por sequência
sequence_length = 30

# Caminho onde estão armazenados os novos dados de teste
DATA_PATH_NEW = 'MP_Data_Novos'

# Mapeamento classe → índice numérico
label_map = {label: num for num, label in enumerate(actions)}

# Listas para armazenar os dados carregados
sequences_new = []
labels_new = []
person_ids = []   # guarda a identidade da pessoa (nome da pasta)
# 0=beatriz | 1=pessoa1 | 2=pessoa2

# Lista de pessoas (pastas) encontradas no diretório
people = sorted(os.listdir(DATA_PATH_NEW))
print("Pessoas encontradas:", people)


# Carregamento das sequências (vídeos) frame a frame.
# Cada pessoa possui pastas organizadas por gesto.
# Dentro de cada gesto há pastas numeradas contendo os 30 frames.
for person in people:
    person_path = os.path.join(DATA_PATH_NEW, person)
    if not os.path.isdir(person_path):
        continue

    for action in actions:
        action_path = os.path.join(person_path, action)
        if not os.path.isdir(action_path):
            continue

        # Cada pasta numérica é um vídeo/sequência
        for seq_folder in np.array(os.listdir(action_path)).astype(int):
            seq_path = os.path.join(action_path, str(seq_folder))
            window = []

            # Carrega cada frame (0.npy até 29.npy)
            for frame_num in range(sequence_length):
                frame_path = os.path.join(seq_path, f"{frame_num}.npy")
                if not os.path.isfile(frame_path):
                    raise FileNotFoundError(f"Faltando frame {frame_num} em {frame_path}")
                res = np.load(frame_path)
                window.append(res)

            sequences_new.append(window)
            labels_new.append(label_map[action])
            person_ids.append(person)

# Converte listas para arrays numpy
X_full_new = np.array(sequences_new)       # (N, 30, D_full)
labels_new = np.array(labels_new)          # (N,)
person_ids = np.array(person_ids)          # (N,)

print("Shape X_full_new:", X_full_new.shape)
print("Total sequências novas:", len(X_full_new))

# Mesmos índices usados no treino para selecionar pose + mãos.
# A MediaPipe fornece todas as landmarks, mas usamos apenas
# o necessário para manter consistência com os dados treinados.
pose_offset  = 0
face_offset  = 33 * 4
handL_offset = face_offset + 468 * 3
handR_offset = handL_offset + 21 * 3

# Landmarks superiores do corpo (ombros → pulsos)
pose_keep = np.array([11, 12, 13, 14, 15, 16])

# Converte landmarks em índices de coordenadas (x, y, z, visibility)
idx_pose_sup = (pose_keep[:, None] * 4 + np.array([0, 1, 2])).reshape(-1)

# Landmarks das mãos
idx_handL = np.arange(handL_offset, handL_offset + 21 * 3)
idx_handR = np.arange(handR_offset, handR_offset + 21 * 3)

# Índices finais usados como input do modelo
keep_idx = np.concatenate([idx_pose_sup, idx_handL, idx_handR])
print("keep_idx size:", keep_idx.size)   # deve dar 144

# Número de features esperado por frame
D0 = keep_idx.size  # 144

# Aplica seleção de features e corrige possíveis erros de estrutura
X_new = apply_keep_idx(X_full_new, keep_idx)
X_new = fix_time_leak(X_new, D0)

# Converte para float32 (igual ao treino)
X_new = np.ascontiguousarray(X_new).astype(np.float32)

# Converte rótulos para one-hot encoding
num_classes = len(actions)
y_new = to_categorical(labels_new, num_classes=num_classes).astype("float32")

# Verificações
print("X_new shape:", X_new.shape)   # (N, 30, 144)
print("y_new shape:", y_new.shape)   # (N, num_classes)

print("len(sequences_new) =", len(sequences_new))
print("len(labels_new)    =", len(labels_new))
print("len(person_ids)    =", len(person_ids))
print("X_full_new.shape[0] =", X_full_new.shape[0])

# Carrega o modelo treinado previamente.
model = load_model("checkpoints/final_model-v2.keras")

print(model.input_shape)

# Avaliação global do modelo nas novas pessoas (dados não vistos).
loss, acc = model.evaluate(X_new, y_new, batch_size=16, verbose=1) # batch_size aqui controla apenas quantas amostras são processadas por vez, não altera precisão
print(f"Acurácia global em NOVAS PESSOAS: {acc:.4f}")

# Predição das classes
y_pred_probs = model.predict(X_new)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = labels_new

# Cálculo da acurácia por pessoa, útil para avaliar generalização.
for person in np.unique(person_ids):
    mask = (person_ids == person)
    acc_person = np.mean(y_pred[mask] == y_true[mask])
    print(f"Pessoa {person}: {acc_person:.4f} de acurácia (N={mask.sum()})")

# Relatório completo de métricas por classe (precisão, recall, F1).
print(classification_report(
    y_true, y_pred, target_names=actions
))

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusão:\n", cm)
