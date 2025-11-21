import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

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

# classes/gestos
actions = np.array(['bom-bem', 'dia', 'oi', 'joia', 'eu', 'amo', 
                    'voce', 'obrigado', 'desculpa', 'pessoa', 'brasil'])

# quantidade de frames
sequence_length = 30

# pasta onde os dados para o testes estão
DATA_PATH_NEW = 'MP_Data_Novos'

# mapeamento das classes
label_map = {label: num for num, label in enumerate(actions)}

sequences_new = []
labels_new = []
person_ids = []   # para sabermos de quem é cada sequência
# 0=beatriz | 1=pessoa1 | 2=pessoa2

people = sorted(os.listdir(DATA_PATH_NEW))
print("Pessoas encontradas:", people)

for person in people:
    person_path = os.path.join(DATA_PATH_NEW, person)
    if not os.path.isdir(person_path):
        continue

    for action in actions:
        action_path = os.path.join(person_path, action)
        if not os.path.isdir(action_path):
            continue

        # cada pasta numérica aqui é uma sequência (vídeo)
        for seq_folder in np.array(os.listdir(action_path)).astype(int):
            seq_path = os.path.join(action_path, str(seq_folder))
            window = []

            for frame_num in range(sequence_length):
                frame_path = os.path.join(seq_path, f"{frame_num}.npy")
                if not os.path.isfile(frame_path):
                    raise FileNotFoundError(f"Faltando frame {frame_num} em {frame_path}")
                res = np.load(frame_path)
                window.append(res)

            sequences_new.append(window)
            labels_new.append(label_map[action])
            person_ids.append(person)

X_full_new = np.array(sequences_new)       # (N, 30, D_full)
labels_new = np.array(labels_new)          # (N,)
person_ids = np.array(person_ids)          # (N,)

print("Shape X_full_new:", X_full_new.shape)
print("Total sequências novas:", len(X_full_new))

# mesmas configurações do treino
pose_offset  = 0
face_offset  = 33 * 4
handL_offset = face_offset + 468 * 3
handR_offset = handL_offset + 21 * 3

pose_keep = np.array([11, 12, 13, 14, 15, 16])
idx_pose_sup = (pose_keep[:, None] * 4 + np.array([0, 1, 2])).reshape(-1)

idx_handL = np.arange(handL_offset, handL_offset + 21 * 3)
idx_handR = np.arange(handR_offset, handR_offset + 21 * 3)

keep_idx = np.concatenate([idx_pose_sup, idx_handL, idx_handR])
print("keep_idx size:", keep_idx.size)   # deve dar 144

D0 = keep_idx.size  # 144

X_new = apply_keep_idx(X_full_new, keep_idx)
X_new = fix_time_leak(X_new, D0)
X_new = np.ascontiguousarray(X_new).astype(np.float32)

num_classes = len(actions)
y_new = to_categorical(labels_new, num_classes=num_classes).astype("float32")

# verificações para controle
print("X_new shape:", X_new.shape)   # (N, 30, 144)
print("y_new shape:", y_new.shape)   # (N, num_classes)

print("len(sequences_new) =", len(sequences_new))
print("len(labels_new)    =", len(labels_new))
print("len(person_ids)    =", len(person_ids))
print("X_full_new.shape[0] =", X_full_new.shape[0])

# carrega modelo
model = load_model("checkpoints/final_model-v2.keras")

print(model.input_shape)

# testes
loss, acc = model.evaluate(X_new, y_new, batch_size=16, verbose=1) # batch_size aqui controla apenas quantas amostras são processadas por vez, não altera precisão
print(f"Acurácia global em NOVAS PESSOAS: {acc:.4f}")

y_pred_probs = model.predict(X_new)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = labels_new

for person in np.unique(person_ids):
    mask = (person_ids == person)
    acc_person = np.mean(y_pred[mask] == y_true[mask])
    print(f"Pessoa {person}: {acc_person:.4f} de acurácia (N={mask.sum()})")

print(classification_report(
    y_true, y_pred, target_names=actions
))

cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusão:\n", cm)
