import cv2
import numpy as np
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

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

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# setup dos diretórios
DATA_PATH = os.path.join('MP_Data_Novos/pessoa2') # Pasta para arquivos exportados

# Gestos
actions = np.array(['bom-bem', 'dia', 'oi', 'joia', 'eu', 'amo', 'voce', 'obrigado', 'desculpa', 'pessoa', 'brasil'])

# quantidade de vídeos para cada gesto
no_sequences = 10

# quantidade de frames em cada vídeo
sequence_length = 30

# Por onde começa (para novos vídeos)
start_folder = 0

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

# cria os diretórios
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions:
        for sequence in range(start_folder, start_folder+no_sequences):
            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                image, results = mediapipe_detection(frame, holistic)

                draw_styled_landmarks(image, results)
                
                if frame_num == 0: 
                    cv2.putText(image, 'COMEÇANDO A COLETA', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Frames para {} Vídeo número {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Frames para {} Vídeo número {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow('OpenCV Feed', image)
                
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()
qq
cap.release()
cv2.destroyAllWindows()