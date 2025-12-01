import cv2
import numpy as np
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Modelo Holistic: detecta pose, face e mãos simultaneamente
mp_drawing = mp.solutions.drawing_utils  # Utilitários para desenhar os landmarks na imagem

def draw_styled_landmarks(image, results):
    # Desenha os landmarks detectados pelo MediaPipe na imagem com estilos customizados

    # Rosto
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          ) 
    # Desenha pose corporal (33 pontos: cabeça, ombros, cotovelos, pulsos, quadril, etc)
    mp_drawing.draw_landmarks(
        image, 
        results.pose_landmarks,  # landmarks da pose
        mp_holistic.POSE_CONNECTIONS,  # conexões entre pontos (linhas do esqueleto)
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  # estilo dos pontos
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)  # estilo das linhas
    ) 
    
    # Desenha mão esquerda (21 pontos por mão)
    mp_drawing.draw_landmarks(
        image, 
        results.left_hand_landmarks,  # landmarks da mão esquerda
        mp_holistic.HAND_CONNECTIONS,  # conexões entre dedos
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),  # cor rosa/roxo
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    ) 
    
    # Desenha mão direita (21 pontos)
    mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks,  # landmarks da mão direita
        mp_holistic.HAND_CONNECTIONS,  # conexões entre dedos
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),  # cor laranja
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    ) 

def extract_keypoints(results):
    # Extrai as coordenadas (x, y, z) de todos os landmarks detectados e concatena em um único vetor

    # Extrai pose (33 landmarks com x, y, z, visibility)
    # Se não detectar pose, preenche com zeros
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*4)
    
    # Extrai face (468 landmarks com x, y, z)
    # Comentado no draw mas extraído aqui - pode ser usado em versões futuras
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
           if results.face_landmarks else np.zeros(468*3)
    
    # Extrai mão esquerda (21 landmarks com x, y, z)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21*3)
    
    # Extrai mão direita (21 landmarks com x, y, z)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21*3)
    
    # Concatena tudo em um único vetor de 1662 elementos
    return np.concatenate([pose, face, lh, rh])

def apply_keep_idx(X, keep_idx):
    # Aplica seleção de features, reduzindo de 1662 para 144 features relevantes

    # Se já está reduzido (mesma largura de keep_idx), devolve como está
    if X.shape[2] == keep_idx.size:
        return X
    # Segurança: garante que os índices cabem no shape atual
    assert np.max(keep_idx) < X.shape[2], \
        f"max(keep_idx)={np.max(keep_idx)} >= D={X.shape[2]}"
    
    # Seleciona apenas as colunas (features) especificadas em keep_idx
    return X[:, :, keep_idx]

def fix_time_leak(X, D0):
    """
    Conserta casos onde o tempo (T) foi concatenado na última dimensão por frame.
    Garante saída (N, T, D0) real, com cada frame contendo exatamente D0 features.
    
    Este bug pode ocorrer quando arrays são mal formatados durante concatenação.
    
    """
    N, T, Dtot = X.shape

    # Caso 1: Dtot == T*D0 (clássico "vazou tudo" para o último eixo)
    # Exemplo: shape (100, 30, 4320) onde 4320 = 30 * 144
    if Dtot == T * D0 and Dtot != D0:
        # Reconstrói pegando o bloco correto de cada timestep
        X_fixed = np.stack([X[:, t, t*D0:(t+1)*D0] for t in range(T)], axis=1)
        return X_fixed

    # Caso 2: shape parece ok (Dtot == D0), mas o buffer interno tem problemas
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
                # Se essa sequência já tem o número certo de elementos, só copia
                if seq.size == exp_elems:
                    X_fixed[i] = np.ascontiguousarray(seq)
                else:
                    # Reconstrói por frame: pega o bloco correto do frame t
                    for t in range(T):
                        X_fixed[i, t, :] = seq[t, t*D0:(t+1)*D0]
            return X_fixed

    # Caso 3: já está ok, apenas garante que está contíguo na memória
    return np.ascontiguousarray(X)

def mediapipe_detection(image, model):
    # Processa uma imagem através do modelo MediaPipe Holistic
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # MediaPipe trabalha com RGB
    image.flags.writeable = False  # Otimização: desabilita escrita durante processamento
    results = model.process(image)  # Detecta pose, face e mãos
    image.flags.writeable = True  # Reabilita escrita
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Volta para BGR (padrão OpenCV)
    return image, results

# Diretório onde os dados serão salvos
DATA_PATH = os.path.join('MP_Data_Novos/pessoa2')  # pessoa2: identificador do participante

# Lista de gestos a coletar (11 classes de sinais em LIBRAS)
actions = np.array(['bom-bem', 'dia', 'oi', 'joia', 'eu', 'amo', 'voce', 'obrigado', 'desculpa', 'pessoa', 'brasil'])

# Quantidade de vídeos (sequências) para cada gesto
no_sequences = 10  # 10 repetições de cada gesto para aumentar variabilidade

# Quantidade de frames em cada vídeo/sequência
sequence_length = 30  # 30 frames = ~1 segundo a 30 FPS

# Por onde começar a numeração (útil para adicionar mais dados depois)
start_folder = 0  # Começa do vídeo 0

# Mapeamento: nome do gesto → número (para treinamento)
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

# Criação os diretórios

# Estrutura: MP_Data_Novos/pessoa2/gesto/sequencia_id/frame.npy
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass  # Diretório já existe, ignora erro

# Inicialização da webcam
cap = cv2.VideoCapture(0)  # 0 = webcam padrão do sistema

# Loop de coleta

# Context manager: garante que o modelo Holistic será fechado corretamente
with mp_holistic.Holistic(
    min_detection_confidence=0.5,  # Confiança mínima para DETECTAR landmarks (primeira vez)
    min_tracking_confidence=0.5    # Confiança mínima para RASTREAR landmarks (frames seguintes)
) as holistic:
    
    # Loop externo: percorre cada gesto/ação
    for action in actions:
        # Loop médio: percorre cada sequência/vídeo do gesto
        for sequence in range(start_folder, start_folder+no_sequences):
            # Loop interno: percorre cada frame da sequência
            for frame_num in range(sequence_length):

                # Captura frame da webcam
                ret, frame = cap.read()  # ret: sucesso?, frame: imagem BGR

                # Processa frame com MediaPipe (detecta pose + mãos + face)
                image, results = mediapipe_detection(frame, holistic)

                # Desenha os landmarks detectados na imagem (feedback visual)
                draw_styled_landmarks(image, results)
                
                # PRIMEIRO FRAME: Mostra mensagem de início e pausa 2 segundos
                if frame_num == 0: 
                    cv2.putText(
                        image, 
                        'COMEÇANDO A COLETA',  # texto grande
                        (120,200),  # posição (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX,  # fonte
                        1,  # tamanho
                        (0,255, 0),  # cor verde (BGR)
                        4,  # espessura
                        cv2.LINE_AA  # antialiasing
                    )
                    cv2.putText(
                        image, 
                        'Frames para {} Vídeo número {}'.format(action, sequence),  # informação do gesto
                        (15,12),  # canto superior esquerdo
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5,  # tamanho menor
                        (0, 0, 255),  # cor vermelha
                        1,  # espessura menor
                        cv2.LINE_AA
                    )

                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)  # Pausa de 2 segundos para usuário se preparar
                
                # FRAMES SEGUINTES: Apenas mostra informação do gesto
                else: 
                    cv2.putText(image, 'Frames para {} Vídeo número {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow('OpenCV Feed', image)
                
                # Salva os keypoints
                keypoints = extract_keypoints(results)  # Extrai 1662 valores numéricos
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)  # Salva como arquivo .npy (formato NumPy eficiente)

                # Permite sair pressionando 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    # Libera recursos da webcam                
    cap.release()
    cv2.destroyAllWindows()

# Redundância: garante que recursos são liberados (caso haja break no loop)
cap.release()
cv2.destroyAllWindows()