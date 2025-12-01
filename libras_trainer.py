# ========= Início =========
import os, sys, traceback, faulthandler, threading, numpy as np
from PIL import ImageFont, ImageDraw, Image

# Habilita o rastreamento de falhas em todas as threads
# Útil para debug de crashes que não geram exceções Python normais
faulthandler.enable(all_threads=True)

# Hook personalizado para capturar exceções "não-levantáveis"
# (ex: erros em __del__, callbacks de threads)
def _unraisable_hook(unraisable):
    print("UNRAISABLE:", unraisable.exc_type, unraisable.exc_value, "in", unraisable.object, file=sys.stderr)
sys.unraisablehook = _unraisable_hook

# Hook para capturar exceções em threads secundárias
def _thread_excepthook(args):
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
threading.excepthook = _thread_excepthook

# Reduz logs verbosos do TensorFlow (apenas erros críticos)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Desabilita transformações de hardware do OpenCV no Windows
# Aumenta estabilidade da captura de vídeo
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

# Silencia logs do OpenCV
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

# Hook adicional para garantir que exceções sejam registradas mesmo em callbacks Tkinter
def _plain_excepthook(exc_type, exc, tb):
    import traceback, sys
    try:
        traceback.print_exception(exc_type, exc, tb, file=sys.__stderr__)
    except Exception as ee:
        try:
            sys.__stderr__.write(f"excepthook falhou: {ee!r}\n")
        except:
            pass
sys.excepthook = _plain_excepthook

print("[BOOT] after hooks/env (Tk paths set)", flush=True)

# --- Interface Gráfica ---
import tkinter as tk
from tkinter import ttk
from collections import deque # Estrutura de dados para buffer circular

import cv2
cv2.setNumThreads(1)  # Força uso de apenas 1 thread para maior estabilidade no Windows

# --- Machine Learning ---
import tensorflow as tf

print("[BOOT] libs importadas", flush=True)

# =========================
# Configurações do modelo
# =========================
# Caminhos dos arquivos do modelo treinado
MODEL_PATHS = ["checkpoints/final_model-v2.keras"]
CLASSES_PATH = "classes.npy"

# PARÂMETROS DA LSTM
SEQLEN = 30 # Tamanho da sequência temporal (30 frames por gesto)

# PARÂMETROS DE SUAVIZAÇÃO E CONFIANÇA
SMOOTH_K = 5  # Janela de suavização temporal (média dos últimos 5 frames)
CONF_THRESH = 0.95 # Threshold de confiança mínima para aceitar uma predição (95%)

# PARÂMETROS DE DETECÇÃO DE MOVIMENTO
MIN_LANDMARKS = 1  # Mínimo de landmarks detectados (pelo menos 1 mão visível)
MOTION_EPS    = 5e-4  # Epsilon para detectar movimento (evita reconhecer gestos parados)
MOTION_MIN_FRAMES = 6 # Número mínimo de frames para calcular movimento

# PARÂMETROS DE VALIDAÇÃO DE PREDIÇÕES
MARGIN_THRESH = 0.30 # Margem mínima entre a classe mais provável e a segunda (30%)
ENTROPY_MAX   = 1.0 # Entropia máxima permitida (mede incerteza da predição)

# PARÂMETROS DE ACEITAÇÃO DE GESTOS
TARGET_ACCEPT = 0.80 # Acurácia mínima para aceitar um gesto (80%)
TARGET_STREAK = 8 # Número consecutivo de frames com acurácia >= 80% (evita falsos positivos)

# Lista de classes/gestos que o modelo reconhece
actions = np.array([
    'bom-bem', 'dia', 'oi', 'joia', 'eu', 'amo',
    'voce', 'obrigado', 'desculpa', 'pessoa', 'brasil'
])
np.save(CLASSES_PATH, actions)

# =========================
# Fases do curso 
# =========================
PHASES = [
    {"title": "1) Oi, tudo bem",      "sequence": ['oi', 'bom-bem', 'joia'], "phrase": "“Oi, tudo bem?”",     "video": "assets/oi_tudo_bem.mp4"},
    {"title": "2) Eu sou brasileiro", "sequence": ['pessoa', 'brasil'],      "phrase": "“Eu sou brasileiro”", "video": "assets/eu_sou_brasileiro.mp4"},
    {"title": "3) Obrigado",          "sequence": ['obrigado'],              "phrase": "“Obrigado(a)”",       "video": "assets/obrigado.mp4"},
    {"title": "4) Eu amo você",       "sequence": ['eu', 'amo', 'voce'],     "phrase": "“Eu amo você”",       "video": "assets/eu_amo_voce.mp4"},
    {"title": "5) Desculpa",          "sequence": ['desculpa'],              "phrase": "“Desculpa”",          "video": "assets/desculpa.mp4"},
    {"title": "6) Bom dia",           "sequence": ['bom-bem', 'dia'],        "phrase": "“Bom dia”",           "video": "assets/bom_dia.mp4"},
]

# =========================
# Utilitários
# =========================
def load_model_and_classes():
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    model = None

    # Tenta carregar de cada caminho até conseguir
    for p in MODEL_PATHS:
        try:
            model = tf.keras.models.load_model(p)
            print(f"[OK] Modelo carregado: {p}", flush=True)
            break
        except Exception as e:
            print(f"[!] Não foi possível carregar {p}: {e}", flush=True)
    if model is None:
        raise RuntimeError("Nenhum modelo foi carregado. Verifique os caminhos.")
    return model, classes

def count_landmarks(results):
    # Conta o num total de landmarks detectados pelo MediaPipe
    # Verifica apenas as mãos (esquerda e direita), pois são essenciais para LIBRAS
    c = 0
    if results.left_hand_landmarks and results.left_hand_landmarks.landmark:
        c += len(results.left_hand_landmarks.landmark)
    if results.right_hand_landmarks and results.right_hand_landmarks.landmark:
        c += len(results.right_hand_landmarks.landmark)
    return c

def motion_energy_last(seq_buf, k=6):
    # calcula a energia de movimentos nos últimos K frames
    # A energia de movimento mede o quanto os landmarks se moveram entre frames.
    # Valores baixos indicam que a pessoa está parada (gesto não está sendo feito).
    n = min(len(seq_buf), k)
    if n < 2: return 0.0

    # Empilha os últimos n frames em um array 3D
    x = np.stack(list(seq_buf)[-n:], axis=0)

    # Calcula diferença entre frames consecutivos
    dx = np.diff(x, axis=0)

    # Retorna a norma média (magnitude do movimento)
    return float(np.mean(np.linalg.norm(dx, axis=1)))

def motion_energy(seq_buf):
    # Calcula a energia de movimento em toda a sequência do buffer, similar a motion_energy_last, mas considera todos os frames disponíveis
    if len(seq_buf) < 2: return 0.0
    x = np.stack(seq_buf, axis=0)
    dx = np.diff(x, axis=0)
    e = np.mean(np.linalg.norm(dx, axis=1))
    return float(e)

def entropy(p):
    # Calcula a entropia de Shannon de uma distribuição de probabilidade
    # Entropia mede a "incerteza" da predição:
    # Entropia baixa: modelo confiante (ex: [0.95, 0.03, 0.02] → entropia ~0.3)
    # Entropia alta: modelo confuso (ex: [0.4, 0.35, 0.25] → entropia ~1.1)
    p = np.clip(p, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))

def should_abstain(p, conf_thresh=0.99):
    # Decide se o modelo deve se abster de fazer uma predição
    maxp = float(np.max(p))

    # Critério 1: Confiança abaixo do threshold
    if maxp < conf_thresh:
        return True
    
    # Critério 2: Margem insuficiente entre top-2 classes
    sorted_p = np.sort(p)[::-1]
    margin = float(sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))
    if margin < MARGIN_THRESH:
        return True
    
    # Critério 3: Entropia muito alta (modelo confuso)
    if entropy(p) > ENTROPY_MAX:
        return True
    return False

# Variáveis globais para modelo e classes
# (carregadas posteriormente na inicialização da GUI)
model = None
classes = np.load(CLASSES_PATH, allow_pickle=True)

def build_keep_idx():
    # Constrói os índices dos landmarks relevantes para LIBRAS
    # MediaPipe Holistic retorna 1662 valores, mas usamos apenas 144:
    # - 6 pontos da POSE (ombros, cotovelos, punhos) × 3 coords × 1 visibility = 18
    # - 21 pontos da mão esquerda × 3 coords = 63
    # - 21 pontos da mão direita × 3 coords = 63
    # Total: 18 + 63 + 63 = 144 features

    # Offsets de cada região no vetor completo
    pose_offset  = 0
    face_offset  = 33 * 4
    handL_offset = face_offset + 468 * 3
    handR_offset = handL_offset + 21 * 3

    # Índices dos pontos da pose que queremos (parte superior do corpo)
    # 11-12: ombros, 13-14: cotovelos, 15-16: pulsos
    pose_keep = np.array([11, 12, 13, 14, 15, 16])

    # Expande para pegar [x, y, z] de cada ponto (ignora visibility aqui)
    idx_pose_sup = (pose_keep[:, None] * 4 + np.array([0, 1, 2])).reshape(-1)

    # Índices de todas as coordenadas das mãos
    idx_handL = np.arange(handL_offset, handL_offset + 21 * 3)
    idx_handR = np.arange(handR_offset, handR_offset + 21 * 3)

    # Concatena todos os índices relevantes
    keep_idx = np.concatenate([idx_pose_sup, idx_handL, idx_handR])
    return keep_idx

# Gera os índices uma vez no início
KEEP_IDX = build_keep_idx()
D0 = KEEP_IDX.size  # 144 features

def apply_keep_idx_feat(feat1662):
    # Extrai apenas as 144 features relevantes do vetor completo de 1662
    return feat1662[KEEP_IDX]

class TemporalSmoother:
    # Classe para suavização temporal das predições.
    
    # Mantém um buffer circular com as últimas K predições e retorna a média.  Isso reduz "ruído" nas predições frame-a-frame, tornando o reconhecimento mais estável e confiável.

    def __init__(self, k=5, num_classes=None):
        self.k = k
        self.buf = deque(maxlen=k)
        self.num_classes = num_classes
    def push(self, probs):
        # Adiciona uma nova predição e retorna a média suavizada
        self.buf.append(probs)

        # Calcula média das últimas K predições
        avg = np.mean(self.buf, axis=0)

        # Classe com maior probabilidade média
        cls = int(np.argmax(avg))

        # Confiança da classe escolhida
        conf = float(np.max(avg))

        return avg, cls, conf
    def clear(self):
        # Limpa o buffer (usado ao resetar ou trocar de fase).
        self.buf.clear()

# =========================
# MediaPipe
# =========================

# MediaPipe é inicializado apenas quando necessário (modo lazy loading)
# para economizar recursos e acelerar a inicialização da aplicação
mp_holistic = None 

def extract_features_holistic(results):
    # Extrai features de um frame processado pelo MediaPipe Holistic.
    
    # O MediaPipe retorna landmarks de:
    # - Pose (33 pontos do corpo)
    # - Face (468 pontos faciais)
    # - Mãos esquerda e direita (21 pontos cada)

    # coordenadas (x, y, z)

    def flatten_landmarks(landmarks, include_visibility=False):
        # Converte lista de landmarks em array numpy flat
        if landmarks is None:
            return None
        out = []
        for lm in landmarks:
            x = lm.x  # Coordenada horizontal [0, 1]
            y = lm.y  # Coordenada vertical [0, 1]
            z = lm.z  # Profundidade (distância da câmera)

            if include_visibility:
                # Visibility indica confiança da detecção [0, 1]
                v = getattr(lm, "visibility", 0.0)
                out.extend([x, y, z, v])
            else:
                out.extend([x, y, z])
        return np.array(out, dtype=np.float32)

    # ---- Extração da POSE (33 pontos × 4 = 132 valores) ----
    if results.pose_landmarks and results.pose_landmarks.landmark:
        pose = flatten_landmarks(results.pose_landmarks.landmark, include_visibility=True)
    else:
        # Se não detectou pose, preenche com zeros
        pose = np.zeros(33 * 4, dtype=np.float32)

    # ---- Extração da FACE (468 pontos × 3 = 1404 valores) ----
    if results.face_landmarks and results.face_landmarks.landmark:
        face = flatten_landmarks(results.face_landmarks.landmark, include_visibility=False)
    else:
        face = np.zeros(468 * 3, dtype=np.float32)

    # ---- Extração da MÃO ESQUERDA (21 pontos × 3 = 63 valores) ----
    if results.left_hand_landmarks and results.left_hand_landmarks.landmark:
        lh = flatten_landmarks(results.left_hand_landmarks.landmark, include_visibility=False)
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    # ---- Extração da MÃO DIREITA (21 pontos × 3 = 63 valores) ----
    if results.right_hand_landmarks and results.right_hand_landmarks.landmark:
        rh = flatten_landmarks(results.right_hand_landmarks.landmark, include_visibility=False)
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    # Concatena tudo em um único vetor de features
    feat = np.concatenate([pose, face, lh, rh], axis=0)

    # Validação: deve ter exatamente 1662 valores
    assert feat.shape[0] == 1662, f"Esperado 1662, obtido {feat.shape[0]}"
    return feat

# =========================
# Tkinter GUI com Fases
# =========================
class LibrasFasesGUI(tk.Tk):
    # Implementa um sistema de fases para ensino progressivo de LIBRAS.
    # Cada fase contém uma sequência de gestos que o usuário deve executar.
    
    # Estados da aplicação:
    # - INTRO: Tela inicial com vídeo demonstrativo
    # - DETECT: Modo de detecção com webcam ativa
    # - DONE: Tela de conclusão de fase
    # - FINAL: Tela de conclusão do curso completo

    def __init__(self):
        # Inicializa a interface gráfica e todos os componentes
        print("[TK] criando janela", flush=True)
        super().__init__()

        # ---- Configuração da janela principal ----
        self.title("Treinador de LIBRAS — Fases")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.geometry("1024x720")

        # Captura exceções de callbacks do Tkinter
        self.report_callback_exception = self._report_callback_exception

        # ------- Estado da aplicação -------
        self.phases = PHASES
        self.phase_idx = 0      # Fase atual (0 = primeira fase)
        self.step_idx = 0       # Gesto atual dentro da fase
        self.target_streak = 0  # Quantos frames seguidos acertou o gesto
        self.state = "INTRO"    # Estado inicial: tela de introdução

        # ------- Carregamento do modelo LSTM -------
        global model, classes
        if model is None:
            print("[BOOT] carregando modelo/classes...", flush=True)
            model, classes = load_model_and_classes()

            # Validação: número de saídas do modelo deve bater com número de classes
            assert model.output_shape[-1] == len(classes), "número de saídas do modelo ≠ nº de classes"
            print("Ordem das classes:", list(classes), flush=True)
        self.model = model
        self.classes = classes

        # ---- Inicialização de captura de vídeo ----
        # Webcam só é aberta quando entrar no modo DETECT (economiza recursos)
        self.cap = None
        self.holistic = None

        # ---- Buffers para processamento temporal ----
        self.seq_buf = deque(maxlen=SEQLEN)  # Buffer circular com últimos 30 frames
        self.smoother = TemporalSmoother(k=SMOOTH_K, num_classes=len(self.classes))
        self.last_avg = None  # Última predição média (para debugging)

        # ---- Container principal ----
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        # Configuração do grid: linha 0 = header fixo, linha 1 = conteúdo expansível
        self.container.grid_rowconfigure(0, weight=0)  # Header não expande
        self.container.grid_rowconfigure(1, weight=1)  # Conteúdo expande
        self.container.grid_columnconfigure(0, weight=1)

        # ---- Header (título e status) ----
        self.header = ttk.Frame(self.container)
        self.header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

        # Label do título da fase (esquerda)
        self.lbl_title = ttk.Label(self.header, text="", font=("Arial", 18, "bold"))
        self.lbl_title.pack(side="left")

        # Label de status (direita)
        self.lbl_status = ttk.Label(self.header, text="", foreground="#0a84ff", font=("Arial", 12, "bold"))
        self.lbl_status.pack(side="right")

        # ---- Frames "telas" empilhados ----
        # Todas as telas ocupam o mesmo espaço (linha 1 do grid)
        # Usamos tkraise() para mostrar apenas uma por vez
        self.frame_intro  = ttk.Frame(self.container)
        self.frame_detect = ttk.Frame(self.container)
        self.frame_done   = ttk.Frame(self.container)

        for f in (self.frame_intro, self.frame_detect, self.frame_done):
            f.grid(row=1, column=0, sticky="nsew")

        # --- INTRO (Vídeo demonstrativo) ---
        self.intro_center = ttk.Frame(self.frame_intro)
        self.intro_center.place(relx=0.5, rely=0.5, anchor="center")

        # Label com a frase em português
        self.lbl_phrase   = ttk.Label(self.intro_center, text="", font=("Arial", 16))
        self.lbl_phrase.pack(pady=(0,10))

        # Label para exibir o vídeo tutorial
        self.preview_label = tk.Label(self.intro_center, bg="black", width=900, height=506)
        self.preview_label.pack()

        # Dica de instrução
        self.hint_label = ttk.Label(
            self.intro_center,
            text="Assista ao vídeo e, quando estiver pronto, clique em “Estou pronto”.",
            font=("Arial", 12)
        )
        self.hint_label.pack(pady=(10,12))

        # Botão para iniciar detecção
        style = ttk.Style()
        style.configure('Ready.TButton', font=('Arial', 14, 'bold'), padding=15)
        
        self.btn_ready = ttk.Button(
            self.intro_center, 
            text="✓ Estou pronto", 
            command=self.start_detect,
            style='Ready.TButton'
        )
        self.btn_ready.pack(pady=10)

        # --- DETECT (Detecção em tempo real) ---
        # Label para exibir o feed da webcam
        self.video_label = tk.Label(self.frame_detect, bg="black")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # Controles na parte inferior
        self.controls_detect = ttk.Frame(self.frame_detect)
        self.controls_detect.pack(side="bottom", pady=20)
        style = ttk.Style()
        style.configure('Large.TButton', font=('Arial', 11), padding=10)
        
        # Botão: Reiniciar fase
        ttk.Button(
            self.controls_detect, 
            text="🔄 Reiniciar fase", 
            command=self.reset_phase,
            style='Large.TButton'
        ).pack(side="left", padx=8)
        
        # Botão: Fase anterior (só aparece se não for a primeira fase)
        self.btn_prev_phase = ttk.Button(
            self.controls_detect, 
            text="⬅ Fase anterior", 
            command=self.prev_phase,
            style='Large.TButton'
        )
        self.btn_prev_phase.pack(side="left", padx=8)
        
        # Botão: Voltar à introdução (volta para o vídeo demonstrativo)
        ttk.Button(
            self.controls_detect, 
            text="🏠 Voltar à introdução", 
            command=self.back_to_intro,
            style='Large.TButton'
        ).pack(side="left", padx=8)

        # --- DONE (Fase concluída) ---
        self.done_center = ttk.Frame(self.frame_done)
        self.done_center.place(relx=0.5, rely=0.5, anchor="center")

        # Título de conclusão
        self.lbl_done = ttk.Label(self.done_center, text="🎉 Fase concluída!", font=("Arial", 20, "bold"))
        self.lbl_done.pack(pady=(0,10))

        # Subtítulo com lista de gestos executados
        self.lbl_done_sub = ttk.Label(self.done_center, text="", font=("Arial", 12))
        self.lbl_done_sub.pack(pady=(0,16))

        # Container de botões
        btns = ttk.Frame(self.done_center)
        btns.pack()
        
        style = ttk.Style()
        style.configure('Action.TButton', font=('Arial', 12), padding=12)
        
        # Botão: Repetir fase atual
        ttk.Button(
            btns, 
            text="🔁 Repetir esta fase", 
            command=self.reset_phase,
            style='Action.TButton'
        ).pack(side="left", padx=10)
        
        # Botão: Avançar para próxima fase
        ttk.Button(
            btns, 
            text="➡ Próxima fase", 
            command=self.next_phase,
            style='Action.TButton'
        ).pack(side="left", padx=10)
        
        # Botão: Sair da aplicação
        ttk.Button(
            btns, 
            text="❌ Sair", 
            command=self.on_close,
            style='Action.TButton'
        ).pack(side="left", padx=10)

        # --- FINAL (Todas as fases concluídas) ---
        self.frame_final = ttk.Frame(self.container)
        self.frame_final.grid(row=1, column=0, sticky="nsew")
        
        # Layout centralizado
        self.final_center = ttk.Frame(self.frame_final)
        self.final_center.place(relx=0.5, rely=0.5, anchor="center")
        
        # Título de parabéns
        self.lbl_final_title = ttk.Label(
            self.final_center, 
            text="🎊 PARABÉNS! 🎊", 
            font=("Arial", 28, "bold"),
            foreground="#00aa00"
        )
        self.lbl_final_title.pack(pady=(0,20))
        
        # Mensagem de conclusão do curso
        self.lbl_final_msg = ttk.Label(
            self.final_center,
            text="Você completou todas as fases do curso de LIBRAS!\n\nContinue praticando para melhorar ainda mais.",
            font=("Arial", 14),
            justify="center"
        )
        self.lbl_final_msg.pack(pady=(0,30))
        
        # Botões finais
        final_btns = ttk.Frame(self.final_center)
        final_btns.pack()
        
        style = ttk.Style()
        style.configure('Final.TButton', font=('Arial', 13, 'bold'), padding=15)
        
        # Botão: Recomeçar curso do zero
        ttk.Button(
            final_btns, 
            text="🔄 Recomeçar do início", 
            command=self.restart_course,
            style='Final.TButton'
        ).pack(side="left", padx=12)
        
        # Botão: Sair da aplicação
        ttk.Button(
            final_btns, 
            text="👋 Sair", 
            command=self.on_close,
            style='Final.TButton'
        ).pack(side="left", padx=12)

        # ---- Variáveis do reprodutor de vídeo tutorial ----
        self.tutorial_cap = None        # Objeto VideoCapture do vídeo tutorial
        self.tutorial_path = None       # Caminho do vídeo atual
        self.tutorial_running = False   # Flag indicando se o vídeo está rodando

        # ---- Configurações de exibição ----
        self.video_w_target = 960  # Largura alvo para redimensionamento do vídeo

        # ---- Inicialização final ----
        self.update_phase_labels()      # Atualiza labels com informações da fase atual
        self.show_state("INTRO")        # Inicia no estado INTRO
        print("[STATE] INTRO", flush=True)

        # Inicia o loop principal após 10ms (permite que a GUI seja montada primeiro)
        self.after(10, self.main_loop)

    # ------- Métodos auxiliares da interface -------
    def _report_callback_exception(self, exc, val, tb):
        # Hook personalizado para capturar exceções em callbacks do Tkinter.
        
        # Tkinter por padrão apenas imprime exceções no console. Este método garante que exceções sejam registradas em arquivo para debugging.

        # Imprime no console
        traceback.print_exception(exc, val, tb)

        # Tenta salvar em arquivo de log
        try:
            with open("fatal.log", "a", encoding="utf-8") as f:
                traceback.print_exception(exc, val, tb, file=f)
        except:
            pass

    # ------- Gerenciamento de fases e estados -------
    def current_phase(self):
        # Retorna o dicionário da fase atual
        return self.phases[self.phase_idx]

    def current_target(self):
        # Retorna o gesto alvo atual (próximo gesto que o usuário deve fazer)
        seq = self.current_phase()["sequence"]
        if self.step_idx < len(seq):
            return seq[self.step_idx]
        return None

    def update_phase_labels(self):
        # Atualiza todos os labels da interface com informações da fase/gesto atual
        p = self.current_phase()

        # Atualiza título
        self.lbl_title.config(text=p["title"])

        # Atualiza status conforme o estado atual
        tgt = self.current_target()
        if self.state == "DETECT":
            self.lbl_status.config(text=f"Faça o gesto: {tgt}" if tgt else "Fase concluída.")
        elif self.state == "INTRO":
            self.lbl_status.config(text="Introdução da fase")
        else:
            self.lbl_status.config(text="")
        
        # Atualiza frase em português
        self.lbl_phrase.config(text=p.get("phrase", ""))

        # Atualiza lista de gestos (para tela DONE)
        seq_str = "  ·  ".join(p["sequence"])
        self.lbl_done_sub.config(text=f"Você executou: {seq_str}")

        # Atualiza visibilidade dos botões
        self.update_button_visibility()

    def update_button_visibility(self):
        # Controla a visibilidade do botão "Fase anterior" - botão só aparece quando não está na 1ª fase
        if self.phase_idx == 0:
            # Primeira fase: esconde o botão
            self.btn_prev_phase.pack_forget()
        else:
            # Outras fases: mostra o botão
            self.btn_prev_phase.pack(side="left", padx=4)

    def reset_phase(self):
        # Reinicia a fase atual do zero - limpa todos os buffers e contadores, e volta para a tela de introdução (vídeo demonstrativo) da mesma fase
        self.step_idx = 0           # Volta para o primeiro gesto da fase
        self.target_streak = 0      # Zera contador de acertos consecutivos
        self.smoother.clear()       # Limpa buffer do suavizador temporal
        self.seq_buf.clear()        # Limpa buffer de sequência de frames
        self.show_state("INTRO")    # Volta para tela de introdução
    
    # ---------- Renderização de texto unicode ------------
    @staticmethod
    def _pick_font_path():
        # Busca uma fonte TrueType instalada no sistema operacional

        # Lista de fontes candidatas em diferentes sistemas
        candidates = [
            "C:/Windows/Fonts/arial.ttf",                           # Windows - Arial
            "C:/Windows/Fonts/seguiemj.ttf",                        # Windows - Segoe UI Emoji
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     # Linux - DejaVu
            "/Library/Fonts/Arial Unicode.ttf",                     # macOS - Arial Unicode
        ]

        # Retorna o primeiro caminho que existir
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def draw_text_unicode(img_bgr, text, org, font_size=32, color=(255,255,255)):
        # Desenha texto Unicode em uma imagem usando Pillow

        # OpenCV não suporta bem caracteres Unicode (acentos, emojis, etc). Esta função usa Pillow (PIL) para renderizar texto com suporte completo a Unicode e fontes TrueType

        # Busca fonte TrueType no sistema
        font_path = LibrasFasesGUI._pick_font_path()

        # Carrega fonte (ou usa padrão se não encontrar)
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

        # Pillow trabalha com RGB, então convertemos
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Converte cor de BGR para RGB (Pillow usa RGB)
        rgb = (int(color[2]), int(color[1]), int(color[0]))

        # Desenha o texto
        draw.text(org, text, font=font, fill=rgb)

        # Converte de volta para BGR (formato OpenCV)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ------- Navegação entre fases -------
    def prev_phase(self):
        # Volta para a fase anterior - só funciona se não estiver na 1ª
        if self.phase_idx > 0:
            self.phase_idx -= 1
            self.reset_phase()

    def next_phase(self):
        # Avança para a próxima fase ou finaliza o curso - ae estiver na última fase, mostra a tela de conclusão do curso (FINAL)

        if self.phase_idx >= len(self.phases) - 1:
            # Última fase concluída - mostra tela final
            self.show_state("FINAL")
        else:
            # Avança para próxima fase
            self.phase_idx += 1
            self.reset_phase()

    # ------- Gerenciamento de estados da aplicação -------
    def show_state(self, new_state):
        # Gerencia a transição entre estados da aplicação - ao trocar de estado, libera recursos do estado anterior e inicializa recursos do novo estado.

        # ---- Limpeza do estado anterior ----
        if self.state == "DETECT":
            self.stop_camera()      # Libera webcam
        if self.state == "INTRO":
            self.stop_tutorial()    # Para reprodução do vídeo

        # ---- Atualização para novo estado ----
        self.state = new_state
        self.update_phase_labels()

        # ---- Inicialização específica de cada estado ----
        if new_state == "INTRO":
            self.start_tutorial()           # Inicia reprodução do vídeo
            self.frame_intro.tkraise()      # Mostra tela de introdução

        elif new_state == "DETECT":
            self.start_camera()             # Abre webcam e MediaPipe
            self.frame_detect.tkraise()     # Mostra tela de detecção
            print("[STATE] DETECT", flush=True)
        
        elif new_state == "DONE":
            self.frame_done.tkraise()       # Mostra tela de conclusão
            print("[STATE] DONE", flush=True)
        
        elif new_state == "FINAL":
            self.frame_final.tkraise()      # Mostra tela final
            self.lbl_title.config(text="Curso Concluído")
            self.lbl_status.config(text="")
            print("[STATE] FINAL - Curso completo!", flush=True)

    def start_detect(self):
        # Inicia o modo de detecção
        self.show_state("DETECT")

    def back_to_intro(self):
        # Volta para a tela de introdução (vídeo demonstrativo)
        self.show_state("INTRO")

    # ------- Gerenciamento do vídeo tutorial -------
    def start_tutorial(self):
        # Inicia a reprodução do vídeo tutorial da fase atual - tentar abrir o arquivo de vídeo especificado na configuração da fase
        path = self.current_phase().get("video")
        self.tutorial_path = path
        
        if path:
            try:
                # Tenta abrir com FFMPEG primeiro
                cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    # Fallback: tenta com codec padrão
                    cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError(f"Não foi possível abrir o vídeo: {path}")
                
                self.tutorial_cap = cap
                self.tutorial_running = True
                print(f"[TUTORIAL] aberto: {path}", flush=True)
            
            except Exception as e:
                print("[!] Erro no vídeo da fase:", e, flush=True)
                self.tutorial_cap = None
                self.tutorial_running = False
       
        else:
            # Fase sem vídeo configurado
            self.tutorial_cap = None
            self.tutorial_running = False

    def stop_tutorial(self):
        # Para a reprodução do vídeo tutorial e libera recursos
        self.tutorial_running = False
        try:
            if self.tutorial_cap:
                self.tutorial_cap.release()
        except Exception:
            pass
        self.tutorial_cap = None

    def tutorial_loop(self):
        # Loop de reprodução do vídeo tutorial - lê frames do vídeo e exibe na interface. Quando o vídeo termina, reinicia automaticamente do início

        # Caso não tenha vídeo disponível
        if not self.tutorial_running or self.tutorial_cap is None:
            # Cria imagem preta com mensagem
            img = np.zeros((506, 900, 3), dtype=np.uint8)
            cv2.putText(img, "Sem vídeo desta fase.", (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
            self._render_on_label(self.preview_label, img)
            return

        # Lê próximo frame do vídeo
        ret, frame = self.tutorial_cap.read()

        # Se chegou ao fim do vídeo ou erro de leitura
        if not ret or frame is None:
            # Reinicia vídeo do início (loop)
            self.tutorial_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.tutorial_cap.read()

            # Se ainda assim falhar, para o tutorial
            if not ret or frame is None:
                self.tutorial_running = False
                return

        # Redimensiona frame para largura alvo mantendo proporção
        disp = self._letterbox(frame, target_w=900)

        # Renderiza na interface
        self._render_on_label(self.preview_label, disp)

    # ------- Gerencimento da câmera e MediaPipe -------
    def start_camera(self):
        # Inicializa a webcam e o MediaPipe Holistic para detecção
        
        global mp_holistic

        # ---- Inicialização do MediaPipe (apenas na primeira vez) ----
        if mp_holistic is None:
            import mediapipe as mp
            mp_holistic = mp.solutions.holistic

        # ---- Criação da instância Holistic ----
        if self.holistic is None:
            try:
                self.holistic = mp_holistic.Holistic(
                    static_image_mode=False,        # Modo vídeo (não imagens estáticas)
                    model_complexity=1,              # Complexidade média (0=leve, 2=pesado)
                    enable_segmentation=False,       # Desabilita segmentação (não usamos)
                    refine_face_landmarks=False,     # Não refina pontos faciais (economiza recursos)
                    min_detection_confidence=0.5,    # Confiança mínima para detectar pessoa
                    min_tracking_confidence=0.5      # Confiança mínima para rastrear entre frames
                )
                print("[MP] Holistic criado", flush=True)
            except Exception as e:
                traceback.print_exc()
                self.lbl_status.config(text=f"Falha ao iniciar MediaPipe: {e}")
                return

        # ---- Abertura da webcam ----
        print("[CAM] abrindo webcam (MSMF)...", flush=True)
        try:
            # Tenta abrir com MSMF (Microsoft Media Foundation - mais estável no Windows)
            self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

            # Se MSMF falhar, tenta DSHOW (DirectShow - fallback)
            if (not self.cap) or (not self.cap.isOpened()):
                print("[CAM] MSMF falhou, tentando DSHOW...", flush=True)
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            # Se ambos falharem
            if (not self.cap) or (not self.cap.isOpened()):
                self.lbl_status.config(text="Não foi possível abrir a webcam.")
                self.cap = None
                print("[CAM] webcam NÃO abriu", flush=True)
            else:
                print("[CAM] webcam aberta", flush=True)
        except Exception as e:
            print(f"[ERROR] Exceção ao abrir webcam: {e}", flush=True)
            traceback.print_exc()

    def stop_camera(self):
        # Libera a webcam e seus recursos
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    # ------- Loops principais -------
    def main_loop(self):
        # Loop principal da aplicação (roda a cada 10ms)

        # Delega para o loop apropriado conforme o estado:
        # - INTRO: tutorial_loop() - reproduz vídeo
        # - DETECT: detect_loop() - processa webcam e faz predições
        # - DONE/FINAL: não faz nada (telas estáticas)
        
        # Usa self.after() para agendamento não-bloqueante compatível com Tkinter

        if self.state == "INTRO":
            self.tutorial_loop()
        elif self.state == "DETECT":
            self.detect_loop()

        # Agenda próxima execução em 10ms (~100 FPS máximo)
        self.after(10, self.main_loop)

    def detect_loop(self):
        # Loop de detecção de gestos em tempo real
        try:
            # ---- Verificação de webcam ativa ----
            if not self.cap:
                return

            # ---- Captura frame da webcam ----
            ret, frame = self.cap.read()
            if not ret:
                self.lbl_status.config(text="Sem vídeo da câmera.")
                return

            # ---- Processamento com MediaPipe ----
            # Converte BGR (OpenCV) para RGB (MediaPipe)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Desabilita escrita para otimização (MediaPipe não modifica)
            img_rgb.flags.writeable = False

            try:
                # Processa frame para detectar landmarks
                results = self.holistic.process(img_rgb)
            except Exception as e:
                traceback.print_exc()
                self.lbl_status.config(text=f"Falha no MediaPipe: {e}")
                return

            # Reabilita escrita (vamos desenhar na imagem depois)
            img_rgb.flags.writeable = True


            # ---- Extração de features ----
            # Conta quantos landmarks de mãos foram detectados
            num_lm = count_landmarks(results)

            # Extrai todas as features (1662 valores)
            feat1662 = extract_features_holistic(results)

            # Reduz para apenas 144 features relevantes
            feat144 = apply_keep_idx_feat(feat1662)

            # Adiciona ao buffer circular (mantém últimos 30 frames)
            self.seq_buf.append(feat144)

            # ---- Variáveis para feedback visual ----
            pred_text = "Observando..."  # Texto padrão
            color = (0, 200, 0)          # Verde padrão (BGR)
            target_lbl = self.current_target()  # Gesto que deve ser feito

            # ---- Detecção de condições especiais ----
            # Verifica se há mãos suficientes na imagem
            no_hands = (num_lm < MIN_LANDMARKS)

            # Verifica se a pessoa está parada (sem movimento)
            if len(self.seq_buf) >= MOTION_MIN_FRAMES:
                still = (motion_energy_last(self.seq_buf, k=MOTION_MIN_FRAMES) < MOTION_EPS)
            else:
                still = False

            # ---- Nenhuma mão visível ----
            if no_hands:
                self.smoother.clear()
                self.target_streak = 0
                pred_text = "Entre na câmera"
                color = (0, 200, 255)
                self.lbl_status.config(text="Entre na câmera: posicione ao menos 1 mão visível.")

            # ---- Mãos visíveis mas paradas ----
            elif still:
                self.smoother.clear()
                self.target_streak = 0
                pred_text = "Parado"
                color = (0, 200, 255)  # Amarelo
                if target_lbl:
                    self.lbl_status.config(text=f"Mova a mão para reconhecer '{target_lbl}'.")

            # ---- Movimento detectado + buffer cheio → FAZER PREDIÇÃO ----
            elif len(self.seq_buf) >= SEQLEN:
                # Prepara sequência para o modelo (últimos 30 frames)
                seq_arr = np.stack(self.seq_buf, axis=0).astype(np.float32)
                x_in = np.expand_dims(seq_arr[-SEQLEN:], axis=0) # Shape: (1, 30, 144)

                # Faz predição com o modelo LSTM
                probs = self.model.predict(x_in, verbose=0)[0]

                # Aplica suavização temporal (média das últimas K predições)
                avg, cls, conf = self.smoother.push(probs)

                # ---- Se há um gesto alvo (estamos em uma fase) ----
                if target_lbl is not None:
                    # Encontra índice do gesto alvo na lista de classes
                    try:
                        target_idx = int(np.where(self.classes == target_lbl)[0][0])
                    except Exception:
                        target_idx = None

                    # Pega probabilidade do gesto alvo
                    p_target = float(avg[target_idx]) if target_idx is not None else 0.0

                    # ---- Validação: gesto está sendo feito corretamente? ----
                    if p_target >= TARGET_ACCEPT:
                        # Acertou! Incrementa contador de acertos consecutivos
                        self.target_streak += 1
                    else:
                        # Errou ou confiança baixa, zera contador
                        self.target_streak = 0

                    # ---- Validação final: manteve acerto por tempo suficiente? ----
                    if self.target_streak >= TARGET_STREAK:
                        # SUCESSO! Gesto reconhecido com confiança
                        pred_text = f"{target_lbl}  {p_target*100:.1f}%"
                        color = (0, 255, 0) # Verde
                        self.lbl_status.config(text=f"Boa! Reconhecido: {target_lbl}")

                        # Reseta contadores
                        self.target_streak = 0
                        self.smoother.clear()

                        # Avança para próximo gesto da fase
                        self.step_idx += 1

                        # Verifica se completou todos os gestos da fase
                        if self.step_idx >= len(self.current_phase()["sequence"]):
                            if self.phase_idx >= len(self.phases) - 1:
                                # Última fase concluída → tela final
                                self.show_state("FINAL")
                            else:
                                # Fase concluída → tela DONE
                                self.show_state("DONE")
                            return
                        
                        # Atualiza labels para próximo gesto
                        self.update_phase_labels()

                    # ---- Gesto ainda não validado (acertos insuficientes) ----
                    else:
                        # Verifica se deve se abster (confiança muito baixa)
                        if should_abstain(avg, conf_thresh=CONF_THRESH):
                            pred_text = "Analisando..."
                            color = (0, 200, 255) # Amarelo
                            if target_lbl:
                                self.lbl_status.config(text=f"Mantenha '{target_lbl}' por um instante.")
                        else:
                            # Mostra qual gesto foi detectado
                            label = str(self.classes[cls])
                            pred_text = f"{label}  {conf*100:.1f}%"
                            color = (0, 255, 0)  # Verde
                            if target_lbl:
                                self.lbl_status.config(text=f"Faça o gesto: {target_lbl}")

                # ---- Modo livre (sem gesto alvo) ----
                else:
                    if should_abstain(avg, conf_thresh=CONF_THRESH):
                        pred_text = "Aguardando…"
                        color = (0, 200, 255)
                    else:
                        label = str(self.classes[cls])
                        pred_text = f"{label}  {conf*100:.1f}%"
                        color = (0, 255, 0)

            # ---- Coletando frames (buffer ainda não está cheio) ----
            else:
                if target_lbl:
                    self.lbl_status.config(text=f"Coletando… alvo: {target_lbl}")

            
            # ---- Renderização do feedback visual ----
            
            base = frame.copy()
            h, w = base.shape[:2]

            # ---- Desenha HUD (Head-Up Display) semitransparente ----
            hud = base.copy()
            cv2.rectangle(hud, (10, 10), (w - 10, 60), (0, 0, 0), -1)  # Retângulo preto
            alpha = 0.6  # Transparência
            frame_hud = cv2.addWeighted(hud, alpha, base, 1 - alpha, 0)

            # ---- Desenha texto principal com suporte Unicode ----
            frame_hud = self.draw_text_unicode(frame_hud, pred_text, (20, 20), font_size=32, color=color)

            # ---- Desenha indicador do gesto alvo (se houver) ----
            if target_lbl:
                text_alvo = f"Alvo: {target_lbl}"

                # Estima tamanho do texto para criar caixa de fundo
                font_path = self._pick_font_path()
                if font_path:
                    from PIL import ImageFont, Image, ImageDraw
                    font = ImageFont.truetype(font_path, 24)

                    # Usa Pillow para medir dimensões do texto
                    dummy = Image.new('RGB', (1, 1))
                    draw = ImageDraw.Draw(dummy)
                    bbox = draw.textbbox((0, 0), text_alvo, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    # Fallback: estimativa manual
                    text_width = len(text_alvo) * 14
                    text_height = 24
                
                # Desenha retângulo escuro semitransparente atrás do texto
                overlay = frame_hud.copy()
                padding = 8
                cv2.rectangle(overlay, 
                            (20 - padding, 70 - padding), 
                            (20 + text_width + padding, 70 + text_height + padding), 
                            (0, 0, 0), -1)
                frame_hud = cv2.addWeighted(overlay, 0.7, frame_hud, 0.3, 0)
                
                # Desenha o texto do alvo
                frame_hud = self.draw_text_unicode(frame_hud, text_alvo, (20, 60), font_size=24, color=(255,255,255))
            
            # ---- Informações de debug ----
            y_dbg = 120
            scale = 0.6
            dbg_color = (200, 200, 200)

            # Mostra número de landmarks detectados
            cv2.putText(frame_hud, f"LM={num_lm} (min {MIN_LANDMARKS})", (20, y_dbg), cv2.FONT_HERSHEY_SIMPLEX, scale, dbg_color, 1, cv2.LINE_AA); y_dbg += 20

            disp = self._letterbox(frame_hud, self.video_w_target)
            self._render_on_label(self.video_label, disp)
        except Exception as e:
            # Captura e registra qualquer exceção no loop
            print(f"[ERROR] Exceção no detect_loop: {e}", flush=True)
            traceback.print_exc()
            self.lbl_status.config(text=f"Erro na detecção: {e}")

    # ------- Utilitários de renderização -------
    def _letterbox(self, img, target_w):
        # Redimensiona imagem mantendo proporção (letterbox) - calcula escala baseada na largura alvo e redimensiona a imagem mantendo a proporção original (não distorce).
        h, w = img.shape[:2]
        scale = target_w / float(w)
        new_w = target_w
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _render_on_label(self, tk_label, bgr_img):
        # Renderiza imagem OpenCV (BGR) em um Label do Tkinter - Tkinter usa PIL/ImageTk para exibir imagens. Esta função converte de BGR (OpenCV) para RGB (PIL) e atualiza o Label

        # Import tardio para evitar conflitos (ImageTk depende de tkinter inicializado)
        from PIL import Image, ImageTk

        # Converte BGR → RGB
        disp_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # Cria objeto PIL Image
        im = Image.fromarray(disp_rgb)

        # Cria PhotoImage para Tkinter
        imgtk = ImageTk.PhotoImage(image=im)

        # Armazena referência (evita garbage collection)
        tk_label.imgtk = imgtk

        # Atualiza Label
        tk_label.configure(image=imgtk)

    # ------- Finalização e limpeza -------
    def on_close(self):
        # Método chamado ao fechar a aplicação
        try:
            self.stop_tutorial()
            self.stop_camera()
            if hasattr(self, "holistic") and self.holistic:
                self.holistic.close()
        except Exception:
            pass
        self.destroy()
    
    def restart_course(self):
        # Reinicia o curso desde a primeira fase
        self.phase_idx = 0
        self.reset_phase()

# ------- Ponto de entrada da aplicação -------
if __name__ == "__main__":
    # Ponto de entrada principal do programa - inicializa a aplicação Tkinter e inicia o loop de eventos
    print("[MAIN] start", flush=True)

    try:
        # Cria instância da aplicação
        app = LibrasFasesGUI()
    except Exception as e:
        # Se falhar na inicialização, registra erro e re-lança exceção
        import traceback, sys
        traceback.print_exc()
        sys.__stderr__.write(f"[TK-INIT] falhou: {e!r}\n")
        raise

    # Inicia loop de eventos do Tkinter (bloqueia até fechar janela)
    app.mainloop()
    print("[MAIN] exit ok", flush=True)
