# ========= Início =========
import os, sys, traceback, faulthandler, threading, numpy as np
from PIL import ImageFont, ImageDraw, Image
faulthandler.enable(all_threads=True)

def _unraisable_hook(unraisable):
    print("UNRAISABLE:", unraisable.exc_type, unraisable.exc_value, "in", unraisable.object, file=sys.stderr)
sys.unraisablehook = _unraisable_hook

def _thread_excepthook(args):
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
threading.excepthook = _thread_excepthook

# reduzir ruído e desarmar coisas instáveis 
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

# ======== FIX TCL/TK (teste) ========
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

# ========= imports de libs =========
# ---- GUI ----
import tkinter as tk
from tkinter import ttk
from collections import deque

import cv2
cv2.setNumThreads(1)  # estabilidade no Windows 
import tensorflow as tf

print("[BOOT] libs importadas", flush=True)

# =========================
# Configurações do modelo
# =========================
MODEL_PATHS = ["checkpoints/final_model-v2.keras"]
CLASSES_PATH = "classes.npy"
SEQLEN = 30
SMOOTH_K = 5
CONF_THRESH = 0.95
MIN_LANDMARKS = 1
MOTION_EPS    = 5e-4
MARGIN_THRESH = 0.30
ENTROPY_MAX   = 1.0
MOTION_MIN_FRAMES = 6

TARGET_ACCEPT = 0.80
TARGET_STREAK = 8 # evita falsos positivos

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
    c = 0
    if results.left_hand_landmarks and results.left_hand_landmarks.landmark:
        c += len(results.left_hand_landmarks.landmark)
    if results.right_hand_landmarks and results.right_hand_landmarks.landmark:
        c += len(results.right_hand_landmarks.landmark)
    return c

def motion_energy_last(seq_buf, k=6):
    n = min(len(seq_buf), k)
    if n < 2: return 0.0
    x = np.stack(list(seq_buf)[-n:], axis=0)
    dx = np.diff(x, axis=0)
    return float(np.mean(np.linalg.norm(dx, axis=1)))

def motion_energy(seq_buf):
    if len(seq_buf) < 2: return 0.0
    x = np.stack(seq_buf, axis=0)
    dx = np.diff(x, axis=0)
    e = np.mean(np.linalg.norm(dx, axis=1))
    return float(e)

def entropy(p):
    p = np.clip(p, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))

def should_abstain(p, conf_thresh=0.99):
    maxp = float(np.max(p))
    if maxp < conf_thresh:
        return True
    sorted_p = np.sort(p)[::-1]
    margin = float(sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0))
    if margin < MARGIN_THRESH:
        return True
    if entropy(p) > ENTROPY_MAX:
        return True
    return False

model = None # carrega depois
classes = np.load(CLASSES_PATH, allow_pickle=True)

def build_keep_idx():
    pose_offset  = 0
    face_offset  = 33 * 4
    handL_offset = face_offset + 468 * 3
    handR_offset = handL_offset + 21 * 3
    pose_keep = np.array([11, 12, 13, 14, 15, 16])  # ombros/cotovelos/pulsos
    idx_pose_sup = (pose_keep[:, None] * 4 + np.array([0, 1, 2])).reshape(-1)
    idx_handL = np.arange(handL_offset, handL_offset + 21 * 3)
    idx_handR = np.arange(handR_offset, handR_offset + 21 * 3)
    keep_idx = np.concatenate([idx_pose_sup, idx_handL, idx_handR])
    return keep_idx

KEEP_IDX = build_keep_idx()
D0 = KEEP_IDX.size  # 144

def apply_keep_idx_feat(feat1662):
    return feat1662[KEEP_IDX]

class TemporalSmoother:
    def __init__(self, k=5, num_classes=None):
        self.k = k
        self.buf = deque(maxlen=k)
        self.num_classes = num_classes
    def push(self, probs):
        self.buf.append(probs)
        avg = np.mean(self.buf, axis=0)
        cls = int(np.argmax(avg))
        conf = float(np.max(avg))
        return avg, cls, conf
    def clear(self):
        self.buf.clear()

# =========================
# MediaPipe
# =========================
mp_holistic = None # inicia quando precisar

def extract_features_holistic(results):
    def flatten_landmarks(landmarks, include_visibility=False):
        if landmarks is None:
            return None
        out = []
        for lm in landmarks:
            x = lm.x; y = lm.y; z = lm.z
            if include_visibility:
                v = getattr(lm, "visibility", 0.0)
                out.extend([x, y, z, v])
            else:
                out.extend([x, y, z])
        return np.array(out, dtype=np.float32)

    if results.pose_landmarks and results.pose_landmarks.landmark:
        pose = flatten_landmarks(results.pose_landmarks.landmark, include_visibility=True)
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    if results.face_landmarks and results.face_landmarks.landmark:
        face = flatten_landmarks(results.face_landmarks.landmark, include_visibility=False)
    else:
        face = np.zeros(468 * 3, dtype=np.float32)

    if results.left_hand_landmarks and results.left_hand_landmarks.landmark:
        lh = flatten_landmarks(results.left_hand_landmarks.landmark, include_visibility=False)
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    if results.right_hand_landmarks and results.right_hand_landmarks.landmark:
        rh = flatten_landmarks(results.right_hand_landmarks.landmark, include_visibility=False)
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    feat = np.concatenate([pose, face, lh, rh], axis=0)
    assert feat.shape[0] == 1662, f"Esperado 1662, obtido {feat.shape[0]}"
    return feat

# =========================
# Tkinter GUI com Fases
# =========================
class LibrasFasesGUI(tk.Tk):
    def __init__(self):
        print("[TK] criando janela", flush=True)
        super().__init__()
        self.title("Treinador de LIBRAS — Fases")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.geometry("1024x720")

        # Captura exceções de callbacks do Tkinter
        self.report_callback_exception = self._report_callback_exception

        # ------- Estado -------
        self.phases = PHASES
        self.phase_idx = 0
        self.step_idx = 0
        self.target_streak = 0
        self.state = "INTRO"  # INTRO | DETECT | DONE | SUCCESS

        # ------- Vídeo/Modelo -------
        global model, classes
        if model is None:
            print("[BOOT] carregando modelo/classes...", flush=True)
            model, classes = load_model_and_classes()
            assert model.output_shape[-1] == len(classes), "número de saídas do modelo ≠ nº de classes"
            print("Ordem das classes:", list(classes), flush=True)
        self.model = model
        self.classes = classes

        # webcam abre apenas quando DETECT
        self.cap = None
        self.holistic = None

        self.seq_buf = deque(maxlen=SEQLEN)
        self.smoother = TemporalSmoother(k=SMOOTH_K, num_classes=len(self.classes))
        self.last_avg = None

        # ------- Layout base -------
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.container.grid_rowconfigure(0, weight=0)  # header
        self.container.grid_rowconfigure(1, weight=1)  # conteúdo cresce
        self.container.grid_columnconfigure(0, weight=1)

        self.header = ttk.Frame(self.container)
        self.header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))

        self.lbl_title = ttk.Label(self.header, text="", font=("Arial", 18, "bold"))
        self.lbl_title.pack(side="left")
        self.lbl_status = ttk.Label(self.header, text="", foreground="#0a84ff", font=("Arial", 12, "bold"))
        self.lbl_status.pack(side="right")

        # Frames “telas” empilhados via grid (linha 1)
        self.frame_intro  = ttk.Frame(self.container)
        self.frame_detect = ttk.Frame(self.container)
        self.frame_done   = ttk.Frame(self.container)

        for f in (self.frame_intro, self.frame_detect, self.frame_done):
            f.grid(row=1, column=0, sticky="nsew")

        # --- INTRO ---
        self.intro_center = ttk.Frame(self.frame_intro)
        self.intro_center.place(relx=0.5, rely=0.5, anchor="center")

        self.lbl_phrase   = ttk.Label(self.intro_center, text="", font=("Arial", 16))
        self.lbl_phrase.pack(pady=(0,10))

        self.preview_label = tk.Label(self.intro_center, bg="black", width=900, height=506)
        self.preview_label.pack()

        self.hint_label = ttk.Label(
            self.intro_center,
            text="Assista ao vídeo e, quando estiver pronto, clique em “Estou pronto”.",
            font=("Arial", 12)
        )
        self.hint_label.pack(pady=(10,12))

        self.btn_ready = ttk.Button(self.intro_center, text="Estou pronto", command=self.start_detect)
        self.btn_ready.pack()

        # --- DETECT ---
        self.video_label = tk.Label(self.frame_detect, bg="black")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        self.controls_detect = ttk.Frame(self.frame_detect)
        self.controls_detect.pack(side="bottom", pady=10)
        ttk.Button(self.controls_detect, text="Reiniciar fase", command=self.reset_phase).pack(side="left", padx=4)
        # ttk.Button(self.controls_detect, text="Fase anterior", command=self.prev_phase).pack(side="left", padx=4)
        self.btn_prev_phase = ttk.Button(self.controls_detect, text="Fase anterior", command=self.prev_phase)
        self.btn_prev_phase.pack(side="left", padx=4)
        ttk.Button(self.controls_detect, text="Voltar à introdução", command=self.back_to_intro).pack(side="left", padx=4)

        # --- DONE ---
        self.done_center = ttk.Frame(self.frame_done)
        self.done_center.place(relx=0.5, rely=0.5, anchor="center")

        self.lbl_done = ttk.Label(self.done_center, text="🎉 Fase concluída!", font=("Arial", 20, "bold"))
        self.lbl_done.pack(pady=(0,10))

        self.lbl_done_sub = ttk.Label(self.done_center, text="", font=("Arial", 12))
        self.lbl_done_sub.pack(pady=(0,16))

        btns = ttk.Frame(self.done_center)
        btns.pack()
        ttk.Button(btns, text="Repetir esta fase", command=self.reset_phase).pack(side="left", padx=6)
        ttk.Button(btns, text="Próxima fase", command=self.next_phase).pack(side="left", padx=6)
        ttk.Button(btns, text="Sair", command=self.on_close).pack(side="left", padx=6)

        # --- SUCCESS ---
        self.frame_final = ttk.Frame(self.container)
        self.frame_final.grid(row=1, column=0, sticky="nsew")
        
        # Layout da tela final
        self.final_center = ttk.Frame(self.frame_final)
        self.final_center.place(relx=0.5, rely=0.5, anchor="center")
        
        self.lbl_final_title = ttk.Label(
            self.final_center, 
            text="🎊 PARABÉNS! 🎊", 
            font=("Arial", 28, "bold"),
            foreground="#00aa00"
        )
        self.lbl_final_title.pack(pady=(0,20))
        
        self.lbl_final_msg = ttk.Label(
            self.final_center,
            text="Você completou todas as fases do curso de LIBRAS!\n\nContinue praticando para melhorar ainda mais.",
            font=("Arial", 14),
            justify="center"
        )
        self.lbl_final_msg.pack(pady=(0,30))
        
        final_btns = ttk.Frame(self.final_center)
        final_btns.pack()
        ttk.Button(final_btns, text="Recomeçar do início", command=self.restart_course).pack(side="left", padx=6)
        ttk.Button(final_btns, text="Sair", command=self.on_close).pack(side="left", padx=6)

        # vídeo didático player (INTRO)
        self.tutorial_cap = None
        self.tutorial_path = None
        self.tutorial_running = False

        self.video_w_target = 960

        self.update_phase_labels()
        self.show_state("INTRO")
        print("[STATE] INTRO", flush=True)
        self.after(10, self.main_loop)

    # Tkinter: hook de exceções
    def _report_callback_exception(self, exc, val, tb):
        traceback.print_exception(exc, val, tb)
        try:
            with open("fatal.log", "a", encoding="utf-8") as f:
                traceback.print_exception(exc, val, tb, file=f)
        except:
            pass

    # ------- helpers de fase/estado -------
    def current_phase(self):
        return self.phases[self.phase_idx]

    def current_target(self):
        seq = self.current_phase()["sequence"]
        if self.step_idx < len(seq):
            return seq[self.step_idx]
        return None

    def update_phase_labels(self):
        p = self.current_phase()
        self.lbl_title.config(text=p["title"])
        tgt = self.current_target()
        if self.state == "DETECT":
            self.lbl_status.config(text=f"Faça o gesto: {tgt}" if tgt else "Fase concluída.")
        elif self.state == "INTRO":
            self.lbl_status.config(text="Introdução da fase")
        else:
            self.lbl_status.config(text="")
        self.lbl_phrase.config(text=p.get("phrase", ""))
        seq_str = "  ·  ".join(p["sequence"])
        self.lbl_done_sub.config(text=f"Você executou: {seq_str}")

        # Mostrar/ocultar botão "Fase anterior" conforme a fase atual
        self.update_button_visibility()

    def update_button_visibility(self):
        if self.phase_idx == 0:
            # Primeira fase: esconde o botão
            self.btn_prev_phase.pack_forget()
        else:
            # Outras fases: mostra o botão
            self.btn_prev_phase.pack(side="left", padx=4)

    def reset_phase(self):
        self.step_idx = 0
        self.target_streak = 0
        self.smoother.clear()
        self.seq_buf.clear()
        self.show_state("INTRO")
    
    # ---------- fontes ------------
    @staticmethod
    def _pick_font_path():
        # tenta achar uma TTF comum no SO
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/seguiemj.ttf",  # emoji/extended
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def draw_text_unicode(img_bgr, text, org, font_size=32, color=(255,255,255)):
        """
        Desenha 'text' (Unicode) em img_bgr na posição org=(x,y) usando Pillow.
        Retorna uma nova imagem BGR.
        """
        font_path = LibrasFasesGUI._pick_font_path()
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

        # Pillow trabalha em RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Pillow usa RGB; convertendo cor BGR -> RGB
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        draw.text(org, text, font=font, fill=rgb)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    
    def prev_phase(self):
        if self.phase_idx > 0:
            self.phase_idx -= 1
            self.reset_phase()

    def next_phase(self):
        # self.phase_idx = (self.phase_idx + 1) % len(self.phases)
        # self.reset_phase()

        if self.phase_idx >= len(self.phases) - 1:
            # Está na última fase, mostra tela final
            self.show_state("FINAL")
        else:
            # Avança para próxima fase
            self.phase_idx += 1
            self.reset_phase()

    # ------- controle de estado / recursos -------
    def show_state(self, new_state):
        if self.state == "DETECT":
            self.stop_camera()
        if self.state == "INTRO":
            self.stop_tutorial()

        self.state = new_state
        self.update_phase_labels()

        if new_state == "INTRO":
            self.start_tutorial()
            self.frame_intro.tkraise()
        elif new_state == "DETECT":
            self.start_camera()
            self.frame_detect.tkraise()
            print("[STATE] DETECT", flush=True)
        elif new_state == "DONE":
            self.frame_done.tkraise()
            print("[STATE] DONE", flush=True)
        elif new_state == "FINAL":
            self.frame_final.tkraise()
            self.lbl_title.config(text="Curso Concluído")
            self.lbl_status.config(text="")
            print("[STATE] FINAL - Curso completo!", flush=True)

    def start_detect(self):
        self.show_state("DETECT")

    def back_to_intro(self):
        self.show_state("INTRO")

    # ------- tutorial video -------
    def start_tutorial(self):
        path = self.current_phase().get("video")
        self.tutorial_path = path
        if path:
            try:
                cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
                if not cap.isOpened():
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
            self.tutorial_cap = None
            self.tutorial_running = False

    def stop_tutorial(self):
        self.tutorial_running = False
        try:
            if self.tutorial_cap:
                self.tutorial_cap.release()
        except Exception:
            pass
        self.tutorial_cap = None

    def tutorial_loop(self):
        if not self.tutorial_running or self.tutorial_cap is None:
            img = np.zeros((506, 900, 3), dtype=np.uint8)
            cv2.putText(img, "Sem vídeo desta fase.", (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
            self._render_on_label(self.preview_label, img)
            return
        ret, frame = self.tutorial_cap.read()
        if not ret or frame is None:
            self.tutorial_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.tutorial_cap.read()
            if not ret or frame is None:
                self.tutorial_running = False
                return
        disp = self._letterbox(frame, target_w=900)
        self._render_on_label(self.preview_label, disp)

    # ------- câmera/detecção -------
    def start_camera(self):
        global mp_holistic
        if mp_holistic is None:
            import mediapipe as mp
            mp_holistic = mp.solutions.holistic

        if self.holistic is None:
            try:
                self.holistic = mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[MP] Holistic criado", flush=True)
            except Exception as e:
                traceback.print_exc()
                self.lbl_status.config(text=f"Falha ao iniciar MediaPipe: {e}")
                return

        if self.holistic is None:
            try:
                self.holistic = mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("[MP] Holistic criado", flush=True)
            except Exception as e:
                traceback.print_exc()
                self.lbl_status.config(text=f"Falha ao iniciar MediaPipe: {e}")
                return

        print("[CAM] abrindo webcam (MSMF)...", flush=True)
        self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if (not self.cap) or (not self.cap.isOpened()):
            print("[CAM] MSMF falhou, tentando DSHOW...", flush=True)
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if (not self.cap) or (not self.cap.isOpened()):
            self.lbl_status.config(text="Não foi possível abrir a webcam.")
            self.cap = None
            print("[CAM] webcam NÃO abriu", flush=True)
        else:
            print("[CAM] webcam aberta", flush=True)

    def stop_camera(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

    # ------- loops principais -------
    def main_loop(self):
        if self.state == "INTRO":
            self.tutorial_loop()
        elif self.state == "DETECT":
            self.detect_loop()
        self.after(10, self.main_loop)

    def detect_loop(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.lbl_status.config(text="Sem vídeo da câmera.")
            return

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        try:
            results = self.holistic.process(img_rgb)
        except Exception as e:
            traceback.print_exc()
            self.lbl_status.config(text=f"Falha no MediaPipe: {e}")
            return
        img_rgb.flags.writeable = True

        num_lm = count_landmarks(results)
        feat1662 = extract_features_holistic(results, frame.shape)
        feat144 = apply_keep_idx_feat(feat1662)
        self.seq_buf.append(feat144)

        pred_text = "Observando..."
        color = (0, 200, 0)
        target_lbl = self.current_target()

        no_hands = (num_lm < MIN_LANDMARKS)
        if len(self.seq_buf) >= MOTION_MIN_FRAMES:
            still = (motion_energy_last(self.seq_buf, k=MOTION_MIN_FRAMES) < MOTION_EPS)
        else:
            still = False

        if no_hands:
            self.smoother.clear()
            self.target_streak = 0
            pred_text = "Entre na câmera"
            color = (0, 200, 255)
            self.lbl_status.config(text="Entre na câmera: posicione ao menos 1 mão visível.")
        elif still:
            self.smoother.clear()
            self.target_streak = 0
            pred_text = "Parado"
            color = (0, 200, 255)
            if target_lbl:
                self.lbl_status.config(text=f"Mova a mão para reconhecer '{target_lbl}'.")
        elif len(self.seq_buf) >= SEQLEN:
            seq_arr = np.stack(self.seq_buf, axis=0).astype(np.float32)
            x_in = np.expand_dims(seq_arr[-SEQLEN:], axis=0)
            probs = self.model.predict(x_in, verbose=0)[0]
            avg, cls, conf = self.smoother.push(probs)

            if target_lbl is not None:
                try:
                    target_idx = int(np.where(self.classes == target_lbl)[0][0])
                except Exception:
                    target_idx = None
                p_target = float(avg[target_idx]) if target_idx is not None else 0.0

                if p_target >= TARGET_ACCEPT:
                    self.target_streak += 1
                else:
                    self.target_streak = 0

                if self.target_streak >= TARGET_STREAK:
                    pred_text = f"{target_lbl}  {p_target*100:.1f}%"
                    color = (0, 255, 0)
                    self.lbl_status.config(text=f"Boa! Reconhecido: {target_lbl}")
                    self.target_streak = 0
                    self.smoother.clear()
                    self.step_idx += 1
                    if self.step_idx >= len(self.current_phase()["sequence"]):
                        self.show_state("DONE")
                        return
                    self.update_phase_labels()
                else:
                    if should_abstain(avg, conf_thresh=CONF_THRESH):
                        pred_text = "Analisando..."
                        color = (0, 200, 255)
                        if target_lbl:
                            self.lbl_status.config(text=f"Mantenha '{target_lbl}' por um instante.")
                    else:
                        label = str(self.classes[cls])
                        pred_text = f"{label}  {conf*100:.1f}%"
                        color = (0, 255, 0)
                        if target_lbl:
                            self.lbl_status.config(text=f"Faça o gesto: {target_lbl}")
            else:
                if should_abstain(avg, conf_thresh=CONF_THRESH):
                    pred_text = "Aguardando…"
                    color = (0, 200, 255)
                else:
                    label = str(self.classes[cls])
                    pred_text = f"{label}  {conf*100:.1f}%"
                    color = (0, 255, 0)
        else:
            if target_lbl:
                self.lbl_status.config(text=f"Coletando… alvo: {target_lbl}")

        base = frame.copy()
        h, w = base.shape[:2]

        hud = base.copy()
        cv2.rectangle(hud, (10, 10), (w - 10, 60), (0, 0, 0), -1)
        alpha = 0.6
        frame_hud = cv2.addWeighted(hud, alpha, base, 1 - alpha, 0)

        # --- Texto com UNICODE via Pillow ---
        frame_hud = self.draw_text_unicode(frame_hud, pred_text, (20, 20), font_size=32, color=color)

        if target_lbl:
            frame_hud = self.draw_text_unicode(frame_hud, f"Alvo: {target_lbl}", (20, 60), font_size=24, color=(255,255,255))
        
        y_dbg = 120
        scale = 0.6
        dbg_color = (200, 200, 200)
        cv2.putText(frame_hud, f"LM={num_lm} (min {MIN_LANDMARKS})", (20, y_dbg), cv2.FONT_HERSHEY_SIMPLEX, scale, dbg_color, 1, cv2.LINE_AA); y_dbg += 20
        # cv2.putText(frame_hud, f"motion={motion_energy(self.seq_buf):.2e} (eps={MOTION_EPS:.1e})", (20, y_dbg), cv2.FONT_HERSHEY_SIMPLEX, scale, dbg_color, 1, cv2.LINE_AA); y_dbg += 20

        disp = self._letterbox(frame_hud, self.video_w_target)
        self._render_on_label(self.video_label, disp)


    # ------- util de render -------
    def _letterbox(self, img, target_w):
        h, w = img.shape[:2]
        scale = target_w / float(w)
        new_w = target_w
        new_h = int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _render_on_label(self, tk_label, bgr_img):
        # Import tardio para evitar conflitos (ImageTk depende de tkinter)
        from PIL import Image, ImageTk
        disp_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(disp_rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        tk_label.imgtk = imgtk
        tk_label.configure(image=imgtk)

    def on_close(self):
        try:
            self.stop_tutorial()
            self.stop_camera()
            if hasattr(self, "holistic") and self.holistic:
                self.holistic.close()
        except Exception:
            pass
        self.destroy()
    
    def restart_course(self):
        self.phase_idx = 0
        self.reset_phase()

# ========= entrypoint =========
if __name__ == "__main__":
    print("[MAIN] start", flush=True)
    try:
        app = LibrasFasesGUI()
    except Exception as e:
        import traceback, sys
        traceback.print_exc()
        sys.__stderr__.write(f"[TK-INIT] falhou: {e!r}\n")
        raise
    app.mainloop()
    print("[MAIN] exit ok", flush=True)
