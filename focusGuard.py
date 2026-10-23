"""Focus Guard v2: head pose + phone/book detection + proof-of-work challenges.
FIRST USE
  1. Install (Python 3.10+):   pip install -U opencv-python mediapipe pynput ultralytics
     (first run downloads the models, ~1 min; RT-DETR needs a reasonably fast CPU -- see "Slow computer" below)
  2. Run:                      python focus_guard.py
  3. Click the video window so it has keyboard focus. The app starts in SETUP: nothing beeps or alarms yet.
  4. Teach it your desk (needs at least 7 samples; more = fewer false alarms):
       a. Put on everything you normally wear/have around you: headphones, usual shirt, books on the desk...
       b. Put your phone OUT of the camera's view, press N and move around for ~15 s (turn your head, lean, move your hands).
          Every second a sample is saved, and anything that looks like a phone but is not (headphones, shirt print...) is remembered
          and ignored from then on. Press N again to stop.
       c. Now hold your REAL phone in view, in a few positions (in front, low, to the side, turned). While a red box is on it press P.
          If the detector misses it, press L, drag a rectangle around the phone and press ENTER.
          (Boxes are only drawn during SETUP; red = counted as phone, grey = seen but ignored.)
  5. Look at the screen the way you normally sit and press S. Focus Guard is now active: it warns at ~8 s and alarms at ~16 s of
     phone / head turned away / looking down without showing a book or notes (it then asks you to SHOW YOUR BOOK/NOTES).
  6. During use:  B = break   F = false alarm (appeal, limited per hour; also teaches it that box was wrong)
                  C = recalibrate sitting position   N / P / L = add more samples any time   Q = quit
  Headphones or something else still triggers it? Press F during the alarm, or press N again with that object on and phone out of view.
  Your phone gets ignored? Press P or L on it a few times.
  Slow computer (the corner of the video shows ms per detection cycle; over 1500 ms is too slow):  echo yolov8s.pt > detector.txt
  Samples are stored in the finetune/ folder next to the script (delete it to start from scratch).

ADVANCED
Teach it your desk (fixes persistent false alarms: shirt print, headphones, ...):
  N = start/stop recording phone-free frames as hard negatives (phone out of view!). Wear your headphones, move around:
      every phone-like box seen is also remembered as a LOOK-ALIKE and vetoed from then on (instant, no training needed)
  P = save a positive when the box on screen is really your phone   L = draw the box yourself (phone the detector missed)
  python focus_guard.py --train   -> fine-tunes YOLO on those samples (phone_finetuned.pt; --bench tests it and uses it only if it wins)
Default detector is RT-DETR (much better with a turned phone, but heavier). The corner of the video shows ms per detection cycle;
if that is >1500 ms, run:  echo yolov8s.pt > detector.txt   (or use --bench below to choose with data).
Pick the detector with data, not guesses:
  python focus_guard.py --record  -> films 3 short labelled clips of your situations (no phone / phone up / phone turned)
  python focus_guard.py --bench   -> replays them through each detector + the real pipeline, writes the winner to detector.txt
"""
import cv2, time, threading, os, sys, glob, shutil, random, urllib.request
import numpy as np
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
try:                                # leave ~2 CPU cores for the camera + face tracking
    import torch; torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
except Exception:
    pass

try:
    from pynput import keyboard, mouse
except ImportError:
    keyboard = mouse = None
try:
    import winsound
except ImportError:
    winsound = None

# ---- config ----
YAW_T, PITCH_T = 0.15, 0.12     # deviation from calibrated baseline
AWAY_S = 30                     # face gone this long (and not head-down) -> AWAY, no alarm
INPUT_GRACE = 3                 # s after typing/mouse that excuses looking down
WARN_S, ALARM_S = 8, 16         # bad-time thresholds: soft beep / loud alarm
DOWN_S = 8                      # head-down seconds before a challenge
CHALLENGE_S = 6                 # time to show a book/notes
BOOK_HOLD = 1.0                 # book must be seen this long to pass
GRANT_S = 180                   # reading status granted per accepted proof
READ_BUDGET = 1800              # max granted seconds per rolling hour
SETUP_MIN = 7                   # samples (N / P / L) needed before S can start the session
APPEALS_MAX = 3                 # false-alarm appeals per rolling hour
BREAK_S = 600
PHONE_CONF, BOOK_CONF, PHONE_HOLD = 0.3, 0.35, 1.0
PHONE_AR = 1.5                  # boxes squarer than this need PHONE_SQUARE_EXTRA more confidence (a turned phone has a squarish box)
PHONE_SQUARE_EXTRA = 0.1
MAX_PHONE_AREA = 0.30            # a phone box covering more than this share of the frame is never a phone (door, wall, book...)
SOFT_HEAD = 0.35                # head-wide boxes (band + both cups): needs DETR_CONF + this = 0.75
EAR_MAX_H = 1.5                 # box beside the head at ear height and up to this many face heights tall = probably headphones...
EAR_CONF = 0.85                 # ...so it only counts as a phone with at least this score (and the look-alike memory still applies)
SOFT_GEOM = 0.2                 # squarish / worn-looking / bigger-than-a-head geometry demands at most this much MORE confidence (never a veto: a phone held close to the camera looks the same)
DETR_CONF = 0.4                 # RT-DETR scored real phones 0.5-0.9 and false ones <=0.35 in my tests; YOLO-type models use PHONE_CONF
PHONE_MIN_PX = 24               # ignore phone boxes smaller than this (px, longest side) unless cut by the frame edge
PHONE_EDGE = 0.2                # lower conf + no shape rule for boxes cut by the frame edge
BOOK_GRACE = 2.0                # s without a book sighting before its timer resets
PHONE_WIN, PHONE_DUTY, PHONE_MIN_S = 6.0, 0.45, 1.0  # phone must be seen for >=45% of the last 6s (time-weighted). Simulated: ~2 false events/h at a 10% false-hit rate
TRACK_ARM, TRACK_ARM_S = 2, 4.0 # tracking of weak hits starts only after 2 confident hits within 4s
FT_DIR, FT_WEIGHTS = "finetune", "phone_finetuned.pt"
CROP_DIR = FT_DIR + "/crops"
MEM_SIM, MEM_MARGIN, MEM_MIN_NEG = 0.85, 0.03, 6   # look-alike memory: veto at >=0.85 similarity to a remembered look-alike (needs >=6 samples)
MIN_POS, MIN_NEG, FT_EPOCHS, NEG_EVERY = 15, 60, 40, 1.0
PHONE_LIVE = 2.0                 # phone only counts as 'present' if seen within this many s (timer is kept longer)
PHONE_TRACK, TRACK_S = 0.12, 6.0     # once a phone is confirmed, near-by weak hits keep it alive for this long
YOLO_EVERY = 0.3                # s between detections (saves CPU)
CLIP_DIR, CLIP_FPS, CLIP_SIZE, DET_FILE = "clips", 10, (1280, 720), "detector.txt"
DEFAULT_DET = "rtdetr-l.pt"
CANDIDATES = ["rtdetr-l.pt", "yolov8s.pt", "yolo11s.pt", "yoloe-11s-seg.pt", "yolov8s-worldv2.pt"]   # --bench tries these (open-vocab ones need internet once)
OV_VOCAB = ["cell phone", "book", "headphones", "earbuds", "tablet", "smart watch", "laptop", "remote control", "wallet", "text on a t-shirt"]

def finetune():
    """python focus_guard.py --train : fine-tune YOLO on your own positives + hard negatives (head only, CPU is fine)."""
    from ultralytics import YOLO as Y
    imgs = glob.glob(f"{FT_DIR}/images/*.jpg")
    npos = sum(os.path.basename(i).startswith("pos_") for i in imgs); nneg = len(imgs) - npos
    print(f"{npos} phone samples, {nneg} no-phone samples")
    if npos < MIN_POS or nneg < MIN_NEG:
        print(f"Need >= {MIN_POS} phone (P key) and >= {MIN_NEG} no-phone (N key / F key) samples first."); return
    base = Y("yolov8s.pt")
    yml = os.path.abspath(f"{FT_DIR}/data.yaml")
    with open(yml, "w") as fh:
        fh.write(f"path: {os.path.abspath(FT_DIR)}\ntrain: images\nval: images\nnames:\n")
        fh.writelines(f"  {k}: {v}\n" for k, v in base.names.items())
    base.train(data=yml, epochs=FT_EPOCHS, imgsz=640, batch=8, freeze=10, optimizer="AdamW", lr0=5e-4, workers=0,
               project=os.path.abspath("finetune_runs"), name="run", exist_ok=True, plots=False, val=False)
    shutil.copy("finetune_runs/run/weights/last.pt", FT_WEIGHTS)
    print(f"Saved {FT_WEIGHTS}. Run: python focus_guard.py --bench   (it is used only if it beats the other detectors on your clips).")
if "--train" in sys.argv:
    finetune(); raise SystemExit

CLIP_PLAN = [("neg_work", "NO PHONE in view. Work as usual (headphones, shirt, desk).", 25),
             ("pos_upright", "HOLD your PHONE up, screen toward the camera. Move it around.", 20),
             ("pos_turned", "PHONE sideways / back to camera / low at desk edge. Turn it around.", 20)]
def record_clips():
    """python focus_guard.py --record : film short labelled clips of YOUR situations (replayed by --bench)."""
    os.makedirs(CLIP_DIR, exist_ok=True)
    cam = cv2.VideoCapture(0); cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    def show(img, lines):
        img = img.copy()
        for i, l in enumerate(lines):
            cv2.putText(img, l, (15, 40 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Record clips", img)
        return cv2.waitKey(1) & 0xFF
    for name, msg, secs in CLIP_PLAN:
        while True:                                            # wait for SPACE
            ok, f = cam.read()
            if not ok:
                print("camera error"); return
            k = show(f, [f"Next clip: {name}", msg, "SPACE = start   Q = quit"])
            if k == 32:
                break
            if k == ord("q"):
                cam.release(); cv2.destroyAllWindows(); return
        out = cv2.VideoWriter(f"{CLIP_DIR}/{name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, CLIP_SIZE)
        t0 = last = time.time()
        while time.time() - t0 < secs + 3:
            ok, f = cam.read()
            if not ok:
                break
            el = time.time() - t0
            if el < 3:                                         # 3 s to get ready
                show(f, [f"{name}: starting in {3 - int(el)}", msg]); continue
            if time.time() - last >= 1 / CLIP_FPS:
                out.write(cv2.resize(f, CLIP_SIZE)); last = time.time()
            show(f, [f"REC {name}  {int(secs + 3 - el)}s", msg])
        out.release()
    cam.release(); cv2.destroyAllWindows()
    print(f"Clips saved in {CLIP_DIR}/. Now run: python focus_guard.py --bench")
if "--record" in sys.argv:
    record_clips(); raise SystemExit

# ---- input activity ----
last_input = time.time()
def touch(*_):
    global last_input
    last_input = time.time()
if keyboard:
    keyboard.Listener(on_press=touch).start()
    mouse.Listener(on_move=touch, on_click=touch).start()

# ---- sound ----
last_beep = 0
def beep(level):
    global last_beep
    now = time.time()
    if now - last_beep < (5 if level == 1 else 2):
        return
    last_beep = now
    def run():
        if winsound:
            winsound.Beep(800, 300) if level == 1 else winsound.Beep(1800, 1200)
        else:
            print("\a", end="", flush=True)
    threading.Thread(target=run, daemon=True).start()

# ---- face landmarks (Tasks API) ----
MODEL = "face_landmarker.task"
if not os.path.exists(MODEL):
    print("Downloading face model (one time)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL)
landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO, num_faces=2))

face_geo = None   # (cx, cy, w, h) of main face, normalized; used to ignore ear-level objects
def measure(frame, ts=None):
    """(yaw, pitch) of the largest face or None. pitch > 0 = looking down."""
    global face_geo
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = landmarker.detect_for_video(img, int(time.time() * 1000) if ts is None else ts)
    if not res.face_landmarks:
        face_geo = None
        return None
    lm = max(res.face_landmarks, key=lambda f: abs(f[454].x - f[234].x))  # ignore bystanders
    face_geo = ((lm[234].x + lm[454].x) / 2, (lm[10].y + lm[152].y) / 2,
                abs(lm[454].x - lm[234].x), abs(lm[152].y - lm[10].y))
    width = abs(lm[454].x - lm[234].x) + 1e-6
    yaw = (lm[1].x - (lm[234].x + lm[454].x) / 2) / width
    eye_y = (lm[33].y + lm[263].y) / 2
    pitch = (lm[1].y - eye_y) / (lm[152].y - eye_y + 1e-6)
    return yaw, pitch

# ---- object detection ----
def load_detector(spec):
    """COCO models know phone/book only. Open-vocabulary models (yoloe/world) also see look-alikes that compete with 'phone'."""
    m = YOLO(spec); low = os.path.basename(spec).lower()
    if "yoloe" in low:
        m.set_classes(OV_VOCAB, m.get_text_pe(OV_VOCAB))
    elif "world" in low:
        m.set_classes(OV_VOCAB)
    else:
        tile = "rtdetr" in low                                                  # RT-DETR: see raw_dets()
        return dict(name=spec, yolo=m, classes=[67, 73], kind={67: "phone", 73: "book"}, open=False, tile=tile,
                    conf=DETR_CONF if tile else PHONE_CONF)
    return dict(name=spec, yolo=m, classes=None, open=True, tile=False, conf=PHONE_CONF,
                kind={i: "phone" if i == 0 else "book" if i == 1 else "neg" for i in range(len(OV_VOCAB))})

def chosen_detector():
    if os.path.exists(DET_FILE):
        return open(DET_FILE).read().strip()
    return DEFAULT_DET
try:
    DET = load_detector(chosen_detector())
except Exception as e:
    print("could not load", chosen_detector(), "->", str(e)[:100], "| falling back to yolov8s.pt")
    DET = load_detector("yolov8s.pt")
print("detector:", DET["name"])
boxes, rejected = [], []

def on_head(x1, y1, x2, y2, W, H):
    """Reason this box is probably worn on the head (headphones, headband, hair, nose...) or None. Uses face geometry."""
    if not face_geo:
        return None
    cx, cy, fw, fh = face_geo
    u, v = ((x1 + x2) / 2 / W - cx) / fw, ((y1 + y2) / 2 / H - cy) / fh        # centre, in face widths / heights
    bw, bh = (x2 - x1) / W / fw, (y2 - y1) / H / fh                              # size, in face widths / heights
    if v < -0.45 and abs(u) < 1.4:
        return "on head"                                                         # above the brow: never a phone
    if 0.35 < abs(u) < 1.6 and abs(v) < 0.8 and bh < EAR_MAX_H and not DET["open"]:
        return "at ear"                                                          # beside the head at ear height = headphone cup (+hand/band merged in the box)
    if abs(u) < 0.5 and abs(v) < 0.5 and bw < 0.6 and bh < 0.6:
        return "on face"
    if abs(u) < 0.8 and abs(v) < 1.0 and bw > 1.2:
        return "head-wide"                                                       # band + both cups in one box
    return None

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

def padded_crop(img, box, pad=0.1):
    x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - pad * w)), max(0, int(y1 - pad * h)); x2, y2 = min(img.shape[1], int(x2 + pad * w)), min(img.shape[0], int(y2 + pad * h))
    return img[y1:y2, x1:x2] if x2 - x1 >= 12 and y2 - y1 >= 12 else None

class LookAlikes:
    """Remembers what YOUR phone look-alikes (headphones, shirt print...) look like: image embeddings of phone-like boxes saved while no
    phone was around (N / F keys). A candidate nearly identical to one of them, and not closer to your real phone (P key), is vetoed.
    Embeddings: yolov8n-cls features, standardised with statistics of random patches of your own camera frames."""
    def __init__(self):
        self.net = self.mu = self.sd = None; self.neg, self.pos, self.pool = [], [], []; self.ready = self.failed = False
    def _raw(self, img):
        v = self.net(cv2.resize(img, (224, 224)), embed=[8], verbose=False)[0]
        return np.asarray(v.cpu().numpy() if hasattr(v, "cpu") else v).ravel()
    def _emb(self, img):
        v = (self._raw(img) - self.mu) / self.sd
        return v / (np.linalg.norm(v) + 1e-9)
    def feed(self, frame):                                   # called every detection cycle until ready
        if self.ready or self.failed:
            return
        H, W = frame.shape[:2]
        for _ in range(6):
            sz = random.randint(40, min(200, H - 1, W - 1)); x, y = random.randint(0, W - sz), random.randint(0, H - sz)
            self.pool.append(frame[y:y + sz, x:x + sz].copy())
        if len(self.pool) < 60:
            return
        try:
            self.net = YOLO("yolov8n-cls.pt")
            raws = np.stack([self._raw(p) for p in self.pool]); self.mu, self.sd = raws.mean(0), raws.std(0) + 1e-6
            for kind, lst in (("neg", self.neg), ("pos", self.pos)):
                for fn in sorted(glob.glob(f"{CROP_DIR}/{kind}_*.jpg"))[-300:]:
                    im = cv2.imread(fn)
                    if im is not None:
                        lst.append(self._emb(im))
            self.ready = True
            print(f"look-alike memory ready: {len(self.neg)} look-alikes, {len(self.pos)} phone crops")
        except Exception as e:
            self.failed = True; print("look-alike memory off:", str(e)[:100])
        self.pool = []
    def add(self, kind, frame, box):
        crop = padded_crop(frame, box)
        if crop is None:
            return
        os.makedirs(CROP_DIR, exist_ok=True)
        cv2.imwrite(f"{CROP_DIR}/{kind}_{int(time.time() * 1000)}_{random.randint(0, 999)}.jpg", crop)
        if self.ready:                                       # before ready the saved file is loaded at start-up instead
            (self.neg if kind == "neg" else self.pos).append(self._emb(crop))
    def check(self, frame, box):
        neg = list(self.neg)
        if not self.ready or len(neg) < MEM_MIN_NEG:
            return None
        crop = padded_crop(frame, box)
        if crop is None:
            return None
        e = self._emb(crop); sn = float((np.stack(neg) @ e).max()); pos = list(self.pos)
        sp = float((np.stack(pos) @ e).max()) if pos else -1.0
        return f"look-alike {sn:.2f}" if sn >= MEM_SIM and sn > sp + MEM_MARGIN else None
MEM = LookAlikes()

def raw_dets(frame, crop):
    """Raw detector output as (class id, conf, x1, y1, x2, y2) in full-frame pixels.
    RT-DETR squashes its input to a square, which would distort a 16:9 frame, so it sees two overlapping square tiles instead."""
    FH, W = frame.shape[:2]
    kw = {} if DET["classes"] is None else {"classes": DET["classes"]}
    if DET["tile"]:
        side = min(W, FH); out = []
        for x0 in ([0] if W <= FH else [0, W - FH]):
            r = DET["yolo"](np.ascontiguousarray(frame[:side, x0:x0 + side]), verbose=False, conf=0.1, imgsz=640, **kw)[0]
            out += [(int(b.cls[0]), float(b.conf[0]), int(b.xyxy[0][0]) + x0, int(b.xyxy[0][1]), int(b.xyxy[0][2]) + x0, int(b.xyxy[0][3]))
                    for b in r.boxes]
        out.sort(key=lambda o: -o[1]); keep = []
        for o in out:                                          # the overlap strip is seen twice: keep the stronger copy
            if not any(k[0] == o[0] and iou(k[2:], o[2:]) > 0.5 for k in keep):
                keep.append(o)
        return keep
    y0 = FH // 2 if crop else 0
    r = DET["yolo"](frame[y0:], verbose=False, conf=0.1, imgsz=960, **kw)[0]
    return [(int(b.cls[0]), float(b.conf[0]), int(b.xyxy[0][0]), int(b.xyxy[0][1]) + y0, int(b.xyxy[0][2]), int(b.xyxy[0][3]) + y0)
            for b in r.boxes if not (crop and int(b.xyxy[0][1]) <= 3)]    # boxes cut by the crop line belong to the full pass

def detect(frame, crop=False, track=None):
    """One detector pass. crop=True looks at the bottom half only (lap/desk). Returns (phone?, book?, boxes, confident phones, rejected)."""
    FH, W = frame.shape[:2]
    ph = bk = False; out = []; strong = []; raw_books = []; rej = []; negs = []
    for cid, p, x1, y1, x2, y2 in raw_dets(frame, crop):
        kind = DET["kind"].get(cid)
        w, h = x2 - x1, y2 - y1
        ar = max(w, h) / max(1, min(w, h))
        edge = x1 <= 4 or x2 >= W - 4 or y2 >= FH - 4 or y1 <= 4   # partly outside the frame
        if kind == "neg":
            if p >= 0.25:
                negs.append((OV_VOCAB[cid], p, x1, y1, x2, y2))
        elif kind == "phone":
            if not edge and max(w, h) < PHONE_MIN_PX:
                continue
            if w * h > MAX_PHONE_AREA * W * FH:
                if p >= 0.25:
                    rej.append(("too big", p, x1, y1, x2, y2))
                continue
            why = on_head(x1, y1, x2, y2, W, FH); soft = 0.0
            ear = why == "at ear"
            if ear:
                why = None                                                       # not a veto any more: only a very confident box counts here
            if why == "head-wide":                                               # hint, not proof: depth is unknown
                soft, why = SOFT_HEAD, None
            if face_geo and max(w, h) > 1.4 * face_geo[3] * FH:
                if max(w, h) > 3.5 * face_geo[3] * FH:
                    why = why or "too big"                                       # laptop/monitor-sized: hard veto
                else:
                    soft = SOFT_GEOM
            if why:
                if p >= 0.25:
                    rej.append((why, p, x1, y1, x2, y2))
                continue
            near = False
            if track:                                   # same place as the confirmed phone -> weak hit is enough
                tx, ty = (track[0] + track[2]) / 2, (track[1] + track[3]) / 2
                tsz = max(track[2] - track[0], track[3] - track[1])
                near = abs((x1 + x2) / 2 - tx) < 1.5 * tsz and abs((y1 + y2) / 2 - ty) < 1.5 * tsz \
                       and 0.5 < max(w, h) / tsz < 2      # similar size, so tracking can't drift onto a book
            need = PHONE_EDGE if edge else DET["conf"] + (0.15 if crop else 0) + min(SOFT_HEAD, (0 if ar >= PHONE_AR else PHONE_SQUARE_EXTRA) + soft)
            if ear:
                need = max(need, EAR_CONF)
            strict = p >= need
            if not (strict or (near and p >= PHONE_TRACK)):
                if p >= 0.25:
                    rej.append((f"needs {need:.2f}", p, x1, y1, x2, y2))
                continue
            out.append((67, p, x1, y1, x2, y2))
            if strict:
                strong.append((67, p, x1, y1, x2, y2))
        elif kind == "book":
            if p >= 0.2:
                raw_books.append((x1, y1, x2, y2))
            if p >= BOOK_CONF:
                out.append((73, p, x1, y1, x2, y2))
                bk = True
    # a phone box covered by a book box is a book; one that loses to a look-alike class (headphones...) is not a phone
    def lost_to(o):
        for bb in raw_books:
            inter = max(0, min(o[4], bb[2]) - max(o[2], bb[0])) * max(0, min(o[5], bb[3]) - max(o[3], bb[1]))
            ab, ao = (bb[2] - bb[0]) * (bb[3] - bb[1]), (o[4] - o[2]) * (o[5] - o[3])
            if iou(o[2:], bb) > 0.4:
                return "book"
            if inter > 0.7 * ab and ao > ab:
                return "contains book"                                           # a phone box can't swallow a whole book
        for n in negs:
            if iou(o[2:], n[2:]) > 0.3 and n[1] >= 0.8 * o[1]:
                return n[0]
        return None
    keep = []
    for o in out:
        why = lost_to(o) if o[0] == 67 else None
        if not why and o[0] == 67:
            why = MEM.check(frame, o[2:])
        if why:
            rej.append((why, o[1], *o[2:]))
        else:
            keep.append(o)
    out = keep; strong = [o for o in strong if o in out]
    rej += [n for n in negs if n[1] >= 0.4]                                      # show look-alikes the model sees
    ph = any(o[0] == 67 for o in out)
    return ph, bk, out, strong, rej

class WeakTrack:
    """After repeated confident hits, weak hits at the same spot keep a phone 'seen' for a while (it flickers when moved/turned)."""
    def __init__(self):
        self.box, self.t, self.hist = None, -1e9, deque()
    def box_for(self, now):
        return self.box if now - self.t < TRACK_S else None
    def feed(self, now, strong):
        if strong:
            self.hist.append(now)
        while self.hist and now - self.hist[0] > TRACK_ARM_S:
            self.hist.popleft()
        if strong and (self.box_for(now) is not None or len(self.hist) >= TRACK_ARM):
            self.box, self.t = max(strong, key=lambda o: o[1])[2:], now

def detect_cycle(f, wt, now, i):
    """The app's detection step (also used by --bench): full frame, plus a bottom-half pass every other cycle."""
    tr = wt.box_for(now)
    ph, bk, b2, st, rj = detect(f, track=tr)
    if not ph and i % 2 and not DET["tile"]:
        ph2, bk2, b3, st2, rj2 = detect(f, crop=True, track=tr)
        ph, bk, b2, st, rj = ph or ph2, bk or bk2, b2 + b3, st + st2, rj + rj2
    wt.feed(now, st)
    return ph, bk, b2, rj

class PhoneTimer:
    """Sustained agreement: phone must be present in >= PHONE_DUTY of recent cycles; scattered hits drain instead of piling up."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.hist, self.t, self.gap, self.duty = deque(), 0.0, 0.0, 0.0
    def update(self, now, hit, step, box=None):
        self.hist.append((now, bool(hit), step, box))
        while self.hist and now - self.hist[0][0] > PHONE_WIN:
            self.hist.popleft()
        span = sum(h[2] for h in self.hist)                    # time-weighted, so slow detectors are judged fairly
        hits = [h for h in self.hist if h[1]]
        if any(h[3] for h in hits):                            # a real phone stays in one area; false boxes jump around -> count the biggest cluster
            def near(a, b):
                sz = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
                return abs((a[0] + a[2]) - (b[0] + b[2])) / 2 < 1.5 * sz and abs((a[1] + a[3]) - (b[1] + b[3])) / 2 < 1.5 * sz
            on = max((sum(g[2] for g in hits if g[3] and near(h[3], g[3])) for h in hits if h[3]), default=0.0)
        else:
            on = sum(h[2] for h in hits)
        self.duty = on / max(span, 1e-6)
        self.gap = 0.0 if hit else self.gap + step
        if span >= PHONE_MIN_S and self.duty >= PHONE_DUTY:
            self.t += step
        else:
            self.t = max(0.0, self.t - step)
    @property
    def active(self):
        return self.t >= PHONE_HOLD and self.gap <= PHONE_LIVE

def save_sample(kind, img, phone_box=None, book_boxes=()):
    """Write a YOLO-format training sample. 'neg' = no phone anywhere in this frame (empty phone labels)."""
    os.makedirs(f"{FT_DIR}/images", exist_ok=True); os.makedirs(f"{FT_DIR}/labels", exist_ok=True)
    name = f"{kind}_{int(time.time() * 1000)}"; H, W = img.shape[:2]
    line = lambda c, b: f"{c} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}"
    lines = ([line(67, phone_box)] if phone_box else []) + [line(73, b) for b in book_boxes]   # keep books alive (proof flow)
    cv2.imwrite(f"{FT_DIR}/images/{name}.jpg", img)
    with open(f"{FT_DIR}/labels/{name}.txt", "w") as fh:
        fh.write("\n".join(lines))

def used(now):
    return sum(d for t, d in grants if now - t < 3600)

def bench():
    """python focus_guard.py --bench : replay recorded clips through each candidate detector and the app's real pipeline."""
    global DET, face_geo
    clips = sorted(glob.glob(f"{CLIP_DIR}/*.mp4"))
    if not any("pos_" in c for c in clips) or not any("neg_" in c for c in clips):
        print("Need clips first: python focus_guard.py --record"); return
    data, base_ts = {}, 0
    for c in clips:                                    # sample frames + face geometry once (independent of the detector)
        vc = cv2.VideoCapture(c); vfps = vc.get(cv2.CAP_PROP_FPS) or CLIP_FPS
        every, n, fr = max(1, round(YOLO_EVERY * vfps)), 0, []
        while True:
            ok, f = vc.read()
            if not ok:
                break
            if n % every == 0:
                measure(f, base_ts + int(n / vfps * 1000)); fr.append((n / vfps, f, face_geo))
            n += 1
        base_ts += int(n / vfps * 1000) + 1000; data[c] = fr
        print(f"{os.path.basename(c)}: {len(fr)} samples")
    cands = CANDIDATES + ([FT_WEIGHTS] if os.path.exists(FT_WEIGHTS) else [])
    results = []
    print(f"\n{'detector':22s} {'clip':14s} {'raw hit':>8s} {'alarm on':>9s} {'first alarm':>12s}")
    for spec in cands:
        try:
            DET = load_detector(spec)
        except Exception as e:
            print(f"{spec:22s} skipped: {str(e)[:80]}"); continue
        row = {}
        for c, fr in data.items():
            wt, pt = WeakTrack(), PhoneTimer(); hits = act = tot = 0; first = None; prev_t = fr[0][0] - YOLO_EVERY
            for i, (t, f, fg) in enumerate(fr):
                face_geo = fg
                ph, bk, b2, rj = detect_cycle(f, wt, t, i)
                pt.update(t, ph, t - prev_t); prev_t = t
                if t >= 2.0:                           # ignore the first 2 s (settling)
                    tot += 1; hits += bool(ph); act += pt.active
                    if pt.active and first is None:
                        first = t
            row[c] = (hits / max(tot, 1), act / max(tot, 1), first)
            print(f"{spec:22s} {os.path.basename(c)[:-4]:14s} {row[c][0]:8.0%} {row[c][1]:9.0%} {('%.1fs' % first) if first else '-':>12s}")
        pos = [v[1] for c, v in row.items() if "pos_" in c]; neg = [v[1] for c, v in row.items() if "neg_" in c]
        results.append((sum(pos) / len(pos) - 3 * sum(neg) / len(neg), spec))
    if not results:
        print("no detector could run"); return
    results.sort(reverse=True)
    print("\nscore = (alarm% on phone clips) - 3 x (alarm% on no-phone clips)")
    for sc, spec in results:
        print(f"  {sc:+.2f}  {spec}")
    open(DET_FILE, "w").write(results[0][1])
    print(f"\nWinner written to {DET_FILE}: {results[0][1]}  (delete that file to go back to the default)")
if "--bench" in sys.argv:
    bench(); raise SystemExit

# ---- state ----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # more pixels = small phones visible
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# ---- background threads: the camera and YOLO never block the focus loop ----
latest = {"frame": None, "id": 0, "dead": False}
stop = False
def grabber():
    fails = 0
    while not stop:
        ok, f = cap.read()
        if ok:
            latest["frame"], latest["id"], fails = f, latest["id"] + 1, 0
        else:
            fails += 1
            if fails > 100:
                latest["dead"] = True; return
            time.sleep(0.02)
threading.Thread(target=grabber, daemon=True).start()

wt = WeakTrack()
det_out = {"seq": 0, "t": time.time(), "ph": False, "bk": False, "boxes": [], "rej": [], "frame": None, "ms": 0.0}
def detector():
    global det_out
    i, seen = 0, -1
    while not stop:
        t0 = time.time()
        f = latest["frame"]
        if f is None or latest["id"] == seen:
            time.sleep(0.02); continue
        seen = latest["id"]
        try:
            ph, bk, b2, rj = detect_cycle(f, wt, t0, i); i += 1
            MEM.feed(f)
            det_out = {"seq": det_out["seq"] + 1, "t": time.time(), "ph": ph, "bk": bk, "boxes": b2, "rej": rj, "frame": f, "ms": (time.time() - t0) * 1000}
        except Exception as e:
            print("detector error:", e); time.sleep(1)
        pace = YOLO_EVERY if globals().get("fps", 30.0) >= 15 else 3 * YOLO_EVERY   # back off if the face loop slows down
        time.sleep(max(0.0, pace - (time.time() - t0)))
threading.Thread(target=detector, daemon=True).start()
seen_id, seen_seq, fps = -1, 0, 30.0

base = None
mode, break_end = "SETUP", 0
lost = bad = down_t = book_t = last_pitch = last_yaw = 0.0
book_gap = boxes_t = last_neg = det_ms = 0.0
pt = PhoneTimer()
last_det = None
neg_rec = False
ft_pos = len(glob.glob(f"{FT_DIR}/images/pos_*.jpg")); ft_neg = len(glob.glob(f"{FT_DIR}/images/neg_*.jpg"))
yolo_i = 0
last_phone, last_phone_t = None, 0.0
challenge_end = grant_end = no_challenge_until = note_until = 0.0
note = ""
grants, appeals = [], []
stats = dict(total=0.0, bad=0.0, alarms=0, passed=0, failed=0, appeals=0)
alarmed = False
prev = last_yolo = time.time()

while True:
    if latest["dead"]:
        break
    f = latest["frame"]
    if f is None or latest["id"] == seen_id:
        time.sleep(0.005)
        continue
    seen_id = latest["id"]; frame = f.copy()
    now = time.time(); dt = now - prev; prev = now
    fps = 0.9 * fps + 0.1 / max(dt, 1e-3)

    d = det_out
    if d["seq"] != seen_seq:                           # a new detection result arrived
        seen_seq = d["seq"]
        ph, bk, b2 = d["ph"], d["bk"], d["boxes"]
        if b2 or d["rej"] or now - boxes_t > 1.0:      # keep boxes on screen between passes
            boxes, rejected, boxes_t = b2, d["rej"], now
        step = min(d["t"] - last_yolo, 1.5); last_yolo = d["t"]
        last_det = d; det_ms = d["ms"]
        pbox = max((o for o in b2 if o[0] == 67), key=lambda o: o[1], default=None)
        pt.update(now, ph, step, pbox[2:] if pbox else None)
        if bk:
            book_t += step; book_gap = 0
        else:
            book_gap += step
            if book_gap > BOOK_GRACE:
                book_t = 0

    if neg_rec and last_det and mode != "BREAK" and now - last_neg >= NEG_EVERY and last_det["frame"] is not None:
        save_sample("neg", last_det["frame"], None, [o[2:] for o in last_det["boxes"] if o[0] == 73 and o[1] >= 0.5])
        for o in last_det["boxes"]:                    # every phone-like box seen while no phone is around = a look-alike
            if o[0] == 67:
                MEM.add("neg", last_det["frame"], o[2:])
        for r in last_det["rej"]:                      # also the ones currently below the bar (they may pass later)
            if r[0].startswith("needs"):
                MEM.add("neg", last_det["frame"], r[2:])
        last_neg = now; ft_neg += 1

    m = measure(frame)
    status, color, dbg = "FOCUSED", (0, 200, 0), ""
    unfocused = False; counted = True

    if mode == "SETUP":                                # no alarms, no timers: teach it your desk first
        status, color, bad, counted = "SETUP", (255, 200, 0), 0, False
        pt.reset(); wt.box = None; challenge_end = grant_end = 0.0
    elif mode == "BREAK":
        left = int(break_end - now)
        status, color, bad, counted = f"BREAK {left//60}:{left%60:02d}", (200, 200, 0), 0, False
        if left <= 0:
            mode = "FOCUS"; beep(1)
    else:
        yaw = pitch = 0.0
        if m:
            lost = 0
            if base is None:
                base = m
            yaw, pitch = m[0] - base[0], m[1] - base[1]
            last_pitch, last_yaw = pitch, yaw
            dbg = f"yaw {yaw:+.2f} pitch {pitch:+.2f}"
        else:
            lost += dt
        # face vanished while head was going down = probably head-down, not left
        lost_down = m is None and last_pitch > 0.7 * PITCH_T and lost < 60
        lost_turned = m is None and abs(last_yaw) > 0.7 * YAW_T and lost < 60  # vanished mid-turn (~90 deg)

        if m is None and lost > AWAY_S and not lost_down and not lost_turned:
            status, color, bad, down_t, challenge_end, counted = "AWAY (paused)", (150, 150, 150), 0, 0, 0, False
        else:
            typing = now - last_input < INPUT_GRACE
            down = (pitch > PITCH_T or lost_down) and not typing
            turned = abs(yaw) > YAW_T or lost_turned
            down_t = down_t + dt if down else 0
            reason, rate, hold = None, 1, False

            if pt.active:                                  # phone overrides every excuse
                reason, rate, challenge_end = "Phone detected", 2, 0
            elif turned:                                   # turning away is never excused
                reason, challenge_end = "Turned away", 0
            elif challenge_end:
                if book_t >= BOOK_HOLD:                    # proof accepted
                    grant_end = now + GRANT_S; grants.append((now, GRANT_S))
                    challenge_end, down_t = 0, 0; stats["passed"] += 1
                elif now > challenge_end:                  # proof failed
                    challenge_end, no_challenge_until, bad = 0, now + 60, ALARM_S + 1
                    stats["failed"] += 1; reason = "No proof shown"
                else:
                    hold = True
                    status, color = f"SHOW YOUR BOOK/NOTES {int(challenge_end - now) + 1}s", (0, 215, 255)
            elif now < grant_end and down:                 # reading excuse only applies while actually looking down
                status, color = f"READING {int(grant_end - now)}s", (200, 150, 0)
            elif down and down_t > DOWN_S:
                if now < no_challenge_until or used(now) + GRANT_S > READ_BUDGET:
                    reason = "Looking down, unproven"
                else:
                    challenge_end = now + CHALLENGE_S; beep(1); hold = True
                    status, color = "SHOW YOUR BOOK/NOTES", (0, 215, 255)

            if reason:
                bad = min(bad + dt * rate, ALARM_S + 2)     # capped so recovery is quick
                status, color = reason, (0, 165, 255)
            elif not hold:
                bad = max(0, bad - 3 * dt)
            unfocused = reason is not None or bad > WARN_S
            if bad > ALARM_S:
                status, color = "FOCUS UP!", (0, 0, 255); beep(2)
                if not alarmed:
                    stats["alarms"] += 1; alarmed = True
            elif bad > WARN_S:
                beep(1)
            if bad <= WARN_S:
                alarmed = False

    if counted:
        stats["total"] += dt
        stats["bad"] += dt if unfocused else 0

    cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"{fps:.0f}fps {mode} bad={bad:.0f}s phone={pt.t:.1f}(duty {pt.duty:.0%}) book={book_t:.1f} {dbg}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{DET['name']} {det_ms:.0f}ms/cycle", (frame.shape[1] - 330, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    if mode == "SETUP":
        n = ft_pos + ft_neg
        lines = ["SETUP: nothing will alarm until you press S",
                 "N: record (phone OUT of view, wear headphones, move around)   P/L: mark your phone when it is on screen",
                 f"samples {n}/{SETUP_MIN}  " + ("-> press S to START" if n >= SETUP_MIN else "(S unlocks at %d)" % SETUP_MIN)]
        for i, t_ in enumerate(lines):
            cv2.putText(frame, t_, (10, 160 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if i == 2 and n >= SETUP_MIN else (255, 255, 0), 2)
    if neg_rec:
        cv2.putText(frame, "REC NEG (phone must be out of view)", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(frame, f"samples: {ft_pos} phone / {ft_neg} no-phone | look-alikes remembered: {len(MEM.neg) if MEM.ready else 'loading...'}", (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    if now < note_until:
        cv2.putText(frame, note, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    # Detection rectangles: hidden while working, kept for later (debug). Drawn only in SETUP, because P (save phone) needs to see the box.
    # To always show them, change the condition below to `if True:`
    if mode == "SETUP":
        for label, p, x1, y1, x2, y2 in rejected:          # grey = seen but NOT counted (and why)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 140, 140), 1)
            cv2.putText(frame, f"x {label} {p:.2f}", (x1, min(y2 + 14, frame.shape[0] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
        for c, p, x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255) if c == 67 else (255, 150, 0), 2)
            cv2.putText(frame, f"{'phone' if c == 67 else 'book'} {p:.2f} ar{max(x2 - x1, y2 - y1) / max(1, min(x2 - x1, y2 - y1)):.1f}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Focus Guard", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    elif k == ord("c"):                                # recalibrate + clear any stuck state
        base = None
        bad = book_t = book_gap = down_t = 0.0
        pt.reset(); wt.box = None
        challenge_end = grant_end = 0.0
    elif k == ord("s") and mode == "SETUP":
        if ft_pos + ft_neg >= SETUP_MIN:
            mode, neg_rec, base = "FOCUS", False, None; beep(1)
            note, note_until = "Focus Guard started", now + 3
        else:
            note, note_until = f"Need {SETUP_MIN - ft_pos - ft_neg} more samples (N / P / L)", now + 3
    elif k == ord("b") and mode != "SETUP":
        mode, break_end = ("FOCUS", 0) if mode == "BREAK" else ("BREAK", now + BREAK_S)
        challenge_end = 0
    elif k == ord("n"):
        neg_rec = not neg_rec
        note, note_until = ("Recording no-phone frames: keep your phone OUT of view" if neg_rec else "Stopped recording"), now + 3
    elif k == ord("p"):
        ph_boxes = [o for o in (last_det or {}).get("boxes", []) if o[0] == 67]
        if last_det and ph_boxes and now - last_det["t"] < 1.5:
            best = max(ph_boxes, key=lambda o: o[1])
            save_sample("pos", last_det["frame"], best[2:], [o[2:] for o in last_det["boxes"] if o[0] == 73 and o[1] >= 0.5])
            MEM.add("pos", last_det["frame"], best[2:])
            ft_pos += 1; note, note_until = f"Saved phone sample #{ft_pos}", now + 2
        else:
            note, note_until = "No phone box on screen to save", now + 2
    elif k == ord("l"):                                # draw the box yourself: for phones the detector missed (turned, edge-on...)
        raw = f.copy(); win = "Draw the phone box, then ENTER (C = cancel)"
        rx, ry, rw, rh = cv2.selectROI(win, raw, False)
        cv2.destroyWindow(win)
        if rw > 5 and rh > 5:
            save_sample("pos", raw, (rx, ry, rx + rw, ry + rh), []); ft_pos += 1
            MEM.add("pos", raw, (rx, ry, rx + rw, ry + rh))
            note, note_until = f"Saved hand-labelled phone #{ft_pos}", time.time() + 2
        prev = time.time()
    elif k == ord("f"):
        appeals = [t for t in appeals if now - t < 3600]
        if pt.active and last_det and last_det["frame"] is not None:   # the phone alarm was wrong -> hard negative
            save_sample("neg", last_det["frame"], None, [o[2:] for o in last_det["boxes"] if o[0] == 73 and o[1] >= 0.5]); ft_neg += 1
            for o in last_det["boxes"]:                # the false alarm's box is a look-alike: remember it
                if o[0] == 67:
                    MEM.add("neg", last_det["frame"], o[2:])
            pt.reset()
        if len(appeals) < APPEALS_MAX:
            appeals.append(now); stats["appeals"] += 1
            bad, challenge_end, no_challenge_until = 0, 0, now + 30
            note, note_until = f"Appeal logged ({len(appeals)}/{APPEALS_MAX} this hour)", now + 3
        else:
            note, note_until = "Appeal limit reached", now + 3

stop = True
time.sleep(0.3)
cap.release()
cv2.destroyAllWindows()
t = max(stats["total"], 1)
print(f"Focus {100 * (1 - stats['bad'] / t):.0f}% of {t / 60:.1f} min | alarms {stats['alarms']} | "
      f"proofs passed {stats['passed']} failed {stats['failed']} | appeals {stats['appeals']}")