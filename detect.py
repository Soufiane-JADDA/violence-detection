import cv2, torch, time, threading
from torchvision import transforms
from model import ViolenceDetector
from collections import deque
from playsound import playsound
import numpy as np

# ── constants ──────────────────────────────────────────────────────────
SEQ_LEN       = 20      # frames per clip
MOTION_SIZE   = 64      # down-scaled side length for diff test
MOTION_THRESH = 7.0     # mean-abs-diff threshold (0-255 scale)

# ── utils ──────────────────────────────────────────────────────────────
def play_alert_sound():
    try: playsound("police.wav")
    except Exception as e: print("🔇 sound error:", e)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess_clip(frames):
    if len(frames) > SEQ_LEN:
        frames = frames[-SEQ_LEN:]
    while len(frames) < SEQ_LEN:
        frames.insert(0, frames[0])
    tensors = [transform(f).unsqueeze(0) for f in frames]
    return torch.cat(tensors).unsqueeze(0).to(device)   # (1,T,C,H,W)

def frame_gray_small(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (MOTION_SIZE, MOTION_SIZE))

# ── device & model ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔧 device:", device)
model  = ViolenceDetector().to(device)
ckpt   = torch.load("checkpoints/violence_detector_best/best_model.pt", map_location=device)
model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
model.eval()

# ── webcam init ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(3, cv2.CAP_V4L2)
assert cap.isOpened(), "Webcam not found"

cv2.startWindowThread()
cv2.namedWindow("Violence Detection", cv2.WINDOW_NORMAL)

frame_buffer, diff_queue = [], deque(maxlen=SEQ_LEN)
hist_queue  = deque(maxlen=15)
cooldown, last_gray = 0, None
font = cv2.FONT_HERSHEY_SIMPLEX
start = time.time()

print("🟢 real-time detection (press q to quit)")

while True:
    ok, frame_bgr = cap.read()
    if not ok: break

    # ── cheap motion detection ──
    cur_gray = frame_gray_small(frame_bgr)
    if last_gray is None:
        diff_val = MOTION_THRESH + 1     # force first clip
    else:
        diff_val = np.mean(cv2.absdiff(cur_gray, last_gray))
    last_gray = cur_gray

    if diff_val >= MOTION_THRESH:
        frame_buffer.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        diff_queue.append(diff_val)

    # keep buffer length
    if len(frame_buffer) > SEQ_LEN:
        frame_buffer = frame_buffer[-SEQ_LEN:]

    # ── default overlay ──
    label_text, color, prob = "Waiting…", (255,255,255), 0.0

    # run model only when we have a full fresh clip
    if len(frame_buffer) == SEQ_LEN and diff_val >= MOTION_THRESH:
        clip = preprocess_clip(frame_buffer)
        with torch.no_grad():
            logits = model(clip)
            prob   = torch.softmax(logits,1)[0,1].item()
            pred   = int(torch.argmax(logits,1))

        hist_queue.append(int(prob>0.80 and pred==1))
        if sum(hist_queue) >= 12:
            label_text, color = "🚨 VIOLENCE", (0,0,255)
            if cooldown==0:
                threading.Thread(target=play_alert_sound, daemon=True).start()
                cooldown = 30
        else:
            label_text, color = f"🟢 Normal ({prob:.2f})", (0,255,0)
        cooldown = max(cooldown-1, 0)

    # ── drawing ──
    fps = 1.0/(time.time()-start); start = time.time()
    #cv2.rectangle(frame_bgr,(10,80),(10+int(prob*200),100),color,-1)
    cv2.putText(frame_bgr,f"FPS:{fps:.1f}", (10,70), font,0.6,(200,200,200),1)
    cv2.putText(frame_bgr,label_text,(10,40),font,0.7,color,2)
    cv2.imshow("Violence Detection", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()
