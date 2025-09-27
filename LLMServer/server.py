import asyncio, struct, json, numpy as np, websockets, torch
from pathlib import Path
import traceback
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# === UI event log (JSONL) ===
from datetime import datetime
LOG_PATH = Path("ui_logs/hase_events.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
OVERRIDE_PATH = Path("ui_logs/threshold_override.json")
# --- add near your imports ---
import aiohttp, asyncio

SYSTEM_PROMPT = """You are a supportive, culturally-neutral coach for live social interaction.
Output exactly one sentence (<= 12 words), no emojis, no slang, no medical claims, no judgments,
no imperatives stronger than 'try/consider'. Tone: calm, discreet, inclusive. If uncertain, suggest a gentle breathing cue.
Examples: "Try a slow breath before your next sentence." "Consider pausing briefly to gather your thoughts." """

OLLAMA_MODEL = "llama3.1:8b"
LLM_SEM = asyncio.Semaphore(3)

async def gen_prompt(session: aiohttp.ClientSession, text_hint: str, timeout: float = 15.0) -> str:
    # Increased timeout to 5 seconds to give the model more time to load
    payload = {
        "model": OLLAMA_MODEL,
        "options": {"temperature": 0.6, "top_p": 0.9, "num_predict": 24},
        "system": SYSTEM_PROMPT,
        "prompt": f"Context: {text_hint}\nGoal: return the single most broadly applicable cue.",
        "stream": False 
    }
    
    resp = "" # Default value
    
    try:
        logging.info(f"Sending request to Ollama with model {OLLAMA_MODEL}...")
        async with session.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=timeout) as r:
            
            logging.info(f"Received response with status code: {r.status}")
            
            # Check if the server responded with an error code
            if r.status != 200:
                error_text = await r.text()
                logging.error(f"Ollama server returned an error: {error_text}")
                # Set a default response because the call failed
                return "Consider pausing for a moment."

            # If status is OK, parse the JSON
            out = await r.json()
            logging.info(f"Received JSON payload: {out}")
            
            # Safely get the response from the JSON
            resp = (out.get("response", "") or "").strip().split("\n")[0]
            if not resp:
                logging.warning("JSON response was successful but the 'response' key was empty.")

    except asyncio.TimeoutError:
        logging.error(f"Request timed out after {timeout} seconds. The model might be loading or the server is slow.")
    except aiohttp.ClientConnectorError as e:
        logging.error(f"Connection refused. Is the Ollama server running? Details: {e}")
    except Exception as e:
        # This will catch any other error, like invalid JSON, etc.
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

    # --- Last-mile guardrails (unchanged) ---
    if not resp:
        return "Try a slow breath before your next sentence."
    resp = resp.replace("—","-")
    if resp.count(".") > 1:
        resp = resp.split(".")[0] + "."
    words = resp.split()
    if len(words) > 12:
        resp = "Try a slow breath before your next sentence."
    if not resp.endswith("."):
        resp += "."
        
    logging.info(f"Final generated response: '{resp}'")
    return resp
def current_thresholds(default_social, default_arousal):
    try:
        obj = json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
        return float(obj.get("theta_social", default_social)), float(obj.get("theta_arousal", default_arousal))
    except Exception:
        return default_social, default_arousal

def _safe_jsonable(x):
    import numpy as np, torch, math
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_safe_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _safe_jsonable(v) for k,v in x.items()}
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "fiu":  # numeric array
            return x.tolist()
        return str(x.shape)
    if torch.is_tensor(x):
        if x.numel() <= 64:
            return x.detach().cpu().tolist()
        return f"tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
    if isinstance(x, (datetime,)):
        return x.isoformat()
    try:
        return float(x)
    except Exception:
        return str(x)

def log_event(status: str, **data):
    rec = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status": status,
        **{k: _safe_jsonable(v) for k,v in data.items()},
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ======== Config: paths to your artifacts ========
ART_DIR = Path(".")
C_SOCIAL_PT       = "C_social_Classifier//C_social_final.pt"
THRESH_SOCIAL_JSON= "C_social_Classifier//threshold_final.json"
AROUSAL_ANCHORS_PT= "S_arousal//arousal_text_anchors.pt"
AROUSAL_THR_JSON  = "S_arousal//arousal_threshold.json"

# ======== Binary frame layout (matches your client) ========
HDR_FMT  = "<I B B H d I I I I I I I"  # magic,ver,flags,res,t0, imu_hz,imu_n,imu_c,imu_cnt, mfcc_n,mfcc_c,mfcc_cnt
HDR_SIZE = struct.calcsize(HDR_FMT)
MAGIC    = 0x48415345  # 'HASE'

from imagebind.models import imagebind_model
import torch.nn.functional as F

# device
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"STATE: main - Using device: {device}")
log_event("server_starting", device=str(device))
ib = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()
imu_pre   = ib.modality_preprocessors["imu"].to(device).eval()
imu_trunk = ib.modality_trunks["imu"].to(device).eval()
imu_head  = ib.modality_heads["imu"].to(device).eval()



# ======== Small helpers ========
def warn_imu_zero(imu):
    print("STATE: warn_imu_zero - Checking if IMU data is all zeros.")
    # quick check: are all zeros?
    is_zero = np.allclose(imu, 0.0)
    if is_zero:
        print("STATE: warn_imu_zero - IMU data is all zeros.")
    return is_zero

def choose_prompt(p_social, s_arousal, hase):
    print(f"STATE: choose_prompt - Received p_social={p_social}, s_arousal={s_arousal}, hase={hase}")
    if not hase:
        prompt = "You might pause and notice your breath."
        print(f"STATE: choose_prompt - Condition not met (hase is False), returning default prompt: '{prompt}'")
        return prompt
    # you can tailor text using scores if you want
    prompt = "Try a slow breath before your next sentence."
    print(f"STATE: choose_prompt - Condition met (hase is True), returning intervention prompt: '{prompt}'")
    return prompt

# ======== Load thresholds & anchors ========
def load_threshold(fp):
    print(f"STATE: load_threshold - Loading threshold from: {fp}")
    with open(fp, "r") as f:
        threshold = float(json.load(f)["threshold"])
        print(f"STATE: load_threshold - Loaded value: {threshold}")
        return threshold

def load_arousal_anchors(path, device):
    import torch
    try:
        st = torch.load(path, map_location="cpu")
    except Exception:
        st = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(st, dict):
        for k in ("state_dict", "_orig_mod", "module"):
            if k in st and isinstance(st[k], dict):
                st = st[k]
    else:
        raise RuntimeError(f"Unsupported anchors object type: {type(st)}")

    e_high = st.get("e_high", st.get("high", None))
    e_low  = st.get("e_low",  st.get("low",  None))
    if e_high is None or e_low is None:
        raise KeyError(f"Missing 'e_high'/'e_low' in anchors file (keys: {list(st.keys())[:8]})")

    e_high = torch.as_tensor(e_high, dtype=torch.float32, device=device)
    e_low  = torch.as_tensor(e_low,  dtype=torch.float32, device=device)
    e_high = torch.nn.functional.normalize(e_high.view(1, -1), dim=-1)
    e_low  = torch.nn.functional.normalize(e_low.view(1, -1),  dim=-1)
    return e_high, e_low



def load_c_social(c_ckpt: Path, device: torch.device):
    print(f"STATE: load_c_social - Loading social classifier from: {c_ckpt} for device: {device}")
    from ConcatClassifier import ConcatClassifier
    model = ConcatClassifier(d=512, hidden=256, num_layers=2, fusion="concat", dropout=0.10).to(device)
    print("STATE: load_c_social - ConcatClassifier model initialized.")

    ckpt = torch.load(c_ckpt, map_location="cuda")
    state = ckpt.get("state_dict", ckpt)
    print(f"STATE: load_c_social - Checkpoint loaded. State dict keys: {list(state.keys())[:5]}")

    # Strip known wrappers: torch.compile => _orig_mod., DataParallel => module.
    def strip_prefixes(k: str):
        for p in ("_orig_mod.", "module."):
            if k.startswith(p):
                return k[len(p):]
        return k

    fixed = {strip_prefixes(k): (v.to(device) if torch.is_tensor(v) else v) for k, v in state.items()}
    print(f"STATE: load_c_social - Prefixes stripped from keys. Example fixed keys: {list(fixed.keys())[:5]}")

    # Try strict load; if it fails, print diagnostics and re-raise.
    try:
        missing, unexpected = model.load_state_dict(fixed, strict=False)
        print("STATE: load_c_social - load_state_dict (strict=False) completed.")
        if missing:
            print("[load_c_social] Missing keys:", missing)
            print("STATE: load_c_social - WARNING: Missing keys in state_dict:", missing)
        if unexpected:
            print("[load_c_social] Unexpected keys:", unexpected)
            print("STATE: load_c_social - WARNING: Unexpected keys in state_dict:", unexpected)
        # Re-run strict check if you prefer hard failure:
        model.load_state_dict(fixed, strict=True)
        print("STATE: load_c_social - load_state_dict (strict=True) successful.")
    except RuntimeError as e:
        print("[load_c_social] load_state_dict failed:", e)
        # Helpful: show a few sample keys
        print("example saved keys:", list(state.keys())[:5])
        print("example fixed  keys:", list(fixed.keys())[:5])
        raise

    model.eval()
    print("STATE: load_c_social - Model set to eval mode. Load complete.")
    return model
def resize_pos_(pre, new_T=800, k=8):
    L_new = new_T // k
    pe = pre.pos_embed
    ncls = pre.num_cls_tokens
    cls = pe[:, :ncls, :]
    pos = pe[:, ncls:, :].transpose(1,2)
    pos = F.interpolate(pos, size=L_new, mode="linear", align_corners=False)
    pre.pos_embed = torch.nn.Parameter(torch.cat([cls, pos.transpose(1,2)], dim=1))
    print(f"STATE: ib_imu_e512 - Resized positional embeddings to handle T={new_T}.")


@torch.inference_mode()
def ib_imu_e512(X_imu6, device):
    """
    IMU -> ImageBind IMU pre+trunk -> CLS -> L2 -> [N,512]
    Assumes X_imu6 is [N,800,6] float32, time-major.
    """
    print(f"STATE: ib_imu_e512 - Starting inference. Input shape: {X_imu6.shape}")

    imu_pre   = ib.modality_preprocessors["imu"].to(device).eval()
    imu_trunk = ib.modality_trunks["imu"].to(device).eval()
    print("STATE: ib_imu_e512 - ImageBind models loaded.")


    resize_pos_(imu_pre, 800, 8)

    x = torch.tensor(X_imu6, dtype=torch.float32, device=device).permute(0,2,1)  # [N,6,800]
    print(f"STATE: ib_imu_e512 - Converted numpy input to tensor. Shape: {x.shape}")
    toks = imu_pre(x)["trunk"]["tokens"]
    out  = imu_trunk(toks)
    if isinstance(out, dict):
        for v in out.values():
            if torch.is_tensor(v): out = v; break
    e = out[:,0,:]
    e = torch.nn.functional.normalize(e, dim=-1)
    if e.shape[-1] != 512:
        print(f"STATE: ib_imu_e512 - WARNING: Trunk output dim is {e.shape[-1]}, slicing to 512.")
        e = e[:, :512]
    print(f"STATE: ib_imu_e512 - Inference complete. Final embedding shape: {e.shape}")
    return e


@torch.inference_mode()
def imu_to_shared_embedding(X_imu6):  # X_imu6: [N,800,6] float32 numpy
    x = torch.tensor(X_imu6, dtype=torch.float32, device=device).permute(0,2,1)  # [B,6,800]
    toks = imu_pre(x)["trunk"]["tokens"]     # tokens for trunk
    out  = imu_trunk(toks)                   # expect [B, L, D] or dict -> tensor

    if isinstance(out, dict):
        # pick first tensor if trunk returns a dict
        for v in out.values():
            if torch.is_tensor(v):
                out = v
                break
    # NOTE: do NOT slice to CLS here; the head wants a sequence [B, L, D]
    z = imu_head(out)                        # ✅ feed [B, L, D] to head

    if isinstance(z, dict):
        for v in z.values():
            if torch.is_tensor(v):
                z = v
                break

    # L2-normalize in the shared space
    z = torch.nn.functional.normalize(z, dim=-1)  # [B, D_shared] (head usually collapses L)
    return z

@torch.inference_mode()
def score_arousal(e_shared, e_high, e_low):
    if e_shared.ndim == 1:
        e_shared = e_shared.view(1, -1)
    e_shared = torch.nn.functional.normalize(e_shared, dim=-1)
    s_high = (e_shared @ e_high.t()).squeeze(-1)
    s_low  = (e_shared @ e_low.t()).squeeze(-1)
    return (s_high - s_low)  # tensor [N]


@torch.inference_mode()
def c_social_prob(model, e_imu512, X_mfcc, device):
    """
    model: ConcatClassifier
    e_imu512: torch [1,512]
    X_mfcc:   numpy [430,13] already CMVN and resampled on device
    """
    print(f"STATE: c_social_prob - Starting inference. IMU emb shape: {e_imu512.shape}, MFCC shape: {X_mfcc.shape}")
    e = e_imu512.to(device)
    m = torch.from_numpy(X_mfcc[None, ...]).to(device).float()  # [1,430,13]
    print(f"STATE: c_social_prob - Inputs converted to tensors. e shape: {e.shape}, m shape: {m.shape}")
    logits, _ = model(e, m)
    p = torch.sigmoid(logits)
    result = float(p.item())
    print(f"STATE: c_social_prob - Inference complete. Logits: {logits.item():.4f}, Probability: {result:.4f}")
    return result

@torch.inference_mode()
def s_arousal(e_shared, e_high, e_low):
    print(f"STATE: s_arousal - Calculating score. Shared emb shape: {e_shared.shape}")
    ei = torch.nn.functional.normalize(e_shared.view(1,-1), dim=-1)
    score = float((ei @ e_high.t() - ei @ e_low.t()).item())
    print(f"STATE: s_arousal - Calculation complete. Score: {score:.4f}")
    return score

# ======== Server handler ========
def make_handler(device, model, theta_social, theta_arousal, e_high, e_low,session: aiohttp.ClientSession):
    print("STATE: make_handler - Creating websocket handler.")
    async def handler(ws):
        print(f"STATE: handler - Client connected from {getattr(ws, 'remote_address', 'unknown')}")
        log_event("client_connected", remote=str(getattr(ws, "remote_address", None)))

        try:
            async for msg in ws:
                print(f"\nSTATE: handler - Message received. Type: {type(msg)}, Size: {len(msg)} bytes. ")
                if not isinstance(msg, (bytes, bytearray)) or len(msg) < HDR_SIZE:
                    print(f"STATE: handler - Skipping invalid message (not bytes or too small).")
                    continue

                (magic, ver, flags, _res, t0,
                 imu_hz, imu_n, imu_c, imu_cnt,
                 mfcc_n, mfcc_c, mfcc_cnt) = struct.unpack(HDR_FMT, msg[:HDR_SIZE])
                print(f"STATE: handler - Header unpacked: magic={hex(magic)}, ver={ver}, t0={t0:.2f}, imu_shape=({imu_n},{imu_c}), mfcc_shape=({mfcc_n},{mfcc_c})")

                if magic != MAGIC or ver != 1:
                    print(f"STATE: handler - Bad header received (magic={hex(magic)}, ver={ver}). Skipping.")
                    continue

                off = HDR_SIZE
                imu  = np.frombuffer(msg, dtype="<f4", count=imu_cnt,  offset=off).reshape(imu_n,  imu_c).copy()
                off += imu_cnt * 4
                mfcc = np.frombuffer(msg, dtype="<f4", count=mfcc_cnt, offset=off).reshape(mfcc_n, mfcc_c).copy()
                print(f"STATE: handler - Data parsed. IMU numpy shape: {imu.shape}, MFCC numpy shape: {mfcc.shape}")
                log_event("window_rx", t0=t0, imu_shape=imu.shape, mfcc_shape=mfcc.shape, imu_hz=imu_hz)

                # IMU: accel (cols 0–2), gyro (cols 3–5)
                accel = imu[:, 0:3]                 # m/s^2
                gyro  = imu[:, 3:6]                 # rad/s

                acc_mean = accel.mean(axis=0)       # [ax, ay, az]
                acc_std  = accel.std(axis=0)        # std per axis
                gyr_mean = gyro.mean(axis=0)        # [gx, gy, gz]
                gyr_std  = gyro.std(axis=0)

                print(
                    f"   accel mean/std = "
                    f"{acc_mean.mean():.3f}/{acc_std.mean():.3f}  "  # overall means
                    f"(per-axis μ=[{acc_mean[0]:.3f},{acc_mean[1]:.3f},{acc_mean[2]:.3f}] "
                    f"σ=[{acc_std[0]:.3f},{acc_std[1]:.3f},{acc_std[2]:.3f}])"
                )
                print(
                    f"   gyro  mean/std = "
                    f"{gyr_mean.mean():.3f}/{gyr_std.mean():.3f}  "
                    f"(per-axis μ=[{gyr_mean[0]:.3f},{gyr_mean[1]:.3f},{gyr_mean[2]:.3f}] "
                    f"σ=[{gyr_std[0]:.3f},{gyr_std[1]:.3f},{gyr_std[2]:.3f}])"
                )

                # MFCC: quick summary across time (frames) for each coefficient
                mfcc_mean = mfcc.mean(axis=0)       # [13]
                mfcc_std  = mfcc.std(axis=0)        # [13]
                mfcc_all_zero = np.allclose(mfcc, 0.0)

                print(
                    f"   MFCC all-zeros? {'yes' if mfcc_all_zero else 'no'}  "
                    f"mean/std (avg over 13 dims) = {mfcc_mean.mean():.4f}/{mfcc_std.mean():.4f}"
                )
                # quick sanity
                if warn_imu_zero(imu):
                    print("! IMU window appears all zeros")
                print(f"[SERVER RX] from={getattr(ws, 'remote_address', None)} "
                      f"t0={t0:.0f}  IMU={imu.shape}@{imu_hz}Hz  MFCC={mfcc.shape}  bytes={len(msg)}",
                      flush=True)
                # ---- Inference per window ----
                print("STATE: handler - Starting inference pipeline...")
                # IMU -> embeddings
                e512   = ib_imu_e512(imu[None, ...],device)                   # torch [1,512]  (for C_social)
                e_share= imu_to_shared_embedding(imu[None, ...])       # torch [1,D]    (for arousal)
                log_event("embeddings_done")

                # C_social
                p_soc  = c_social_prob(model, e512, mfcc, device=device)

                # S_arousal
                s_ar   = float(score_arousal(e_share.squeeze(0), e_high, e_low).item())
                theta_social_active, theta_arousal_active = current_thresholds(theta_social, theta_arousal)
                hase = (p_soc >= theta_social_active) and (s_ar >= theta_arousal_active)

                #hase   = (p_soc >= theta_social) and (s_ar >= theta_arousal)

                hint = f"Scores: p_social={p_soc:.3f}, s_arousal={s_ar:.3f}. User is speaking."
                if hase:
                    txt = await gen_prompt(session, hint)    # LLM-generated cue
                else:
                    # optionally still produce a gentle, neutral non-intervention line
                    txt = "Consider pausing briefly to gather your thoughts."
                print("STATE: handler - Prompt chosen.")

                print(f"t0={t0:.0f}  p_social={p_soc:.3f}  s_arousal={s_ar:.3f}  hase={hase}")
                log_event("scores", p_social=p_soc, s_arousal=s_ar)
                response_payload = {
                    "type":"prompt",
                    "text": txt,
                    "hase_window": bool(hase),
                    "scores": {"p_social": p_soc, "s_arousal": s_ar, "t0": t0,
                               "theta_social": theta_social_active, "theta_arousal": theta_arousal_active}
                }
                print(f"STATE: handler - Sending response payload: {response_payload}")
                await ws.send(json.dumps(response_payload))
                print("STATE: handler - Response sent. Waiting for next message.")
                log_event("decision", hase=bool(hase), text=txt)


        except websockets.ConnectionClosedOK:
            print("STATE: handler - Client closed cleanly")
        except websockets.ConnectionClosedError as e:
            print(f"STATE: handler - Client closed with error: {e}")
        except Exception:
            print("STATE: handler - Exception traceback:", flush=True)
            traceback.print_exc()
        finally:
            log_event("client_disconnected", remote=str(getattr(ws, "remote_address", None)))

            print(f"STATE: handler - Client disconnected from {getattr(ws, 'remote_address', 'unknown')}")
    return handler

async def main():
    print("STATE: main - Starting main execution.")
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"STATE: main - Using device: {device}")
    
    # load artifacts once
    print("STATE: main - Loading artifacts...")
    theta_social  = load_threshold(THRESH_SOCIAL_JSON)
    theta_arousal = load_threshold(AROUSAL_THR_JSON)
    model = load_c_social(C_SOCIAL_PT, device)
    e_high, e_low = load_arousal_anchors(AROUSAL_ANCHORS_PT, device)
    print("STATE: main - All artifacts loaded successfully.")
    print(f"θ_social={theta_social:.4f}  θ_arousal={theta_arousal:.4f}")
    log_event("models_loaded", theta_social=theta_social, theta_arousal=theta_arousal)

    session = aiohttp.ClientSession() 
    # start server (keepalive off so Unity doesn't time out)
    handler = make_handler(device, model, theta_social, theta_arousal, e_high, e_low, session)
    print("STATE: main - Server handler created.")
    print("Listening on ws://0.0.0.0:8765")
    log_event("server_listening", url="ws://0.0.0.0:8765")
    async with websockets.serve(handler, "0.0.0.0", 8765, ping_interval=None, ping_timeout=None, max_size=2**27):
        print("STATE: main - Websocket server started. Awaiting connections...")
        await asyncio.Future() # run forever

if __name__ == "__main__":
    print("STATE: __main__ - Script started.")
    import websockets
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSTATE: __main__ - Server stopped by user (KeyboardInterrupt).")