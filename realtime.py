import sys
import cv2
import torch
import numpy as np
import argparse

from ESPCN import ESPCN
from FSRCNN import FSRCNN

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = select_device()
USE_FP16 = (DEVICE.type == "cuda")  # important: not on mps/cpu
print(f"Device: {DEVICE}, use_fp16: {USE_FP16}")

try:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def load_upscaler_from_checkpoint(checkpoint_path: str, device: torch.device, use_fp16: bool):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_class = ckpt["model_class"]
    if model_class == "ESPCN":
        model = ESPCN(scale=ckpt.get("scale", 4))
    elif model_class == "FSRCNN":
        model = FSRCNN()
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    model = model.half() if use_fp16 else model.float()
    return model

def bgr_to_tensor_rgb01(frame_bgr: np.ndarray, device: torch.device, use_fp16: bool) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    t = t.float().div_(255.0)
    if use_fp16:
        t = t.half()
    return t

def tensor01rgb_to_bgr8(out: torch.Tensor) -> np.ndarray:
    out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    out = (out * 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

@torch.inference_mode()
def upscale_frame(frame_bgr: np.ndarray, model: torch.nn.Module, device: torch.device, use_fp16: bool) -> np.ndarray:
    x = bgr_to_tensor_rgb01(frame_bgr, device, use_fp16=use_fp16)
    y = model(x)
    return tensor01rgb_to_bgr8(y)

def open_camera(index: int, width: int, height: int, fps: int):
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Éviter BUFFERSIZE/FOURCC sur macOS (souvent non supportés)
    if sys.platform != "darwin":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

    print(f"Camera opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {cap.get(cv2.CAP_PROP_FPS)}fps")
    return cap

def create_side_by_side(original: np.ndarray, upscaled: np.ndarray) -> np.ndarray:
    # Redimensionner upscaled à la même hauteur que l'original pour comparaison
    h, w = original.shape[:2]
    up_h, up_w = upscaled.shape[:2]
    
    # Garder le ratio de l'upscaled, mais ajuster la hauteur
    ratio = h / up_h
    new_up_w = int(up_w * ratio)
    upscaled_resized = cv2.resize(upscaled, (new_up_w, h), interpolation=cv2.INTER_LINEAR)
    
    # Concatener horizontalement : Original | Upscaled
    comparison = np.hstack([original, upscaled_resized])
    
    # Ajouter des labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "SR Upscaled", (w + 10, 30), font, 1, (255, 0, 0), 2)
    
    return comparison

def run_realtime(checkpoint_path: str, cam_index: int = 0, width: int = 1280, height: int = 720, target_fps: int = 30):
    model = load_upscaler_from_checkpoint(checkpoint_path, DEVICE, USE_FP16)
    cap = open_camera(cam_index, width, height, target_fps)

    cv2.namedWindow("SR Comparison", cv2.WINDOW_AUTOSIZE)

    # Warm-up si possible
    ok, frame = cap.read()
    if ok:
        for _ in range(3):
            _ = upscale_frame(frame, model, DEVICE, USE_FP16)

    failed_reads = 0
    while True:
        # si la fenêtre est fermée par l'utilisateur → sortir proprement
        if cv2.getWindowProperty("SR Comparison", cv2.WND_PROP_VISIBLE) < 1:
            break

        ok, frame = cap.read()
        if not ok:
            failed_reads += 1
            if failed_reads > 30:  # 1s à ~30fps → abandonner proprement
                print("Camera read failed repeatedly; exiting.")
                break
            continue
        failed_reads = 0

        try:
            # Original frame
            original = frame.copy()
            
            # SR upscale
            upscaled = upscale_frame(frame, model, DEVICE, USE_FP16)
            
            # Side-by-side comparison
            comparison = create_side_by_side(original, upscaled)
            
        except Exception as e:
            print(f"Inference error: {e}")
            break

        cv2.imshow("SR Comparison", comparison)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC ou q
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cam-index", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--target-fps", type=int, default=30)
    args = p.parse_args()

    run_realtime(
        checkpoint_path=args.checkpoint,
        cam_index=args.cam_index,
        width=args.width,
        height=args.height,
        target_fps=args.target_fps
    )