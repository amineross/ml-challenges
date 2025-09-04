import cv2
import torch
import numpy as np
import argparse

from ESPCN import ESPCN
from FSRCNN import FSRCNN

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = select_device()
try:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def load_upscaler_from_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model_class = ckpt["model_class"]
    if model_class == "ESPCN":
        model = ESPCN()
    elif model_class == "FSRCNN":
        model = FSRCNN()
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    return model

def bgr_to_tensor_rgb01(frame_bgr, device, use_fp16):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    t = t.float() / 255.0
    if use_fp16 and device.type != "cpu":
        t = t.half()
    return t

def tensor_rgb01_to_bgr(out):
    out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    out = (out * 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

@torch.inference_mode()
def upscale_frame(model, frame_bgr, device, use_fp16):
    if use_fp16 and device.type != "cpu":
        model = model.half()
    else:
        model = model.float()

    x = bgr_to_tensor_rgb01(frame_bgr, device, use_fp16=use_fp16)
    y = model(x)
    return tensor_rgb01_to_bgr(y)

def open_camera(index, width, height, fps=30):
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, 1)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    except Exception:
        pass
    return cap

def run_realtime(checkpoint_path, cam_index=0, width=640, height=480, target_fps=30):
    device = DEVICE
    model = load_upscaler_from_checkpoint(checkpoint_path, device)
    cap = open_camera(cam_index, width, height, target_fps)

    ok, frame = cap.read()
    is_device_cpu = device.type == "cpu"
    if ok:
        for _ in range(3):
            _ = upscale_frame(model, frame, device, use_fp16=(not is_device_cpu))
    
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        up = upscale_frame(model, frame, device, use_fp16=(not is_device_cpu))
        cv2.imshow("upscaled", up)
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cam-index", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--target-fps", type=int, default=30)
    args = p.parse_args()

    run_realtime(
        checkpoint_path=args.checkpoint,
        cam_index=args.cam_index,
        width=args.width,
        height=args.height,
        target_fps=args.target_fps
    )