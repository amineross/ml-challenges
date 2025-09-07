import sys
import cv2
import torch
import numpy as np
import argparse

from ESPCN import ESPCN
from FSRCNN import FSRCNN
from EDSRLITE import EDSRLITE

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = select_device()
USE_FP16 = (DEVICE.type == "cuda")  # important: pas sur mps/cpu
print(f"périphérique: {DEVICE}, use_fp16: {USE_FP16}")

try:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def load_upscaler_from_checkpoint(checkpoint_path: str, device: torch.device, use_fp16: bool):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_class = ckpt["model_class"]
    scale = 4
    
    if model_class == "ESPCN":
        model = ESPCN(scale=scale)
    elif model_class == "FSRCNN":
        model = FSRCNN()
    elif model_class == "EDSRLITE":
        model = EDSRLITE()
    else:
        raise ValueError(f"classe de modèle inconnue: {model_class}")

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    model = model.half() if use_fp16 else model.float()
    return model, scale


def bgr_to_tensor_rgb01(frame_bgr: np.ndarray, device: torch.device, use_fp16: bool) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # éviter torch.from_numpy sous numpy 2
    t = torch.utils.dlpack.from_dlpack(rgb.__dlpack__())  # [H,W,3], uint8, cpu
    t = t.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    dtype = torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32
    t = t.to(device=device, dtype=dtype)
    t = t.div(255.0)
    return t


def tensor01rgb_to_bgr8(out: torch.Tensor) -> np.ndarray:
    # convertir [1,3,H,W] float(0..1) vers uint8 hwc sur cpu sans .numpy()
    x = (
        out.clamp(0, 1)
        .mul(255.0)
        .to(torch.uint8)
        .squeeze(0)
        .permute(1, 2, 0)
        .contiguous()
        .to("cpu")
    )
    rgb8 = np.from_dlpack(x)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)


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
        raise RuntimeError("impossible d'ouvrir la caméra")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # éviter buffersize/fourcc sur macos (souvent non supportés)
    if sys.platform != "darwin":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

    print(f"caméra ouverte: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {cap.get(cv2.CAP_PROP_FPS)}fps")
    return cap


def downscale_area(img: np.ndarray, scale: int) -> np.ndarray:
    # area c'est mieux pour réduire la résolution (moins d'aliasing que bicubique)
    h, w = img.shape[:2]
    new_h, new_w = h // scale, w // scale
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_side_by_side_hr(left_hr: np.ndarray, right_hr: np.ndarray, display_height: int = 480) -> np.ndarray:
    # left_hr: remontée bicubique depuis basse-res; right_hr: remontée modèle
    h_hr, w_hr = right_hr.shape[:2]

    # calculer les dimensions d'affichage avec le même ratio
    display_ratio = display_height / h_hr
    display_width = int(w_hr * display_ratio)

    left_display = cv2.resize(left_hr, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
    right_display = cv2.resize(right_hr, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

    comparison = np.hstack([left_display, right_display])

    # labels plus clairs
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 if display_height < 500 else 1.0
    thickness = 1 if display_height < 500 else 2
    cv2.putText(comparison, "baseline: reduit puis bicubique vers taille camera", (10, 25), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(comparison, "modele: reduit puis neural vers taille camera", (display_width + 10, 25), font, font_scale, (255, 0, 0), thickness)

    return comparison


def run_realtime(checkpoint_path: str, cam_index: int = 0, width: int = 1280, height: int = 720, target_fps: int = 30, display_height: int = 480, downscale: int | None = None):
    model, model_scale = load_upscaler_from_checkpoint(checkpoint_path, DEVICE, USE_FP16)
    # par défaut utiliser model_scale sauf override utilisateur
    forced_downscale = downscale or model_scale
    print(f"facteur d'échelle du modèle: {model_scale}x")
    print(f"facteur de simulation basse-res: {forced_downscale}x")
    if forced_downscale != model_scale:
        print(f"attention: facteur downscale {forced_downscale}x différent de l'échelle modèle {model_scale}x; qualité peut se dégrader")

    cap = open_camera(cam_index, width, height, target_fps)

    cv2.namedWindow("comparaison sr", cv2.WINDOW_AUTOSIZE)

    # préchauffage quelques passes
    ok, frame = cap.read()
    if ok:
        lr = downscale_area(frame, forced_downscale)
        for _ in range(3):
            _ = upscale_frame(lr, model, DEVICE, USE_FP16)

    failed_reads = 0
    while True:
        if cv2.getWindowProperty("comparaison sr", cv2.WND_PROP_VISIBLE) < 1:
            break

        ok, frame = cap.read()
        if not ok:
            failed_reads += 1
            if failed_reads > 30:
                print("lecture caméra a échoué plusieurs fois; on sort.")
                break
            continue
        failed_reads = 0

        try:
            # 1) simuler entrée basse-res depuis frame caméra
            lr = downscale_area(frame, forced_downscale)

            # 2) baseline: remontée bicubique vers résolution capture
            h_cap, w_cap = frame.shape[:2]
            left_hr = cv2.resize(lr, (w_cap, h_cap), interpolation=cv2.INTER_CUBIC)
            
            # 3) remontée modèle depuis lr vers hr
            right_hr = upscale_frame(lr, model, DEVICE, USE_FP16)
            if right_hr.shape[:2] != (h_cap, w_cap):
                # utiliser cubique quand on redimensionne la sortie modèle vers résolution capture
                right_hr = cv2.resize(right_hr, (w_cap, h_cap), interpolation=cv2.INTER_CUBIC)

            # 4) affichage côte à côte à taille contrôlée
            comparison = create_side_by_side_hr(left_hr, right_hr, display_height)
            
        except Exception as e:
            print(f"erreur inférence: {e}")
            break

        cv2.imshow("comparaison sr", comparison)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
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
    p.add_argument("--display-height", type=int, default=480, help="hauteur fenêtre d'affichage (garde traitement à résolution complète)")
    p.add_argument("--downscale", type=int, default=None, help="facteur pour créer entrée basse-res depuis caméra; par défaut échelle du modèle")
    args = p.parse_args()

    run_realtime(
        checkpoint_path=args.checkpoint,
        cam_index=args.cam_index,
        width=args.width,
        height=args.height,
        target_fps=args.target_fps,
        display_height=args.display_height,
        downscale=args.downscale
    )