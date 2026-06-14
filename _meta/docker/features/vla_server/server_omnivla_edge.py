import os
import sys
from collections import deque
from io import BytesIO
import numpy as np
import torch
import clip
from PIL import Image

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.environ.get(
    "VLA_MODEL_DIR",
    os.path.join(_SCRIPT_DIR, "model"),
)
sys.path.insert(0, _SCRIPT_DIR)        # enables: from model.utils_policy import ...
sys.path.insert(0, _MODEL_DIR)         # enables: from model_omnivla_edge import ... inside utils_policy.py
from model.utils_policy import load_model, transform_images_PIL_mask, transform_images_map

_WEIGHTS_DEFAULT = os.path.join(_MODEL_DIR, "omnivla-edge.pth")

_IMGSIZE = (96, 96) #for efficientnet encoder    
_IMGSIZE_CLIP = (224, 224) #for clip encoder

_MODALITY_ID = 7 #lang instruction input
_METRIC_WP_SPACING = 0.1 #scaling the waypoint output to meter?

# load model parameters
_MODEL_PARAMS = {}
_MODEL_PARAMS["model_type"] = "omnivla-edge"    
_MODEL_PARAMS["len_traj_pred"] = 8
_MODEL_PARAMS["learn_angle"] = True
_MODEL_PARAMS["context_size"] = 5
_MODEL_PARAMS["obs_encoder"] = "efficientnet-b0"
_MODEL_PARAMS["encoding_size"] = 256
_MODEL_PARAMS["obs_encoding_size"] = 1024   
_MODEL_PARAMS["goal_encoding_size"] = 1024   
_MODEL_PARAMS["late_fusion"] = False         
_MODEL_PARAMS["mha_num_attention_heads"] = 4   
_MODEL_PARAMS["mha_num_attention_layers"] = 4   
_MODEL_PARAMS["mha_ff_dim_factor"] = 4 
_MODEL_PARAMS["clip_type"] = "ViT-B/32"

# per-robot observation history, keyed by session id (robot namespace). context_size+1 frames.
_CONTEXT_LEN = _MODEL_PARAMS["context_size"] + 1
_context: dict[str, deque] = {}

app = FastAPI()


def _load()->None:
    global device, model, text_encoder, mask_360_pil_96, mask_360_pil_224
    weights_path = os.environ.get("VLA_WEIGHTS", _WEIGHTS_DEFAULT)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[vla_server] loading {weights_path} on {device}", flush=True)
    model, text_encoder, _ = load_model(
        weights_path,
        _MODEL_PARAMS,
        device,
    )
    text_encoder = text_encoder.to(device).eval()
    model = model.to(device).eval()

    mask_360_pil_96 = np.ones((96, 96, 3), dtype=np.float32)
    mask_360_pil_224 = np.ones((224, 224, 3), dtype=np.float32)
    print("[vla_server] ready", flush=True)
    
@app.on_event("startup")
def on_startup()->None:
    _load()

@app.get("/health")
def health()->JSONResponse:
    return JSONResponse({"status":"ok"})

@app.post("/reset")
async def reset(session: str=Form(...))->JSONResponse:
    _context.pop(session, None)
    return JSONResponse({"status":"ok"})

@app.post("/act")
async def act(
    image: UploadFile=File(...),
    instruction: str=Form(...),
    session: str=Form(...)
)->JSONResponse:
    img=await image.read()

    # Load current image
    current_image_PIL = Image.open(BytesIO(img)).convert("RGB")

    current_image_PIL_96 = current_image_PIL.resize(_IMGSIZE)
    current_image_PIL_224 = current_image_PIL.resize(_IMGSIZE_CLIP)

    #-----omnivla-edge inference code reuse-------
    # rolling per-robot observation history, pad with the oldest frame until the window fills.
    history = _context.setdefault(session, deque(maxlen=_CONTEXT_LEN))
    history.append(current_image_PIL_96)
    context_queue = list(history)
    if len(context_queue) < _CONTEXT_LEN:
        context_queue = [context_queue[0]] * (_CONTEXT_LEN - len(context_queue)) + context_queue
    #obs_images = transform_images_PIL(context_queue)
    obs_images = transform_images_PIL_mask(context_queue, mask_360_pil_96)        
    obs_images = torch.split(obs_images.to(device), 3, dim=1)
    obs_image_cur = obs_images[-1].to(device) 
    obs_images = torch.cat(obs_images, dim=1).to(device)     

    #cur_large_img = transform_images_PIL(current_image_PIL_224).to(device)                 
    cur_large_img = transform_images_PIL_mask(current_image_PIL_224, mask_360_pil_224).to(device) 
        
    #Dummy satellite image
    satellite_cur = Image.new("RGB", (352, 352), color=(0, 0, 0)) 
    satellite_goal = Image.new("RGB", (352, 352), color=(0, 0, 0))         
    current_map_image = transform_images_map(satellite_cur)
    goal_map_image = transform_images_map(satellite_goal)
    map_images = torch.cat((current_map_image.to(device), goal_map_image.to(device), obs_image_cur), axis=1)  
    
    # Egocentric goal image
    dummy_goal = Image.new("RGB", _IMGSIZE, color=(0, 0, 0))
    goal_image = transform_images_PIL_mask(dummy_goal, mask_360_pil_96).to(device)
    goal_pose_torch = torch.zeros(1, 4, dtype=torch.float32, device=device)


    # Language instruction
    obj_inst_lan = clip.tokenize(instruction, truncate=True).to(device) 
    modality_id = torch.tensor([_MODALITY_ID]).to(device)
    
    with torch.no_grad():
        feat_text_lan = text_encoder.encode_text(obj_inst_lan)
        predicted_actions, distances, mask_number = model(
            obs_images,
            goal_pose_torch,
            map_images,
            goal_image,
            modality_id,
            feat_text_lan,
            cur_large_img,
        )


    actions = predicted_actions.float().cpu().numpy()[0]
    actions[:, :2] *= _METRIC_WP_SPACING
    steps = [
        {"x": float(a[0]), "y": float(a[1]), "yaw": float(np.arctan2(a[3], a[2]))}
        for a in actions
    ]

    return JSONResponse({"actions": {"mobile": {"waypoints": steps}}})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OmniVLA-edge inference server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    if args.weights:
        os.environ["VLA_WEIGHTS"] = args.weights
    uvicorn.run(app, host=args.host, port=args.port)