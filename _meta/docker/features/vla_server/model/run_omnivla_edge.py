# ===============================================================
# OmniVLA edge Inference
# ===============================================================
# 
# Sample inference code for OmniVLA edge
# if you want to control the robot, you need to update the current state such as pose and image in "run_omnivla_edge" and comment out "break" in "run".
#
# ---------------------------
# Paths and System Setup
# ---------------------------
import sys, os
sys.path.insert(0, '..')

import time, math, json
from typing import Optional, Tuple, Type, Dict
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utm
import argparse
import yaml
import clip

from utils_policy import transform_images_map, load_model, transform_images_PIL, transform_images_PIL_mask

# ===============================================================
# Utility Functions
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")

def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: "InferenceConfig",
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return module

# ===============================================================
# Inference Class
# ===============================================================
class Inference:
    def __init__(self, save_dir, lan_inst_prompt, goal_utm, goal_compass, goal_image_PIL):
        self.tick_rate = 3
        self.lan_inst_prompt = lan_inst_prompt
        self.goal_utm = goal_utm
        self.goal_compass = goal_compass
        self.goal_image_PIL = goal_image_PIL
        self.count_id = 0
        self.linear, self.angular = 0.0, 0.0
        self.datastore_path_image = save_dir
    # ----------------------------
    # Static Utility Methods
    # ----------------------------
    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    # ----------------------------
    # Main Loop
    # ----------------------------
    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.time()
        while True:
            if time.time() - start_time > loop_time:
                self.tick()
                start_time = time.time()
                break

    def tick(self):
        self.linear, self.angular = self.run_omnivla()

    # ----------------------------
    # OmniVLA Inference
    # ----------------------------
    def run_omnivla(self):
        thres_dist = 30.0
        metric_waypoint_spacing = 0.1

        # Load current GPS & heading
        current_lat = 37.87371258374039
        current_lon = -122.26729417226024
        current_compass = 270.0
        cur_utm = utm.from_latlon(current_lat, current_lon)
        cur_compass = -float(current_compass) / 180.0 * math.pi  # inverted compass

        # Local goal position
        delta_x, delta_y = self.calculate_relative_position(
            cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1]
        )
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
        radius = np.sqrt(relative_x**2 + relative_y**2)
        if radius > thres_dist:
            relative_x *= thres_dist / radius
            relative_y *= thres_dist / radius

        goal_pose_torch = torch.from_numpy(np.array([
            relative_y / metric_waypoint_spacing,
            -relative_x / metric_waypoint_spacing,
            np.cos(self.goal_compass - cur_compass),
            np.sin(self.goal_compass - cur_compass)
        ])).unsqueeze(0).float().to(device)
               
        # Overwriting "goal_pose_torch" to test only. If you want to use the GPS signal to calculate "goal_pose_torch", you need to comment out the following block.
        yaw_ang = -90.0
        goal_pose_torch = torch.from_numpy(np.array([
            1.0 / metric_waypoint_spacing,
            -10.0 / metric_waypoint_spacing,
            np.cos(yaw_ang/180.0*3.1415),
            np.sin(yaw_ang/180.0*3.1415)
        ])).unsqueeze(0).float().to(device)

        # Load current image
        current_image_path = "./inference/current_img.jpg"
        current_image_PIL = Image.open(current_image_path).convert("RGB")

        current_image_PIL_96 = current_image_PIL.resize(imgsize)
        current_image_PIL_224 = current_image_PIL.resize(imgsize_clip)
        
        #In this test code, we feed same images for the observation history, assuming that the robot stopped at the current location.
        context_queue = [current_image_PIL_96, current_image_PIL_96, current_image_PIL_96, current_image_PIL_96, current_image_PIL_96, current_image_PIL_96]
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
        
        # Language instruction
        lan_inst = self.lan_inst_prompt if lan_prompt else "xxxx"
        obj_inst_lan = clip.tokenize(lan_inst, truncate=True).to(device) 

        # Egocentric goal image
        #goal_image = transform_images_PIL(goal_image_PIL).to(device)        
        goal_image = transform_images_PIL_mask(goal_image_PIL, mask_360_pil_96).to(device)

        batch = {}
        batch["obs_images"] = obs_images
        batch["goal_pose_torch"] = goal_pose_torch
        batch["map_images"] = map_images
        batch["goal_image"] = goal_image  
        batch["obj_inst_lan"] = obj_inst_lan      
        batch["cur_large_img"] = cur_large_img                    

        # Run forward pass
        actions, modality_id = self.run_forward_pass(
            model=model.eval(),
            batch=batch,
            device_id=device,
            mode="train",
            idrun=self.count_id,
        )
        self.count_id += 1

        waypoints = actions.float().cpu().numpy()

        # Select waypoint
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        # PD controller
        EPS = 1e-8
        DT = 1 / 3
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT

        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)

        # Velocity limitation
        maxv, maxw = 0.3, 0.3
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        else:
            if np.abs(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if np.abs(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
                else:
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)

        # Save behavior
        self.save_robot_behavior(
            current_image_PIL, self.goal_image_PIL, goal_pose_torch[0].cpu(), waypoints[0],
            linear_vel_value_limit, angular_vel_value_limit, metric_waypoint_spacing, modality_id.cpu().numpy()
        )

        print("linear angular", linear_vel_value_limit, angular_vel_value_limit)
        return linear_vel_value_limit, angular_vel_value_limit

    # ----------------------------
    # Save Robot Behavior Visualization
    # ----------------------------
    def save_robot_behavior(self, cur_img, goal_img, goal_pose, waypoints,
                            linear_vel, angular_vel, metric_waypoint_spacing, mask_number):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2, 2)
        ax_ob = fig.add_subplot(gs[0, 0])
        ax_goal = fig.add_subplot(gs[1, 0])
        ax_graph_pos = fig.add_subplot(gs[:, 1])

        ax_ob.imshow(np.array(cur_img).astype(np.uint8))
        ax_goal.imshow(np.array(goal_img).astype(np.uint8))

        x_seq = waypoints[:, 0] #generated trajectory is on the robot coordinate. X is front and Y is left. 
        y_seq_inv = -waypoints[:, 1]           
        ax_graph_pos.plot(np.insert(y_seq_inv, 0, 0.0), np.insert(x_seq, 0, 0.0), linewidth=4.0, markersize=12, marker='o', color='blue')

        # Mask annotation
        mask_type = int(mask_number[0])
        mask_texts = [
            "satellite only", "pose and satellite", "satellite and image", "all",
            "pose only", "pose and image", "image only", "language only", "language and pose"
        ]
        if mask_type < len(mask_texts):
            ax_graph_pos.annotate(mask_texts[mask_type], xy=(1.0, 0.0), xytext=(-20, 20), fontsize=18, textcoords='offset points')

        ax_ob.set_title("Egocentric current image", fontsize=18)
        ax_goal.set_title("Egocentric goal image", fontsize=18)
        ax_graph_pos.tick_params(axis='x', labelsize=15) 
        ax_graph_pos.tick_params(axis='y', labelsize=15) 
        
        if int(mask_number[0]) == 1 or int(mask_number[0]) == 3 or int(mask_number[0]) == 4 or int(mask_number[0]) == 5 or int(mask_number[0]) == 8:
            ax_graph_pos.plot(-goal_pose[1], goal_pose[0], marker = '*', color='red', markersize=15)  
        else:                           
            ax_graph_pos.set_xlim(-3.0, 3.0)
            ax_graph_pos.set_ylim(-0.1, 10.0)
        ax_graph_pos.set_xlim(-3.0, 3.0)
        ax_graph_pos.set_ylim(-0.1, 10.0)
                        
        ax_graph_pos.set_title("Normalized generated 2D trajectories from OmniVLA", fontsize=18)
        
        save_path = os.path.join(self.datastore_path_image, f"{self.count_id}_ex_omnivla_edge.jpg")
        plt.savefig(save_path)

    # ----------------------------
    # Run Forward Pass
    # ----------------------------
    def run_forward_pass(self, model, batch, device_id, mode="vali", idrun=0) -> Tuple[torch.Tensor, Dict[str, float]]:

        #Setup masking
        if pose_goal and satellite and image_goal and not lan_prompt:
            modality_id = 3
        elif not pose_goal and satellite and not image_goal and not lan_prompt:
            modality_id = 0
        elif pose_goal and not satellite and not image_goal and not lan_prompt:
            modality_id = 4
        elif pose_goal and satellite and not image_goal and not lan_prompt:
            modality_id = 1
        elif not pose_goal and satellite and image_goal and not lan_prompt:
            modality_id = 2
        elif pose_goal and not satellite and image_goal and not lan_prompt:
            modality_id = 5            
        elif not pose_goal and not satellite and image_goal and not lan_prompt:
            modality_id = 6
        elif not pose_goal and not satellite and not image_goal and lan_prompt:
            modality_id = 7
        elif pose_goal and not satellite and not image_goal and lan_prompt:
            modality_id = 8       
        elif not pose_goal and not satellite and image_goal and lan_prompt:
            modality_id = 9                                
        modality_id_select = torch.tensor([modality_id]).to(device)     

        bimg, _, _, _ = batch["goal_image"].size()
        with torch.no_grad():  
            feat_text_lan = text_encoder.encode_text(batch["obj_inst_lan"])             
            predicted_actions, distances, mask_number = model(batch["obs_images"].repeat(bimg, 1, 1, 1), batch["goal_pose_torch"].repeat(bimg,1), batch["map_images"].repeat(bimg, 1, 1, 1), batch["goal_image"], modality_id_select.repeat(bimg), feat_text_lan.repeat(bimg, 1), batch["cur_large_img"].repeat(bimg, 1, 1, 1))                               
        print("Generated action chunk", predicted_actions)
        # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
        return predicted_actions, modality_id_select

                
# ===============================================================
# Inference Configuration
# ===============================================================
class InferenceConfig:
    resume: bool = True
    vla_path: str = "./omnivla-original"
    resume_step: Optional[int] = 120000    
    #vla_path: str = "./omnivla-finetuned-cast"    
    #resume_step: Optional[int] = 210000
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

def define_model(cfg: InferenceConfig) -> None:
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Loading OpenVLA Model `{cfg.vla_path}`")

    # GPU setup
    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPOSE_DIM: {POSE_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    
    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device_id) #            trust_remote_code=True,
    
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device_id)
    
    pose_projector = init_module(
        ProprioProjector,
        "pose_projector",
        cfg,
        device_id,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},            
    )
    
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead_idcat,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},            
            to_bf16=True,
        )            
 
    # Get number of vision patches
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()    
    NUM_PATCHES += 1 #for goal pose

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor

# ===============================================================
# Main Entry
# ===============================================================
if __name__ == "__main__":
    # select modality
    pose_goal = False
    satellite = False
    image_goal = False
    lan_prompt = True

    imgsize = (96, 96)    
    imgsize_clip = (224, 224)

    # Goal definitions
    # language prompt
    lan_inst_prompt = "blue trash bin"
    
    # GPS signal
    goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
    goal_utm = utm.from_latlon(goal_lat, goal_lon)
    goal_compass = -float(goal_compass) / 180.0 * math.pi
    
    # Egocentric goal image
    goal_image_PIL = Image.open("./inference/goal_img.jpg").convert("RGB").resize(imgsize)

    Front_foward = True

    # load model parameters
    model_params = {}
    model_params["model_type"] = "omnivla-edge"    
    model_params["len_traj_pred"] = 8
    model_params["learn_angle"] = True
    model_params["context_size"] = 5
    model_params["obs_encoder"] = "efficientnet-b0"
    model_params["encoding_size"] = 256
    model_params["obs_encoding_size"] = 1024   
    model_params["goal_encoding_size"] = 1024   
    model_params["late_fusion"] = False         
    model_params["mha_num_attention_heads"] = 4   
    model_params["mha_num_attention_layers"] = 4   
    model_params["mha_ff_dim_factor"] = 4 
    model_params["clip_type"] = "ViT-B/32"   
    
    context_size = model_params["context_size"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    MODEL_WEIGHTS_PATH = "./omnivla-edge"
    ckpth_path = MODEL_WEIGHTS_PATH + "/" + "omnivla-edge.pth"  
    if os.path.exists(ckpth_path):
        print(f"Loading model from {ckpth_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
    model, text_encoder, preprocess = load_model(
        ckpth_path,
        model_params,
        device,
    )
    text_encoder = text_encoder.to(device).eval()
    model = model.to(device).eval()

    #mask setting
    #mask_360 = np.loadtxt(open(DIR_loc + "/train/mask_360view.csv", "rb"), delimiter=",", skiprows=0)
    #Memo: Depending on your camera type, we observe the fisheye image type masking can work well for OmniVLA-edge. Following is no mask case.
    mask_360_pil_96 = np.ones((96, 96, 3), dtype=np.float32)
    mask_360_pil_224 = np.ones((224, 224, 3), dtype=np.float32)

    # Run inference
    inference = Inference(
        save_dir="./inference",
        lan_inst_prompt=lan_inst_prompt,
        goal_utm=goal_utm,
        goal_compass=goal_compass,
        goal_image_PIL=goal_image_PIL,
    )
    inference.run()
