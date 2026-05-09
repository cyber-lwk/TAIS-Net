from pipeline.stablevsr_pipeline import StableVSRPipeline               
from diffusers import DDPMScheduler, ControlNetModel
from accelerate.utils import set_seed
from PIL import Image
import os
import argparse
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from pathlib import Path
import torch
import ssl
import ssl

# ssl._create_default_https_context = ssl._create_unverified_context()


def center_crop(im, size=128):
    width, height = im.size   # Get dimensions
    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2
    return im.crop((left, top, right, bottom))

# get arguments
parser = argparse.ArgumentParser(description="Test code for StableVSR.")
parser.add_argument("--out_path", default='./StableVSR_results/', type=str, help="Path to output folder.")
parser.add_argument("--in_path", type=str, required=False, help="Path to input folder (containing sets of LR images).")
parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of sampling steps")
parser.add_argument("--controlnet_ckpt", type=str, default=None, help="Path to your folder with the controlnet checkpoint.")
args = parser.parse_args()

print("Run with arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")

# set parameters
set_seed(42)
device = torch.device('cuda')
model_id = 'checkpoints/StableVSR'
# args.controlnet_ckpt = None
# 加载 controlnet 的模型
controlnet_model = ControlNetModel.from_pretrained(args.controlnet_ckpt if args.controlnet_ckpt is not None else model_id, subfolder='controlnet') # your own controlnet model
# 加载 stablevsr 的流水线模型
pipeline = StableVSRPipeline.from_pretrained(model_id, controlnet=controlnet_model)
# pipeline = StableVSRPipeline.from_pretrained(model_id)
# 加载 scheduler 调度器
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')
pipeline.scheduler = scheduler
pipeline = pipeline.to(device)
pipeline.enable_xformers_memory_efficient_attention()
#加载光流估计模型 模型参数为默认参数
of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
of_model.requires_grad_(False)
of_model = of_model.to(device)

# iterate for every video sequence in the input folder
seqs = sorted(os.listdir(args.in_path))                                     #读取需要评估的视频序列目录
for seq in seqs:
    print(f"Processing sequence {seq}")
    frame_names = sorted(os.listdir(os.path.join(args.in_path, seq)))       #当前视频序列帧文件名
    frames = []
    for frame_name in frame_names:
        frame = Path(os.path.join(args.in_path, seq, frame_name))
        frame = Image.open(frame)
        frame = center_crop(frame,size=64)
        frames.append(frame)                                                #读取所有的视频帧数据

    # upscale frames using StableVSR
    frames = pipeline('', frames, num_inference_steps=args.num_inference_steps, guidance_scale=0, of_model=of_model).images
    frames = [frame[0] for frame in frames]
    
    # save upscaled sequences
    # 保存超分处理之后的结果
    seq = Path(seq)
    target_path = os.path.join(args.out_path, seq.parent.name, seq.name)
    os.makedirs(target_path, exist_ok=True)
    for frame, name in zip(frames, frame_names):
        frame.save(os.path.join(target_path, name))
