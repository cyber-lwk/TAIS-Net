import os
import sys; print('Python %s on %s' % (sys.version, sys.platform))
workdir = os.path.abspath(os.path.join(os.path.abspath(__file__),"../../../../"))
print(f"workdir:{workdir}")
sys.path.extend([f'{workdir}', f'{workdir}/diffusers/src'])
from pipeline.stablevsr_pipeline import *
from diffusers import DDPMScheduler,DDIMScheduler
# , ControlNetModel)
# from diffusers.modules.SamplesReview import XStartReviewCrossFFT as XStartReview
from diffusers.modules.SamplesReview import XStartReviewCrossFFT as XStartReview
from diffusers.models.controlnet import ControlNetModel
from diffusers.models.autoencoder_kl import AutoencoderKLIND
from diffusers.modules.LLIFModule import IND,to_pixel_samples,make_coord
from diffusers.models.unet_2d_condition import UNet2DMultiScaleConditionModel,UNet2DConditionModel
from accelerate.utils import set_seed
from PIL import Image
import os
import argparse
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from pathlib import Path
import torch
import ssl
import ssl
from safetensors.torch import load_file
# ssl._create_default_https_context = ssl._create_unverified_context()
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import pyiqa
from DISTS_pytorch import DISTS
from torchvision.models.optical_flow import raft_large as raft
from torchvision.transforms import ToTensor, CenterCrop
from basicsr.utils.logger import get_root_logger
from tqdm import tqdm
from util.flow_utils import get_flow
def center_crop(im, size=128):
    width, height = im.size   # Get dimensions
    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2
    return im.crop((left, top, right, bottom))

# get arguments
parser = argparse.ArgumentParser(description="Test code for TAISNet.")
parser.add_argument("--out_path", default='./TAISNet_results/', type=str, help="Path to output folder.")
parser.add_argument("--in_path", type=str,default="data/UrbanDroneVSR/val/BIx4/", required=False, help="Path to input folder (containing sets of LR images).")
parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of sampling steps")
parser.add_argument("--num_lamda_steps", type=int, default=20, help="Number of lamda steps")
parser.add_argument("--controlnet_ckpt", type=str, default=None, help="Path to your folder with the controlnet checkpoint.")
parser.add_argument("--unet_ckpt", type=str, default=None, help="Path to your folder with the unet checkpoint.")
parser.add_argument("--vaedecoder_ckpt", type=str, default=None, help="Path to your folder with the vae checkpoint.")
parser.add_argument("--xstartreview_ckpt", type=str, default=None, help="Path to your folder with the HSGM checkpoint.")
parser.add_argument("--reviewmode", type=str, default="latest", help="Path to your folder with the HSGM mode.")
parser.add_argument(
    "--batch_len",
    type=int,
    default=7,
    help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
)
parser.add_argument("--gt_path", type=str, default='data/UrbanDroneVSR/val/GT', help="Path to folder with GT frames.")
parser.add_argument("--gt_size", type=int, default=None, help="Size of GT frames.")

# parser.add_argument("--miavsrnet_ckpt", type=str, default=None, help="Path to your folder with the controlnet checkpoint.")

args = parser.parse_args()
os.makedirs(args.out_path, exist_ok=True)
logger = get_root_logger(logger_name=__file__, log_file=os.path.join(args.out_path,f"test_log.txt"))
print("Run with arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
    logger.info(f"  {arg}: {value}")


# set parameters
set_seed(42)
device = torch.device('cuda')
model_id = 'checkpoints/TAISNetVSR'
# args.controlnet_ckpt = None
# 加载 controlnet 的模型
# controlnet_model = ControlSpatialNetModel()
controlnet_model = ControlNetModel.from_pretrained(args.controlnet_ckpt if args.controlnet_ckpt is not None else model_id, subfolder='controlnet') # your own controlnet model
xstartreviewnet_model = XStartReview.from_pretrained(args.xstartreview_ckpt if args.xstartreview_ckpt is not None else model_id, subfolder='XStartReview') # your own controlnet model
xstartreviewnet_model = xstartreviewnet_model.to(device)
vae_model = AutoencoderKLIND.from_pretrained(args.vaedecoder_ckpt if args.vaedecoder_ckpt is not None else model_id, subfolder='MultiVAEINDDecoder',low_cpu_mem_usage=False,device_map=None) # your own controlnet model
vae_model = vae_model.to(device)
unet_model = UNet2DMultiScaleConditionModel.from_pretrained(args.unet_ckpt if args.unet_ckpt is not None else model_id, subfolder='unet',low_cpu_mem_usage=False,device_map=None) # your own controlnet model
unet_model = unet_model.to(device)

# # UNet_ori
# unet_ori = UNet2DConditionModel.from_pretrained(
#     model_id, subfolder="unet", low_cpu_mem_usage=False,
#     device_map=None
# )
# unet_ori = unet_ori.to(device)
#
# controlnet_model = ControlRevampNetModel.from_pretrained(args.controlnet_ckpt if args.controlnet_ckpt is not None else model_id, subfolder='controlnet',
#                                                          low_cpu_mem_usage=False,device_map=None) # your own controlnet model
# miavsr_model = MIAVSR.from_pretrained(args.miavsrnet_ckpt if args.miavsrnet_ckpt is not None else model_id, subfolder='miavsrnet') # your own controlnet model


# # 加载 stablevsr 的流水线模型
# # set up the models
# miavsr_model = MIAVSR(mid_channels=64,
#                embed_dim=120,
#                depths=[6, 6, 6, 6],
#                num_heads=[6, 6, 6, 6],
#                window_size=[3, 8, 8],
#                num_frames=3,
#                cpu_cache_length=100,
#                is_low_res_input=True,
#                use_mask=True,
#                spynet_path='checkpoints/MIAModel/flownet/spynet_sintel_final-3d2a1287.pth')
# # miavsr_model.load_state_dict(load_file(args.miavsrnet_ckpt if args.miavsrnet_ckpt is not None else "checkpoints/MIAModel/MIAVSR_REDS_x4.pth"), strict=False)
#
# miavsr_model.load_state_dict(torch.load(args.miavsrnet_ckpt if args.miavsrnet_ckpt is not None else "checkpoints/MIAModel/MIAVSR_REDS_x4.pth"), strict=False)
#
# pipeline = StableVSRPipeline.from_pretrained(model_id, controlnet=controlnet_model,unet=unet_model)
pipeline = StableVSRMutiScalePipeline.from_pretrained(model_id, controlnet=controlnet_model,unet=unet_model,newvaenet=vae_model,)
# pipeline = StableVSRMutiScalePipeline.from_pretrained(model_id, controlnet=controlnet_model)
# pipeline = StableMIAVSRPipeline.from_pretrained(model_id)
# 加载 scheduler 调度器
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler')
scheduler.reviewmodel = xstartreviewnet_model
# scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
pipeline.scheduler = scheduler
pipeline = pipeline.to(device)
pipeline.enable_xformers_memory_efficient_attention()
#加载光流估计模型 模型参数为默认参数
of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
of_model.requires_grad_(False)
of_model = of_model.to(device)

#################################精度评价###################################################
lpips = LPIPS(normalize=True)
dists = DISTS()
psnr = PSNR(data_range=1)
ssim = SSIM(data_range=1)
musiq = pyiqa.create_metric('musiq', device='cuda', as_loss=False)
niqe = pyiqa.create_metric('niqe', device='cuda', as_loss=False)
clip = pyiqa.create_metric('clipiqa', device='cuda', as_loss=False)

# of_model = raft(pretrained=True).to(device)
lpips = lpips.to(device)
dists = dists.to(device)
psnr = psnr.to(device)
ssim = ssim.to(device)

##############################################精度评价###################################################################
lpips_dict = {}
psnr_dict = {}
ssim_dict = {}
dists_dict = {}
musiq_dict = {}
niqe_dict = {}
clip_dict = {}
tlpips_dict = {}
tof_dict = {}
tt = ToTensor()

total = 0
gt_path = args.gt_path
for root, dirs, files in os.walk(gt_path):
    total += len(files)

pbar = tqdm(total=total, ncols=100)
# iterate for every video sequence in the input folder
seqs = sorted(os.listdir(args.in_path))                                     #读取需要评估的视频序列目录
for seq in seqs:
    print(f"Processing sequence {seq}")
    # 存放评价指标
    # ims_rec = sorted(os.listdir(os.path.join(rec_path, seq)))
    # ims_gt = sorted(os.listdir(os.path.join(gt_path, seq)))

    lpips_dict[seq] = []
    psnr_dict[seq] = []
    ssim_dict[seq] = []
    dists_dict[seq] = []
    musiq_dict[seq] = []
    niqe_dict[seq] = []
    clip_dict[seq] = []
    tlpips_dict[seq] = []
    tof_dict[seq] = []

    # 读取加载数据
    frame_names = sorted(os.listdir(os.path.join(args.in_path, seq)))       #当前视频序列帧文件名
    frames = []
    gt_frames = []
    for idx,frame_name in enumerate(frame_names):
        frame = Path(os.path.join(args.in_path, seq, frame_name))
        frame = Image.open(frame)
                                            #读取所有的视频帧数据
        gt = Image.open(os.path.join(gt_path, seq, frame_name))
        gt_szie = args.gt_size
        if gt_szie is not None:
            frame = center_crop(frame, size= gt_szie//4)
            gt = center_crop(gt, gt_szie)
        frames.append(frame)
        gt_frames.append(gt)
        # if idx>10:
        #     break


    # upscale frames using TAISNet
    frames = pipeline('', frames,GT_images=gt_frames, num_inference_steps=args.num_inference_steps,lambda_step = args.num_lamda_steps, guidance_scale=0, of_model=of_model,reviewmode=args.reviewmode).images
    frames = [frame[0] for frame in frames]

    # 进行精度评价
    for i, (im_rec, im_gt) in enumerate(zip(frames, gt_frames)):
        gt = tt(im_gt).unsqueeze(0).to(device)
        rec = tt(im_rec).unsqueeze(0).to(device)

        psnr_value = psnr(gt, rec)
        ssim_value = ssim(gt, rec)
        lpips_value = lpips(gt, rec)
        dists_value = dists(gt, rec)
        musiq_value = musiq(rec)
        niqe_value = niqe(rec)
        clip_value = clip(rec)
        if i > 0:
            tlpips_value = (lpips(gt, prev_gt) - lpips(rec, prev_rec)).abs()
            tlpips_dict[seq].append(tlpips_value.item())
            tof_value = (get_flow(of_model, rec, prev_rec) - get_flow(of_model, gt, prev_gt)).abs().mean()
            tof_dict[seq].append(tof_value.item())

        psnr_dict[seq].append(psnr_value.item())
        ssim_dict[seq].append(ssim_value.item())
        lpips_dict[seq].append(lpips_value.item())
        dists_dict[seq].append(dists_value.item())
        musiq_dict[seq].append(musiq_value.item())
        niqe_dict[seq].append(niqe_value.item())
        clip_dict[seq].append(clip_value.item())

        prev_rec = rec
        prev_gt = gt
        pbar.update()

    # save upscaled sequences
    # 保存超分处理之后的结果
    seq = Path(seq)
    target_path = os.path.join(args.out_path, seq.parent.name, seq.name)
    os.makedirs(target_path, exist_ok=True)
    for frame, name in zip(frames, frame_names):
        frame.save(os.path.join(target_path, name))

pbar.close()
# 打印当前序列的评价指标
for seq in seqs:
    mean_lpips = np.round(np.mean(lpips_dict[seq]), 3)
    mean_dists = np.round(np.mean(dists_dict[seq]), 3)
    mean_psnr = np.round(np.mean(psnr_dict[seq]), 2)
    mean_ssim = np.round(np.mean(ssim_dict[seq]), 3)
    mean_musiq = np.round(np.mean(musiq_dict[seq]), 2)
    mean_niqe = np.round(np.mean(niqe_dict[seq]), 2)
    mean_clip = np.round(np.mean(clip_dict[seq]), 3)
    mean_tlpips = np.round(np.mean(tlpips_dict[seq]) * 1e3, 2)
    mean_tof = np.round(np.mean(tof_dict[seq]) * 1e1, 3)
    # seq
    logger.info(
        f"seq:{seq} \n"
        f'PSNR: {mean_psnr}, SSIM: {mean_ssim}, LPIPS: {mean_lpips}, DISTS: {mean_dists}, MUSIQ: {mean_musiq}, CLIP: {mean_clip}, NIQE: {mean_niqe}, tLPIPS: {mean_tlpips}, tOF: {mean_tof}')
    print(
        f"seq:{seq} \n"
        f'PSNR: {mean_psnr}, SSIM: {mean_ssim}, LPIPS: {mean_lpips}, DISTS: {mean_dists}, MUSIQ: {mean_musiq}, CLIP: {mean_clip}, NIQE: {mean_niqe}, tLPIPS: {mean_tlpips}, tOF: {mean_tof}')



mean_lpips = np.round(np.mean([np.mean(lpips_dict[key]) for key in lpips_dict.keys()]), 3)
mean_dists = np.round(np.mean([np.mean(dists_dict[key]) for key in dists_dict.keys()]), 3)
mean_psnr = np.round(np.mean([np.mean(psnr_dict[key]) for key in psnr_dict.keys()]), 2)
mean_ssim = np.round(np.mean([np.mean(ssim_dict[key]) for key in ssim_dict.keys()]), 3)
mean_musiq = np.round(np.mean([np.mean(musiq_dict[key]) for key in musiq_dict.keys()]), 2)
mean_niqe = np.round(np.mean([np.mean(niqe_dict[key]) for key in niqe_dict.keys()]), 2)
mean_clip = np.round(np.mean([np.mean(clip_dict[key]) for key in clip_dict.keys()]), 3)
mean_tlpips = np.round(np.mean([np.mean(tlpips_dict[key]) for key in tlpips_dict.keys()]) * 1e3, 2)
mean_tof = np.round(np.mean([np.mean(tof_dict[key]) for key in tof_dict.keys()]) * 1e1, 3)

logger.info(
    f'PSNR: {mean_psnr}, SSIM: {mean_ssim}, LPIPS: {mean_lpips}, DISTS: {mean_dists}, MUSIQ: {mean_musiq}, CLIP: {mean_clip}, NIQE: {mean_niqe}, tLPIPS: {mean_tlpips}, tOF: {mean_tof}')

print(
    f'PSNR: {mean_psnr}, SSIM: {mean_ssim}, LPIPS: {mean_lpips}, DISTS: {mean_dists}, MUSIQ: {mean_musiq}, CLIP: {mean_clip}, NIQE: {mean_niqe}, tLPIPS: {mean_tlpips}, tOF: {mean_tof}')
