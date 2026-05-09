#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import warnings
warnings.filterwarnings("ignore")
import torchvision
import cv2
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from omegaconf import OmegaConf
# from dataset.reds_dataset import REDSRecurrentDataset
from dataset.uavdrone_dataset import UAVDroneRecurrentDataset as REDSRecurrentDataset
# from dataset.visdrone_dataset import VISDroneRecurrentDataset as REDSRecurrentDataset

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from einops import rearrange
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from util.flow_utils import get_flow, flow_warp

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel
)
from diffusers.models.autoencoder_kl import AutoencoderKLIND
from diffusers.models.unet_2d_condition import UNet2DMultiScaleConditionModel
from pipeline.stablevsr_pipeline import StableVSRPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from scheduler.ddpm_scheduler import DDPMScheduler

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import pyiqa
from DISTS_pytorch import DISTS
from torchvision.models.optical_flow import raft_large as raft
from torchvision.transforms import ToTensor, CenterCrop
# from diffusers.modules.SamplesReview import XStartReview
# from diffusers.modules.SamplesReview import XStartReviewDF as XStartReview
from diffusers.modules.SamplesReview import XStartReviewCrossFFT as XStartReview
# from diffusers.modules.SamplesReview import XStartReviewCrossFreq as XStartReview
best_psnr = 0
best_lpips = 1e6

#将传入的图像列表根据设置的行列进行排列
#注意，传入的图像列表数目需要与行列乘积一致
#默认大小一致 取第一张图像的大小
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
def center_crop(im, size=128):
    width, height = im.size   # Get dimensions
    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2
    return im.crop((left, top, right, bottom))
def center_crop_tensor(im, size=128):
    width, height,_ = im.shape   # Get dimensions
    left = int((width - size)/2)
    top = int((height - size)/2)
    right = int((width + size)/2)
    bottom = int((height + size)/2)
    return im[left:right, top:bottom,:]

from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
import torchvision
import cv2


noise = torch.randn((1, 4, 64, 64))
def log_validation(vae,noise_scheduler, encoder_hidden_states, unet, xstartreview, args, accelerator, weight_dtype, step, of_model):
    logger.info("Running validation... ")  # 正在执行验证操作
    logger_file = get_root_logger(logger_name=__file__, log_file=os.path.join(args.output_dir,f"train_log.txt"))

    lpips = LPIPS(normalize=True)
    dists = DISTS()
    psnr = PSNR(data_range=1)
    ssim = SSIM(data_range=1)
    musiq = pyiqa.create_metric('musiq', device='cuda', as_loss=False)
    niqe = pyiqa.create_metric('niqe', device='cuda', as_loss=False)
    clip = pyiqa.create_metric('clipiqa', device='cuda', as_loss=False)

    # of_model = raft(pretrained=True).to(device)
    lpips = lpips.to(accelerator.device)
    dists = dists.to(accelerator.device)
    psnr = psnr.to(accelerator.device)
    ssim = ssim.to(accelerator.device)                            #正在执行验证操作

    noise_scheduler.set_timesteps(20)  # 设置时间步
    timesteps = noise_scheduler.timesteps
    latents_pred_prev_store = []
    validation_images = args.validation_image[0]
    frames = validation_images.split(';')
    # 读取视频帧并进行中心裁切操作
    gt_frames = []
    lq_frames = []
    for i, frame in enumerate(frames):
        if not os.path.exists(frame):
            continue
        images_times = []
        seq, name = str(frame).split("/")[-2:]
        img = cv2.imread(frame)
        img = img.astype(np.float32) / 255.
        # img_tensor = img_tensor(img, bgr2rgb=True)      #rgb 格式
        if args.gt_size is not None:
            img = center_crop_tensor(img, size=args.gt_size//4)
        img = np.array(img)
        # img = img.astype(np.float32) / 255.
        img_tensor = img2tensor(img, bgr2rgb=True)  # rgb 格式

        lq_frames.append(img_tensor)

        gt_dir = args.gt_path
        # lq_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(frame))), "val_sharp_bicubic/X4")
        gt_filename = os.path.join(gt_dir, seq, name)
        img = cv2.imread(gt_filename)
        img = img.astype(np.float32) / 255.
        # img_tensor = img_tensor(img, bgr2rgb=True)      #rgb 格式
        if args.gt_size is not None:
            img = center_crop_tensor(img, size=args.gt_size)
        img = np.array(img)
        # img = img.astype(np.float32) / 255.
        img_tensor_lq = img2tensor(img, bgr2rgb=True)  # rgb 格式
        gt_frames.append(img_tensor_lq)
        # Sample noise that we'll add to the latents
        # 为隐空间的图像添加噪声，模拟扩散过程。

    lpips_dict_ori = []
    psnr_dict_ori = []
    ssim_dict_ori = []
    dists_dict_ori = []
    musiq_dict_ori = []
    niqe_dict_ori = []
    clip_dict_ori = []
    tlpips_dict_ori = []
    tof_dict_ori = []

    lpips_dict_new = []
    psnr_dict_new = []
    ssim_dict_new = []
    dists_dict_new = []
    musiq_dict_new = []
    niqe_dict_new = []
    clip_dict_new = []
    tlpips_dict_new = []
    tof_dict_new = []

    tt = ToTensor()
    unet.eval()
    with torch.no_grad():
        for idx_f,(gt_img,lq_img) in tqdm(enumerate(zip(gt_frames,lq_frames))):
            x0_prev_historys=[]
            gt_tensor = gt_img.unsqueeze(0).to(accelerator.device)
            lq_tensor = lq_img.unsqueeze(0).to(accelerator.device)
            gt = 2 * gt_tensor - 1
            lq = 2 * lq_tensor - 1
            b, t, _, _ = lq_tensor.shape

            # Convert images to latent space
            # 将高质量图像和前一帧图像编码到隐空间。 1 3 256 256
            latents = vae.encode(gt.to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            global noise
            noise = noise.to(accelerator.device)
            # noise = torch.randn_like(latents).to(accelerator.device)  # 真值参考的噪声
            img = noise.clone().to(accelerator.device)
            img_2 = img.clone().to(accelerator.device)
            for tt in tqdm(timesteps):
                bsz = latents.shape[0]
                timestep = tt.expand((bsz,)).to(latents.device)
                # if tt > 1:
                #     noisy_latents_step = noise_scheduler.add_noise(latents, noise, timestep - 1)
                # else:
                #     noisy_latents_step = latents

                noisy_latents_cat1 = torch.cat([img, lq], dim=1)  # 去噪的输入条件
                noisy_latents_cat2 = torch.cat([img_2, lq], dim=1)  # 去噪的输入条件
                # Get the text embedding for conditioning
                # tokenization = tokenizer('', max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                # encoder_hidden_states = text_encoder(tokenization)[0]
                #获取真值
                noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)
                target = noise_scheduler.get_velocity(latents, noise, timestep)

                latnet_targets = []
                for idx, tt in enumerate(timestep):
                    output_prev = noise_scheduler.step(target[[idx]], tt, noisy_latents)
                    # 在这里更新 lantent
                    latnet_pred_prev1, x0_est_pred1 = output_prev.prev_sample, output_prev.pred_original_sample
                    # XStart_pred1_prevs.append(x0_est_pred1)
                    latnet_targets.append(latnet_pred_prev1)
                latnet_targets = torch.cat(latnet_targets)


                # 使用 U-Net 模型预测噪声残差。当前预测的
                model_pred_1 = unet(
                    noisy_latents_cat1,
                    timestep,
                    # class_labels = noise_level,
                    encoder_hidden_states=encoder_hidden_states[:b].detach(),
                    num_scales=1
                ).sample  # 通过前一时刻的噪声输入预测前一时刻添加的噪声
                if isinstance(model_pred_1,list):
                    model_pred_1 = model_pred_1[-1]

                # 使用 U-Net 模型预测噪声残差。当前预测的
                model_pred_2 = unet(
                    noisy_latents_cat2,
                    timestep,
                    # class_labels = noise_level,
                    encoder_hidden_states=encoder_hidden_states[:b].detach(),
                    num_scales=1
                ).sample  # 通过前一时刻的噪声输入预测前一时刻添加的噪声
                if isinstance(model_pred_2,list):
                    model_pred_2 = model_pred_2[-1]

                latnet_pred1_prevs = []
                # XStart_pred1_prevs = []
                for idx, tt in enumerate(timestep):
                    output_prev = noise_scheduler.step(model_pred_1[[idx]], tt, img[[idx]])
                    # 在这里更新 lantent
                    latnet_pred_prev1, x0_est_pred1 = output_prev.prev_sample, output_prev.pred_original_sample
                    # XStart_pred1_prevs.append(x0_est_pred1)
                    latnet_pred1_prevs.append(latnet_pred_prev1)
                latnet_pred1_prevs = torch.cat(latnet_pred1_prevs)
                # XStart_pred1_prevs = torch.cat(XStart_pred1_prevs)
                img = latnet_pred1_prevs                                            # 原始输入预测的上一步输出



                latnet_pred2_prevs = []
                XStart_pred2_prevs = []
                for idx, tt in enumerate(timestep):
                    output_prev = noise_scheduler.step(model_pred_2[[idx]], tt, img_2[[idx]])
                    # 在这里更新 lantent
                    latnet_pred_prev, x0_est_pred = output_prev.prev_sample, output_prev.pred_original_sample
                    XStart_pred2_prevs.append(x0_est_pred)                  #预测的上一步的X0
                    if len(x0_prev_historys) > 0:
                        x0_prev_last = x0_prev_historys[-1]    #和最后一个做合并
                        x0_merger= xstartreview(x0_est_pred,x0_prev_last,(timestep[idx-1],tt))
                        output_prev_merge = noise_scheduler.step(model_pred_2[[idx]], tt, img_2[[idx]],x0_estmate=x0_merger)
                        latnet_pred_merge, x0_est_merge = output_prev_merge.prev_sample, output_prev_merge.pred_original_sample
                        latnet_pred2_prevs.append(latnet_pred_merge)
                    else:
                        latnet_pred2_prevs.append(latnet_pred_prev)
                latnet_pred2_prevs = torch.cat(latnet_pred2_prevs)
                XStart_pred2_prevs = torch.cat(XStart_pred2_prevs)      # 保存的是之前预测的结果 也可以保存优化之后的结果 暂时先保存原始结果吧
                img_2 = latnet_pred2_prevs

                x0_prev_historys.append(XStart_pred2_prevs)


                in_diff_ori = torch.abs(target - model_pred_1)  # 当前和之前的差值
                in_diff_post = torch.abs(target - model_pred_2)  # 当前和之前的差值
                GT_diff = torch.abs(latnet_targets - latnet_pred1_prevs)  # 当前和之前的差值
                GT_diffPost = torch.abs(latnet_targets - latnet_pred2_prevs)  # 当前和之前的差值

                Post_diffPost = torch.abs(latnet_pred1_prevs - latnet_pred2_prevs)  # 当前和之前的差值

                # 四通道的数据  model_pred noisy_latents model_pred_post target
                visual_images = [torch.cat([latents[[0], [t]], noisy_latents[[0], [t]],
                                            model_pred_1[[0], [t]], target[[0], [t]], in_diff_ori[[0], [t]],
                                            model_pred_2[[0], [t]], target[[0], [t]], in_diff_post[[0], [t]],
                                            latnet_pred1_prevs[[0], [t]], latnet_targets[[0], [t]], GT_diff[[0], [t]],
                                            latnet_pred2_prevs[[0], [t]], latnet_targets[[0], [t]], GT_diffPost[[0], [t]],
                                            Post_diffPost[[0], [t]]]
                ) for t in range(4)]

                grid_images = []
                for visual in visual_images:
                    visual = visual.unsqueeze(1)
                    grid_image = torchvision.utils.make_grid(visual, normalize=True, nrow=30)
                    grid_image = (grid_image.permute(1, 2, 0) * 255.0).cpu().contiguous().detach().numpy().astype(
                        np.uint8)
                    grid_image = cv2.applyColorMap(grid_image, cv2.COLORMAP_JET)
                    grid_images.append(grid_image)
                    cc = 0
                grid_image = np.concatenate(grid_images, axis=0)
                h, w, c = grid_image.shape
                pad = np.zeros((20, w, c))
                grid_image = np.concatenate([grid_image, pad], axis=0)

                # grid_image = np.pad(grid_image, ((10, 10), (2, 2)), mode='constant', constant_values=0)
                images_times.append(grid_image)
            image_save = np.concatenate(images_times, axis=0)
            image_save = image_save.astype(np.uint8)
            dir = os.path.join(args.output_dir, "images", "train")
            os.makedirs(dir, exist_ok=True)
            name = f"{idx_f:04d}_{step:04d}_latent.png"
            cv2.imwrite(os.path.join(dir, name), image_save)

            img_rgb = vae.decode(img / vae.config.scaling_factor, return_dict=False,num_scales=1)[0]
            img_2_rgb = vae.decode(img_2 / vae.config.scaling_factor, return_dict=False,num_scales=1)[0]
            #处理多个输入情况
            if isinstance(img_rgb,list):
                img_rgb = img_rgb[-1]
                img_2_rgb = img_2_rgb[-1]


            img_rgb = (img_rgb / 2 + 0.5).clamp(0, 1)
            img_2_rgb = (img_2_rgb / 2 + 0.5).clamp(0, 1)
            gt = (gt / 2 + 0.5).clamp(0, 1)

            #获取评价指标
            psnr_value_ori = psnr(gt, img_rgb)
            ssim_value_ori = ssim(gt, img_rgb)
            lpips_value_ori = lpips(gt, img_rgb)
            dists_value_ori = dists(gt, img_rgb)
            musiq_value_ori = musiq(img_rgb)
            niqe_value_ori = niqe(img_rgb)
            clip_value_ori = clip(img_rgb)

            psnr_dict_ori.append(psnr_value_ori.item())
            ssim_dict_ori.append(ssim_value_ori.item())
            lpips_dict_ori.append(lpips_value_ori.item())
            dists_dict_ori.append(dists_value_ori.item())
            musiq_dict_ori.append(musiq_value_ori.item())
            niqe_dict_ori.append(niqe_value_ori.item())
            clip_dict_ori.append(clip_value_ori.item())


            psnr_value = psnr(gt, img_2_rgb)
            ssim_value = ssim(gt, img_2_rgb)
            lpips_value = lpips(gt, img_2_rgb)
            dists_value = dists(gt, img_2_rgb)
            musiq_value = musiq(img_2_rgb)
            niqe_value = niqe(img_2_rgb)
            clip_value = clip(img_2_rgb)

            psnr_dict_new.append(psnr_value.item())
            ssim_dict_new.append(ssim_value.item())
            lpips_dict_new.append(lpips_value.item())
            dists_dict_new.append(dists_value.item())
            musiq_dict_new.append(musiq_value.item())
            niqe_dict_new.append(niqe_value.item())
            clip_dict_new.append(clip_value.item())


    psnr_mean_ori =np.mean(psnr_dict_ori)
    mean_ssim_ori = np.mean(ssim_dict_ori)
    lpips_mean_ori =np.mean(lpips_dict_ori)
    mean_dists_ori = np.mean(dists_dict_ori)
    mean_musiq_ori = np.mean(musiq_dict_ori)
    mean_clip_ori = np.mean(clip_dict_ori)
    mean_niqe_ori = np.mean(niqe_dict_ori)
    # mean_tlpips_ori = np.mean(tlpips_dict_ori)
    # mean_tof_ori = np.mean(tof_dict_ori)

    psnr_mean_new =np.mean(psnr_dict_new)
    mean_ssim_new = np.mean(ssim_dict_new)
    lpips_mean_new =np.mean(lpips_dict_new)
    mean_dists_new = np.mean(dists_dict_new)
    mean_musiq_new = np.mean(musiq_dict_new)
    mean_clip_new = np.mean(clip_dict_new)
    mean_niqe_new = np.mean(niqe_dict_new)
    # mean_tlpips_new = np.mean(tlpips_dict_new)
    # mean_tof_new = np.mean(tof_dict_new)

    logger.info(
        f'PSNR: {psnr_mean_ori}, SSIM: {mean_ssim_ori}, LPIPS: {lpips_mean_ori}, DISTS: {mean_dists_ori}, MUSIQ: {mean_musiq_ori}, '
        f'CLIP: {mean_clip_ori}, NIQE: {mean_niqe_ori}\n'
        f'PSNRNew: {psnr_mean_new}, SSIMNew: {mean_ssim_new}, LPIPSNew: {lpips_mean_new}, DISTSNew: {mean_dists_new}, MUSIQNew: {mean_musiq_new}, '
        f'CLIPNew: {mean_clip_new}, NIQENew: {mean_niqe_new}\n')

    logger_file.info(
        f'PSNR: {psnr_mean_ori}, SSIM: {mean_ssim_ori}, LPIPS: {lpips_mean_ori}, DISTS: {mean_dists_ori}, MUSIQ: {mean_musiq_ori}, '
        f'CLIP: {mean_clip_ori}, NIQE: {mean_niqe_ori}\n'
        f'PSNRNew: {psnr_mean_new}, SSIMNew: {mean_ssim_new}, LPIPSNew: {lpips_mean_new}, DISTSNew: {mean_dists_new}, MUSIQNew: {mean_musiq_new}, '
        f'CLIPNew: {mean_clip_new}, NIQENew: {mean_niqe_new}\n')    # unet.train()

    os.makedirs(os.path.join(args.output_dir, "Metric"), exist_ok=True)
    global best_psnr, best_lpips
    if psnr_mean_new > best_psnr:
        best_psnr = psnr_mean_new
        save_path = os.path.join(args.output_dir, "Metric", f"checkpoint-psnrbest")
        accelerator.save_state(save_path)
        logger.info(f"Step:{step} Saved state to {save_path}")
        # bestpsnr_step = step
    if lpips_mean_new > best_lpips:
        best_lpips = lpips_mean_new
        save_path = os.path.join(args.output_dir, "Metric", f"checkpoint-lpipsbest")
        accelerator.save_state(save_path)
        logger.info(f"Step:{step} Saved state to {save_path}")
        # bestpsnr_step = step




def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-2-1',
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-2-1',
        required=True,
        help="Path to pretrained vae model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--vaedecoder_ckpt",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--unet_ckpt",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_path",
        type=str,
        default=None,
        help=(
            "The path to the config file related to the dataset."
        ),
    )    
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--gt_path", type=str, default='data/REDS/val_sharp/', help="Path to folder with GT frames.")
    parser.add_argument("--gt_size", type=int, default=None, help="Size of GT frames.")
    parser.add_argument(
        "--batch_len",
        type=int,
        default=7,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None and args.dataset_config_path is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir` or `dataset_config_path`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )


    return args

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    #XStartReview网络
    xstartreview = XStartReview()

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path, subfolder="vae", revision=args.revision)
    # unet = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    # )

    vae = AutoencoderKLIND.from_pretrained(args.vaedecoder_ckpt, subfolder="vae", revision=args.revision)
    unet = UNet2DMultiScaleConditionModel.from_pretrained(
        args.unet_ckpt, subfolder="unet", revision=args.revision
    )

    # if args.controlnet_model_name_or_path:
    #     logger.info("Loading existing controlnet weights")
    #     controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    # else:
    #     logger.info("Initializing controlnet weights from unet")
    #     controlnet = ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(64,128,256,))

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "XStartReview"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = xstartreview.from_pretrained(input_dir, subfolder="XStartReview")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    # of_model vae unet text_encoder 模型参数全部冻结 仅训练 controlNet网络
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    of_model.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    xstartreview.train()
    #是否使用 xformer 加快运行效率，减小内存占用 但是会降低质量的一致性
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            xstartreview.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    #梯度检查
    if args.gradient_checkpointing:
        xstartreview.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(xstartreview).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(xstartreview).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = xstartreview.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    # train_dataset = make_train_dataset(args, tokenizer, accelerator)
    dataset_opts = OmegaConf.load(args.dataset_config_path)
    train_dataset = REDSRecurrentDataset(dataset_opts['dataset']['train'])

    #加载训练数据集
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        # collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)    #重新计算一个epoch多少步
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    #加载学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    xstartreview, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        xstartreview, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    of_model.to(accelerator.device, dtype=weight_dtype)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)              #总共需要多少个epoch

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    # 从 checkpoint 中恢复
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0
    # 进度条 回显进度
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None


    with torch.no_grad():
        tokenization = tokenizer([''] * args.train_batch_size, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        encoder_hidden_states = text_encoder(tokenization.input_ids.to(accelerator.device))[0]

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):        #定义了训练的外层循环（遍历所有训练轮数）和内层循环（遍历每个批次的数据）。
            # 使用 accelerator.accumulate 上下文管理器，确保梯度累积在多个前向传播中进行，然后在执行一次反向传播。
            with accelerator.accumulate(xstartreview):
                # inference_steps = random.randint(10,100)                #假设实际采样步数为10 - 100 步之间
                inference_steps = random.choice([20,30,50])
                noise_scheduler.set_timesteps(inference_steps)                #设置时间步
                timesteps = noise_scheduler.timesteps                         #获取更新后的timesteps
                #获取两个t
                t_idx=random.randint(1,inference_steps-1)
                time_prev=timesteps[t_idx]                                      #较小时刻
                time_after = timesteps[t_idx-1]                                 #较大时刻
                # Prepare images
                lq = batch['lq'] 
                gt = batch['gt']
                gt = 2 * gt - 1
                lq = 2 * lq - 1          # 从批次中提取低质量图像（lq）和高质量图像（gt），并将它们归一化到 [-1, 1] 范围内。
                b, t, _, _, _ = lq.shape

                # 选择当前时间步的图像。 #中间帧
                gt = gt[:, t // 2, ...]
                lq = lq[:, t // 2, ...]
                # upscaled_lq_cur = upscaled_lq[:, t // 2, ...]

                # Convert images to latent space
                # 将高质量图像和前一帧图像编码到隐空间。 1 3 256 256
                latents = vae.encode(gt.to(dtype=weight_dtype)).latent_dist.sample()            # 10 4 64 64
                latents = latents * vae.config.scaling_factor

                #
                time_afters = time_after.expand((b,)).to(accelerator.device)
                noise = torch.randn_like(latents).to(accelerator.device)  # 真值参考的噪声
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)   #前向加噪过程
                noisy_latents = noise_scheduler.add_noise(latents, noise, time_afters)

                noisy_latents_cat = torch.cat([noisy_latents, lq], dim=1)  # 去噪的输入条件
                model_pred_prevtime = unet(
                    noisy_latents_cat,
                    time_afters,
                    # class_labels = noise_level,
                    encoder_hidden_states=encoder_hidden_states[:b].detach(),
                    num_scales=1,
                ).sample  # 通过前一时刻的噪声输入预测前一时刻添加的噪声
                if isinstance(model_pred_prevtime,list):
                    model_pred_prevtime = model_pred_prevtime[-1]
                latnet_pred_afters = []
                x0_pred_afters = []
                for idx, tt in enumerate(time_afters):
                    output_prev = noise_scheduler.step(model_pred_prevtime[[idx]], tt, noisy_latents[[idx]])
                    # 在这里更新 lantent
                    latnet_pred_prev, x0_est_prev = output_prev.prev_sample, output_prev.pred_original_sample
                    latnet_pred_afters.append(latnet_pred_prev)
                    x0_pred_afters.append(x0_est_prev)
                latnet_pred_afters = torch.cat(latnet_pred_afters).to(accelerator.device)
                x0_pred_afters = torch.cat(x0_pred_afters).to(accelerator.device)
                #后一个时刻
                time_prevs = time_prev.expand((b,)).to(accelerator.device)
                noisy_latents_aftercat = torch.cat([latnet_pred_afters, lq], dim=1)  # 去噪的输入条件
                model_pred_prevtime = unet(
                    noisy_latents_aftercat,
                    time_prevs,
                    # class_labels = noise_level,
                    encoder_hidden_states=encoder_hidden_states[:b].detach()
                ).sample  # 通过前一时刻的噪声输入预测前一时刻添加的噪声
                if isinstance(model_pred_prevtime,list):
                    model_pred_prevtime = model_pred_prevtime[-1]
                # latnet_pred_prevs = []
                x0_pred_prevs = []
                for idx, tt in enumerate(time_afters):
                    output_prev = noise_scheduler.step(model_pred_prevtime[[idx]], tt, noisy_latents[[idx]])
                    # 在这里更新 lantent
                    latnet_pred_prev, x0_est_prev = output_prev.prev_sample, output_prev.pred_original_sample
                    # latnet_pred_prevs.append(latnet_pred_prev)
                    x0_pred_prevs.append(x0_est_prev)
                # latnet_pred_prevs = torch.cat(latnet_pred_prevs)
                x0_pred_prevs = torch.cat(x0_pred_prevs).to(accelerator.device)

                x0_final = xstartreview(x0_pred_prevs,x0_pred_afters,(time_afters,time_prevs))
                latent_denoises = []
                for idx, tt in enumerate(time_afters):
                    output_prev = noise_scheduler.step(model_pred_prevtime[[idx]], tt, noisy_latents[[idx]],x0_estmate=x0_final[[idx]])
                    # 在这里更新 lantent
                    latnet_pred_prev, x0_est_prev = output_prev.prev_sample, output_prev.pred_original_sample
                    latent_denoises.append(latnet_pred_prev)
                latent_denoises = torch.cat(latent_denoises).to(accelerator.device)

                target = noise_scheduler.get_velocity(latents, noise, time_prevs)
                noisy_latents = noise_scheduler.add_noise(latents, noise, time_prevs)
                latnet_target_prevs = []
                for idx, tt in enumerate(time_prevs):
                    output_prev = noise_scheduler.step(target[[idx]], tt, noisy_latents[[idx]])
                    # 在这里更新 lantent
                    latnet_pred_prev, x0_est_prev = output_prev.prev_sample, output_prev.pred_original_sample
                    latnet_target_prevs.append(latnet_pred_prev)
                latnet_target_prevs = torch.cat(latnet_target_prevs).to(accelerator.device)

                #OK latent_denoises latnet_target_prevs

                # 计算预测值和目标值之间的均方误差损失。
                loss = F.mse_loss(latent_denoises.float(), latnet_target_prevs.float(), reduction="mean")
                # 执行反向传播，更新模型参数，并调整学习率。
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = xstartreview.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # 在主进程中保存模型权重。
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    # # 定期执行验证，并记录验证结果
                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            noise_scheduler,
                            encoder_hidden_states,
                            unet,
                            xstartreview,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            of_model
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(xstartreview)
        controlnet.save_pretrained(args.output_dir)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
