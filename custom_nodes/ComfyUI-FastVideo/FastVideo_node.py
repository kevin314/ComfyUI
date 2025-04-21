from __future__ import annotations
import torch
import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
import logging
import glob
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch
#os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
sys.path.insert(0, "/workspace/FastVideo")


import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import comfy.clip_vision
import comfy.model_management
from comfy.cli_args import args
import importlib
import folder_paths
import latent_preview
import node_helpers
import subprocess

from fastvideo.v1.logger import init_logger
from fastvideo import VideoGenerator

logger = init_logger(__name__)

MAX_RESOLUTION = 16384

class FastVideoSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_gpus": ("INT", {"default": 2, "min": 1, "max": 16}),
                "master_port": ("INT", {"default": 29503}),
                "model_path": ("STRING", {"default": "FastHunyuan-diffusers"}),
                "prompt": ("STRING",
                           {"default": "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest. The playful yet serene atmosphere is complemented by soft natural light filtering through the petals. Mid-shot, warm and cheerful tones."}),
                #"prompt_path": ("STRING", {"default": os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")}),
                "output_path": ("STRING", {"default": "/workspace/ComfyUI/outputs_video/"}),
                "height": ("INT", {"default": 720}),
                "width": ("INT", {"default": 1280}),
                "num_frames": ("INT", {"default": 45}),
                "num_inference_steps": ("INT", {"default": 6}),
                "guidance_scale": ("FLOAT", {"default": 1.0}),
                "embedded_cfg_scale": ("FLOAT", {"default": 6.0}),
                "flow_shift": ("INT", {"default": 17}),
                "seed": ("INT", {"default": 1024}),
                "sp_size": ("INT", {"default": 2}),
                "tp_size": ("INT", {"default": 2}),
                "vae_sp": ("BOOLEAN", {"default": True}),
                "fps": ("INT", {"default": 24}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "launch_inference"
    #FUNCTION = "generate_video"
    CATEGORY = "fastvideo"

    def load_output_video(self, output_dir):
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files = []

        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(output_dir, ext)))

        if not video_files:
            logger.warning("No video files found in output directory: %s", output_dir)
            return ""

        video_files.sort()
        return video_files[0]

    def launch_inference_old(
        self,
        num_gpus,
        master_port,
        model_path,
        prompt,
        output_path,
        height,
        width,
        num_frames,
        num_inference_steps,
        guidance_scale,
        embedded_cfg_scale,
        flow_shift,
        seed,
        sp_size,
        tp_size,
        vae_sp,
        fps,
    ):
        current_env = os.environ.copy()
        python_executable = sys.executable

        main_script = "custom_nodes/fastvideo/v1/sample/v1_fastvideo_inference.py"

        current_env["PYTHONIOENCODING"] = "utf-8"
        current_env["FASTVIDEO_ATTENTION_BACKEND"] = ""
        current_env["MODEL_BASE"] = model_path

        cmd = [
            python_executable, "-m", "torch.distributed.run",
            f"--nnodes=1",
            f"--nproc_per_node={num_gpus}",
            f"--master_port={master_port}",
            main_script,
            "--sp_size", str(sp_size),
            "--tp_size", str(tp_size),
            "--height", str(height),
            "--width", str(width),
            "--num_frames", str(num_frames),
            "--num_inference_steps", str(num_inference_steps),
            "--guidance_scale", str(guidance_scale),
            "--embedded_cfg_scale", str(embedded_cfg_scale),
            "--flow_shift", str(flow_shift),
            #"--prompt_path", prompt_path,
            "--prompt", str(prompt),
            "--seed", str(seed),
            "--output_path", output_path,
            "--model_path", model_path,
            "--fps", str(fps),
        ]

        if vae_sp:
            cmd.append("--vae-sp")

        logger.info("Launching FastVideo inference with %d GPU(s)", num_gpus)
        logger.info("Command: %s", " ".join(cmd))

        process = subprocess.Popen(
            cmd,
            env=current_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=0,
            encoding='utf-8',
            errors='replace'
        )

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line.strip())

        process.wait()

        #video_path = self.load_output_video(output_path)
        video_path = os.path.join(output_path, f"{prompt[:100]}.mp4")
        print("VIDEOPATHX1", video_path)
        return (video_path,)

    # global generator
    # generator = VideoGenerator.from_pretrained(
    #     model_path="FastVideo/FastHunyuan-diffusers",
    #     num_gpus=2
    # )

    def launch_inference(
        self,
        num_gpus,
        master_port,
        model_path,
        prompt,
        output_path,
        height,
        width,
        num_frames,
        num_inference_steps,
        guidance_scale,
        embedded_cfg_scale,
        flow_shift,
        seed,
        sp_size,
        tp_size,
        vae_sp,
        fps,
    ):
        print("test")
        generator = VideoGenerator.from_pretrained(
            model_path="FastVideo/FastHunyuan-diffusers",
            num_gpus=2
        )
        generator.generate_video(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            seed=seed
        )

        output_path = os.path.join(args.output_path, f"{prompt[:100]}.mp4")
        return(output_path,)

# Register the custom node
NODE_CLASS_MAPPINGS = {
    "FastVideoSampler": FastVideoSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastVideoSampler": "Fast Video Sampler",
}
