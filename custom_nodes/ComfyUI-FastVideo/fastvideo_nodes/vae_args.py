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
    CATEGORY = "fastvideo"

    generator = None

    def load_output_video(self, output_dir):
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files = []

        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(output_dir, ext)))

        if not video_files:
            print("No video files found in output directory: %s", output_dir)
            return ""

        video_files.sort()
        return video_files[0]

    def load_args(
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

        current_env["PYTHONIOENCODING"] = "utf-8"
        current_env["FASTVIDEO_ATTENTION_BACKEND"] = ""
        current_env["MODEL_BASE"] = model_path


        # fastvideo_local_path = os.path.abspath("custom_nodes/ComfyUI-FastVideo")
        # pythonpath = fastvideo_local_path + os.pathsep + current_env.get("PYTHONPATH", "")
        # current_env["PYTHONPATH"] = pythonpath

        if self.generator is None:
            self.generator = VideoGenerator.from_pretrained(
                model_path="FastVideo/FastHunyuan-diffusers",
                num_gpus=num_gpus,
                output_path=output_path,
                tp_size=tp_size,
                sp_size=sp_size,
            )

        self.generator.generate_video(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            seed=seed
        )

        output_path = os.path.join(output_path, f"{prompt[:100]}.mp4")
        return(output_path,)
