class LoadFastVideoInferenceArgs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",
                           {"default": "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest. The playful yet serene atmosphere is complemented by soft natural light filtering through the petals. Mid-shot, warm and cheerful tones."}),
                "output_path": ("STRING", {"default": "/workspace/ComfyUI/outputs_video/"}),
                "height": ("INT", {"default": 720}),
                "width": ("INT", {"default": 1280}),
                "num_frames": ("INT", {"default": 45}),
                "num_inference_steps": ("INT", {"default": 6}),
                "guidance_scale": ("FLOAT", {"default": 1.0}),
                "flow_shift": ("INT", {"default": 17}),
                "seed": ("INT", {"default": 1024}),
                "fps": ("INT", {"default": 24}),
            }
        }

    RETURN_TYPES = ("INFERENCE_ARGS",)
    RETURN_NAMES = ("inference_args",)
    FUNCTION = "load_args"
    CATEGORY = "fastvideo"

    def load_args(
        self,
        prompt,
        output_path,
        height,
        width,
        num_frames,
        num_inference_steps,
        guidance_scale,
        flow_shift,
        seed,
        fps,
    ):
        args = {
            "prompt": prompt,
            "output_path": output_path,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "seed": seed,
            "fps": fps,
        }
        print("BIG ARGS1", args)
        return(args,)

# # Register the custom node
# NODE_CLASS_MAPPINGS = {
#     "LoadFastVideoInferenceArgs": LoadFastVideoInferenceArgs,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "LoadFastVideoInferenceArgs": "Load FastVideo Inference Arguments",
# }
