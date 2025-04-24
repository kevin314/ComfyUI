class VAEConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "scale_factor": ("INT", {"default": 8}),
                "sp": ("INT", {"default": True}),
                "tiling": ("BOOLEAN", {"default": True}),
                "precision": ("STRING", {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("VAE_CONFIG",)
    RETURN_NAMES = ("vae_config",)
    FUNCTION = "set_args"
    CATEGORY = "fastvideo"

    def set_args(
        self,
        scale_factor,
        sp,
        tiling,
        precision
    ):
        args = {
            "scale_factor": scale_factor,
            "sp": sp,
            "tiling": tiling,
            "precision": precision,
        }
        return(args,)
