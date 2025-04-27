class VAEConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "scale_factor": ("INT", {"default": 8}),
                "sp": ("INT", {"default": True}),
                "tiling": ([True, False], {"default": True}),
                "precision": (["fp16", "bf16"], {"default": "fp16"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, scale_factor=None, **kwargs):
        # Handle None value for scale_factor
        if scale_factor is None:
            # This is valid - we'll use the default value in the set_args method
            return True
        
        # For non-None values, ensure it's a valid integer
        try:
            int(scale_factor)
            return True
        except (ValueError, TypeError):
            return f"scale_factor must be an integer, got {type(scale_factor).__name__}"

    RETURN_TYPES = ("VAE_CONFIG",)
    RETURN_NAMES = ("vae_config",)
    FUNCTION = "set_args"
    CATEGORY = "fastvideo"

    def set_args(
        self,
        scale_factor=None,
        sp=None,
        tiling=None,
        precision=None
    ):
        # Use default values if None or "auto" is provided
        if scale_factor == -99999:
            scale_factor = None
        
        args = {
            "scale_factor": scale_factor,
            "sp": sp if sp is not None else True,
            "tiling": tiling if tiling is not None else True,
            "precision": precision,
        }
        print('vae args', args)
        return(args,)
