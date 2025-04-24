class DITConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "precision": ("STRING", {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("DIT_CONFIG",)
    RETURN_NAMES = ("dit_config",)
    FUNCTION = "set_args"
    CATEGORY = "fastvideo"

    def set_args(
        self,
        precision
    ):
        args = {
            "precision": precision,
        }
        return(args,)
