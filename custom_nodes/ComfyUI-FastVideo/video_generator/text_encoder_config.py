class TextEncoderConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "precision": ("STRING", {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("TEXT_ENCODER_CONFIG",)
    RETURN_NAMES = ("text_encoder_config",)
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
