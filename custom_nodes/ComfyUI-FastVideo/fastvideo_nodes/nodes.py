from .inference_args import LoadFastVideoInferenceArgs
from .FastVideo_node import FastVideoSampler

NODE_CLASS_MAPPINGS = {
    "FastVideoSampler": FastVideoSampler,
    "LoadInferenceArgs": LoadFastVideoInferenceArgs
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastVideoSampler": "Fast Video Sampler",
    "LoadInferenceArgs": "Load FastVideo Inference Arguments"
}