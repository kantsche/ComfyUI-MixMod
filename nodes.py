from .guider import MixModGuider


# COMPONENTS

class MixModGuiderComponentNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"model": ("MODEL",),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",), 
                     "base_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "cfg": ("FLOAT", {"default": 7.5, "min": -100.0, "max": 100.0, "step": 0.1}),
                    },
                "optional": 
                    {
                     "prev_component": ("COMPONENT",), 
                     "options": ("OPTIONS",),
                    }
                }

    RETURN_TYPES = ("COMPONENT",)
    FUNCTION = "create_component"
    CATEGORY = "mixmod/components"

    def create_component(self, model, positive, negative, base_weight, cfg, options=None, prev_component=None):
        component = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "weight": base_weight,
            "cfg": cfg,
            "options": options,
            "prev": prev_component
        }
        return (component,)


class MixModGuiderComponentPipelineNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"pipeline": ("PIPELINE",),
                     "base_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "cfg": ("FLOAT", {"default": 7.5, "min": -100.0, "max": 100.0, "step": 0.1}),
                     
                    },
                "optional": 
                    {
                     "prev_component": ("COMPONENT",), 
                     "options": ("OPTIONS",),
                    }
                }

    RETURN_TYPES = ("COMPONENT",)
    FUNCTION = "create_component"
    CATEGORY = "mixmod/components"

    def create_component(self, pipeline, base_weight, cfg, options=None, prev_component=None):
        component = {
            "model": pipeline["model"],
            "positive": pipeline["positive"],   
            "negative": pipeline["negative"],
            "weight": base_weight,
            "cfg": cfg,
            "options": options,
            "prev": prev_component,
        }
        return (component,)
    

class MixModPipelineNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"model": ("MODEL",),  
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                    }
                }
    
    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "create_pipeline"
    CATEGORY = "mixmod/components"

    def create_pipeline(self, model, positive, negative):
        pipeline = {
            "model": model,
            "positive": positive,
            "negative": negative,
        }
        return (pipeline,)

    
class MixModGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"component": ("COMPONENT",),
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "mixmod/components"

    def create_guider(self, component):
        # This is a placeholder for the actual guider implementation
        # The actual implementation will be filled in by the user
        guider = MixModGuider(component, mode="team")
        return (guider,)


# OPTIONS

class MixModOptionsUncondDecayNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"uncond_decay": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                "optional":{
                     "prev_options": ("OPTIONS",),
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    FUNCTION = "options"
    CATEGORY = "mixmod/options"

    def options(self, uncond_decay, prev_options=None): 
        options = {
            "uncond_decay": uncond_decay,
            "prev": prev_options
        }
        return (options,)


class MixModOptionsMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"mask": ("MASK",),
                    },
                "optional":{
                     "prev_options": ("OPTIONS",),
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    FUNCTION = "options"
    CATEGORY = "mixmod/options"

    def options(self, mask, prev_options=None):
        options = {
            "mask": mask,
            "prev": prev_options
        }
        return (options,)

class MixModOptionsSchedulerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                     "start_step": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                     "end_step": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
                    },
                "optional":{
                     "prev_options": ("OPTIONS",),
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    FUNCTION = "options"
    CATEGORY = "mixmod/options"    

    def options(self, start_step, end_step, prev_options=None):
        options = {
            "start_step": start_step,
            "end_step": end_step,
            "prev": prev_options
        }
        return (options,)


class MixModOptionsScaleNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                     "scale_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                "optional":{
                     "prev_options": ("OPTIONS",),
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    FUNCTION = "options"
    CATEGORY = "mixmod/options"

    def options(self, scale_factor, prev_options=None):
        options = {
            "scale_factor": scale_factor,
            "prev": prev_options
        }
        return (options,)

class MixModOptionsMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                     "mask": ("MASK",),
                    },
                "optional":{
                     "prev_options": ("OPTIONS",),
                    }
                }

    RETURN_TYPES = ("OPTIONS",)
    FUNCTION = "options"
    CATEGORY = "mixmod/options"

    def options(self, mask, prev_options=None):
        options = {
            "mask": mask,
            "prev": prev_options
        }
        return (options,)


# SPECIAL GUIDERS


class MixModStyleInjectionGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"component": ("COMPONENT",),
                     "image": ("IMAGE",),
                     "noise_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "start_step": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
                     "end_step": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "mixmod/special"

    def create_guider(self, component, image, noise_weight=1.0, start_step=4, end_step=-1):
        guider = MixModGuider(component, mode="styleinjection", image=image, noise_weight=noise_weight, start_step=start_step, end_step=end_step)
        return (guider,)
    
class MixModFFTGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"component": ("COMPONENT",),
                     "mode": (["teamfft", "2model_fft", "2m2f", "teamfft4freq", "fft_overlap", "fft_full"],),
                     "ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "mixmod/special"

    def create_guider(self, component, mode="team", ratio=1.0):
        # This is a placeholder for the actual guider implementation
        # The actual implementation will be filled in by the user
        guider = MixModGuider(component, mode=mode, ratio=ratio)
        return (guider,)


class MixModBandFFTGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"component": ("COMPONENT",),
                     "bands": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "mixmod/special"

    def create_guider(self, component, bands=2):
        # This is a placeholder for the actual guider implementation
        # The actual implementation will be filled in by the user
        guider = MixModGuider(component, mode="bandfft", bands=bands)
        return (guider,)
           

class MixModDynamicMaskGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"component": ("COMPONENT",),
                     "decay_factor": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 0.99, "step": 0.01}),
                     "threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
                     "blur_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "mixmod/special"

    def create_guider(self, component, decay_factor=0.9, threshold=0.1, blur_sigma=0.5):
        guider = MixModGuider(component, mode="dynamic_mask", decay_factor=decay_factor, threshold=threshold, blur_sigma=blur_sigma)
        return (guider,)

class MixModDynamicMaskAlternativeGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"component": ("COMPONENT",),
                     "threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
                     "blur_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                     "invert": ("BOOLEAN", {"default": False}), 
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "mixmod/special"

    def create_guider(self, component, threshold=0.1, blur_sigma=0.5, invert=False):
        guider = MixModGuider(component, mode="dynamic_mask_alternative", threshold=threshold, blur_sigma=blur_sigma, invert=invert)
        return (guider,)

class MixModDepthGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"component": ("COMPONENT",),
                     "sharpness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "start": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                    },
                "optional":
                    {
                     "invert": ("BOOLEAN", {"default": False}),
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "mixmod/special"

    def create_guider(self, component, sharpness=0.0, invert=False, start=0):
        guider = MixModGuider(component, mode="foreground_background", sharpness=sharpness, invert=invert, start=start)
        return (guider,)
    

class MixModHighResGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"component": ("COMPONENT",),
                     "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "start_step": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                     "end_step": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "mixmod/special"

    def create_guider(self, component, ratio=0.5, start_step=0, end_step=-1, mode="team", image=None):
        # Convert end_step = -1 to None for the wrapper function
        actual_end_step = None if end_step == -1 else end_step
        # This is a placeholder for the actual guider implementation
        # The actual implementation will be filled in by the user
        guider = MixModGuider(component, mode=mode, ratio=ratio, highres_mode=True, start_step=start_step, end_step=actual_end_step)
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "MixModGuiderComponentNode": MixModGuiderComponentNode,
    "MixModGuiderComponentPipelineNode": MixModGuiderComponentPipelineNode,
    "MixModGuiderNode": MixModGuiderNode,
    "MixModFFTGuiderNode": MixModFFTGuiderNode,
    "MixModBandFFTGuiderNode": MixModBandFFTGuiderNode,
    "MixModHighResGuiderNode": MixModHighResGuiderNode,
    "MixModOptionsMaskNode": MixModOptionsMaskNode,
    "MixModOptionsSchedulerNode": MixModOptionsSchedulerNode,
    "MixModOptionsScaleNode": MixModOptionsScaleNode,
    "MixModPipelineNode": MixModPipelineNode,
    "MixModDynamicMaskGuiderNode": MixModDynamicMaskGuiderNode,
    "MixModDynamicMaskAlternativeGuiderNode": MixModDynamicMaskAlternativeGuiderNode,
    "MixModDepthGuiderNode": MixModDepthGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MixModGuiderComponentNode": "MixMod Guider Component",
    "MixModGuiderComponentPipelineNode": "MixMod Guider Component Pipeline",
    "MixModGuiderNode": "MixMod Guider",
    "MixModFFTGuiderNode": "MixMod FFT Guider",
    "MixModBandFFTGuiderNode": "MixMod Band FFT Guider",
    "MixModHighResGuiderNode": "MixMod High Res Guider",
    "MixModOptionsMaskNode": "MixMod Options Mask",
    "MixModOptionsSchedulerNode": "MixMod Options Scheduler",
    "MixModOptionsScaleNode": "MixMod Options Scale",
    "MixModPipelineNode": "MixMod Pipeline",
    "MixModDynamicMaskGuiderNode": "MixMod Dynamic Mask Guider",
    "MixModDynamicMaskAlternativeGuiderNode": "MixMod Dynamic Mask Alternative Guider",
    "MixModDepthGuiderNode": "MixMod Depth Guider",
}