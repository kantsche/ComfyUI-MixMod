import torch
import math
import comfy.samplers
import comfy.utils
import nodes
import logging

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
                    }
                }

    RETURN_TYPES = ("COMPONENT",)
    FUNCTION = "create_component"
    CATEGORY = "mixmod/conditioning"

    def create_component(self, model, positive, negative, base_weight, cfg, mask=None, prev_component=None):
        component = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "weight": base_weight,
            "cfg": cfg,
            "mask": mask,
            "prev": prev_component
        }
        return (component,)

class MixModGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"component": ("COMPONENT",),
                     "mode": (["team", "2model_fft"],),
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "mixmod/conditioning"

    def create_guider(self, component, mode, image=None,):
        # This is a placeholder for the actual guider implementation
        # The actual implementation will be filled in by the user
        guider = MixModGuider(component, mode,  image, component)
        return (guider,)
    

class MixModGuider(comfy.samplers.CFGGuider):
    def __init__(self, component, mode, vae, image=None, prev_guider=None):

        self.component = component
        self.mode = mode
        self.vae = vae
        self.image = image
        self.prev_guider = prev_guider
        self.original_conds = {}
        self.prepared_models = []
        # Initialize other necessary attributes
        self.models = []
        self.weights = []
        self.cfgs = []
        self.masks = []
        # Process component chain
        current = component
        conds = {}
        while current is not None:
            self.models.append(current["model"])
            if len(self.models) == 1:
                self.model_patcher = self.models[0]
                self.model_options = self.models[0].model_options
            self.weights.append(current["weight"])
            self.cfgs.append(current["cfg"])
            self.masks.append(current["mask"])
            
            # Add to original_conds with indexed keys
            idx = len(self.models)
            conds.update({f"positive_{idx}": current["positive"], f"negative_{idx}": current["negative"]})
            
            current = current.get("prev")
        
        self.inner_set_conds(conds)

        weight_sum = sum(self.weights)
        
        # Normalize weights if sum is not zero
        if weight_sum > 0:
            self.weights = [w / weight_sum for w in self.weights]
    
    def set_mode(self, mode):
        self.mode = mode



    def predict_noise(self, x, timestep, model_options={}, seed=None):
        pred = []
        for i in range(len(self.models)):
            model_options = self.models[i].model_options
            cond = [self.conds.get(f"positive_{i+1}", None), self.conds.get(f"negative_{i+1}", None)]
            
            pred.append(comfy.samplers.calc_cond_batch(self.prepared_models[i][0], cond, x, timestep, model_options))

            for fn in model_options.get("sampler_pre_cfg_function", []):
                args = {"conds":cond, "conds_out": out, "cond_scale": self.cfgs[i], "timestep": timestep,
                        "input": x, "sigma": timestep, "model": self.prepared_models[i][0], "model_options": model_options}
                out  = fn(args)

        return self.mix_function(pred, x, timestep)

    def mix_function(self, pred, x, timestep):

        if(self.mode == "team"):
            #            mix = (uncond_pred * ratio + baduncond * (1.0 - ratio)) + (cond_pred - uncond_pred) * cond_scale + (badcond - baduncond) * cond_scale_2
            # Use pred array with self.weights and self.cfgs
            # First, initialize with the unconditioned prediction from the first model
            mix = torch.zeros_like(pred[0][1])

            for i in range(len(pred)):
                mix = mix + pred[i][1] * self.weights[i]
            
            # For each model, apply the guidance according to weights and cfg values
            for i in range(len(pred)):
                # Extract conditional and unconditional predictions for this model
                c_pred = pred[i][0]
                u_pred = pred[i][1]
                
                # Apply the CFG formula with the corresponding weight and cfg
                # Add the weighted difference between conditional and unconditional
                mix = mix + (c_pred - u_pred) * self.cfgs[i]

        elif(self.mode == "2model_fft"):
            # Apply FFT-based frequency splitting
            # Need at least 2 models for FFT mode
            if len(pred) < 2:
                return pred[0][1]  # Return unconditioned prediction if only one model

            # Get shapes from first prediction
            _, _, h, w = pred[0][0].shape
            
            # Use the first model for low frequencies and second model for high frequencies
            # Extract predictions from first model
            c_pred_1 = pred[0][0]  # Conditioned prediction
            u_pred_1 = pred[0][1]  # Unconditioned prediction
            
            # Extract predictions from second model
            c_pred_2 = pred[1][0]  # Conditioned prediction
            u_pred_2 = pred[1][1]  # Unconditioned prediction
            
            # Convert to frequency domain
            fft_u_pred_1 = torch.fft.rfft2(u_pred_1)
            fft_c_pred_1 = torch.fft.rfft2(c_pred_1)
            fft_u_pred_2 = torch.fft.rfft2(u_pred_2)
            fft_c_pred_2 = torch.fft.rfft2(c_pred_2)
            
            # Use weight of first model as frequency split ratio
            ratio = abs(self.weights[0])
            
            # Calculate frequency split point
            split_h = int(h * ratio)
            split_w = int(w * ratio / 2)  # rfft2 returns half the width in frequency domain
            
            # Create masks for low and high frequencies
            mask_low = torch.ones_like(fft_u_pred_1, dtype=torch.float32)
            mask_high = torch.ones_like(fft_u_pred_1, dtype=torch.float32)
            
            # Set high frequencies to zero in low mask
            mask_low[:, :, split_h:, :] = 0
            mask_low[:, :, :, split_w:] = 0
            
            # Set low frequencies to zero in high mask
            mask_high[:, :, :split_h, :split_w] = 0
            
            # Apply masks and compute CFG in frequency domain using the corresponding CFG values
            fft_result_low = (fft_u_pred_1 * mask_low) + ((fft_c_pred_1 - fft_u_pred_1) * self.cfgs[0] * mask_low)
            fft_result_high = (fft_u_pred_2 * mask_high) + ((fft_c_pred_2 - fft_u_pred_2) * self.cfgs[1] * mask_high)
            
            # Combine low and high frequency results
            fft_result = fft_result_low + fft_result_high
            
            # Convert back to spatial domain
            mix = torch.fft.irfft2(fft_result, s=(h, w))

        return mix
    
    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        # Get the device from the first model
        device = self.models[0].load_device
        
        # Prepare the models, conditions, and loaded models for each model
        prepared_models = []
        all_loaded_models = []
        
        for i, model in enumerate(self.models):
            # Get relevant conditions for this model
            model_conds = {k: v for k, v in self.conds.items() if k.endswith(f"_{i+1}")}
            
            # Prepare sampling for this model
            inner_model, prepared_conds, loaded_models = comfy.sampler_helpers.prepare_sampling(model, noise.shape, model_conds, model.model_options)
            
            # Save the prepared data
            self.prepared_models.append((inner_model, prepared_conds))
            
            all_loaded_models.extend(loaded_models)
            
            # Update the conditions for this model
            for k, v in prepared_conds.items():
                self.conds[k] = v

        self.inner_model = self.prepared_models[0][0]
        # Prepare the device-related tensors
        if denoise_mask is not None:
            denoise_mask = comfy.sampler_helpers.prepare_mask(denoise_mask, noise.shape, device)

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)
        
        # Set model options for the main model patcher
        comfy.samplers.cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())
        
        try:
            # Pre-run all models
            for model in self.models:
                model.pre_run()
                
            # Set the inner_model to be a list of prepared models
            
            # Run the inner sample with prepared models
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            # Clean up all models
            for model in self.models:
                model.cleanup()
            del self.inner_model
            # Clean up models and loaded models
            comfy.sampler_helpers.cleanup_models(self.conds, all_loaded_models)
            
            
        return output




NODE_CLASS_MAPPINGS = {
    "MixModGuiderComponentNode": MixModGuiderComponentNode,
    "MixModGuiderNode": MixModGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MixModGuiderComponentNode": "MixMod Guider Component",
    "MixModGuiderNode": "MixMod Guider",
}