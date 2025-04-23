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
                     
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "mixmod/conditioning"

    def create_guider(self, component, image=None,):
        # This is a placeholder for the actual guider implementation
        # The actual implementation will be filled in by the user
        guider = MixModGuider(component, mode="team", image=image)
        return (guider,)    

class MixModFFTGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"component": ("COMPONENT",),
                     "mode": (["teamfft", "2model_fft"],),
                     "ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "mixmod/conditioning"

    def create_guider(self, component, mode="team", image=None, ratio=1.0):
        # This is a placeholder for the actual guider implementation
        # The actual implementation will be filled in by the user
        guider = MixModGuider(component, mode=mode, image=image, ratio=ratio)
        return (guider,)
    

class MixModGuider(comfy.samplers.CFGGuider):
    def __init__(self, component, mode="team", vae=None, image=None, prev_guider=None, ratio=1.0):

        self.component = component
        self.mode = mode
        self.vae = vae
        self.image = image
        self.prev_guider = prev_guider
        self.ratio = ratio
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
            cond = [self.conds.get(f"positive_{i+1}", None), self.conds.get(f"negative_{i+1}", None)]
            
            pred.append(comfy.samplers.calc_cond_batch(self.prepared_models[i][0], cond, x, timestep, model_options))

            for fn in model_options.get("sampler_pre_cfg_function", []):
                args = {"conds":cond, "conds_out": pred, "cond_scale": self.cfgs[i], "timestep": timestep,
                        "input": x, "sigma": timestep, "model": self.prepared_models[i][0], "model_options": model_options}
                pred[i]  = fn(args)

        return self.mix_function(self.prepared_models[0][0], pred, x, timestep, model_options)

    def mix_function(self, model, pred, x, timestep, model_options):
        
        if(self.mode == "team"):
            #            mix = (uncond_pred * ratio + baduncond * (1.0 - ratio)) + (cond_pred - uncond_pred) * cond_scale + (badcond - baduncond) * cond_scale_2
            # Use pred array with self.weights and self.cfgs
            # First, initialize with the unconditioned prediction from the first model
            mix = torch.zeros_like(pred[0][1])
            c_predf = torch.zeros_like(pred[0][0])
            u_predf = torch.zeros_like(pred[0][1])

            for i in range(len(pred)):
                mix = mix + pred[i][1] * self.weights[i]
            
            # For each model, apply the guidance according to weights and cfg values
            for i in range(len(pred)):
                # Extract conditional and unconditional predictions for this model
                c_pred = pred[i][0]
                u_pred = pred[i][1]

                c_predf = c_predf + c_pred
                u_predf = u_predf + u_pred
                
                # Apply the CFG formula with the corresponding weight and cfg
                # Add the weighted difference between conditional and unconditional
                mix = mix + (c_pred - u_pred) * self.cfgs[i]


            for fn in model_options.get("sampler_post_cfg_function", []):
                args = {"denoised": mix, "cond": self.conds.get(f"positive_1", None), "uncond": self.conds.get(f"negative_1", None), "cond_scale": self.cfgs[0], "model": model, "uncond_denoised": pred[0][1], "cond_denoised": pred[0][0],
                        "sigma": timestep, "model_options": model_options, "input": x}
                mix = fn(args)

        if(self.mode == "teamfft"):
            #            mix = (uncond_pred * ratio + baduncond * (1.0 - ratio)) + (cond_pred - uncond_pred) * cond_scale + (badcond - baduncond) * cond_scale_2
            # Use pred array with self.weights and self.cfgs
            # First, get shape information from the first model's predictions
            _, _, h, w = pred[0][0].shape
            
            # Initialize an empty tensor in frequency domain for the result
            fft_mix = torch.zeros_like(torch.fft.rfft2(pred[0][1]))
            
            # First loop: mix the unconditioned predictions in frequency domain
            for i in range(len(pred)):
                # Convert unconditional prediction to frequency domain
                fft_u_pred = torch.fft.rfft2(pred[i][1])
                # Add weighted contribution
                fft_mix = fft_mix + fft_u_pred * self.weights[i]
            
            # Second loop: apply the guidance in frequency domain
            for i in range(len(pred)):
                # Convert conditional and unconditional predictions to frequency domain
                fft_c_pred = torch.fft.rfft2(pred[i][0])
                fft_u_pred = torch.fft.rfft2(pred[i][1])
                
                # Apply the CFG formula in frequency domain
                fft_mix = fft_mix + (fft_c_pred - fft_u_pred) * self.cfgs[i]
            
            # Convert the final result back to spatial domain
            mix = torch.fft.irfft2(fft_mix, s=(h, w))

        if(self.mode == "teamfft4freq"):
            # Get shape information from the first model's predictions
            _, _, h, w = pred[0][0].shape
            
            # Initialize an empty tensor in frequency domain for the result
            fft_mix = torch.zeros_like(torch.fft.rfft2(pred[0][1]))
            
            # Get frequency domain dimensions
            freq_h, freq_w = fft_mix.shape[2], fft_mix.shape[3]
            
            # Create frequency band masks - dividing into 4 frequency bands
            # Low, low-mid, high-mid, and high frequencies
            masks = []
            
            # Calculate coordinates for frequency space
            h_coords = torch.linspace(0, 1, freq_h, device=fft_mix.device)
            w_coords = torch.linspace(0, 1, freq_w, device=fft_mix.device)
            h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')
            
            # Calculate frequency magnitude (distance from DC component)
            freq_magnitude = torch.sqrt(h_grid**2 + w_grid**2)
            
            # Normalize to [0,1] range
            max_freq = torch.sqrt(torch.tensor(2.0, device=fft_mix.device))
            normalized_freq = freq_magnitude / max_freq
            
            # Define frequency band boundaries
            band_boundaries = [0.0, 0.25, 0.5, 0.75, 1.0]
            
            # Create masks for each frequency band with smooth transitions
            transition_width = 0.05  # 5% transition width
            
            for i in range(4):  # 4 frequency bands
                band_start = band_boundaries[i]
                band_end = band_boundaries[i+1]
                
                # Calculate transition regions
                trans_start = max(0.0, band_start - transition_width/2)
                trans_end = min(1.0, band_end + transition_width/2)
                
                # Create mask
                mask = torch.zeros_like(normalized_freq)
                
                # Full strength in the main band region
                main_region = (normalized_freq >= band_start) & (normalized_freq <= band_end)
                mask = torch.where(main_region, torch.ones_like(mask), mask)
                
                # Smooth transition at the start of the band
                if band_start > 0:
                    start_trans = (normalized_freq >= trans_start) & (normalized_freq < band_start)
                    mask = torch.where(
                        start_trans,
                        (normalized_freq - trans_start) / (band_start - trans_start),
                        mask
                    )
                
                # Smooth transition at the end of the band
                if band_end < 1.0:
                    end_trans = (normalized_freq > band_end) & (normalized_freq <= trans_end)
                    mask = torch.where(
                        end_trans,
                        1.0 - (normalized_freq - band_end) / (trans_end - band_end),
                        mask
                    )
                
                masks.append(mask)
            
            # Ensure masks sum to 1.0 at each point for perfect reconstruction
            mask_sum = sum(masks)
            masks = [m / (mask_sum + 1e-8) for m in masks]  # Add small epsilon to avoid division by zero
            
            # Process each frequency band separately
            for band_idx in range(4):
                band_mix = torch.zeros_like(fft_mix)
                
                # First loop: mix the unconditioned predictions in this frequency band
                for i in range(len(pred)):
                    # Convert unconditional prediction to frequency domain
                    fft_u_pred = torch.fft.rfft2(pred[i][1])
                    # Add weighted contribution for this frequency band
                    band_mix = band_mix + fft_u_pred * self.weights[i] * masks[band_idx]
                
                # Second loop: apply the guidance in this frequency band
                for i in range(len(pred)):
                    # Convert conditional and unconditional predictions to frequency domain
                    fft_c_pred = torch.fft.rfft2(pred[i][0])
                    fft_u_pred = torch.fft.rfft2(pred[i][1])
                    
                    # Apply the CFG formula in this frequency band
                    band_mix = band_mix + (fft_c_pred - fft_u_pred) * self.cfgs[i] * masks[band_idx]
                
                # Add this band's contribution to the final mix
                fft_mix = fft_mix + band_mix
            
            # Convert the final result back to spatial domain
            mix = torch.fft.irfft2(fft_mix, s=(h, w))

        elif(self.mode == "2model_fft"):
            # Apply FFT-based frequency splitting
            # Need at least 2 models for FFT mode
            if len(pred) < 2:
                return pred[0][0] * self.cfgs[0] + pred[0][1] * (1 - self.cfgs[0])  # Return properly conditioned prediction
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
            # Ensure ratio is between 0.05 and 0.95 to guarantee some frequency coverage for both models
            ratio = max(0.05, min(0.95, abs(self.ratio)))
            
            # Calculate frequency split point
            split_h = int(h * ratio)
            split_w = int(w * ratio / 2)  # rfft2 returns half the width in frequency domain
            
            # Create masks for low and high frequencies
            mask_low = torch.ones_like(fft_u_pred_1, dtype=torch.float32)
            mask_high = torch.ones_like(fft_u_pred_1, dtype=torch.float32)
            
            # Create transition area for smooth blending
            transition_h = max(1, int(h * 0.05))  # 5% transition zone
            transition_w = max(1, int(w * 0.05 / 2))  # 5% transition zone
            
            # Set high frequencies to zero in low mask with smooth transition
            for i in range(mask_low.shape[2]):
                for j in range(mask_low.shape[3]):
                    if i >= split_h + transition_h or j >= split_w + transition_w:
                        mask_low[:, :, i, j] = 0.0
                    elif i >= split_h or j >= split_w:
                        # Calculate smooth transition value
                        fade_h = 1.0 - min(1.0, (i - split_h) / transition_h) if i >= split_h else 1.0
                        fade_w = 1.0 - min(1.0, (j - split_w) / transition_w) if j >= split_w else 1.0
                        mask_low[:, :, i, j] = fade_h * fade_w
            
            # Set low frequencies to zero in high mask with inverse smooth transition
            for i in range(mask_high.shape[2]):
                for j in range(mask_high.shape[3]):
                    if i <= split_h - transition_h and j <= split_w - transition_w:
                        mask_high[:, :, i, j] = 0.0
                    elif i <= split_h or j <= split_w:
                        # Calculate smooth transition value - inverse of the low mask
                        fade_h = min(1.0, (split_h - i) / transition_h) if i <= split_h else 0.0
                        fade_w = min(1.0, (split_w - j) / transition_w) if j <= split_w else 0.0
                        mask_high[:, :, i, j] = 1.0 - (fade_h * fade_w)
            
            # Ensure masks sum to 1.0 at each point for full spectrum coverage
            mask_sum = mask_low + mask_high
            mask_low = mask_low / mask_sum
            mask_high = mask_high / mask_sum
            
            # Apply masks and compute CFG in frequency domain using the corresponding CFG values
            fft_result_low = (fft_u_pred_1 * mask_low) + ((fft_c_pred_1 - fft_u_pred_1) * self.cfgs[0] * mask_low)
            fft_result_high = (fft_u_pred_2 * mask_high) + ((fft_c_pred_2 - fft_u_pred_2) * self.cfgs[1] * mask_high)
            
            # Combine low and high frequency results
            fft_result = fft_result_low + fft_result_high
            
            # Convert back to spatial domain
            mix = torch.fft.irfft2(fft_result, s=(h, w))

        elif(self.mode == "fft_overlap"):
            # Split each model into its own frequency band with overlap - optimized version
            _, _, h, w = pred[0][0].shape
            
            # Initialize result in frequency domain
            fft_result = torch.zeros_like(torch.fft.rfft2(pred[0][0]))
            
            # Convert all predictions to frequency domain
            fft_c_preds = []
            fft_u_preds = []
            valid_weights = []
            valid_cfgs = []
            
            for i in range(len(pred)):
                if self.weights[i] <= 0:
                    continue
                
                # Extract predictions for this model
                c_pred = pred[i][0]
                u_pred = pred[i][1]
                
                # Convert to frequency domain
                fft_c_preds.append(torch.fft.rfft2(c_pred))
                fft_u_preds.append(torch.fft.rfft2(u_pred))
                valid_weights.append(self.weights[i])
                valid_cfgs.append(self.cfgs[i])
            
            # Get number of valid models
            num_models = len(valid_weights)
            if num_models == 0:
                # Handle edge case with no valid models
                return pred[0][0]  # Return first model's conditioned output
            elif num_models == 1:
                # With only one model, use it for the full spectrum
                mix = pred[0][0] * valid_cfgs[0] + pred[0][1] * (1 - valid_cfgs[0])
                return mix
            
            # Calculate frequency coordinates
            freq_h, freq_w = fft_result.shape[2], fft_result.shape[3]
            
            # Create normalized coordinate grids - vectorized
            h_coords = torch.linspace(0, 1, freq_h, device=fft_result.device)
            w_coords = torch.linspace(0, 1, freq_w, device=fft_result.device)
            
            # Create meshgrid of coordinates
            h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')
            
            # Calculate frequency magnitude for all points
            freq_magnitude = torch.sqrt(h_grid**2 + w_grid**2)
            
            # Normalize frequency magnitude to range [0, 1]
            max_freq = torch.sqrt(torch.tensor(2.0, device=fft_result.device))  # Maximum possible frequency magnitude
            normalized_freq = freq_magnitude / max_freq
            
            # Define overlap ratio (0.2 means 20% overlap between bands)
            overlap_ratio = 0.2
            
            # Calculate band width for each model, accounting for overlap
            band_width = (1.0 + overlap_ratio) / num_models
            
            # Calculate frequency masks for each model - vectorized
            model_masks = []
            for i in range(num_models):
                # Calculate band start and end points
                band_start = max(0.0, i * band_width - (overlap_ratio * band_width / 2))
                band_end = min(1.0, (i + 1) * band_width + (overlap_ratio * band_width / 2))
                
                # Create initial mask where freq is within the band
                mask = torch.zeros_like(normalized_freq)
                in_band = (normalized_freq >= band_start) & (normalized_freq <= band_end)
                
                # Define transition regions
                start_trans = band_start + (overlap_ratio * band_width / 2)
                end_trans = band_end - (overlap_ratio * band_width / 2)
                
                # Create mask for different regions
                full_region = (normalized_freq >= start_trans) & (normalized_freq <= end_trans)
                start_region = (normalized_freq >= band_start) & (normalized_freq < start_trans)
                end_region = (normalized_freq > end_trans) & (normalized_freq <= band_end)
                
                # Set full strength in middle region
                mask[full_region] = 1.0
                
                # Calculate and set smooth transitions at edges
                if torch.any(start_region):
                    trans_width = start_trans - band_start
                    mask[start_region] = (normalized_freq[start_region] - band_start) / trans_width
                
                if torch.any(end_region):
                    trans_width = band_end - end_trans
                    mask[end_region] = (band_end - normalized_freq[end_region]) / trans_width
                
                model_masks.append(mask)
            
            # Stack masks for easier normalization
            stacked_masks = torch.stack(model_masks)
            
            # Normalize masks to ensure they sum to 1.0 across all models
            sum_masks = torch.sum(stacked_masks, dim=0)
            sum_masks = torch.clamp(sum_masks, min=1e-10)
            
            # Normalize each mask
            normalized_stacked_masks = stacked_masks / sum_masks
            
            # Apply CFG to each frequency band and combine - vectorized
            for i in range(num_models):
                # Get mask, add batch and channel dimensions
                mask = normalized_stacked_masks[i].unsqueeze(0).unsqueeze(0)
                
                # Apply CFG formula with frequency band mask
                fft_u = fft_u_preds[i]
                fft_c = fft_c_preds[i]
                cfg_scale = valid_cfgs[i]
                
                # Add weighted contribution to result
                fft_result += (fft_u + (fft_c - fft_u) * cfg_scale) * mask
            
            # Convert back to spatial domain
            mix = torch.fft.irfft2(fft_result, s=(h, w))

        elif(self.mode == "fft_full"):
            # Full spectrum frequency mixing implementation - optimized version
            _, _, h, w = pred[0][0].shape
            
            # Initialize result in frequency domain
            fft_result = torch.zeros_like(torch.fft.rfft2(pred[0][0]))
            
            # First, convert all predictions to frequency domain
            fft_c_preds = []
            fft_u_preds = []
            valid_model_indices = []
            
            for i in range(len(pred)):
                if self.weights[i] <= 0:
                    # Add placeholders for zero-weight models
                    fft_c_preds.append(None)
                    fft_u_preds.append(None)
                    continue
                
                # Extract predictions for this model
                c_pred = pred[i][0]
                u_pred = pred[i][1]
                
                # Convert to frequency domain
                fft_c_preds.append(torch.fft.rfft2(c_pred))
                fft_u_preds.append(torch.fft.rfft2(u_pred))
                valid_model_indices.append(i)
            
            # Calculate frequency coordinates
            frequency_h, frequency_w = fft_result.shape[2], fft_result.shape[3]
            
            # Create normalized coordinate grids (vectorized)
            h_coords = torch.arange(frequency_h, device=fft_result.device).float() / frequency_h
            w_coords = torch.arange(frequency_w, device=fft_result.device).float() / frequency_w
            
            # Create meshgrid of coordinates
            h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')
            
            # Calculate frequency magnitude for all points at once
            freq_magnitude = torch.sqrt(h_grid**2 + w_grid**2)
            
            # Process each valid model
            if valid_model_indices:
                # Pre-calculate model weights for all frequency points
                all_model_weights = []
                for i in valid_model_indices:
                    # Calculate model influence: higher weights → lower frequencies, lower weights → higher frequencies
                    model_weight = self.weights[i] * (1.0 - freq_magnitude) + (1.0 - self.weights[i]) * freq_magnitude
                    all_model_weights.append(model_weight)
                
                # Convert to tensor for easier operations
                all_model_weights = torch.stack(all_model_weights)
                
                # Normalize weights along model dimension
                weight_sum = all_model_weights.sum(dim=0, keepdim=True)
                # Avoid division by zero
                weight_sum = torch.clamp(weight_sum, min=1e-10)
                normalized_weights = all_model_weights / weight_sum
                
                # Apply weights and CFG to each frequency component
                for idx, i in enumerate(valid_model_indices):
                    # Get model weights for this model
                    model_weights = normalized_weights[idx]
                    
                    # Apply weights to each frequency point (broadcasting)
                    model_weights = model_weights.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                    
                    # Get unconditioned and conditioned predictions
                    fft_u = fft_u_preds[i]
                    fft_c = fft_c_preds[i]
                    
                    # Apply CFG with model-specific weight
                    contrib = fft_u * model_weights + (fft_c - fft_u) * self.cfgs[i] * model_weights
                    
                    # Add to result
                    fft_result += contrib
            
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
    "MixModFFTGuiderNode": MixModFFTGuiderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MixModGuiderComponentNode": "MixMod Guider Component",
    "MixModGuiderNode": "MixMod Guider",
    "MixModFFTGuiderNode": "MixMod FFT Guider",
}
