import torch
import comfy.samplers
import comfy.sampler_helpers
import comfy.utils
import logging
import torch.nn.functional as F
from .utils import sumweights

# Initialize other necessary attributes

import comfy.model_management as model_management
from .utils import load_taesd, ensure_model, create_adaptive_low_res_model_wrapper

# Import mixing functions
from .mixes.fft import mix_teamfft, mix_teamfft4freq, mix_2model_fft, mix_2m2f, mix_fft_overlap, mix_fft_full, mix_bandfft
from .mixes.standard import mix_standard, mix_masked, mix_dynamic_mask, mix_foreground_background, mix_dynamic_mask_alternative


class MixModGuider(comfy.samplers.CFGGuider):
    def __init__(self, component, **kwargs):

        self.component = component
        self.mode = kwargs.get("mode", "team")
        self.ratio = kwargs.get("ratio", 1.0)
        self.original_conds = {}
        self.prepared_models = []
        self.highres_mode = kwargs.get("highres_mode", False)
        self.bands = kwargs.get("bands", None)
        self.kwargs = kwargs
        self.last_latent = None

        # Load the depth model
        if(self.mode == "foreground_background"):
            #check if controlnet_aux is installed
            try:
                from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.depth_anything import DepthAnythingDetector
                self.depth_model = DepthAnythingDetector.from_pretrained().to(model_management.get_torch_device())
                self.vae = load_taesd("taesdxl")
            except Exception as e:
                self.mode = "team"
                raise Exception("Please install Comfyui ControlNet Aux custom node. Falling back to team mode.")
            
            
        self.models = []
        self.weights = []
        self.cfgs = []
        self.masks = []  # Initialize empty masks list
        self.types = []

        self.step = 0
        self.total_steps = 0
        self.start_steps = []
        self.end_steps = []
        
        self.options = []
        self.scale_factors = []
        self.threshold = kwargs.get("threshold", 0.1)  # Dynamic mask threshold

        self.active_masks = []
        self.active_weights = []
        self.active_cfgs = []


        self.foreground_background_masks = []
        self.mask_history = None  # For tracking dynamic masks between steps

        self.seed = -1
        # Process component chain
        current = component
        conds = {}
        while current is not None:
            self.models.append(current.get("model"))
            if len(self.models) == 1:
                self.model_patcher = self.models[0]
                self.model_options = self.models[0].model_options
            self.weights.append(current.get("weight", 1.0))
            self.cfgs.append(current.get("cfg", 7.5))
            self.types.append(current.get("type", "eps"))
            opt = current.get("options", {})
            options = {}
            while opt is not None:
                options.update(opt)
                opt = opt.get("prev")
            self.scale_factors.append(options.get("scale_factor", 0.0))
            self.start_steps.append(options.get("start_step", 0))
            self.end_steps.append(options.get("end_step", -1))
            self.masks.append(options.get("mask", None))
            # Add to original_conds with indexed keys
            idx = len(self.models)
            conds.update({f"positive_{idx}": current["positive"], f"negative_{idx}": current["negative"]})
            
            current = current.get("prev")


        print(self.models)
        print(self.weights)
        print(self.cfgs)
        print(self.types)
        print(self.scale_factors)
        print(self.start_steps)
        print(self.end_steps)
        
        
        self.inner_set_conds(conds)
        
        
       
    
    def set_mode(self, mode):
        self.mode = mode

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        pred = []
        self.active_masks = []
        self.active_weights = []
        self.active_cfgs = []
        self.active_types = []
        for i in range(len(self.models)):
            cond = [self.conds.get(f"positive_{i+1}", None), self.conds.get(f"negative_{i+1}", None)]
            if(self.start_steps[i] <= self.step <= self.end_steps[i] or self.end_steps[i] == -1):
                if(self.scale_factors[i] > 0):
                    model_options["model_function_wrapper"] = create_adaptive_low_res_model_wrapper(
                        min_scale_factor=self.scale_factors[i], 
                        max_scale_factor=self.scale_factors[i]
                    )
                    ensure_model(self.models[i],"normal")
                    pred.append(comfy.samplers.calc_cond_batch(self.prepared_models[i][0], cond, x, timestep, model_options))
                    del model_options["model_function_wrapper"]
                else:
                    ensure_model(self.models[i],"normal")

                    pred.append(comfy.samplers.calc_cond_batch(self.prepared_models[i][0], cond, x, timestep, model_options))
                
                self.active_masks.append(self.masks[i])
                self.active_weights.append(self.weights[i])
                self.active_cfgs.append(self.cfgs[i])
                self.active_types.append(self.types[i])
        if not pred:
            logging.warning(f"No models were active at step {self.step}, using default model 0")
            cond = [self.conds.get(f"positive_1", None), self.conds.get(f"negative_1", None)]
            pred.append(comfy.samplers.calc_cond_batch(self.prepared_models[0][0], cond, x, timestep, model_options))
            self.active_masks.append(self.masks[0])
            self.active_weights.append(self.weights[0])
            self.active_cfgs.append(self.cfgs[0])
            self.active_types.append(self.types[0])
        return self.mix_function(self.prepared_models[0][0], pred, x, timestep, model_options, seed)

    def mix_function(self, model, pred, x, timestep, model_options, seed):
        haspostfunctions = False
        
        postuncond = torch.zeros_like(pred[0][1])
        postcond = torch.zeros_like(pred[0][0])

        
        self.active_weights = sumweights(self.active_weights)
        if(self.mode == "team"):
            if any(mask is not None for mask in self.active_masks):
                mix, postuncond = mix_masked(pred, self.active_weights, self.active_cfgs, self.active_masks,)
            else:
                mix, postuncond = mix_standard(pred, self.active_weights, self.active_cfgs)

        elif self.mode == "dynamic_mask":
            mix, postuncond, new_masks = mix_dynamic_mask(pred, self.active_weights, self.active_cfgs, self.step, self.mask_history, self.kwargs.get("decay_factor", 0.9), self.threshold, self.kwargs.get("blur_sigma", 0.5))
            self.mask_history = new_masks  # Store the updated masks for next step

        elif self.mode == "dynamic_mask_alternative":
            mix, postuncond = mix_dynamic_mask_alternative(pred, self.active_weights, self.active_cfgs, self.depth_model, self.kwargs.get("threshold", 0.1), self.kwargs.get("invert", False), self.kwargs.get("blur_sigma", 1.0))

        elif self.mode == "foreground_background":
            if(self.step == 0):
                self.foreground_background_masks = []

            masks, mix, postuncond = mix_foreground_background(pred, self.active_weights, self.last_latent, self.active_cfgs, vae=self.vae, depth_model=self.depth_model, sharpness=self.kwargs.get("sharpness", 0.0), invert=self.kwargs.get("invert", False), step=self.step, masks=self.foreground_background_masks, startmask=self.kwargs.get("startmask", 5))

            if any(mask is not None for mask in masks):
                self.foreground_background_masks = masks

        elif self.mode == "bandfft":
            mix, postuncond = mix_bandfft(pred, self.active_weights, self.active_cfgs, self.bands)
        
        elif self.mode == "2m2f":
            mix, postuncond = mix_2m2f(pred, self.active_weights, self.active_cfgs, self.ratio)
        
        elif self.mode == "teamfft":
            mix, postuncond = mix_teamfft(pred, self.active_weights, self.active_cfgs)
                
        elif self.mode == "teamfft4freq":
            mix, postuncond = mix_teamfft4freq(pred, self.active_weights, self.active_cfgs)
                
        elif self.mode == "2model_fft":
            mix, postuncond = mix_2model_fft(pred, self.active_weights, self.active_cfgs, self.ratio)
                
        elif self.mode == "fft_overlap":
            mix, postuncond = mix_fft_overlap(pred, self.active_weights, self.active_cfgs, self.ratio)
                
        elif self.mode == "fft_full":
            mix, postuncond = mix_fft_full(pred, self.active_weights, self.active_cfgs)
        else:
            # Fallback to standard mixing if mode is not recognized
            mix, postuncond = mix_standard(pred, self.active_weights, self.active_cfgs)

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {"denoised": mix, "cond": self.conds.get(f"positive_1", None), "uncond": self.conds.get(f"negative_1", None), "cond_scale": self.active_cfgs[0], "model": model, "uncond_denoised": postuncond, "cond_denoised": postcond,
                    "sigma": timestep, "model_options": model_options, "input": x}
            mix = fn(args)
        self.step += 1
        self.seed = seed
        return mix
    
    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        # Get the device from the first model
        device = self.models[0].load_device
        
        # Prepare the models, conditions, and loaded models for each model

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
            # Clean up models and loaded models
            comfy.sampler_helpers.cleanup_models(self.conds, all_loaded_models)
            self.prepared_models = []
            del all_loaded_models
            
            
        return output

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed):
        # Create a wrapper callback that captures intermediate denoised results (x0)
        original_callback = callback
        
        def wrapped_callback(step, x0, x, total_steps):
            # Store the current x0 (denoised latent)
            self.step = step
            self.total_steps = total_steps
            self.last_latent = x0.detach().clone()
            # Call the original callback if provided
            if original_callback is not None:
                original_callback(step, x0, x, total_steps)

        if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        # Process each cond with each prepared inner model separately
        processed_conds = {}
        for i, (inner_model, _) in enumerate(self.prepared_models):
            # Get the conditions for this model (positive_i, negative_i)
            model_conds = {k: v for k, v in self.conds.items() if k.endswith(f"_{i+1}")}
            
            # Process conditions with this model
            processed_model_conds = comfy.samplers.process_conds(inner_model, noise, model_conds, device, latent_image, denoise_mask, seed)
            
            # Add to processed conditions
            processed_conds.update(processed_model_conds)
        
        # Merge back into self.conds
        self.conds = processed_conds

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
        )
        samples = executor.execute(self, sigmas, extra_args, wrapped_callback, noise, latent_image, denoise_mask, disable_pbar)
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        if sigmas.shape[-1] == 0:
            return latent_image

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
        comfy.samplers.preprocess_conds_hooks(self.conds)

        try:
            orig_model_options = self.model_options
            self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
            orig_hook_mode = self.model_patcher.hook_mode
            if comfy.samplers.get_total_hook_groups_in_conds(self.conds) <= 1:
                self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
                
            comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
            
            comfy.samplers.filter_registered_hooks_on_conds(self.conds, self.model_options)
            executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self.outer_sample,
                self,
                comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
            )
            output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            comfy.samplers.cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.model_options = orig_model_options
            self.model_patcher.hook_mode = orig_hook_mode
            self.model_patcher.restore_hook_patches()

        del self.conds
        return output