from __future__ import annotations
import folder_paths
import comfy.utils
import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as F
import math
from comfy import model_management
from comfy import latent_formats


def ensure_model(model_patcher, mode = None):
    if not mode or mode == "disable":
        # Ensure model mode disabled.
        return
    mp = model_patcher
    lm = model_management.LoadedModel(mp)
    found_lm = None
    for list_lm in model_management.current_loaded_models:
        if lm == list_lm:
            found_lm = list_lm
            break
    if getattr(found_lm, "currently_used", False):
        # Model already exists and appears to be loaded.
        return
    if found_lm is not None:
        lm = found_lm
        mp = lm.model
    if mode.startswith("normal"):
        if mode == "normal_unload":
            model_management.unload_all_models()
        model_management.load_models_gpu((mp,))
        return
    model_management.unload_all_models()
    if mode == "lowvram":
        # Logic from comfy.model_management.load_models_gpu
        min_inference_memory = model_management.minimum_inference_memory()
        minimum_memory_required = max(
            min_inference_memory,
            model_management.extra_reserved_memory(),
        )
        loaded_memory = lm.model_loaded_memory()
        current_free_mem = model_management.get_free_memory(lm.device) + loaded_memory

        lowvram_model_memory = max(
            64 * 1024 * 1024,
            (current_free_mem - minimum_memory_required),
            min(
                current_free_mem * model_management.MIN_WEIGHT_MEMORY_RATIO,
                current_free_mem - model_management.minimum_inference_memory(),
            ),
        )
        lowvram_model_memory = max(0.1, lowvram_model_memory - loaded_memory)
    elif mode == "novram":
        lowvram_model_memory = 0.1
    else:
        raise ValueError("Bad ensure_model_mode")
    lm.model_load(lowvram_model_memory)
    model_management.current_loaded_models.insert(0, lm)


def fallback(val, default, *, exclude=None, default_is_fun=False):
    return val if val is not exclude else (default() if default_is_fun else default)


def scale_dim(n, factor=1.0, *, increment=64) -> int:
    return math.ceil((n * factor) / increment) * increment


def sigma_to_float(sigma):
    return sigma.detach().cpu().max().item()

def load_taesd(name):
    sd = {}
    approx_vaes = folder_paths.get_filename_list("vae_approx")

    encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
    decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

    enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
    for k in enc:
        sd["taesd_encoder.{}".format(k)] = enc[k]

    dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder),device=model_management.get_torch_device())
    for k in dec:
        sd["taesd_decoder.{}".format(k)] = dec[k]

    if name == "taesd":
        sd["vae_scale"] = torch.tensor(0.18215)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesdxl":
        sd["vae_scale"] = torch.tensor(0.13025)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesd3":
        sd["vae_scale"] = torch.tensor(1.5305)
        sd["vae_shift"] = torch.tensor(0.0609)
    elif name == "taef1":
        sd["vae_scale"] = torch.tensor(0.3611)
        sd["vae_shift"] = torch.tensor(0.1159)
    
    vae = comfy.sd.VAE(sd=sd)
    return vae

def load_vae(name):
    vae = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", name),device=model_management.get_torch_device())
    vae = comfy.sd.VAE(sd=vae)
    return vae

def create_depth_map(latent_image, depth_model, resolution, **kwargs):
    batch_size = latent_image.shape[0]
    out_tensor = None
    for i, image in enumerate(latent_image):
        np_image = np.asarray(image.cpu() * 255., dtype=np.uint8)
        np_result = depth_model(np_image, output_type="np", detect_resolution=resolution, **kwargs)
        out = torch.from_numpy(np_result.astype(np.float32) / 255.0)
        if out_tensor is None:
            out_tensor = torch.zeros(batch_size, *out.shape, dtype=torch.float32)
        out_tensor[i] = out
                
    return out_tensor

def save_debug_image(tensor, filename, scale=True):
    try:
        # Ensure tensor is detached and on CPU
        tensor = tensor.detach().cpu().float()
        print(tensor.shape)
        # Remove batch dimension if present
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
            
        # Handle 2D tensors (grayscale masks) by converting to RGB
        if tensor.ndim == 2:
            # Convert 2D tensor to 3D by repeating across 3 channels
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
            
        # If scale is True, scale to [0, 255]
        if scale:
            array = (tensor.numpy() * 255).clip(0, 255).astype('uint8')
        else:
            array = tensor.numpy().clip(0, 255).astype('uint8')
        # Ensure shape is (H, W, 3)
        if array.shape[-1] != 3:
            raise ValueError(f"Expected last dimension to be 3 (RGB), got {array.shape[-1]}")
        img = Image.fromarray(array)
        
        # Ensure debug directory exists
        os.makedirs("debug", exist_ok=True)
        
        img.save("debug/" + filename)
        return True
    except Exception as e:
        print(f"Error saving debug image {filename}: {e}")
        return False
    
def process_mask(mask, H, W, device):
    # Convert to tensor if not already
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, device=device, dtype=torch.float32)
    mask = mask.to(device).float()

    # Squeeze singleton dimensions
    mask = mask.squeeze()
    
    # If mask is RGB, convert to grayscale
    if mask.ndim == 3 and mask.shape[-1] == 3:
        # [H, W, 3] -> [3, H, W]
        mask = mask.permute(2, 0, 1)
        mask = mask.mean(dim=0)  # [H, W]
    elif mask.ndim == 4 and mask.shape[-1] == 3:
        # [B, H, W, 3] -> [B, 3, H, W]
        mask = mask.permute(0, 3, 1, 2)
        mask = mask.mean(dim=1)  # [B, H, W]
        mask = mask.squeeze(0)   # Remove batch if present

    # Now mask should be [H, W]
    if mask.shape != (H, W):
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear')[0, 0]

    return mask


def create_adaptive_low_res_model_wrapper( min_scale_factor=0.5, max_scale_factor=1.0, start_step=0, end_step=None):
    """
    Creates a model function wrapper that runs the model at progressively higher resolution as sampling proceeds.
    
    Args:
        min_scale_factor: Minimum scale factor to use (for early steps)
        max_scale_factor: Maximum scale factor to use (for final steps)
        start_step: Step number to start increasing resolution (0 means immediately)
        end_step: Step number when scaling should reach max_scale_factor (None means the last step)
    
    Returns:
        A function wrapper that can be used as model_options['model_function_wrapper']
    """
    def wrapper(apply_model_func, args_dict):
        input_x = args_dict["input"]
        timestep = args_dict["timestep"]
        c = args_dict["c"]
        
        # Get sampling progress from sigmas in transformer_options
        sample_sigmas = c['transformer_options'].get('sample_sigmas', None)
        current_sigma = c['transformer_options'].get('sigmas', None)
        
        # Determine scale factor based on current step
        if sample_sigmas is not None and current_sigma is not None:
            # Get total steps from the length of sample_sigmas
            total_steps = len(sample_sigmas)
            
            # Find current step by matching current_sigma to sample_sigmas
            current_sigma_val = float(current_sigma[0])
            
            # Find the index of the closest sigma value in sample_sigmas
            # This gives us the current step number
            sigma_diffs = [abs(float(s) - current_sigma_val) for s in sample_sigmas]
            current_step = sigma_diffs.index(min(sigma_diffs))
            
            # If end_step is not provided, use the last step
            actual_end_step = end_step if end_step is not None else (total_steps - 1)
            
            # Clamp start and end steps to valid range
            start_step_clamped = max(0, min(start_step, total_steps - 1))
            end_step_clamped = max(start_step_clamped, min(actual_end_step, total_steps - 1))
            
            # Calculate scale factor based on current step
            if current_step < start_step_clamped:
                # Before start step - use minimum scale
                scale_factor = min_scale_factor
            elif current_step > end_step_clamped:
                # After end step - use maximum scale
                scale_factor = max_scale_factor
            else:
                # In between - interpolate between min and max
                step_range = end_step_clamped - start_step_clamped
                if step_range > 0:
                    progress = (current_step - start_step_clamped) / step_range
                    scale_factor = min_scale_factor + progress * (max_scale_factor - min_scale_factor)
                else:
                    # If start_step equals end_step
                    scale_factor = min_scale_factor
        else:
            # Fallback if we can't determine steps
            scale_factor = max_scale_factor
        
        # Original shape
        original_shape = input_x.shape
        
        # Skip resizing if we're at full resolution
        if scale_factor >= 0.99:
            return apply_model_func(input_x, timestep, **c)
        print(f"Scaling down to {scale_factor}x for step {current_step} using bislerp")
        # Calculate new dimensions
        new_h = int(original_shape[2] * scale_factor)
        new_w = int(original_shape[3] * scale_factor)
        
        # Ensure dimensions are at least 8 (minimum size for stable diffusion)
        new_h = max(8, new_h)
        new_w = max(8, new_w)
        
        # Downscale using comfy.utils.common_upscale with bislerp mode
        downscaled_x = comfy.utils.common_upscale(
            input_x,
            new_w,
            new_h,
            'bislerp',
            'disabled'
        )
        
        # Run model at lower resolution
        output = apply_model_func(downscaled_x, timestep, **c)
        
        # Upscale result back to original resolution using bislerp
        upscaled_output = comfy.utils.common_upscale(
            output, 
            original_shape[3],  # width
            original_shape[2],  # height
            'bislerp',
            'disabled'
        )

        """ # #blurring the output
        # Apply a small amount of Gaussian blur to smooth the upscaled output
        # This helps reduce artifacts that can appear from the upscaling process
        kernel_size = 3  # Small kernel for subtle blurring
        sigma = 0.5      # Low sigma for gentle blur
        
        # Create Gaussian kernel
        kernel = torch.tensor([
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]
        ], device=upscaled_output.device).view(1, 1, kernel_size, kernel_size)
        
        Expand kernel for all channels
        channels = upscaled_output.shape[1]
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        
        # Group kernel for batch processing
        kernel = kernel.repeat(1, 1, 1, 1)
        
        # Apply convolution for blurring
        # Use padding='same' to maintain dimensions
        padding = kernel_size // 2
        blurred_output = torch.nn.functional.conv2d(
            upscaled_output, 
            kernel.to(upscaled_output.dtype),
            padding=padding,
            groups=channels
        )
        
        # Use the blurred output instead of the direct upscaled output
        upscaled_output = blurred_output """
        return upscaled_output
        
    return wrapper

def create_noise_injection_wrapper(latent, noise_weight=1.0, callback=None):
    """
    Creates a wrapper that takes an input image, converts it to latent space,
    noises it to match the current timestep/sigma, and injects its noise prediction.
    
    Args:
        image: Input image to inject (tensor in pixel space)
        vae: VAE model for encoding to latent space
        noise_weight: Weight of the injected noise prediction (0-1)
        
    Returns:
        A function wrapper to use as model_options['model_function_wrapper']

        
    """
    def wrapper(apply_model_func, args_dict):
        # Extract args
        input_x = args_dict["input"]
        timestep = args_dict["timestep"] 
        c = args_dict["c"]
        
        
        # Get current sampling parameters
        sigmas = c['transformer_options'].get('sigmas', None)
        sigma_idx = c['transformer_options'].get('sigma_idx', 0)
        current_sigma = sigmas[sigma_idx] if sigmas is not None else None
        
        # Noise the image latent to match current noise level
        if current_sigma is not None:
            # Move tensors to the same device as input_x
            device = input_x.device
            image_latent = latent.to(device)
            current_sigma = current_sigma.to(device)

            image_latent = latent_formats.SDXL().process_in(image_latent)
            
            # Resize image_latent to match input_x dimensions
            if image_latent.shape[2:] != input_x.shape[2:]:
                image_latent = torch.nn.functional.interpolate(
                    image_latent, 
                    size=(input_x.shape[2], input_x.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            
            noise = torch.randn_like(image_latent)
            noised_image_latent = image_latent + noise * current_sigma

            noised_image_latent = (1 - noise_weight) * input_x + noise_weight * noised_image_latent
            
            # Get noise prediction on the image latent
            image_args = args_dict.copy()
            image_args["input"] = noised_image_latent
            combined_pred = apply_model_func(noised_image_latent, timestep, **c)
            
            # Run original model to get base noise prediction
            #original_noise_pred = apply_model_func(input_x, timestep, **c)

            
            
            # Combine predictions
            #combined_pred = (1 - noise_weight) * original_noise_pred + noise_weight * image_noise_pred
            #if callback is not None:
            #    callback(combined_pred)
            return combined_pred
        
        # Fall back to original model if we can't determine sigma
        return apply_model_func(input_x, timestep, **c)
        
    return wrapper

def create_model_wrapper(type, vae15, vaexl):
    """
    Creates a wrapper that takes an input image, converts it to latent space,
    noises it to match the current timestep/sigma, and injects its noise prediction.
    
    Args:
        image: Input image to inject (tensor in pixel space)
        vae: VAE model for encoding to latent space
        noise_weight: Weight of the injected noise prediction (0-1)
        
    Returns:
        A function wrapper to use as model_options['model_function_wrapper']

        
    """
    def wrapper(apply_model_func, args_dict):
        # Extract args
        input_x = args_dict["input"]
        timestep = args_dict["timestep"] 
        c = args_dict["c"]
        
        # Noise the image latent to match current noise level
        if type == "sd15":
            # Get device from input_x
            device = input_x.device
            
            image_latent = convert_latent_format(input_x, "sdxl", "sd15", device)
            
            #
            model_output = apply_model_func(image_latent, timestep, **c)
            model_output = model_output.to(device)
            
            final_output = convert_latent_format(model_output, "sd15", "sdxl", device)
            
            return final_output
        
        # Fall back to original model if type is not SD15
        return apply_model_func(input_x, timestep, **c)
        
    return wrapper

def sumweights(weights):
    weight_sum = sum(weights)
        
        # Normalize weights if sum is not zero
    if weight_sum > 0:
        weights = [w / weight_sum for w in weights]
    return weights

def convert_latent_format(latent, source_format="sd15", target_format="sdxl", device=None):
    """
    Converts latents between SD1.5 and SDXL formats using their RGB factors and scales.
    
    Args:
        latent: Input latent tensor (B, 4, H, W)
        source_format: Source format ("sd15" or "sdxl")
        target_format: Target format ("sd15" or "sdxl")
        device: Device to use (if None, uses the latent's device)
        
    Returns:
        Converted latent tensor (B, 4, H, W)
    """
    if source_format == target_format:
        return latent
    
    if device is None:
        device = latent.device
    
    # Define format parameters
    sd15_scale = 0.18215
    sdxl_scale = 0.13025
    
    sd15_factors = torch.tensor([
        [ 0.3512,  0.2297,  0.3227],
        [ 0.3250,  0.4974,  0.2350],
        [-0.2829,  0.1762,  0.2721],
        [-0.2120, -0.2616, -0.7177]
    ], device=device)
    
    sdxl_factors = torch.tensor([
        [ 0.3651,  0.4232,  0.4341],
        [-0.2533, -0.0042,  0.1068],
        [ 0.1076,  0.1111, -0.0362],
        [-0.3165, -0.2492, -0.2188]
    ], device=device)
    
    sdxl_bias = torch.tensor([0.1084, -0.0175, -0.0011], device=device)
    
    # Unscale the latent based on source format
    if source_format == "sd15":
        unscaled_latent = latent * sd15_scale
    else:  # sdxl
        unscaled_latent = latent * sdxl_scale
    
    # Get dimensions
    B, C, H, W = unscaled_latent.shape
    
    # For simplicity, we'll process each sample in the batch separately
    target_latent = torch.zeros_like(unscaled_latent)
    
    for b in range(B):
        # Get the current sample
        sample = unscaled_latent[b]  # [4, H, W]
        
        # Reshape to [4, H*W]
        sample_flat = sample.reshape(4, -1)
        
        # Convert to RGB representation
        if source_format == "sd15":
            # Calculate pseudo-inverse (4x3 matrix becomes 3x4)
            sd15_pinv = torch.pinverse(sd15_factors)
            # Transform from latent to RGB: [3x4] x [4, H*W] -> [3, H*W]
            rgb = torch.matmul(sd15_pinv, sample_flat)
        else:  # sdxl
            sdxl_pinv = torch.pinverse(sdxl_factors)
            # Transform from latent to RGB: [3x4] x [4, H*W] -> [3, H*W]
            rgb = torch.matmul(sdxl_pinv, sample_flat)
            # Remove bias (for each channel)
            for i in range(3):
                rgb[i] = rgb[i] - sdxl_bias[i]
        
        # Convert RGB to target latent space
        if target_format == "sd15":
            # RGB to SD1.5 latent: [4x3] x [3, H*W] -> [4, H*W]
            out = torch.matmul(sd15_factors, rgb)
            # Apply scale
            out = out / sd15_scale
        else:  # sdxl
            # Add bias
            for i in range(3):
                rgb[i] = rgb[i] + sdxl_bias[i]
            # RGB to SDXL latent: [4x3] x [3, H*W] -> [4, H*W]
            out = torch.matmul(sdxl_factors, rgb)
            # Apply scale
            out = out / sdxl_scale
        
        # Reshape back to [4, H, W] and store
        target_latent[b] = out.reshape(4, H, W)
    
    return target_latent
