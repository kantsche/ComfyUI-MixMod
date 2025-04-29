import torch

from comfy import latent_formats
from ..utils import create_depth_map, save_debug_image, process_mask
import comfy.sd
import numpy as np
import comfy.samplers
from ..utils import sumweights


def mix_standard(pred, weights, cfgs, uncond_decays=[1.0]):
    """
    Standard mixing function that applies CFG for each model based on weights.
    
    Args:
        pred: List of model predictions, each containing [conditional, unconditional]
        weights: List of weights for each model
        cfgs: List of cfg values for each model
        
    Returns:
        mix: The mixed prediction
        postuncond: The unconditioned prediction for post-processing
    """
    print("mix_standard")
    # Initialize with the shape of the first model's prediction
    mix = torch.zeros_like(pred[0][1])

    
    # First, combine all unconditional predictions based on weights
    for i in range(len(pred)):
        mix = mix + pred[i][1] * weights[i]

    # Save the combined unconditional prediction for post-processing
    postuncond = mix.clone()
    
    # For each model, apply the guidance according to weights and cfg values
    for i in range(len(pred)):
        # Extract conditional and unconditional predictions for this model
        c_pred = pred[i][0]
        u_pred = pred[i][1]
        
        # Apply the CFG formula with the corresponding weight and cfg
        # Add the weighted difference between conditional and unconditional
        mix = mix + (c_pred - u_pred) * cfgs[i]
        
    return mix, postuncond

def mix_masked(pred, weights, cfgs, masks=None):
    """
    Mask-based mixing function that applies different models to different regions.
    
    Args:
        pred: List of model predictions, each containing [conditional, unconditional]
        weights: List of weights for each model
        cfgs: List of cfg values for each model
        masks: List of masks (one per model) - should already be properly formatted/feathered
        
    Returns:
        mix: The mixed prediction
        postuncond: The unconditioned prediction for post-processing
    """
    print("mix_masked")
    # Check if we have valid mask in the list
    #if not any(mask is not None for mask in masks):
    #    # Fall back to standard mixing if no valid masks
    #    return mix_standard(pred, weights, cfgs)
    
    # Initialize with zeros
    B, C, H, W = pred[0][0].shape
    mix = torch.zeros_like(pred[0][0])
    postuncond = torch.zeros_like(pred[0][1])
    # Prepare masks
    processed_masks = []
    for i, mask in enumerate(masks):
        try:
            if(mask is not None):
                processed_mask = process_mask(mask, H, W, device=pred[0][0].device)
                processed_masks.append(processed_mask)
            else:
                processed_masks.append(torch.ones((H, W), device=pred[0][0].device))   
        except Exception as e:
            print(f"Error processing mask {i}: {e}")
            processed_masks.append(torch.zeros((H, W), device=pred[0][0].device))
    
    # If we couldn't process any masks, fall back to standard mixing
    if len(processed_masks) == 0:
        return mix_standard(pred, weights, cfgs)
        
    # If we have fewer masks than models, pad with zeros
    while len(processed_masks) < len(pred):
        processed_masks.append(torch.ones_like(processed_masks[0], device=pred[0][0].device))
        
    print(len(processed_masks))
    # Stack and normalize masks
    stacked_masks = torch.stack(processed_masks[:len(pred)])
    mask_sum = torch.sum(stacked_masks, dim=0, keepdim=True)
    mask_sum = torch.clamp(mask_sum, min=1e-6)  # Avoid division by zero
    normalized_masks = stacked_masks / mask_sum
    
    # Apply each model according to its normalized mask
    for i in range(len(pred)):
        if i >= len(normalized_masks):
            continue
            
        # Get mask for this model and add batch and channel dimensions
        model_mask = normalized_masks[i].unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)
        
        # Extract predictions
        c_pred = pred[i][0]
        u_pred = pred[i][1]
        
        # Apply CFG
        model_result = u_pred + (c_pred - u_pred) * cfgs[i]
        
        # Add to result based on mask
        mix = mix + model_result * model_mask
        postuncond = postuncond + u_pred * model_mask
    
    return mix, postuncond

def mix_dynamic_mask(pred, weights, cfgs, step=0, history=None, decay_factor=0.9, threshold=0.1, blur_sigma=0.5, max_update_step=15):
    """
    Dynamic mask-based mixing that evolves masks based on prediction magnitudes over time.
    
    Args:
        pred: List of model predictions, each containing [conditional, unconditional]
        weights: List of weights for each model
        cfgs: List of cfg values for each model
        step: Current generation step (used for mask evolution)
        history: Previous mask history to evolve from (None for first step)
        decay_factor: How much to retain from previous mask (0-1)
        threshold: Threshold for magnitude difference to affect mask
        blur_sigma: Standard deviation for Gaussian blur applied to masks (0 for no blur)
        max_update_step: If set, stops updating masks after this step and uses existing masks
        
    Returns:
        mix: The mixed prediction
        postuncond: The unconditioned prediction for post-processing
        masks: The updated masks to pass to the next step
    """
    # Get dimensions from the first prediction
    B, C, H, W = pred[0][0].shape
    device = pred[0][0].device
    # Initialize output tensors
    mix = torch.zeros_like(pred[0][0])
    postuncond = torch.zeros_like(pred[0][1])
    
    # Initialize or update masks
    if history is None or step == 0:
        # First step: initialize with equal masks or based on weights
        masks = []
        for i in range(len(pred)):
            masks.append(torch.ones((H, W), device=device) * weights[i])
    else:
        # Use previous masks as starting point
        masks = history
        # Save masks to debug folder
        for i, mask in enumerate(masks):
            save_debug_image(mask, f"mask_{i}.png")
    
    # Check if we should update masks based on max_update_step
    should_update_masks = max_update_step is None or step <= max_update_step
    
    if should_update_masks:
        # Calculate magnitudes for each model's prediction
        magnitudes = []
        for i in range(len(pred)):
            # Use the difference between conditional and unconditional as a measure of "importance"
            model_contrib = torch.abs(pred[i][0] - pred[i][1])
            # Average across channels to get a 2D magnitude map
            magnitude = torch.mean(model_contrib, dim=1)[0]  # Take first batch
            magnitudes.append(magnitude)
        
        # Stack magnitudes for comparison
        stacked_magnitudes = torch.stack(magnitudes)
        
        # Find which model has highest magnitude at each pixel
        max_indices = torch.argmax(stacked_magnitudes, dim=0)
        
        # Update masks based on magnitudes
        updated_masks = []
        for i in range(len(pred)):
            # Create binary mask where this model has highest magnitude
            model_dominant = (max_indices == i).float()
            
            # Apply threshold to only update where difference is significant
            magnitude_threshold = torch.max(stacked_magnitudes) * threshold
            significant_diff = (stacked_magnitudes[i] > magnitude_threshold).float()
            
            # Combine with threshold
            update_mask = model_dominant * significant_diff
            
            # Evolve mask gradually (blend old and new)
            if i < len(masks):
                # Decay old mask and add new information
                new_mask = masks[i] * decay_factor + update_mask * (1 - decay_factor)
                updated_masks.append(new_mask)
            else:
                updated_masks.append(update_mask)
        
        # Use the updated masks
        mask_list = updated_masks
    else:
        # Just use the existing masks without updating
        if isinstance(masks, list):
            mask_list = masks
        else:
            # Convert tensor masks back to list if needed
            mask_list = [masks[i] for i in range(masks.shape[0])]
    
    # Normalize masks to sum to 1 at each pixel
    stacked_masks = torch.stack(mask_list)
    mask_sum = torch.sum(stacked_masks, dim=0, keepdim=True)
    mask_sum = torch.clamp(mask_sum, min=1e-6)  # Avoid division by zero
    normalized_masks = stacked_masks / mask_sum
    
    # Apply Gaussian blur to the masks if blur_sigma > 0
    if blur_sigma > 0:
        import torch.nn.functional as F
        import math
        blurred_masks = []
        
        # Create a custom Gaussian blur function since F.gaussian_blur isn't available
        def custom_gaussian_blur(input_tensor, kernel_size, sigma):
            # Create Gaussian kernel
            channels = input_tensor.shape[1]
            kernel_size = [kernel_size, kernel_size]
            sigma = [sigma, sigma]
            
            # Create a grid of coordinates
            x_coord = torch.arange(kernel_size[0], device=input_tensor.device)
            x_grid = x_coord.repeat(kernel_size[1]).view(kernel_size[1], kernel_size[0])
            y_grid = x_grid.t()
            
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
            
            # Calculate the center of the filter
            mean = (torch.tensor(kernel_size, device=input_tensor.device) - 1) / 2.
            variance = torch.tensor(sigma, device=input_tensor.device) ** 2
            
            # Calculate the 2d gaussian kernel
            gaussian_kernel = torch.exp(
                -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance[0])
            )
            
            # Make sure sum of values in gaussian kernel equals 1.
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
            
            # Reshape to 2d depth-wise convolutional weight
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size[0], kernel_size[1])
            gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
            
            # Apply padding
            pad_size = (kernel_size[0] // 2, kernel_size[1] // 2)
            padded_input = F.pad(input_tensor, (pad_size[0], pad_size[0], pad_size[1], pad_size[1]), mode='reflect')
            
            # Apply Gaussian blur using depthwise convolution
            return F.conv2d(padded_input, gaussian_kernel, groups=channels)
        
        for i in range(len(normalized_masks)):
            mask = normalized_masks[i].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            # Apply Gaussian blur (kernel size should be odd and >= 3)
            kernel_size = max(3, int(blur_sigma * 4 + 0.5) * 2 + 1)
            blurred_mask = custom_gaussian_blur(mask, kernel_size, blur_sigma)
            blurred_masks.append(blurred_mask.squeeze(0).squeeze(0))
        
        # Re-normalize after blurring to ensure they still sum to 1
        stacked_blurred_masks = torch.stack(blurred_masks)
        blurred_sum = torch.sum(stacked_blurred_masks, dim=0, keepdim=True)
        blurred_sum = torch.clamp(blurred_sum, min=1e-6)
        normalized_masks = stacked_blurred_masks / blurred_sum
    
    # Apply each model according to its normalized mask
    for i in range(len(pred)):
        # Get mask for this model and add batch and channel dimensions
        model_mask = normalized_masks[i].unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)
        
        # Extract predictions
        c_pred = pred[i][0]
        u_pred = pred[i][1]
        
        # Apply CFG
        model_result = u_pred + (c_pred - u_pred) * cfgs[i]
        
        # Add to result based on mask
        mix = mix + model_result * model_mask
        postuncond = postuncond + u_pred * model_mask
    
    # Return the mixed result, unconditioned prediction, and updated masks for next step
    return mix, postuncond, normalized_masks.detach()

def mix_dynamic_mask_alternative(pred, weights, cfgs, depth_model=None, threshold=0.5, invert=False, blur_sigma=1.0):
    """
    Mixing function that automatically detects foreground/background and applies different models to each.
    
    Args:
        pred: List of model predictions, each containing [conditional, unconditional]
        weights: List of weights for each model
        cfgs: List of cfg values for each model
        depth_model: Optional depth estimation model. If None, uses simple heuristics
        threshold: Threshold for foreground/background separation (0-1)
        invert: If True, inverts the mask (foreground becomes background)
        blur_sigma: Blur amount to apply to mask edges
        
    Returns:
        mix: The mixed prediction
        postuncond: The unconditioned prediction for post-processing
    """
    # Need at least 2 models to do foreground/background mixing
    if len(pred) < 2:
        return mix_standard(pred, weights, cfgs)
    
    # Get dimensions from the first prediction
    B, C, H, W = pred[0][0].shape
    device = pred[0][0].device
    
    # Generate foreground/background mask using prediction features
    if depth_model is None:
        # Simple heuristic: use the magnitude of the conditional prediction
        # as an approximation of feature importance
        c_pred = pred[0][0]  # Use first model's conditional prediction
        
        # Calculate magnitude across channels (approximate depth/saliency)
        magnitude = torch.mean(torch.abs(c_pred), dim=1, keepdim=True)
        
        # Normalize to 0-1 range
        min_val = torch.min(magnitude)
        max_val = torch.max(magnitude)
        normalized = (magnitude - min_val) / (max_val - min_val + 1e-8)
        
        # Apply threshold
        mask = (normalized > threshold).float()
        
        # Invert if requested
        if invert:
            mask = 1.0 - mask
    else:
        # Use provided depth model (not implemented here, would need integration)
        # This would replace the simple heuristic above with actual depth estimation
        mask = torch.ones((B, 1, H, W), device=device) * 0.5
        # mask = depth_model(c_pred)
    
    # Apply Gaussian blur to smooth mask edges
    if blur_sigma > 0:
        import torch.nn.functional as F
        
        # Create a Gaussian kernel
        kernel_size = max(3, int(blur_sigma * 4 + 0.5) * 2 + 1)
        
        # Apply Gaussian blur
        mask = F.pad(mask, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
        mask = F.avg_pool2d(mask, kernel_size, stride=1, padding=0)
    
    # Create two masks: foreground and background
    mask_fg = mask
    mask_bg = 1.0 - mask
    
    # Ensure masks have the correct shape (H, W) without extra dimensions
    if mask_fg.dim() > 2:
        # Extract a 2D mask, squeezing out extra dimensions
        if mask_fg.dim() == 3:  # [C, H, W] or [1, H, W]
            mask_fg = mask_fg.squeeze(0)
        elif mask_fg.dim() == 4:  # [B, C, H, W]
            mask_fg = mask_fg.squeeze(0).squeeze(0)
        
        # If mask still has more than 2 dimensions, use only the first 2D slice
        if mask_fg.dim() > 2:
            mask_fg = mask_fg[0]
            
    # Do the same for background mask
    if mask_bg.dim() > 2:
        if mask_bg.dim() == 3:
            mask_bg = mask_bg.squeeze(0)
        elif mask_bg.dim() == 4:
            mask_bg = mask_bg.squeeze(0).squeeze(0)
        
        if mask_bg.dim() > 2:
            mask_bg = mask_bg[0]
    
    # Create list of masks for each model
    masks = [mask_fg, mask_bg]

    # Create directory if it doesn't exist
    for i, mask in enumerate(masks):
        save_debug_image(mask, f"mask_{i}.png")
    
    # If more than 2 models, create zero masks for the rest
    for i in range(2, len(pred)):
        masks.append(torch.zeros_like(mask_fg))
    
    # Use the existing masked mixing function with our generated masks
    return mix_masked(pred, weights, cfgs, masks)

def mix_foreground_background(pred, weights, last_latent, cfgs, step=0, invert=False, masks=None, vae=None, sharpness=0.0, depth_model=None, startmask=5):
    """
    Mixing function that uses Depth Anything V2 to separate foreground and background.
    
    Args:
        pred: List of model predictions, each containing [conditional, unconditional]
        weights: List of weights for each model
        cfgs: List of cfg values for each model
        invert: If True, inverts the mask (foreground becomes background)
        ckpt_name: The Depth Anything V2 checkpoint to use
        threshold: Threshold for foreground/background separation (0-1)
        blur_sigma: Blur amount to apply to mask edges
        
    Returns:
        mix: The mixed prediction
        postuncond: The unconditioned prediction for post-processing
    """
    
    # Get dimensions from the first prediction
    B, C, H, W = pred[0][0].shape
    #get the higes resolution of the latent image make sure its a power of 2
    res = 2**(max(H, W).bit_length() - 1)
    device = pred[0][0].device
    
    # Get the unconditioned prediction from the first model to generate a depth map
    u_pred = pred[0][1]
    
    # Convert latent to image for depth estimation
    # This is a simplified approach - assumes we're working with latents
    # In a real implementation, you'd need proper latent-to-image conversion
    
    if not any(mask is not None for mask in masks):
        if step == startmask:
            # Use Depth Anything V2 to generate a depth map
            latent_image = last_latent
            try:
                import comfy.model_management as model_management
                from custom_nodes.comfyui_controlnet_aux.utils import common_annotator_call
                
                
                
                # Convert latent to proper image using TaeSDXL Mini VAE
                try:
                    latent_image = latent_formats.SDXL().process_out(latent_image)
                    latent_image = vae.decode(latent_image)
                
                except Exception as e:
                    print(f"Error using TAESD: {e}")
                    
                    
                out_tensor = create_depth_map(latent_image, depth_model, res)

                depth_norm = out_tensor
                
                
            except Exception as e:
                print(f"Error using Depth Anything V2: {e}")
                print("Falling back to simple heuristic for foreground/background separation")
                
                # Fallback: use the magnitude of the conditional prediction as approximation
                c_pred = pred[0][0]
                depth_norm = torch.mean(torch.abs(c_pred), dim=1, keepdim=True)
                depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-8)
            mask = depth_norm * (1.0 + sharpness) - (depth_norm * sharpness).mean()
            mask = torch.clamp(mask, 0.0, 1.0)

            mask = mask
            # Invert if requested
            if invert:
                mask = 1.0 - mask
            
            # Create two masks: foreground and background
            mask_fg = mask
            mask_bg = 1.0 - mask
            
            
            
            # Create list of masks for each model
            masks = [mask_fg, mask_bg]
            
            # If more than 2 models, create zero masks for the rest
            for i in range(2, len(pred)):
                masks.append(torch.zeros_like(mask_fg))

            mix, postuncond = mix_masked(pred, weights, cfgs, masks)
            # Debug: Save masks
            try:
                # Save various debug images
                save_debug_image(latent_image, "last_latent.png")
                
                save_debug_image(mask, "mask.png")
            except Exception as e:
                print(f"Failed to save debug masks: {e}")
        else:
            mix, postuncond = mix_standard(pred, weights, cfgs)
    else:
        mix, postuncond = mix_masked(pred, weights, cfgs, masks)
    torch.cuda.empty_cache()
    # Use the existing masked mixing function with our generated masks
    return masks, mix, postuncond

