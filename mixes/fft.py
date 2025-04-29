import torch

def mix_teamfft(pred, weights, cfgs):
    # First, get shape information from the first model's predictions
    _, _, h, w = pred[0][0].shape
    
    # Initialize an empty tensor in frequency domain for the result
    fft_mix = torch.zeros_like(torch.fft.rfft2(pred[0][1]))
    
    postuncond = None
    
    # First loop: mix the unconditioned predictions in frequency domain
    for i in range(len(pred)):
        # Convert unconditional prediction to frequency domain
        fft_u_pred = torch.fft.rfft2(pred[i][1])
        # Add weighted contribution
        fft_mix = fft_mix + fft_u_pred * weights[i]
    

    postuncond = torch.fft.irfft2(fft_mix, s=(h, w))

    # Second loop: apply the guidance in frequency domain
    for i in range(len(pred)):
        # Convert conditional and unconditional predictions to frequency domain
        fft_c_pred = torch.fft.rfft2(pred[i][0])
        fft_u_pred = torch.fft.rfft2(pred[i][1])
        
        # Apply the CFG formula in frequency domain
        fft_mix = fft_mix + (fft_c_pred - fft_u_pred) * cfgs[i]
    
    # Convert the final result back to spatial domain
    mix = torch.fft.irfft2(fft_mix, s=(h, w))
    
    return mix, postuncond

def mix_teamfft4freq(pred, weights, cfgs):
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
    
    uncondresult = torch.zeros_like(fft_mix)
    postuncond = None
    
    # Process each frequency band separately
    for band_idx in range(4):
        band_mix = torch.zeros_like(fft_mix)
        
        # First loop: mix the unconditioned predictions in this frequency band
        for i in range(len(pred)):
            # Convert unconditional prediction to frequency domain
            fft_u_pred = torch.fft.rfft2(pred[i][1])
            # Add weighted contribution for this frequency band
            band_mix = band_mix + fft_u_pred * weights[i] * masks[band_idx]

        uncondresult = uncondresult + band_mix
        
        # Second loop: apply the guidance in this frequency band
        for i in range(len(pred)):
            # Convert conditional and unconditional predictions to frequency domain
            fft_c_pred = torch.fft.rfft2(pred[i][0])
            fft_u_pred = torch.fft.rfft2(pred[i][1])
            
            # Apply the CFG formula in this frequency band
            band_mix = band_mix + (fft_c_pred - fft_u_pred) * cfgs[i] * masks[band_idx]
        
        # Add this band's contribution to the final mix
        fft_mix = fft_mix + band_mix
    

    postuncond = torch.fft.irfft2(uncondresult, s=(h, w))
        
    # Convert the final result back to spatial domain
    mix = torch.fft.irfft2(fft_mix, s=(h, w))
    
    return mix, postuncond

def mix_bandfft(pred, weights, cfgs, bands=2):
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
    
    #calculate the number of bands boundaries from bands size 
    band_boundaries = [0.0]
    for i in range(bands-1):
        band_boundaries.append(band_boundaries[-1] + 1.0 / bands)
    band_boundaries.append(1.0)
    
    # Create masks for each frequency band with smooth transitions
    transition_width = 0.01  # 1% transition width
    
    for i in range(bands):  # 4 frequency bands
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
    
    uncondresult = torch.zeros_like(fft_mix)
    postuncond = None
    
    # Process each frequency band separately
    for band_idx in range(bands):
        band_mix = torch.zeros_like(fft_mix)
        
        # First loop: mix the unconditioned predictions in this frequency band
        for i in range(len(pred)):
            # Convert unconditional prediction to frequency domain
            fft_u_pred = torch.fft.rfft2(pred[i][1])
            # Add weighted contribution for this frequency band
            band_mix = band_mix + fft_u_pred * weights[i] * masks[band_idx]

        uncondresult = uncondresult + band_mix
        
        # Second loop: apply the guidance in this frequency band
        for i in range(len(pred)):
            # Convert conditional and unconditional predictions to frequency domain
            fft_c_pred = torch.fft.rfft2(pred[i][0])
            fft_u_pred = torch.fft.rfft2(pred[i][1])
            
            # Apply the CFG formula in this frequency band
            band_mix = band_mix + (fft_c_pred - fft_u_pred) * cfgs[i] * masks[band_idx]
        
        # Add this band's contribution to the final mix
        fft_mix = fft_mix + band_mix
    

    postuncond = torch.fft.irfft2(uncondresult, s=(h, w))
        
    # Convert the final result back to spatial domain
    mix = torch.fft.irfft2(fft_mix, s=(h, w))
    
    return mix, postuncond

def mix_2model_fft(pred, weights, cfgs, ratio):
    # Apply FFT-based frequency splitting
    # Need at least 2 models for FFT mode
    if len(pred) < 2:
        mix = pred[0][0] * cfgs[0] + pred[0][1] * (1 - cfgs[0])
        return mix, None  # Return properly conditioned prediction
        
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
    split_ratio = max(0.05, min(0.95, abs(ratio)))
    
    # Calculate frequency split point
    split_h = int(h * split_ratio)
    split_w = int(w * split_ratio / 2)  # rfft2 returns half the width in frequency domain
    
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


    result = fft_u_pred_1 * mask_low + fft_u_pred_2 * mask_high
    postuncond = torch.fft.irfft2(result, s=(h, w))
        
    # Apply masks and compute CFG in frequency domain using the corresponding CFG values
    fft_result_low = (fft_u_pred_1 * mask_low) + ((fft_c_pred_1 - fft_u_pred_1) * cfgs[0] * mask_low)
    fft_result_high = (fft_u_pred_2 * mask_high) + ((fft_c_pred_2 - fft_u_pred_2) * cfgs[1] * mask_high)
    
    # Combine low and high frequency results
    fft_result = fft_result_low + fft_result_high
    
    # Convert back to spatial domain
    mix = torch.fft.irfft2(fft_result, s=(h, w))
    
    return mix, postuncond

def create_low_freq_mask(shape, radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2  # center

    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')
    
    # Distance from center
    dist_from_center = torch.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Create mask: 1 inside radius, 0 outside
    mask = torch.zeros((rows, cols))
    mask[dist_from_center <= radius] = 1

    return mask


def mix_2m2f(pred, weights, cfgs, ratio):
    # Split the prediction into low frequency and high frequency components
    # ratio controls the cutoff frequency between low and high
    
    # Need at least 2 models for this mode
    if len(pred) < 2:
        mix = pred[0][0] * cfgs[0] + pred[0][1] * (1 - cfgs[0])
        return mix, None  # Return properly conditioned prediction
        
    # Get shapes from first prediction
    _, _, h, w = pred[0][0].shape
    
    # Extract predictions from first model (handles low frequencies)
    c_pred_low = pred[0][0]  # Conditioned prediction for low frequencies
    u_pred_low = pred[0][1]  # Unconditioned prediction for low frequencies
    
    # Extract predictions from second model (handles high frequencies)
    c_pred_high = pred[1][0]  # Conditioned prediction for high frequencies
    u_pred_high = pred[1][1]  # Unconditioned prediction for high frequencies
    
    # Convert to frequency domain
    fft_c_low = torch.fft.rfft2(c_pred_low)
    fft_u_low = torch.fft.rfft2(u_pred_low)
    fft_c_high = torch.fft.rfft2(c_pred_high)
    fft_u_high = torch.fft.rfft2(u_pred_high)
    
    # Get frequency domain dimensions
    freq_h, freq_w = fft_c_low.shape[2], fft_c_low.shape[3]
    
    # Create normalized coordinate grids
    h_coords = torch.linspace(0, 1, freq_h, device=fft_c_low.device)
    w_coords = torch.linspace(0, 1, freq_w, device=fft_c_low.device)
    h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')
    
    # Calculate frequency magnitude (normalized distance from DC component)
    freq_magnitude = torch.sqrt(h_grid**2 + w_grid**2)
    max_freq = torch.sqrt(torch.tensor(2.0, device=fft_c_low.device))
    normalized_freq = freq_magnitude / max_freq
    
    # Create frequency masks with smooth transition around cutoff
    cutoff = ratio  # Frequency cutoff point (normalized)
    transition_width = 0.05  # Width of transition band for smooth blending
    
    # Create low frequency mask (1 for low frequencies, 0 for high)
    low_mask = torch.ones_like(normalized_freq)
    # Create smooth transition where freq > (cutoff - transition_width/2)
    transition_start = cutoff - transition_width/2
    transition_end = cutoff + transition_width/2
    
    # Apply smooth transition
    transition_region = (normalized_freq >= transition_start) & (normalized_freq <= transition_end)
    low_mask[transition_region] = 0.5 - 0.5 * torch.sin(
        torch.pi * (normalized_freq[transition_region] - cutoff) / transition_width
    )
    low_mask[normalized_freq > transition_end] = 0.0
    
    # High frequency mask is complement of low frequency mask
    high_mask = 1.0 - low_mask
    
    # Add batch and channel dimensions to masks
    low_mask = low_mask.unsqueeze(0).unsqueeze(0)
    high_mask = high_mask.unsqueeze(0).unsqueeze(0)
    
    # Separate frequency components
    # Low frequency unconditioned prediction
    fft_u_result = fft_u_low * low_mask + fft_u_high * high_mask
    
    # Convert unconditioned result back to spatial domain for returning
    postuncond = torch.fft.irfft2(fft_u_result, s=(h, w))
    
    # Apply CFG separately to low and high frequency components
    fft_low_result = fft_u_low * low_mask + (fft_c_low - fft_u_low) * cfgs[0] * low_mask
    fft_high_result = fft_u_high * high_mask + (fft_c_high - fft_u_high) * cfgs[1] * high_mask
    
    # Combine frequency components
    fft_result = fft_low_result + fft_high_result
    
    # Convert back to spatial domain
    mix = torch.fft.irfft2(fft_result, s=(h, w))
    
    return mix, postuncond

def mix_fft_overlap(pred, weights, cfgs, ratio):
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
        if weights[i] <= 0:
            continue
        
        # Extract predictions for this model
        c_pred = pred[i][0]
        u_pred = pred[i][1]
        
        # Convert to frequency domain
        fft_c_preds.append(torch.fft.rfft2(c_pred))
        fft_u_preds.append(torch.fft.rfft2(u_pred))
        valid_weights.append(weights[i])
        valid_cfgs.append(cfgs[i])
    
    # Get number of valid models
    num_models = len(valid_weights)
    if num_models == 0:
        # Handle edge case with no valid models
        default_mix = pred[0][0] * cfgs[0] + pred[0][1] * (1 - cfgs[0])
        return default_mix, None  # Return first model's conditioned output
    elif num_models == 1:
        # With only one model, use it for the full spectrum
        mix = pred[0][0] * valid_cfgs[0] + pred[0][1] * (1 - valid_cfgs[0])
        return mix, None
    
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
    overlap_ratio = ratio
    
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
    uncondresult = torch.zeros_like(fft_result)
    
    # Apply CFG to each frequency band and combine - vectorized
    for i in range(num_models):
        # Get mask, add batch and channel dimensions
        mask = normalized_stacked_masks[i].unsqueeze(0).unsqueeze(0)
        
        # Apply CFG formula with frequency band mask
        fft_u = fft_u_preds[i]
        uncondresult = uncondresult + fft_u 
        fft_c = fft_c_preds[i]
        cfg_scale = valid_cfgs[i]
        
        # Add weighted contribution to result
        fft_result += (fft_u + (fft_c - fft_u) * cfg_scale) * mask
    

    postuncond = torch.fft.irfft2(uncondresult, s=(h, w))
        
    # Convert back to spatial domain
    mix = torch.fft.irfft2(fft_result, s=(h, w))
    
    return mix, postuncond

def mix_fft_full(pred, weights, cfgs):
    # Full spectrum frequency mixing implementation - optimized version
    _, _, h, w = pred[0][0].shape
    
    # Initialize result in frequency domain
    fft_result = torch.zeros_like(torch.fft.rfft2(pred[0][0]))
    
    # First, convert all predictions to frequency domain
    fft_c_preds = []
    fft_u_preds = []
    valid_model_indices = []
    
    for i in range(len(pred)):
        if weights[i] <= 0:
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
    uncondresult = torch.zeros_like(fft_result)
    
    # Process each valid model
    if valid_model_indices:
        # Pre-calculate model weights for all frequency points
        all_model_weights = []
        for i in valid_model_indices:
            # Calculate model influence: higher weights → lower frequencies, lower weights → higher frequencies
            model_weight = weights[i] * (1.0 - freq_magnitude) + (1.0 - weights[i]) * freq_magnitude
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
            uncondresult = uncondresult + fft_u
            fft_c = fft_c_preds[i]
            
            # Apply CFG with model-specific weight
            contrib = fft_u * model_weights + (fft_c - fft_u) * cfgs[i] * model_weights
            
            # Add to result
            fft_result += contrib


    postuncond = torch.fft.irfft2(uncondresult, s=(h, w))
    
    # Convert back to spatial domain
    mix = torch.fft.irfft2(fft_result, s=(h, w))
    
    return mix, postuncond