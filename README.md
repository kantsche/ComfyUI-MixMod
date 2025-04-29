# ComfyUI-MixMod

ComfyUI-MixMod provides a powerful way to combine multiple models during sampling.

## Features

- Mix multiple models during sampling
- Scheduling everything
- Multiple experimental modes.

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Kantsche/comfyui-mixmod.git
```

2. Restart ComfyUI if it's already running.

## Tips

- Different models excel at different aspects - try mixing a detail-focused model with a composition-focused one
- For FFT modes, base models work well for low frequencies, style models for high frequencies
- Experiment with weights and CFG values to find the best balance
- Schedule models to activate at different sampling steps for creative control

## Compatibility

- Tested with SD 1.5, SDXL, and SD 2.x models
- Works with different model architectures (base, inpainting, etc.)

## Credits

Created by Kantsche
