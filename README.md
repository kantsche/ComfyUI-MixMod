# ComfyUI-MixMod

A custom node extension for ComfyUI that allows mixing multiple models during the sampling process for enhanced image generation.

##Please share workflows or good combinations here!

## Description

ComfyUI-MixMod provides a powerful way to combine multiple models during stable diffusion sampling. This extension introduces two primary mixing modes:

- **Team Mode**: Combines multiple models by weighted averaging of their predictions and applying their respective guidance scales.
- **2Model FFT Mode**: (EXPERIMENTAL) Splits the frequency domain between two models, using one model for low frequencies and another for high frequencies.

## Features

- Mix multiple models during sampling
- Control model contributions with customizable weights
- Set individual guidance scales (CFG) for each model
- Frequency domain splitting for controlling different image characteristics
- Chain multiple components together to create complex model combinations

## Roadmap

- Add more modes
- Make this work with SD1.5, Flux and other models

## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Kantsche/comfyui-mixmod.git
```

2. Restart ComfyUI if it's already running.

## Usage

The extension adds two main nodes to ComfyUI:

### MixMod Guider Component

This node creates a model component that can be chained with other components.

Inputs:
- **model**: The model to use in the mix
- **positive**: Positive conditioning for the model
- **negative**: Negative conditioning for the model
- **base_weight**: The weight of this model in the mix (normalized with other models)
- **cfg**: The guidance scale for this model
- **prev_component** (optional): Another component to chain with

### MixMod Guider

This node creates the final guider that will be used during sampling.

Inputs:
- **component**: The component or chain of components to use
- **mode**: The mixing mode to use (either "team" or "2model_fft")

## Example Workflows

### Basic Model Mixing

1. Add two or more "MixMod Guider Component" nodes
2. Connect your models, conditioning and set weights for each component
3. Chain components together by connecting them via "prev_component"
4. Connect the final component to a "MixMod Guider" node
5. Set the mixing mode
6. Use the resulting guider in your sampling node

### Basic Example Workflow

Here's a simple workflow to mix two models:

```
┌───────────────┐             ┌───────────────┐
│ Load Model A  │             │ Load Model B  │
└───────┬───────┘             └───────┬───────┘
        │                             │
        ▼                             ▼
┌───────────────┐             ┌───────────────┐
│ CLIP Text A   │             │ CLIP Text B   │
└───────┬───────┘             └───────┬───────┘
        │                             │
        ▼                             ▼
┌───────────────────────┐    ┌───────────────────────┐
│ MixMod Component A    │----│ MixMod Component B    │
│ weight: 0.6, cfg: 7.5 │    │ weight: 0.4, cfg: 8.0 │
└────────┬──────────────┘    └───────────────────────┘
         │
         ▼
┌──────────────────┐
│ MixMod Guider    │
│ mode: team       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ KSampler         │
│ (use as guider)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ VAE Decode       │
└────────┬─────────┘
         │
         ▼
      Image
```

## Compatibility

- This extension is compatible with ComfyUI version 1.0.0 and above
- Tested with SD 1.5, SDXL, and SD 2.x models
- Works with different model architectures (base, inpainting, etc.)

## Tips for Best Results

- When using Team Mode, try different weight combinations to find the best blend
- In FFT mode, typically use a high-quality base model for low frequencies and a detail-focused model for high frequencies
- Experiment with different CFG values for each model to control its influence
- For models with very different styles, start with lower weights for specialty models

Example with Ponyv6 and NoobAI:

Only Pony, Pony+Noob, Only Noob
![image](https://github.com/user-attachments/assets/0108c1e4-bf3c-4060-9860-47ae8a52b627)


## Credits

Created by Kantsche
