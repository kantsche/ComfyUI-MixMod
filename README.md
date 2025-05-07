# ComfyUI-MixMod


ComfyUI-MixMod provides a powerful way to combine multiple models during sampling.

VRAM requirement: Fitting two or multiple models.  Around 12gb is minimum for two sdxl. Around 16gb vram for sdxl+pixartsigma with a Q3 t5xxl encoder.

## Features

- Mix multiple models during sampling
- SD1.5+SDXL (use https://huggingface.co/ostris/sdxl-sd1-vae-lora to align the latents)
- SDXL+Pixart sigma for increased prompt adherence
- Scheduling everything
- Multiple experimental modes.

## Please share cool workflows you find under discussion, this stuff is really experimental and still needs discoveries.

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

- Tested with SD 1.5, SDXL and Pixart Sigma
- Works with different model architectures (base, inpainting, etc.)
- Only tested on Windows


Example workflow with pixart sigma:
![ShadowPCDisplay_n735IOn8Gu](https://github.com/user-attachments/assets/a053400d-2b8f-41de-8a44-c6b248867f07)

It improves the prompt adherence of sdxl in general prompts:

![ShadowPCDisplay_0d5BXStu6l](https://github.com/user-attachments/assets/2472d4e9-ed2d-4791-965a-3908003afe2a)



Example with Ponyv6 and NoobAI:

Only Pony, Pony+Noob, Only Noob image
![image](https://github.com/user-attachments/assets/9853bf07-f5e2-405c-bf0d-2c6e2a836511)


## Credits

Created by Kantsche
