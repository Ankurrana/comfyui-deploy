# Workflows

A collection of ComfyUI workflows for image and video generation.

---

## LongCat Image-to-Video

**File:** `longcat-img2video.json`

### Description

This workflow generates **5 to 30 second videos** from a single input image using the LongCat model. LongCat is a state-of-the-art video generation model based on the Wan Video architecture, optimized for creating longer, coherent video sequences.

### What it does

1. **Takes an input image** - Load any image as the starting frame
2. **Encodes the image** - Uses the Wan VAE to encode the image into latent space
3. **Generates video frames** - LongCat diffusion model creates smooth video continuation
4. **Outputs MP4 video** - Saves the result as an H.264 MP4 video file

### Required Models

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| `LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors` | Diffusion | ~8GB | Main video generation model (FP8 quantized) |
| `LongCat_distill_lora_alpha64_bf16.safetensors` | LoRA | ~200MB | Distillation LoRA for faster inference |
| `wan_2.1_vae.safetensors` | VAE | ~300MB | Video encoder/decoder |
| `umt5-xxl-enc-bf16.safetensors` | CLIP/T5 | ~10GB | Text encoder for prompts |

### Required Custom Nodes

- **ComfyUI-WanVideoWrapper** - Core video generation nodes
- **ComfyUI-KJNodes** - Utility nodes
- **ComfyUI-VideoHelperSuite** - Video output/preview nodes

### Settings

- **Resolution:** Configurable (default: 832x480)
- **Frame Rate:** 16 FPS
- **Video Length:** 93 frames (~5.8 seconds at 16fps, adjustable)
- **Attention Mode:** SageAttn (compiled) for faster inference

### VRAM Requirements

- **Minimum:** 12GB VRAM (with FP8 model + offloading)
- **Recommended:** 24GB VRAM for best performance

### Usage

```bash
# Deploy this workflow to a new ComfyUI installation
python deploy_workflow.py -w workflows/longcat-img2video.json -c /path/to/ComfyUI

# Dry run to see what will be installed
python deploy_workflow.py -w workflows/longcat-img2video.json -c /path/to/ComfyUI --dry-run
```

### Tips

- Use high-quality input images for best results
- 16:9 aspect ratio works best
- Increase frame count for longer videos (but requires more VRAM)
- Enable torch compile for faster generation after first run

### Credits

- **LongCat Model:** [Kijai/LongCat-Video_comfy](https://huggingface.co/Kijai/LongCat-Video_comfy)
- **Wan Video:** [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged)
