# Workflow Model Analysis Report

**Generated:** 2026-01-28 00:01

**API Keys:** CIVITAI_API_KEY=Set | HF_TOKEN=Set

## Summary Table

| # | Workflow | Models | Found | Valid | Missing | Failed |
|---|----------|--------|-------|-------|---------|--------|
| 1 | CreArt (Nunchaku) with Upscaler | 3 | 3 | 3 | 0 | 0 |
| 2 | EP22 IMG to Transparent | 0 | 0 | 0 | 0 | 0 |
| 3 | Ep26 LivePortrait Image to Image Ex... | 0 | 0 | 0 | 0 | 0 |
| 4 | EP27 SDXL Image to Digital Painting... | 4 | 3 | 2 | 1 | 1 |
| 5 | Ep49 Wan2 2.1 Vace GGUF Text To Vid... | 2 | 2 | 2 | 0 | 0 |
| 6 | EP51 Cosmos Predict2 2B Text2Image | 3 | 3 | 3 | 0 | 0 |
| 7 | EP53 Flux 1 Kontext Dev - Fantasy G... | 4 | 4 | 4 | 0 | 0 |
| 8 | EP54 Flux Dev - Image Vector Variat... | 2 | 2 | 2 | 0 | 0 |
| 9 | Ep55 Wan 2.1 I2V 480p with FusionX ... | 3 | 3 | 3 | 0 | 0 |
| 10 | Flux Krea txt2img with Upscaler | 3 | 3 | 3 | 0 | 0 |
| 11 | Flux txt2img with Upscaler - 2K Lan... | 3 | 3 | 2 | 0 | 1 |
| 12 | Fluxmania V - Text2Image | 3 | 1 | 0 | 2 | 1 |
| 13 | LongCat img2video - 5 to 30s Video ... | 4 | 4 | 4 | 0 | 0 |
| 14 | LongCat txt2video | 4 | 4 | 4 | 0 | 0 |
| 15 | longcat-img2video | 4 | 4 | 4 | 0 | 0 |
| 16 | LTX 0.95 image2video | 2 | 2 | 2 | 0 | 0 |
| 17 | LTX 0.95 text2video (1) | 2 | 2 | 2 | 0 | 0 |
| 18 | LTX 0.95 text2video | 2 | 2 | 2 | 0 | 0 |
| 19 | ltx | 2 | 2 | 2 | 0 | 0 |
| 20 | txt2ImgLORA | 2 | 1 | 0 | 1 | 1 |

## Missing Models (No URL Found)

### EP27 SDXL Image to Digital Painting with Lora and Control Net and UPSCALER

| Model | Type | Reason |
|-------|------|--------|
| EldritchDigitalArt1.3.safetensors | lora | No URL found in any source |

### Fluxmania V - Text2Image

| Model | Type | Reason |
|-------|------|--------|
| fluxmania_V.safetensors | unet | No URL found in any source |
| clipLCLIPGFullFP32_zer0intVisionCLIPL.safetensors | clip | No URL found in any source |

### txt2ImgLORA

| Model | Type | Reason |
|-------|------|--------|
| Pony Realism Slider.safetensors | lora | No URL found in any source |

## Failed URLs

### EP27 SDXL Image to Digital Painting with Lora and Control Net and UPSCALER

| Model | Source | Error | URL |
|-------|--------|-------|-----|
| Juggernaut_X_RunDiffusion.safetensors | civitai | HTTP 403 | `https://civitai.com/api/download/models/782002` |

### Flux txt2img with Upscaler - 2K Landscape 16-9

| Model | Source | Error | URL |
|-------|--------|-------|-----|
| ae.safetensors | workflow | HTTP 403 | `https://huggingface.co/black-forest-labs/FLUX.1-sc...` |

### Fluxmania V - Text2Image

| Model | Source | Error | URL |
|-------|--------|-------|-----|
| ae.safetensors | workflow | HTTP 403 | `https://huggingface.co/black-forest-labs/FLUX.1-sc...` |

### txt2ImgLORA

| Model | Source | Error | URL |
|-------|--------|-------|-----|
| juggernautXL_ragnarokBy.safetensors | civitai | HTTP 403 | `https://civitai.com/api/download/models/1759168` |

## Statistics

- **Total Workflows:** 20
- **Total Models:** 52
- **URLs Found:** 48 (92.3%)
- **URLs Valid:** 44 (84.6%)
- **Missing:** 4
- **Failed:** 4

## Notes

- **HTTP 403** errors typically mean:
  - CivitAI: Model requires accepting terms on the website first
  - HuggingFace: Gated model requiring license acceptance
- **Missing URLs**: Model not found in ComfyUI Manager, HuggingFace, or CivitAI
- **Timeout**: Server didn't respond in time (may still work)
