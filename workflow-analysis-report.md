# Workflow Model Analysis Report

**Generated:** 2026-01-27 23:54

**API Keys:** CIVITAI_API_KEY=Set | HF_TOKEN=Set

## Summary Table

| # | Workflow | Models | Found | Valid | Missing | Failed |
|---|----------|--------|-------|-------|---------|--------|
| 1 | CreArt (Nunchaku) with Upscaler | 3 | 3 | 3 | 0 | 0 |
| 2 | EP22 IMG to Transparent | 0 | 0 | 0 | 0 | 0 |
| 3 | Ep26 LivePortrait Image to Image Ex... | 0 | 0 | 0 | 0 | 0 |
| 4 | EP27 SDXL Image to Digital Painting... | 4 | 4 | 2 | 0 | 2 |
| 5 | Ep49 Wan2 2.1 Vace GGUF Text To Vid... | 2 | 2 | 2 | 0 | 0 |
| 6 | EP51 Cosmos Predict2 2B Text2Image | 3 | 3 | 3 | 0 | 0 |
| 7 | EP53 Flux 1 Kontext Dev - Fantasy G... | 4 | 4 | 4 | 0 | 0 |
| 8 | EP54 Flux Dev - Image Vector Variat... | 2 | 2 | 2 | 0 | 0 |
| 9 | Ep55 Wan 2.1 I2V 480p with FusionX ... | 3 | 3 | 3 | 0 | 0 |
| 10 | Flux Krea txt2img with Upscaler | 3 | 3 | 3 | 0 | 0 |
| 11 | Flux txt2img with Upscaler - 2K Lan... | 3 | 3 | 2 | 0 | 1 |
| 12 | Fluxmania V - Text2Image | 3 | 3 | 0 | 0 | 3 |
| 13 | LongCat img2video - 5 to 30s Video ... | 4 | 4 | 4 | 0 | 0 |
| 14 | LongCat txt2video | 4 | 4 | 4 | 0 | 0 |
| 15 | longcat-img2video | 4 | 4 | 4 | 0 | 0 |
| 16 | LTX 0.95 image2video | 2 | 2 | 2 | 0 | 0 |
| 17 | LTX 0.95 text2video (1) | 2 | 2 | 2 | 0 | 0 |
| 18 | LTX 0.95 text2video | 2 | 2 | 2 | 0 | 0 |
| 19 | ltx | 2 | 2 | 2 | 0 | 0 |
| 20 | txt2ImgLORA | 2 | 2 | 0 | 0 | 2 |

## Missing Models (No URL Found)

*None - all models have URLs!*

## Failed URLs

### EP27 SDXL Image to Digital Painting with Lora and Control Net and UPSCALER

| Model | Source | Error | URL |
|-------|--------|-------|-----|
| Juggernaut_X_RunDiffusion.safetensors | civitai | HTTP 403 | `https://civitai.com/api/download/models/782002` |
| EldritchDigitalArt1.3.safetensors | civitai | HTTP 403 | `https://civitai.com/api/download/models/42959` |

### Flux txt2img with Upscaler - 2K Landscape 16-9

| Model | Source | Error | URL |
|-------|--------|-------|-----|
| ae.safetensors | workflow | HTTP 403 | `https://huggingface.co/black-forest-labs/FLUX.1-sc...` |

### Fluxmania V - Text2Image

| Model | Source | Error | URL |
|-------|--------|-------|-----|
| clipLCLIPGFullFP32_zer0intVisionCLIPL.safetensors | civitai | HTTP 403 | `https://civitai.com/api/download/models/501240?typ...` |
| fluxmania_V.safetensors | civitai | HTTP 403 | `https://civitai.com/api/download/models/501240?typ...` |
| ae.safetensors | workflow | HTTP 403 | `https://huggingface.co/black-forest-labs/FLUX.1-sc...` |

### txt2ImgLORA

| Model | Source | Error | URL |
|-------|--------|-------|-----|
| Pony Realism Slider.safetensors | civitai | HTTP 403 | `https://civitai.com/api/download/models/2605934` |
| juggernautXL_ragnarokBy.safetensors | civitai | HTTP 403 | `https://civitai.com/api/download/models/1759168` |

## Statistics

- **Total Workflows:** 20
- **Total Models:** 52
- **URLs Found:** 52 (100.0%)
- **URLs Valid:** 44 (84.6%)
- **Missing:** 0
- **Failed:** 8

## Notes

- **HTTP 403** errors typically mean:
  - CivitAI: Model requires accepting terms on the website first
  - HuggingFace: Gated model requiring license acceptance
- **Missing URLs**: Model not found in ComfyUI Manager, HuggingFace, or CivitAI
- **Timeout**: Server didn't respond in time (may still work)
