# ComfyUI Deploy

A utility for quickly deploying ComfyUI workflows on new machines. This tool automatically extracts all required models and custom nodes from workflow JSON files, searches for download URLs, and installs everything to the correct locations.

## Features

- 🔍 **Workflow Analysis**: Parses ComfyUI workflow JSON files to extract all dependencies
- 📦 **Smart Model Search**: Automatically finds model download URLs from:
  - ComfyUI Manager curated list
  - HuggingFace (pattern matching + repo file listing)
  - CivitAI (public API - no key required for most models)
- 🧩 **Custom Node Installation**: Clones required nodes from GitHub and installs dependencies
- 🗂️ **Correct Path Placement**: Automatically places models in the right ComfyUI folders
- 🔐 **Optional Authentication**: Supports HuggingFace and CivitAI tokens for gated models
- 📋 **Dry Run Mode**: Preview what will be installed before making changes
- ⚡ **Parallel Downloads**: Download multiple models simultaneously for faster deployment

## Installation

### Prerequisites

- Python 3.10 or higher
- Git (for cloning custom nodes)
- ComfyUI already installed

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ankurrana/comfyui-deploy.git
cd comfyui-deploy
```

### Step 2: Install Dependencies

```bash
pip install -e .
```

This installs all required packages: `requests`, `rich`, `click`, `pyyaml`, `huggingface-hub`, etc.

## Quick Start

### Example: Deploy LongCat Video Workflow

This example uses the included LongCat image-to-video workflow located in `workflows/`.

#### Step 1: Analyze the Workflow (Dry Run)

First, see what models and custom nodes are required **without downloading anything**:

```bash
python deploy_workflow.py -w workflows/longcat-img2video.json -c /workspace/ComfyUI --dry-run
```

Or using the full flag names:

```bash
python deploy_workflow.py --workflow workflows/longcat-img2video.json --comfyui /workspace/ComfyUI --dry-run
```

**Example output:**
```
======================================================================
PARSING WORKFLOW
======================================================================

📦 Models required: 4
   ❌ LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors (diffusion)
      → models/diffusion_models
   ❌ LongCat_distill_lora_alpha64_bf16.safetensors (lora)
      → models/loras
   ❌ wan_2.1_vae.safetensors (vae)
      → models/vae
   ❌ umt5-xxl-enc-bf16.safetensors (clip)
      → models/clip

🧩 Custom nodes required: 4
   ❌ ComfyUI-WanVideoWrapper
      → https://github.com/kijai/ComfyUI-WanVideoWrapper
   ❌ comfyui-kjnodes
      → https://github.com/kijai/ComfyUI-KJNodes
   ❌ comfyui-videohelpersuite
      → https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
   ❌ rgthree-comfy
      → https://github.com/rgthree/rgthree-comfy

======================================================================
⚠️  DRY RUN - No changes will be made
======================================================================
```

#### Step 2: Deploy the Workflow

Once you've reviewed the dependencies, run the full deployment:

```bash
python deploy_workflow.py --workflow workflows/longcat-img2video.json --comfyui /workspace/ComfyUI
```

Or with parallel downloads for faster speed:

```bash
python deploy_workflow.py --workflow workflows/longcat-img2video.json \--comfyui /workspace/ComfyUI \
  --parallel 8
```

This will:
1. ✅ Install default packages (ComfyUI-Manager, Impact-Pack, etc.)
2. ✅ Clone all required custom nodes to `/workspace/ComfyUI/custom_nodes/`
3. ✅ Install Python dependencies for each custom node
4. ✅ Search for and download all models to their correct folders
5. ✅ Display a summary of what was installed

#### Step 3: Start ComfyUI and Load the Workflow

```bash
cd /workspace/ComfyUI
python main.py
```

Then load `workflows/longcat-img2video.json` in the ComfyUI interface.

## Command Reference

### deploy_workflow.py

```bash
python deploy_workflow.py --workflow WORKFLOW --comfyui COMFYUI [OPTIONS]

Required Arguments:
  --workflow, -w    Path to workflow JSON file
  --comfyui, -c     Path to ComfyUI installation

Options:
  --dry-run, -n     Analyze only, don't download/install anything
  --parallel, -p    Number of parallel downloads (default: 4)
  --no-defaults     Skip installing default packages (ComfyUI-Manager, etc.)
```

### Examples

```bash
# Dry run - see what will be installed
python deploy_workflow.py -w workflow.json -c /workspace/ComfyUI --dry-run

# Standard deployment (4 parallel downloads + default packages)
python deploy_workflow.py -w workflow.json -c /workspace/ComfyUI

# Fast deployment (8 parallel downloads)
python deploy_workflow.py -w workflow.json -c /workspace/ComfyUI --parallel 8

# Skip default packages (only install workflow-specific nodes)
python deploy_workflow.py -w workflow.json -c /workspace/ComfyUI --no-defaults

# Sequential downloads (for slow/metered connections)
python deploy_workflow.py -w workflow.json -c /workspace/ComfyUI --parallel 1
```

### Using the CLI

```bash
# Analyze workflow
comfyui-deploy analyze workflows/longcat-img2video.json

# Deploy with all dependencies
comfyui-deploy deploy workflows/longcat-img2video.json --comfyui-path /workspace/ComfyUI

# Search for a model
comfyui-deploy search "LongCat"
```

## How It Works

### Directory Structure

The tool uses ComfyUI's standard directory structure. You provide the path to your ComfyUI installation, and the tool automatically derives all other paths:

```
ComfyUI/                          ← You provide this path with --comfyui
├── custom_nodes/                 ← Custom nodes installed here
│   ├── ComfyUI-WanVideoWrapper/
│   ├── comfyui-kjnodes/
│   └── ...
├── models/                       ← Models downloaded here
│   ├── checkpoints/              ← SD/SDXL checkpoints
│   ├── loras/                    ← LoRA models
│   ├── vae/                      ← VAE models
│   ├── clip/                     ← CLIP/T5 text encoders
│   ├── diffusion_models/         ← Flux/Wan/LongCat diffusion models
│   ├── controlnet/               ← ControlNet models
│   ├── upscale_models/           ← ESRGAN/upscalers
│   └── ...
├── python_embeded/               ← Portable install Python (Windows)
└── venv/                         ← Virtual environment (Linux/Mac)
```

### Model Type Detection

The tool automatically detects model types from workflow node types and places them in the correct folder:

| Node Type Pattern | Model Type | Target Folder |
|-------------------|------------|---------------|
| `CheckpointLoader*` | checkpoint | `models/checkpoints` |
| `LoraLoader*`, `*LoraSelect*` | lora | `models/loras` |
| `VAELoader*` | vae | `models/vae` |
| `CLIPLoader*`, `*TextEncode*` | clip | `models/clip` |
| `*DiffusionModel*`, `WanVideo*` | diffusion | `models/diffusion_models` |
| `ControlNetLoader*` | controlnet | `models/controlnet` |
| `UpscaleModel*`, `*ESRGAN*` | upscale | `models/upscale_models` |

### Smart Model Search

When a workflow doesn't include download URLs, the tool searches multiple sources:

```
┌─────────────────────────────────────────────────────────┐
│                    Search Flow                           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│ 1. ComfyUI Manager List (instant, curated)              │
│    - Fetches from GitHub: ltdrdata/ComfyUI-Manager      │
│    - ~500+ models with verified URLs                    │
└─────────────────────────────────────────────────────────┘
                           │ not found
                           ▼
┌─────────────────────────────────────────────────────────┐
│ 2. HuggingFace Smart Search                              │
│    - Pattern matching: "longcat" → Kijai/LongCat-Video   │
│    - Lists actual files in candidate repos via API       │
│    - NO API key required                                 │
└─────────────────────────────────────────────────────────┘
                           │ not found
                           ▼
┌─────────────────────────────────────────────────────────┐
│ 3. CivitAI Search                                        │
│    - Searches public API by filename                     │
│    - NO API key required for public models               │
│    - Key only needed for NSFW/gated downloads            │
└─────────────────────────────────────────────────────────┘
```

### Pattern-Based Repository Mapping

The tool maps filename patterns to likely HuggingFace repositories:

| Pattern | Repositories Checked |
|---------|---------------------|
| `longcat*` | `Kijai/LongCat-Video_comfy` |
| `wan*`, `umt5*` | `Comfy-Org/Wan_2.1_ComfyUI_repackaged`, `ALGOTECH/WanVideo_comfy` |
| `flux*` | `black-forest-labs/FLUX.1-dev`, `comfyanonymous/flux_text_encoders` |
| `sd_xl*`, `sdxl*` | `stabilityai/stable-diffusion-xl-base-1.0` |
| `control*` | `lllyasviel/ControlNet-v1-1` |

### Custom Node Installation

1. **Identifies** custom nodes from workflow's `cnr_id` property
2. **Looks up** GitHub URL from known repository mapping
3. **Clones** repository to `ComfyUI/custom_nodes/`
4. **Installs** Python dependencies from `requirements.txt`

Known custom node repositories:
- `ComfyUI-WanVideoWrapper` → `github.com/kijai/ComfyUI-WanVideoWrapper`
- `comfyui-videohelpersuite` → `github.com/Kosinkadink/ComfyUI-VideoHelperSuite`
- `comfyui-kjnodes` → `github.com/kijai/ComfyUI-KJNodes`
- `rgthree-comfy` → `github.com/rgthree/rgthree-comfy`
- And 20+ more...

## Configuration (Optional)

For gated models or higher API rate limits, configure authentication:

```bash
# Set HuggingFace token (for gated models like Flux-dev)
comfyui-deploy config set hf_token YOUR_HF_TOKEN

# Set CivitAI API key (for NSFW models)
comfyui-deploy config set civitai_api_key YOUR_CIVITAI_KEY
```

Or use environment variables:
```bash
export HF_TOKEN=your_token
export CIVITAI_API_KEY=your_key
```

Config file location: `~/.comfyui-deploy/config.yaml`

## Command Reference

### deploy_workflow.py

```bash
python deploy_workflow.py --workflow WORKFLOW --comfyui COMFYUI [--dry-run]

Arguments:
  --workflow, -w    Path to workflow JSON file (required)
  --comfyui, -c     Path to ComfyUI installation (required)
  --dry-run, -n     Analyze only, don't download/install anything
```

### comfyui-deploy CLI

```bash
# Analyze workflow
comfyui-deploy analyze <workflow.json> [--output report.yaml]

# Deploy workflow
comfyui-deploy deploy <workflow.json> --comfyui-path <path>
    [--download-models/--no-download-models]
    [--install-nodes/--no-install-nodes]
    [--dry-run]

# Search for models
comfyui-deploy search <query> [--type lora|checkpoint|vae] [--limit 10]

# Manage configuration
comfyui-deploy config show
comfyui-deploy config set <key> <value>
comfyui-deploy config init

# List installed custom nodes
comfyui-deploy list-nodes --comfyui-path <path>
```

## Example Output

```
======================================================================
PARSING WORKFLOW
======================================================================

📦 Models required: 4
   ❌ LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors (diffusion)
      → models/diffusion_models
   ❌ LongCat_distill_lora_alpha64_bf16.safetensors (lora)
      → models/loras
   ❌ wan_2.1_vae.safetensors (vae)
      → models/vae
   ❌ umt5-xxl-enc-bf16.safetensors (clip)
      → models/clip

🧩 Custom nodes required: 3
   ✅ ComfyUI-WanVideoWrapper
      → https://github.com/kijai/ComfyUI-WanVideoWrapper
   ✅ comfyui-kjnodes
      → https://github.com/kijai/ComfyUI-KJNodes
   ✅ comfyui-videohelpersuite
      → https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
```

## Extending the Tool

### Adding New Model Sources

Edit `src/comfyui_deploy/smart_search.py`:

```python
REPO_PATTERNS = {
    r"my_model_pattern": [
        "username/repo-name",
    ],
}
```

### Adding New Custom Node Mappings

Edit `src/comfyui_deploy/node_installer.py`:

```python
CUSTOM_NODE_REPOS = {
    "my-custom-node": "https://github.com/user/my-custom-node",
}
```

## Setup ComfyUI with Default Packages

The tool includes a setup script that installs essential custom nodes and can update ComfyUI.

### Default Packages

The following packages are configured as defaults in `default-packages.json`:

| Package | Description |
|---------|-------------|
| **ComfyUI-Manager** | Essential node manager - install/update nodes from UI |
| **ComfyUI-Impact-Pack** | Detector, detailer, upscaler, and utility nodes |
| **ComfyUI-Inspire-Pack** | Regional prompting, wildcards, creative tools |
| **rgthree-comfy** | Quality of life nodes - context, reroute, bookmark |
| **ComfyUI-VideoHelperSuite** | Video loading, combining, and export utilities |

### Install Default Packages

```bash
# Install all default packages
python setup_comfyui.py --comfyui /workspace/ComfyUI

# Dry run - see what would be installed
python setup_comfyui.py --comfyui /workspace/ComfyUI --dry-run
```

### Update ComfyUI

```bash
# Update ComfyUI to latest version
python setup_comfyui.py --comfyui /workspace/ComfyUI --update

# Update ComfyUI AND install default packages
python setup_comfyui.py --comfyui /workspace/ComfyUI --update

# Only update (skip default packages)
python setup_comfyui.py --comfyui /workspace/ComfyUI --update --no-defaults
```

### Custom Default Packages

You can customize the default packages by editing `default-packages.json`:

```json
{
  "description": "My custom default packages",
  "custom_nodes": [
    {
      "name": "My-Custom-Node",
      "cnr_id": "my-custom-node",
      "github_url": "https://github.com/user/my-custom-node",
      "description": "Description of what it does"
    }
  ]
}
```

Or use a completely custom config file:

```bash
python setup_comfyui.py --comfyui /workspace/ComfyUI --config my-packages.json
```

## Troubleshooting

### Model not found
- Check if the model name is spelled correctly in the workflow
- Try searching manually: `comfyui-deploy search "model_name"`
- Add the model to the pattern mapping in `smart_search.py`

### Custom node installation fails
- Ensure `git` is installed and in PATH
- Check if the repository URL is correct
- Try cloning manually: `git clone <url> custom_nodes/<name>`

### Permission errors
- Run with administrator privileges (Windows)
- Check folder permissions on the ComfyUI directory

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please submit pull requests for:
- New model repository mappings
- New custom node mappings
- Bug fixes and improvements

