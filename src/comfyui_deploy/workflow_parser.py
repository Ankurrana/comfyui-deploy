"""
Workflow Parser - Extracts models, custom nodes, and dependencies from ComfyUI workflow JSON files.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .node_database import get_node_database, NodeDatabase


@dataclass
class ModelReference:
    """Represents a model file referenced in a workflow."""
    
    filename: str
    node_type: str
    node_id: int
    model_type: str  # checkpoint, lora, vae, clip, controlnet, upscale, diffusion, embedding, etc.
    target_folder: str  # The ComfyUI folder where this model should be placed
    download_url: str | None = None
    source: str | None = None  # huggingface, civitai, github, etc.
    
    def __hash__(self):
        return hash((self.filename, self.model_type))
    
    def __eq__(self, other):
        if not isinstance(other, ModelReference):
            return False
        return self.filename == other.filename and self.model_type == other.model_type


@dataclass
class CustomNodeReference:
    """Represents a custom node pack required by the workflow."""
    
    cnr_id: str  # ComfyUI Node Registry ID
    version: str | None = None
    node_types: list[str] = field(default_factory=list)
    github_url: str | None = None
    
    def __hash__(self):
        return hash(self.cnr_id)
    
    def __eq__(self, other):
        if not isinstance(other, CustomNodeReference):
            return False
        return self.cnr_id == other.cnr_id


@dataclass
class WorkflowDependencies:
    """Complete dependency information for a workflow."""
    
    models: list[ModelReference] = field(default_factory=list)
    custom_nodes: list[CustomNodeReference] = field(default_factory=list)
    embedded_docs: list[str] = field(default_factory=list)  # Documentation found in workflow
    download_urls: dict[str, str] = field(default_factory=dict)  # Model name -> URL
    

class WorkflowParser:
    """Parses ComfyUI workflow JSON files to extract all dependencies."""
    
    # Node type patterns to model folder mappings
    MODEL_TYPE_MAPPINGS = {
        # Checkpoint loaders
        r".*[Cc]heckpoint.*[Ll]oader.*": ("checkpoint", "models/checkpoints"),
        r".*[Uu]net.*[Ll]oader.*": ("unet", "models/diffusion_models"),
        r".*[Dd]iffusion.*[Mm]odel.*[Ll]oader.*": ("diffusion", "models/diffusion_models"),
        
        # LoRA loaders
        r".*[Ll]ora.*[Ss]elect.*": ("lora", "models/loras"),
        r".*[Ll]ora.*[Ll]oader.*": ("lora", "models/loras"),
        
        # VAE loaders
        r".*VAE.*[Ll]oader.*": ("vae", "models/vae"),
        
        # CLIP/Text Encoders
        r".*[Cc][Ll][Ii][Pp].*[Ll]oader.*": ("clip", "models/clip"),
        r".*[Tt]ext.*[Ee]ncod.*": ("clip", "models/clip"),
        r".*T5.*[Ll]oader.*": ("clip", "models/clip"),
        r".*UMT5.*": ("clip", "models/clip"),
        
        # ControlNet
        r".*[Cc]ontrol[Nn]et.*[Ll]oader.*": ("controlnet", "models/controlnet"),
        r".*[Cc]ontrol.*[Nn]et.*[Aa]pply.*": ("controlnet", "models/controlnet"),
        
        # Upscalers
        r".*[Uu]pscale.*[Mm]odel.*[Ll]oader.*": ("upscale", "models/upscale_models"),
        r".*[Ll]atent.*[Uu]pscale.*[Mm]odel.*[Ll]oader.*": ("upscale", "models/latent_upscale_models"),
        r".*ESRGAN.*": ("upscale", "models/upscale_models"),
        
        # IP Adapter
        r".*IP.*[Aa]dapter.*[Ll]oader.*": ("ipadapter", "models/ipadapter"),
        r".*IP.*[Aa]dapter.*[Mm]odel.*": ("ipadapter", "models/ipadapter"),
        
        # Embeddings
        r".*[Ee]mbedding.*": ("embedding", "models/embeddings"),
        
        # CLIP Vision
        r".*[Cc][Ll][Ii][Pp].*[Vv]ision.*": ("clip_vision", "models/clip_vision"),
        
        # Style Models
        r".*[Ss]tyle.*[Mm]odel.*": ("style", "models/style_models"),
        
        # Hypernetworks  
        r".*[Hh]ypernetwork.*": ("hypernetwork", "models/hypernetworks"),
        
        # AnimateDiff
        r".*[Aa]nimate[Dd]iff.*[Ll]oader.*": ("animatediff", "models/animatediff_models"),
        
        # WanVideo specific
        r"WanVideo.*[Mm]odel.*[Ll]oader.*": ("diffusion", "models/diffusion_models"),
        r"WanVideo.*[Ll]ora.*": ("lora", "models/loras"),
        r"WanVideo.*VAE.*": ("vae", "models/vae"),
        r"WanVideo.*[Tt]ext[Ee]ncode.*": ("clip", "models/clip"),
    }
    
    # Widget index hints for specific node types (which widget position contains the model name)
    WIDGET_MODEL_INDEX = {
        "CheckpointLoaderSimple": 0,
        "CheckpointLoader": 0,
        "LoraLoader": 0,
        "LoraLoaderModelOnly": 0,
        "VAELoader": 0,
        "CLIPLoader": 0,
        "ControlNetLoader": 0,
        "UpscaleModelLoader": 0,
        "WanVideoModelLoader": 0,
        "WanVideoLoraSelect": 0,
        "WanVideoVAELoader": 0,
        "WanVideoTextEncodeCached": 0,
    }

    # Node type patterns that indicate a custom node (for nodes without cnr_id)
    # Maps pattern in node type to cnr_id
    NODE_TYPE_TO_CNR_ID = {
        r"\(rgthree\)": "rgthree-comfy",
        r"\(WAS\)": "was-node-suite-comfyui",
        r"\(pysssss\)": "comfyui-custom-scripts",
        r"^KJ": "comfyui-kjnodes",
        r"^VHS_": "comfyui-videohelpersuite",
        r"^Impact": "comfyui-impact-pack",
        r"^Efficiency": "efficiency-nodes-comfyui",
    }

    # Known custom node repositories
    CUSTOM_NODE_REPOS = {
        "ComfyUI-WanVideoWrapper": "https://github.com/kijai/ComfyUI-WanVideoWrapper",
        "comfyui-videohelpersuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "rgthree-comfy": "https://github.com/rgthree/rgthree-comfy",
        "comfyui-kjnodes": "https://github.com/kijai/ComfyUI-KJNodes",
        "comfyui-impact-pack": "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "comfyui-manager": "https://github.com/ltdrdata/ComfyUI-Manager",
        "comfyui-controlnet-aux": "https://github.com/Fannovel16/comfyui_controlnet_aux",
        "comfyui-animatediff-evolved": "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved",
        "was-node-suite-comfyui": "https://github.com/WASasquatch/was-node-suite-comfyui",
        "comfyui-custom-scripts": "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
        "efficiency-nodes-comfyui": "https://github.com/jags111/efficiency-nodes-comfyui",
        "comfy-core": None,  # Built-in, no need to install
    }

    def __init__(self):
        self.dependencies = WorkflowDependencies()
    
    def parse(self, workflow_path: str | Path) -> WorkflowDependencies:
        """Parse a workflow file and extract all dependencies."""
        workflow_path = Path(workflow_path)
        
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow_data = json.load(f)
        
        return self.parse_data(workflow_data)
    
    def parse_data(self, workflow_data: dict[str, Any]) -> WorkflowDependencies:
        """Parse workflow data dictionary and extract all dependencies."""
        self.dependencies = WorkflowDependencies()
        
        nodes = workflow_data.get("nodes", [])
        
        for node in nodes:
            self._process_node(node)
        
        # Extract embedded documentation (often contains download URLs)
        self._extract_embedded_docs(nodes)
        
        # Try to match download URLs to models
        self._match_download_urls()
        
        # Deduplicate
        self.dependencies.models = list(set(self.dependencies.models))
        self.dependencies.custom_nodes = list(set(self.dependencies.custom_nodes))
        
        return self.dependencies
    
    def _process_node(self, node: dict[str, Any]) -> None:
        """Process a single node and extract its dependencies."""
        node_type = node.get("type", "")
        node_id = node.get("id", 0)
        properties = node.get("properties", {})
        widgets_values = node.get("widgets_values", [])
        
        # Extract custom node reference from cnr_id
        cnr_id = properties.get("cnr_id")
        if cnr_id and cnr_id != "comfy-core":
            self._add_custom_node(cnr_id, node_type, properties.get("ver"))
        
        # Also check node type for custom nodes without cnr_id
        # Uses ComfyUI Manager database + local pattern fallback
        if not cnr_id or cnr_id == "comfy-core":
            detected = self._detect_cnr_id_from_type(node_type)
            if detected:
                detected_cnr_id, detected_github_url = detected
                self._add_custom_node(detected_cnr_id, node_type, None, detected_github_url)
        
        # Extract embedded model URLs from properties.models array (ComfyUI format)
        embedded_models = properties.get("models", [])
        embedded_url_map = {}
        embedded_directory_map = {}
        for em in embedded_models:
            if isinstance(em, dict) and em.get("name") and em.get("url"):
                embedded_url_map[em["name"]] = em["url"]
                if em.get("directory"):
                    embedded_directory_map[em["name"]] = em["directory"]
        
        # Extract model references
        model_info = self._get_model_type_and_folder(node_type)
        if model_info:
            model_type, target_folder = model_info
            model_name = self._extract_model_name(node_type, widgets_values)
            
            if model_name and self._is_valid_model_name(model_name):
                # Check for embedded URL and directory override
                download_url = embedded_url_map.get(model_name)
                
                # Use embedded directory if available (more accurate)
                if model_name in embedded_directory_map:
                    embedded_dir = embedded_directory_map[model_name]
                    target_folder = f"models/{embedded_dir}"
                
                model_ref = ModelReference(
                    filename=model_name,
                    node_type=node_type,
                    node_id=node_id,
                    model_type=model_type,
                    target_folder=target_folder,
                    download_url=download_url,
                    source="workflow" if download_url else None,
                )
                self.dependencies.models.append(model_ref)
    
    def _add_custom_node(self, cnr_id: str, node_type: str, version: str | None, github_url: str | None = None) -> None:
        """Add or update a custom node reference."""
        # Use provided github_url, or look up from local mapping
        if github_url is None:
            github_url = self.CUSTOM_NODE_REPOS.get(cnr_id)
        
        # Check if already exists
        existing = next(
            (n for n in self.dependencies.custom_nodes if n.cnr_id == cnr_id),
            None
        )
        if existing:
            if node_type not in existing.node_types:
                existing.node_types.append(node_type)
            # Update github_url if we found one and existing doesn't have it
            if github_url and not existing.github_url:
                existing.github_url = github_url
        else:
            node_ref = CustomNodeReference(
                cnr_id=cnr_id,
                version=version,
                node_types=[node_type],
                github_url=github_url,
            )
            self.dependencies.custom_nodes.append(node_ref)
    
    def _detect_cnr_id_from_type(self, node_type: str) -> tuple[str, str] | None:
        """Detect custom node from node type using ComfyUI Manager database.
        
        First checks the ComfyUI Manager's extension-node-map.json database,
        then falls back to local pattern matching.
        
        Args:
            node_type: The node type string (e.g., 'Label (rgthree)')
            
        Returns:
            Tuple of (cnr_id, github_url) or None if not found
        """
        # First, try ComfyUI Manager's database
        db = get_node_database()
        github_url = db.get_repo_for_node_type(node_type)
        
        if github_url:
            # Extract cnr_id from the GitHub URL
            cnr_id = db.extract_cnr_id_from_url(github_url)
            return (cnr_id, github_url)
        
        # Fall back to local pattern matching
        for pattern, cnr_id in self.NODE_TYPE_TO_CNR_ID.items():
            if re.search(pattern, node_type):
                github_url = self.CUSTOM_NODE_REPOS.get(cnr_id)
                return (cnr_id, github_url)
        
        return None

    def _get_model_type_and_folder(self, node_type: str) -> tuple[str, str] | None:
        """Determine the model type and target folder based on node type."""
        for pattern, (model_type, folder) in self.MODEL_TYPE_MAPPINGS.items():
            if re.match(pattern, node_type, re.IGNORECASE):
                return (model_type, folder)
        return None
    
    def _extract_model_name(self, node_type: str, widgets_values: list) -> str | None:
        """Extract the model filename from widget values."""
        if not widgets_values:
            return None
        
        # Handle both list and dict widget values
        if isinstance(widgets_values, dict):
            # Some nodes store values as dict
            for key, value in widgets_values.items():
                if isinstance(value, str) and self._is_valid_model_name(value):
                    return value
            return None
        
        # Use known index if available
        node_type_base = node_type.split("(")[0].strip()  # Remove any suffix like "(rgthree)"
        if node_type_base in self.WIDGET_MODEL_INDEX:
            idx = self.WIDGET_MODEL_INDEX[node_type_base]
            if idx < len(widgets_values):
                value = widgets_values[idx]
                if isinstance(value, str):
                    return value
        
        # Otherwise, look for safetensors/ckpt/pt files in widget values
        for value in widgets_values:
            if isinstance(value, str) and self._is_valid_model_name(value):
                return value
        
        return None
    
    def _is_valid_model_name(self, name: str) -> bool:
        """Check if a string looks like a valid model filename."""
        if not name:
            return False
        
        model_extensions = (
            ".safetensors", ".ckpt", ".pt", ".pth", ".bin", 
            ".onnx", ".pkl", ".pickle"
        )
        
        return any(name.lower().endswith(ext) for ext in model_extensions)
    
    def _extract_embedded_docs(self, nodes: list[dict]) -> None:
        """Extract documentation embedded in workflow nodes (like MarkdownNote)."""
        for node in nodes:
            node_type = node.get("type", "")
            
            # Check for markdown/note nodes
            if any(note_type in node_type.lower() for note_type in ["markdown", "note", "comment", "text"]):
                widgets_values = node.get("widgets_values", [])
                
                for value in widgets_values:
                    if isinstance(value, str) and len(value) > 50:  # Likely documentation
                        self.dependencies.embedded_docs.append(value)
                        
                        # Extract URLs from the documentation
                        self._extract_urls_from_text(value)
    
    def _extract_urls_from_text(self, text: str) -> None:
        """Extract download URLs from text (e.g., markdown documentation)."""
        # Pattern for HuggingFace direct download links
        hf_pattern = r'https://huggingface\.co/[^\s\)]+\.safetensors[^\s\)]*'
        hf_matches = re.findall(hf_pattern, text)
        
        for url in hf_matches:
            # Clean up URL (remove markdown artifacts)
            clean_url = url.rstrip(')').rstrip(']')
            
            # Extract filename from URL
            filename_match = re.search(r'/([^/]+\.safetensors)', clean_url)
            if filename_match:
                filename = filename_match.group(1)
                self.dependencies.download_urls[filename] = clean_url
        
        # Pattern for CivitAI links
        civitai_pattern = r'https://civitai\.com/[^\s\)]+'
        civitai_matches = re.findall(civitai_pattern, text)
        
        for url in civitai_matches:
            clean_url = url.rstrip(')').rstrip(']')
            # CivitAI links are harder to parse, store them with a generic key
            if "download" in clean_url.lower():
                self.dependencies.download_urls[f"civitai_{hash(clean_url)}"] = clean_url
        
        # Pattern for GitHub releases
        github_pattern = r'https://github\.com/[^\s\)]+/releases/download/[^\s\)]+'
        github_matches = re.findall(github_pattern, text)
        
        for url in github_matches:
            clean_url = url.rstrip(')').rstrip(']')
            filename = clean_url.split('/')[-1]
            if filename:
                self.dependencies.download_urls[filename] = clean_url
    
    def _match_download_urls(self) -> None:
        """Match extracted download URLs to model references."""
        for model in self.dependencies.models:
            # Direct filename match
            if model.filename in self.dependencies.download_urls:
                model.download_url = self.dependencies.download_urls[model.filename]
                model.source = self._detect_source(model.download_url)
            
            # Try partial match (without extension variations)
            else:
                base_name = model.filename.rsplit('.', 1)[0]
                for url_filename, url in self.dependencies.download_urls.items():
                    if base_name in url_filename or base_name.lower() in url.lower():
                        model.download_url = url
                        model.source = self._detect_source(url)
                        break
    
    def _detect_source(self, url: str) -> str:
        """Detect the source platform from a URL."""
        url_lower = url.lower()
        
        if "huggingface.co" in url_lower:
            return "huggingface"
        elif "civitai.com" in url_lower:
            return "civitai"
        elif "github.com" in url_lower:
            return "github"
        elif "drive.google.com" in url_lower:
            return "gdrive"
        else:
            return "direct"
    
    def to_dict(self) -> dict:
        """Convert dependencies to a dictionary for serialization."""
        return {
            "models": [
                {
                    "filename": m.filename,
                    "node_type": m.node_type,
                    "model_type": m.model_type,
                    "target_folder": m.target_folder,
                    "download_url": m.download_url,
                    "source": m.source,
                }
                for m in self.dependencies.models
            ],
            "custom_nodes": [
                {
                    "cnr_id": n.cnr_id,
                    "version": n.version,
                    "node_types": n.node_types,
                    "github_url": n.github_url,
                }
                for n in self.dependencies.custom_nodes
            ],
            "download_urls": self.dependencies.download_urls,
        }


def main():
    """Test the parser with a sample workflow."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python workflow_parser.py <workflow.json>")
        sys.exit(1)
    
    parser = WorkflowParser()
    deps = parser.parse(sys.argv[1])
    
    print("\n=== Models Required ===")
    for model in deps.models:
        print(f"  - {model.filename}")
        print(f"    Type: {model.model_type}")
        print(f"    Folder: {model.target_folder}")
        if model.download_url:
            print(f"    URL: {model.download_url[:80]}...")
        print()
    
    print("\n=== Custom Nodes Required ===")
    for node in deps.custom_nodes:
        print(f"  - {node.cnr_id}")
        if node.github_url:
            print(f"    GitHub: {node.github_url}")
        print(f"    Nodes: {', '.join(node.node_types[:3])}{'...' if len(node.node_types) > 3 else ''}")
        print()


if __name__ == "__main__":
    main()
