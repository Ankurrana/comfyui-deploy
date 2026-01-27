"""
Model Resolver - Advanced model search and resolution strategies.

This module handles the case when workflows don't have embedded download URLs.
It uses multiple strategies to find models:
1. Known model database (pre-mapped popular models)
2. HuggingFace Hub API search
3. CivitAI API search  
4. Fuzzy filename matching
5. Community model lists
"""

import re
from dataclasses import dataclass
from typing import Any

from rich.console import Console

console = Console()


# Comprehensive database of known ComfyUI models and their download sources
# This is the PRIMARY fallback when API search fails
# TEMPORARILY CLEARED FOR TESTING API SEARCH
KNOWN_MODELS_DB = {
    # Database cleared for testing - will rely on API search
}


# Patterns to identify model types from filenames
MODEL_TYPE_PATTERNS = {
    "checkpoint": [
        r".*\.ckpt$",
        r".*_base.*\.safetensors$",
        r".*_refiner.*\.safetensors$",
        r"v\d+-\d+.*\.safetensors$",
    ],
    "lora": [
        r".*lora.*\.safetensors$",
        r".*_lora_.*\.safetensors$",
    ],
    "vae": [
        r".*vae.*\.safetensors$",
        r".*_vae\.safetensors$",
        r"^ae\.safetensors$",
    ],
    "controlnet": [
        r".*control.*\.pth$",
        r".*controlnet.*\.safetensors$",
    ],
    "upscale": [
        r".*ESRGAN.*\.pth$",
        r".*upscale.*\.pth$",
        r".*x\d+.*\.pth$",
    ],
    "clip": [
        r".*clip.*\.safetensors$",
        r".*t5.*\.safetensors$",
        r".*text_encoder.*\.safetensors$",
    ],
}


@dataclass
class ResolvedModel:
    """A model with resolved download information."""
    filename: str
    download_url: str | None
    source: str | None
    model_type: str
    target_folder: str
    size_gb: float | None = None
    requires_auth: bool = False
    confidence: str = "unknown"  # exact, fuzzy, guessed


class ModelResolver:
    """
    Resolves model filenames to download URLs using multiple strategies.
    """
    
    def __init__(self):
        self.known_models = KNOWN_MODELS_DB
        self._build_alias_index()
    
    def _build_alias_index(self):
        """Build an index of model aliases for fuzzy matching."""
        self.alias_index = {}
        
        for filename, info in self.known_models.items():
            # Index by exact filename
            self.alias_index[filename.lower()] = filename
            
            # Index by aliases
            for alias in info.get("aliases", []):
                self.alias_index[alias.lower()] = filename
            
            # Index by base name (without extension)
            base = filename.rsplit(".", 1)[0].lower()
            if base not in self.alias_index:
                self.alias_index[base] = filename
    
    def resolve(self, filename: str, model_type: str | None = None) -> ResolvedModel:
        """
        Resolve a model filename to download information.
        
        Args:
            filename: The model filename from the workflow
            model_type: Hint about the model type
        
        Returns:
            ResolvedModel with download info (URL may be None if not found)
        """
        # Strategy 1: Exact match in known models
        if filename in self.known_models:
            info = self.known_models[filename]
            return ResolvedModel(
                filename=filename,
                download_url=info["url"],
                source=info["source"],
                model_type=info["type"],
                target_folder=info["folder"],
                size_gb=info.get("size_gb"),
                requires_auth=info.get("requires_auth", False),
                confidence="exact",
            )
        
        # Strategy 2: Case-insensitive match
        filename_lower = filename.lower()
        if filename_lower in self.alias_index:
            canonical = self.alias_index[filename_lower]
            info = self.known_models[canonical]
            return ResolvedModel(
                filename=filename,
                download_url=info["url"],
                source=info["source"],
                model_type=info["type"],
                target_folder=info["folder"],
                size_gb=info.get("size_gb"),
                requires_auth=info.get("requires_auth", False),
                confidence="exact",
            )
        
        # Strategy 3: Fuzzy match by base name
        base_name = filename.rsplit(".", 1)[0].lower()
        fuzzy_match = self._fuzzy_match(base_name)
        if fuzzy_match:
            info = self.known_models[fuzzy_match]
            return ResolvedModel(
                filename=filename,
                download_url=info["url"],
                source=info["source"],
                model_type=info["type"],
                target_folder=info["folder"],
                size_gb=info.get("size_gb"),
                requires_auth=info.get("requires_auth", False),
                confidence="fuzzy",
            )
        
        # Strategy 4: Infer type and folder from filename patterns
        inferred_type = model_type or self._infer_type(filename)
        target_folder = self._get_folder_for_type(inferred_type)
        
        return ResolvedModel(
            filename=filename,
            download_url=None,  # Could not resolve
            source=None,
            model_type=inferred_type,
            target_folder=target_folder,
            confidence="guessed",
        )
    
    def _fuzzy_match(self, base_name: str) -> str | None:
        """Try to find a fuzzy match for a model base name."""
        # Remove common suffixes/prefixes for matching
        clean_name = re.sub(r'_(fp16|fp32|bf16|fp8|e4m3fn|scaled|comfy|kj)$', '', base_name, flags=re.IGNORECASE)
        clean_name = re.sub(r'^(comfy_|kj_)', '', clean_name, flags=re.IGNORECASE)
        
        best_match = None
        best_score = 0
        
        for known_filename in self.known_models:
            known_base = known_filename.rsplit(".", 1)[0].lower()
            known_clean = re.sub(r'_(fp16|fp32|bf16|fp8|e4m3fn|scaled|comfy|kj)$', '', known_base, flags=re.IGNORECASE)
            
            # Check for substring match
            if clean_name in known_clean or known_clean in clean_name:
                score = len(set(clean_name.split("_")) & set(known_clean.split("_")))
                if score > best_score:
                    best_score = score
                    best_match = known_filename
        
        return best_match if best_score >= 2 else None
    
    def _infer_type(self, filename: str) -> str:
        """Infer model type from filename patterns."""
        filename_lower = filename.lower()
        
        for model_type, patterns in MODEL_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, filename_lower):
                    return model_type
        
        # Default based on extension
        if filename_lower.endswith(".ckpt"):
            return "checkpoint"
        elif filename_lower.endswith(".safetensors"):
            return "checkpoint"  # Default assumption
        elif filename_lower.endswith(".pth"):
            return "upscale"
        
        return "unknown"
    
    def _get_folder_for_type(self, model_type: str) -> str:
        """Get the target folder for a model type."""
        folder_map = {
            "checkpoint": "models/checkpoints",
            "diffusion": "models/diffusion_models",
            "lora": "models/loras",
            "vae": "models/vae",
            "clip": "models/clip",
            "controlnet": "models/controlnet",
            "upscale": "models/upscale_models",
            "ipadapter": "models/ipadapter",
            "clip_vision": "models/clip_vision",
            "animatediff": "models/animatediff_models",
            "embedding": "models/embeddings",
            "hypernetwork": "models/hypernetworks",
        }
        return folder_map.get(model_type, "models")
    
    def suggest_search_terms(self, filename: str) -> list[str]:
        """
        Generate search terms for manual search when auto-resolution fails.
        """
        suggestions = []
        
        # Base filename without extension
        base = filename.rsplit(".", 1)[0]
        suggestions.append(base)
        
        # Clean version without version suffixes
        clean = re.sub(r'_(fp16|fp32|bf16|fp8|e4m3fn|scaled|v\d+)$', '', base, flags=re.IGNORECASE)
        if clean != base:
            suggestions.append(clean)
        
        # Split by underscores and take meaningful parts
        parts = [p for p in base.split("_") if len(p) > 2 and not p.isdigit()]
        if len(parts) >= 2:
            suggestions.append(" ".join(parts[:3]))
        
        return list(dict.fromkeys(suggestions))  # Remove duplicates, preserve order
    
    def get_search_urls(self, filename: str) -> dict[str, str]:
        """
        Generate URLs for manual search on different platforms.
        """
        search_term = filename.rsplit(".", 1)[0].replace("_", " ")
        
        return {
            "huggingface": f"https://huggingface.co/models?search={search_term}",
            "civitai": f"https://civitai.com/search/models?query={search_term}",
            "github": f"https://github.com/search?q={search_term}+safetensors&type=code",
        }


# Global resolver instance
_resolver: ModelResolver | None = None


def get_resolver() -> ModelResolver:
    """Get or create the global resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = ModelResolver()
    return _resolver
