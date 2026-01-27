"""
Smart Model Search - Enhanced search using repo file listing and pattern matching.

This module improves model discovery by:
1. Mapping filename patterns to known HuggingFace repos
2. Directly listing files in those repos via API
3. Fuzzy matching filenames
4. Fetching from ComfyUI Manager's model list
"""

import re
import time
from dataclasses import dataclass
from typing import Any
from functools import lru_cache

import requests
from rich.console import Console

console = Console()


@dataclass
class SmartSearchResult:
    """Result from smart model search."""
    filename: str
    download_url: str
    repo_id: str
    source: str = "huggingface"
    size_bytes: int | None = None
    confidence: str = "exact"  # exact, fuzzy, pattern


class SmartModelSearch:
    """
    Enhanced model search using multiple strategies.
    """
    
    # Pattern -> List of HuggingFace repos to check
    # Order matters - first match wins
    REPO_PATTERNS = {
        # LongCat Video models
        r"longcat": [
            "Kijai/LongCat-Video_comfy",
        ],
        
        # Wan Video models  
        r"wan.?2\.?1|wanvideo|umt5": [
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "ALGOTECH/WanVideo_comfy",
            "Wan-AI/Wan2.1-T2V-14B",
        ],
        
        # Flux models
        r"flux": [
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
            "comfyanonymous/flux_text_encoders",
        ],
        
        # SDXL models
        r"sd.?xl|sdxl": [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            "stabilityai/sdxl-turbo",
            "stabilityai/sdxl-vae",
            "madebyollin/sdxl-vae-fp16-fix",
        ],
        
        # SD 1.5 / 2.x models
        r"v1-5|sd-?v?1\.?5|stable.?diffusion.?1": [
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/sd-vae-ft-mse-original",
        ],
        r"v2-1|sd-?v?2": [
            "stabilityai/stable-diffusion-2-1",
            "stabilityai/stable-diffusion-2-1-base",
        ],
        
        # ControlNet
        r"control.?net|control_v11": [
            "lllyasviel/ControlNet-v1-1",
            "diffusers/controlnet-canny-sdxl-1.0",
            "lllyasviel/sd_control_collection",
        ],
        
        # IP-Adapter
        r"ip.?adapter": [
            "h94/IP-Adapter",
        ],
        
        # AnimateDiff
        r"animatediff|mm_sd": [
            "guoyww/animatediff",
            "ByteDance/AnimateDiff-Lightning",
        ],
        
        # Text encoders
        r"t5xxl|clip_l|text.?encoder": [
            "comfyanonymous/flux_text_encoders",
            "google/t5-v1_1-xxl",
        ],
        
        # Upscalers
        r"esrgan|upscale|ultrasharp|4x": [
            "ai-forever/Real-ESRGAN",
            "Kim2091/UltraSharp",
        ],
        
        # VAE models
        r"vae": [
            "stabilityai/sd-vae-ft-mse-original",
            "stabilityai/sdxl-vae",
        ],
        
        # Popular community models
        r"realvis|photon|juggernaut": [
            "SG161222/RealVisXL_V4.0",
            "dataautogpt3/Photon",
        ],
    }
    
    # Direct repo mappings for specific model creators/projects
    CREATOR_REPOS = {
        "kijai": [
            "Kijai/LongCat-Video_comfy",
            "Kijai/flux-fp8",
            "Kijai/DepthCrafter-comfy",
        ],
        "comfyanonymous": [
            "comfyanonymous/flux_text_encoders",
        ],
        "comfy-org": [
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "Comfy-Org/stable-diffusion-3.5-fp8",
        ],
        "stabilityai": [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/sdxl-vae",
        ],
    }
    
    # Cache for repo file listings
    _repo_cache: dict[str, list[dict]] = {}
    _cache_ttl = 3600  # 1 hour
    _cache_times: dict[str, float] = {}
    
    def __init__(self, hf_token: str | None = None):
        self.hf_token = hf_token
        self.session = requests.Session()
        if hf_token:
            self.session.headers["Authorization"] = f"Bearer {hf_token}"
    
    def search(self, filename: str, model_type: str | None = None) -> SmartSearchResult | None:
        """
        Search for a model using smart pattern matching and repo listing.
        
        Args:
            filename: The model filename to find
            model_type: Optional hint about model type
        
        Returns:
            SmartSearchResult if found, None otherwise
        """
        filename_lower = filename.lower()
        
        # Strategy 1: Find repos that might contain this model
        candidate_repos = self._find_candidate_repos(filename_lower)
        
        console.print(f"[dim]  Checking {len(candidate_repos)} candidate repos...[/dim]")
        
        # Strategy 2: List files in each repo and look for match
        for repo_id in candidate_repos:
            files = self._list_repo_files(repo_id)
            
            # Look for exact match
            for f in files:
                if f["path"].lower() == filename_lower:
                    return SmartSearchResult(
                        filename=f["path"],
                        download_url=f"https://huggingface.co/{repo_id}/resolve/main/{f['path']}",
                        repo_id=repo_id,
                        size_bytes=f.get("size"),
                        confidence="exact",
                    )
            
            # Look for fuzzy match (same base name, different precision suffix)
            base_name = self._get_base_name(filename)
            for f in files:
                if self._get_base_name(f["path"]) == base_name:
                    return SmartSearchResult(
                        filename=f["path"],
                        download_url=f"https://huggingface.co/{repo_id}/resolve/main/{f['path']}",
                        repo_id=repo_id,
                        size_bytes=f.get("size"),
                        confidence="fuzzy",
                    )
        
        return None
    
    def _find_candidate_repos(self, filename_lower: str) -> list[str]:
        """Find repos that might contain the model based on filename patterns."""
        candidates = []
        
        # Check pattern mappings
        for pattern, repos in self.REPO_PATTERNS.items():
            if re.search(pattern, filename_lower, re.IGNORECASE):
                candidates.extend(repos)
        
        # Check creator mappings (extract potential creator from filename)
        for creator, repos in self.CREATOR_REPOS.items():
            if creator in filename_lower:
                candidates.extend(repos)
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for repo in candidates:
            if repo not in seen:
                seen.add(repo)
                unique.append(repo)
        
        return unique
    
    def _list_repo_files(self, repo_id: str) -> list[dict]:
        """List all files in a HuggingFace repo with caching."""
        # Check cache
        now = time.time()
        if repo_id in self._repo_cache:
            cache_time = self._cache_times.get(repo_id, 0)
            if now - cache_time < self._cache_ttl:
                return self._repo_cache[repo_id]
        
        # Fetch from API
        try:
            response = self.session.get(
                f"https://huggingface.co/api/models/{repo_id}/tree/main",
                timeout=30,
            )
            
            if response.ok:
                files = response.json()
                # Filter to model files only
                model_files = [
                    f for f in files 
                    if f.get("type") == "file" and 
                    any(f.get("path", "").lower().endswith(ext) 
                        for ext in (".safetensors", ".ckpt", ".pt", ".pth", ".bin"))
                ]
                
                # Cache the result
                self._repo_cache[repo_id] = model_files
                self._cache_times[repo_id] = now
                
                return model_files
            else:
                console.print(f"[dim]  Could not list {repo_id}: {response.status_code}[/dim]")
                
        except requests.RequestException as e:
            console.print(f"[dim]  Error listing {repo_id}: {e}[/dim]")
        
        return []
    
    def _get_base_name(self, filename: str) -> str:
        """Extract base model name without precision/format suffixes."""
        name = filename.lower()
        
        # Remove extension
        name = re.sub(r"\.(safetensors|ckpt|pt|pth|bin)$", "", name)
        
        # Remove common suffixes
        name = re.sub(r"_(fp16|fp32|bf16|fp8|e4m3fn|e5m2)$", "", name)
        name = re.sub(r"_(scaled|comfy|kj|pruned|ema)$", "", name)
        
        return name
    
    def list_repo_models(self, repo_id: str) -> list[dict]:
        """List all models in a specific repo."""
        files = self._list_repo_files(repo_id)
        
        results = []
        for f in files:
            results.append({
                "filename": f["path"],
                "download_url": f"https://huggingface.co/{repo_id}/resolve/main/{f['path']}",
                "size_bytes": f.get("size"),
                "repo_id": repo_id,
            })
        
        return results
    
    def get_popular_repos(self) -> dict[str, list[str]]:
        """Get list of popular model repos organized by category."""
        return {
            "checkpoints": [
                "stabilityai/stable-diffusion-xl-base-1.0",
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-3.5-large",
            ],
            "video": [
                "Kijai/LongCat-Video_comfy",
                "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                "ALGOTECH/WanVideo_comfy",
            ],
            "flux": [
                "black-forest-labs/FLUX.1-dev",
                "black-forest-labs/FLUX.1-schnell",
                "comfyanonymous/flux_text_encoders",
            ],
            "controlnet": [
                "lllyasviel/ControlNet-v1-1",
            ],
            "vae": [
                "stabilityai/sdxl-vae",
                "stabilityai/sd-vae-ft-mse-original",
            ],
        }


# ComfyUI Manager model list integration
class ComfyUIManagerModels:
    """
    Fetch models from ComfyUI Manager's curated list.
    """
    
    MODEL_LIST_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json"
    
    _cache: list[dict] | None = None
    _cache_time: float = 0
    _cache_ttl = 3600 * 24  # 24 hours
    
    @classmethod
    def fetch_models(cls) -> list[dict]:
        """Fetch the ComfyUI Manager model list."""
        now = time.time()
        
        if cls._cache and (now - cls._cache_time < cls._cache_ttl):
            return cls._cache
        
        try:
            response = requests.get(cls.MODEL_LIST_URL, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            
            cls._cache = models
            cls._cache_time = now
            
            return models
            
        except requests.RequestException as e:
            console.print(f"[yellow]Could not fetch ComfyUI Manager model list: {e}[/yellow]")
            return cls._cache or []
    
    @classmethod
    def find_model(cls, filename: str) -> dict | None:
        """Find a model by filename in the ComfyUI Manager list."""
        models = cls.fetch_models()
        filename_lower = filename.lower()
        
        for model in models:
            model_filename = model.get("filename", "").lower()
            if model_filename == filename_lower:
                return model
            
            # Check URL for filename
            url = model.get("url", "")
            if filename_lower in url.lower():
                return model
        
        return None


class CivitAISearch:
    """
    Search CivitAI for models - works WITHOUT API key for public models.
    API key only needed for NSFW/gated content downloads.
    """
    
    API_BASE = "https://civitai.com/api/v1"
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.session = requests.Session()
    
    def search(self, filename: str, model_type: str | None = None, limit: int = 5) -> list[dict]:
        """
        Search CivitAI for models matching the filename.
        
        Returns list of dicts with: name, filename, download_url, type, downloads
        """
        # Clean filename for search
        search_term = self._filename_to_search_term(filename)
        
        params = {
            "query": search_term,
            "limit": limit,
        }
        
        # Map model type to CivitAI types
        civitai_type = self._get_civitai_type(model_type)
        if civitai_type:
            params["types"] = civitai_type
        
        try:
            response = self.session.get(
                f"{self.API_BASE}/models",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for model in data.get("items", []):
                versions = model.get("modelVersions", [])
                if not versions:
                    continue
                
                # Get files from latest version
                files = versions[0].get("files", [])
                for file_info in files:
                    file_name = file_info.get("name", "")
                    if not self._is_model_file(file_name):
                        continue
                    
                    download_url = file_info.get("downloadUrl", "")
                    
                    # Add API key to URL if available (needed for some downloads)
                    if self.api_key and download_url:
                        sep = "&" if "?" in download_url else "?"
                        download_url = f"{download_url}{sep}token={self.api_key}"
                    
                    results.append({
                        "name": model.get("name"),
                        "filename": file_name,
                        "download_url": download_url,
                        "type": model.get("type", "").lower(),
                        "downloads": model.get("stats", {}).get("downloadCount", 0),
                        "source": "civitai",
                        "model_id": model.get("id"),
                    })
            
            # Sort by filename similarity and downloads
            results.sort(
                key=lambda x: (
                    self._filename_similarity(filename, x["filename"]),
                    x["downloads"]
                ),
                reverse=True,
            )
            
            return results[:limit]
            
        except requests.RequestException as e:
            console.print(f"[dim]  CivitAI search error: {e}[/dim]")
            return []
    
    def find_best_match(self, filename: str, model_type: str | None = None) -> dict | None:
        """Find the best matching model for a filename."""
        results = self.search(filename, model_type=model_type, limit=5)
        
        if not results:
            return None
        
        # Check for exact filename match
        filename_lower = filename.lower()
        for r in results:
            if r["filename"].lower() == filename_lower:
                return r
        
        # Return best match by score
        return results[0] if results else None
    
    def _filename_to_search_term(self, filename: str) -> str:
        """Convert filename to search term."""
        # Remove extension
        name = re.sub(r"\.(safetensors|ckpt|pt|pth|bin)$", "", filename, flags=re.IGNORECASE)
        # Replace separators with spaces
        name = re.sub(r"[_\-]+", " ", name)
        # Remove precision suffixes
        name = re.sub(r"\s*(fp16|fp32|bf16|fp8|e4m3fn|scaled)\s*", " ", name, flags=re.IGNORECASE)
        return name.strip()
    
    def _get_civitai_type(self, model_type: str | None) -> str | None:
        """Map internal model type to CivitAI type."""
        if not model_type:
            return None
        
        type_map = {
            "checkpoint": "Checkpoint",
            "diffusion": "Checkpoint",
            "lora": "LORA",
            "vae": "VAE",
            "embedding": "TextualInversion",
            "controlnet": "Controlnet",
            "upscale": "Upscaler",
        }
        return type_map.get(model_type.lower())
    
    def _is_model_file(self, filename: str) -> bool:
        """Check if filename is a model file."""
        return filename.lower().endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin"))
    
    def _filename_similarity(self, query: str, candidate: str) -> float:
        """Calculate similarity score between filenames."""
        query_lower = query.lower()
        candidate_lower = candidate.lower()
        
        if query_lower == candidate_lower:
            return 1.0
        if query_lower in candidate_lower:
            return 0.8
        if candidate_lower in query_lower:
            return 0.7
        
        # Word overlap
        query_words = set(re.split(r"[_\-\s.]+", query_lower))
        candidate_words = set(re.split(r"[_\-\s.]+", candidate_lower))
        
        overlap = len(query_words & candidate_words)
        total = len(query_words | candidate_words)
        
        return overlap / total if total > 0 else 0.0


def smart_search(
    filename: str, 
    model_type: str | None = None, 
    hf_token: str | None = None,
    civitai_key: str | None = None,
    sources: list[str] | None = None,
) -> dict | None:
    """
    Convenience function for smart model search across multiple sources.
    
    Args:
        filename: Model filename to search for
        model_type: Optional hint (checkpoint, lora, vae, etc.)
        hf_token: Optional HuggingFace token
        civitai_key: Optional CivitAI API key (not required for public models)
        sources: List of sources to search (default: all)
    
    Returns dict with: filename, download_url, source, confidence
    """
    if sources is None:
        sources = ["comfyui-manager", "huggingface", "civitai"]
    
    # 1. Try ComfyUI Manager first (curated list, fast)
    if "comfyui-manager" in sources:
        manager_result = ComfyUIManagerModels.find_model(filename)
        if manager_result:
            return {
                "filename": manager_result.get("filename", filename),
                "download_url": manager_result.get("url"),
                "source": "comfyui-manager",
                "confidence": "curated",
            }
    
    # 2. Try smart HuggingFace search (pattern matching + repo listing)
    if "huggingface" in sources:
        searcher = SmartModelSearch(hf_token=hf_token)
        result = searcher.search(filename, model_type=model_type)
        
        if result:
            return {
                "filename": result.filename,
                "download_url": result.download_url,
                "repo_id": result.repo_id,
                "source": "huggingface",
                "confidence": result.confidence,
            }
    
    # 3. Try CivitAI search (works without API key for public models)
    if "civitai" in sources:
        console.print(f"[dim]  Searching CivitAI...[/dim]")
        civitai = CivitAISearch(api_key=civitai_key)
        result = civitai.find_best_match(filename, model_type=model_type)
        
        if result:
            return {
                "filename": result["filename"],
                "download_url": result["download_url"],
                "source": "civitai",
                "confidence": "search",
                "model_id": result.get("model_id"),
            }
    
    return None


if __name__ == "__main__":
    # Test the smart search
    test_models = [
        "LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors",
        "LongCat_distill_lora_alpha64_bf16.safetensors",
        "wan_2.1_vae.safetensors",
        "umt5-xxl-enc-bf16.safetensors",
        "sd_xl_base_1.0.safetensors",
    ]
    
    print("Smart Model Search Test")
    print("=" * 60)
    
    for filename in test_models:
        print(f"\nSearching: {filename}")
        result = smart_search(filename)
        
        if result:
            print(f"  ✓ Found! ({result['confidence']})")
            print(f"    URL: {result['download_url'][:70]}...")
        else:
            print(f"  ✗ Not found")
