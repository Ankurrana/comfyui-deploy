"""
Model Database - Registry and search functionality for ComfyUI models across multiple platforms.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests
from rich.console import Console

console = Console()


@dataclass
class ModelSearchResult:
    """Represents a model found from search."""
    
    name: str
    filename: str
    download_url: str
    source: str  # huggingface, civitai, github
    size_bytes: int | None = None
    model_type: str | None = None
    repo_id: str | None = None  # For HuggingFace
    model_id: int | None = None  # For CivitAI
    description: str | None = None
    sha256: str | None = None
    score: float = 0.0  # Search relevance score
    
    def size_str(self) -> str:
        """Return human-readable size."""
        if not self.size_bytes:
            return "Unknown"
        
        for unit in ["B", "KB", "MB", "GB"]:
            if self.size_bytes < 1024:
                return f"{self.size_bytes:.1f} {unit}"
            self.size_bytes /= 1024
        return f"{self.size_bytes:.1f} TB"


class HuggingFaceSearch:
    """Search for models on HuggingFace."""
    
    API_BASE = "https://huggingface.co/api"
    
    def __init__(self, token: str | None = None):
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
    
    def search_models(
        self, 
        query: str, 
        model_type: str | None = None,
        limit: int = 10
    ) -> list[ModelSearchResult]:
        """Search HuggingFace for models matching the query."""
        results = []
        
        # Build search URL
        search_url = f"{self.API_BASE}/models"
        params = {
            "search": query,
            "limit": limit,
            "full": "true",
        }
        
        # Add filters based on model type
        if model_type:
            type_tags = self._get_type_tags(model_type)
            if type_tags:
                params["tags"] = type_tags
        
        try:
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            models = response.json()
            
            for model in models:
                # Get files in the model repo
                repo_id = model.get("id", "")
                files = self._get_model_files(repo_id)
                
                for file_info in files:
                    filename = file_info.get("filename", "")
                    if self._is_model_file(filename):
                        result = ModelSearchResult(
                            name=model.get("id", ""),
                            filename=filename,
                            download_url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
                            source="huggingface",
                            size_bytes=file_info.get("size"),
                            model_type=model_type,
                            repo_id=repo_id,
                            description=model.get("description", ""),
                            sha256=file_info.get("sha256"),
                            score=self._calculate_score(query, filename, model),
                        )
                        results.append(result)
        
        except requests.RequestException as e:
            console.print(f"[yellow]HuggingFace search error: {e}[/yellow]")
        
        # Sort by relevance score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def search_by_filename(self, filename: str, limit: int = 5) -> list[ModelSearchResult]:
        """Search HuggingFace specifically for a filename."""
        # Clean up filename for search
        search_term = filename.replace(".safetensors", "").replace(".ckpt", "")
        search_term = re.sub(r'[_-]', ' ', search_term)
        
        return self.search_models(search_term, limit=limit)
    
    def get_direct_download_url(self, repo_id: str, filename: str) -> str:
        """Get direct download URL for a file in a repo."""
        return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    
    def _get_model_files(self, repo_id: str) -> list[dict]:
        """Get list of files in a model repository."""
        try:
            url = f"{self.API_BASE}/models/{repo_id}/tree/main"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            files = []
            for item in response.json():
                if item.get("type") == "file":
                    files.append({
                        "filename": item.get("path", ""),
                        "size": item.get("size"),
                        "sha256": item.get("oid"),
                    })
            return files
        except requests.RequestException:
            return []
    
    def _is_model_file(self, filename: str) -> bool:
        """Check if a file is a model file."""
        extensions = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")
        return filename.lower().endswith(extensions)
    
    def _get_type_tags(self, model_type: str) -> str | None:
        """Map model type to HuggingFace tags."""
        tag_map = {
            "checkpoint": "diffusers",
            "diffusion": "diffusers",
            "lora": "lora",
            "vae": "vae",
            "clip": "text-to-image",
            "controlnet": "controlnet",
        }
        return tag_map.get(model_type)
    
    def _calculate_score(self, query: str, filename: str, model: dict) -> float:
        """Calculate relevance score for a search result."""
        score = 0.0
        query_lower = query.lower()
        filename_lower = filename.lower()
        
        # Exact filename match
        if query_lower in filename_lower:
            score += 100
        
        # Partial match
        query_parts = query_lower.replace("-", " ").replace("_", " ").split()
        for part in query_parts:
            if part in filename_lower:
                score += 20
        
        # Popularity boost
        downloads = model.get("downloads", 0)
        likes = model.get("likes", 0)
        score += min(downloads / 1000, 50)  # Cap at 50 bonus points
        score += min(likes, 30)
        
        return score


class CivitAISearch:
    """Search for models on CivitAI."""
    
    API_BASE = "https://civitai.com/api/v1"
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
    
    def search_models(
        self, 
        query: str, 
        model_type: str | None = None,
        limit: int = 10
    ) -> list[ModelSearchResult]:
        """Search CivitAI for models matching the query."""
        results = []
        
        params = {
            "query": query,
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
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            for model in data.get("items", []):
                model_versions = model.get("modelVersions", [])
                
                for version in model_versions[:1]:  # Just get latest version
                    for file_info in version.get("files", []):
                        filename = file_info.get("name", "")
                        
                        if self._is_model_file(filename):
                            download_url = file_info.get("downloadUrl", "")
                            
                            # Add API key to download URL if available
                            if self.api_key and download_url:
                                separator = "&" if "?" in download_url else "?"
                                download_url = f"{download_url}{separator}token={self.api_key}"
                            
                            result = ModelSearchResult(
                                name=model.get("name", ""),
                                filename=filename,
                                download_url=download_url,
                                source="civitai",
                                size_bytes=file_info.get("sizeKB", 0) * 1024,
                                model_type=model.get("type", "").lower(),
                                model_id=model.get("id"),
                                description=model.get("description", ""),
                                sha256=file_info.get("hashes", {}).get("SHA256"),
                                score=self._calculate_score(query, filename, model),
                            )
                            results.append(result)
        
        except requests.RequestException as e:
            console.print(f"[yellow]CivitAI search error: {e}[/yellow]")
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def search_by_filename(self, filename: str, limit: int = 5) -> list[ModelSearchResult]:
        """Search CivitAI specifically for a filename."""
        search_term = filename.replace(".safetensors", "").replace(".ckpt", "")
        search_term = re.sub(r'[_-]', ' ', search_term)
        
        return self.search_models(search_term, limit=limit)
    
    def get_model_by_hash(self, sha256: str) -> ModelSearchResult | None:
        """Find a model by its SHA256 hash."""
        try:
            response = self.session.get(
                f"{self.API_BASE}/model-versions/by-hash/{sha256}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            for file_info in data.get("files", []):
                if file_info.get("hashes", {}).get("SHA256", "").lower() == sha256.lower():
                    return ModelSearchResult(
                        name=data.get("model", {}).get("name", ""),
                        filename=file_info.get("name", ""),
                        download_url=file_info.get("downloadUrl", ""),
                        source="civitai",
                        size_bytes=file_info.get("sizeKB", 0) * 1024,
                        model_id=data.get("modelId"),
                        sha256=sha256,
                        score=100,
                    )
        except requests.RequestException:
            pass
        
        return None
    
    def _is_model_file(self, filename: str) -> bool:
        """Check if a file is a model file."""
        extensions = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")
        return filename.lower().endswith(extensions)
    
    def _get_civitai_type(self, model_type: str | None) -> str | None:
        """Map model type to CivitAI type."""
        if not model_type:
            return None
        
        type_map = {
            "checkpoint": "Checkpoint",
            "diffusion": "Checkpoint",
            "lora": "LORA",
            "vae": "VAE",
            "embedding": "TextualInversion",
            "hypernetwork": "Hypernetwork",
            "controlnet": "Controlnet",
            "upscale": "Upscaler",
        }
        return type_map.get(model_type.lower())
    
    def _calculate_score(self, query: str, filename: str, model: dict) -> float:
        """Calculate relevance score for a search result."""
        score = 0.0
        query_lower = query.lower()
        filename_lower = filename.lower()
        name_lower = model.get("name", "").lower()
        
        # Exact filename match
        if query_lower in filename_lower:
            score += 100
        
        # Name match
        if query_lower in name_lower:
            score += 80
        
        # Partial match
        query_parts = query_lower.replace("-", " ").replace("_", " ").split()
        for part in query_parts:
            if len(part) > 2:
                if part in filename_lower:
                    score += 20
                if part in name_lower:
                    score += 15
        
        # Popularity boost
        stats = model.get("stats", {})
        downloads = stats.get("downloadCount", 0)
        rating = stats.get("rating", 0)
        
        score += min(downloads / 100, 50)
        score += rating * 5
        
        return score


class GitHubSearch:
    """Search for custom nodes on GitHub."""
    
    API_BASE = "https://api.github.com"
    
    # Known ComfyUI custom node repositories
    KNOWN_REPOS = {
        "ComfyUI-WanVideoWrapper": "kijai/ComfyUI-WanVideoWrapper",
        "comfyui-videohelpersuite": "Kosinkadink/ComfyUI-VideoHelperSuite",
        "rgthree-comfy": "rgthree/rgthree-comfy",
        "comfyui-kjnodes": "kijai/ComfyUI-KJNodes",
        "comfyui-impact-pack": "ltdrdata/ComfyUI-Impact-Pack",
        "comfyui-manager": "ltdrdata/ComfyUI-Manager",
        "comfyui-controlnet-aux": "Fannovel16/comfyui_controlnet_aux",
        "comfyui-animatediff-evolved": "Kosinkadink/ComfyUI-AnimateDiff-Evolved",
        "was-node-suite-comfyui": "WASasquatch/was-node-suite-comfyui",
        "comfyui-custom-scripts": "pythongosssss/ComfyUI-Custom-Scripts",
        "comfyui-advanced-controlnet": "Kosinkadink/ComfyUI-Advanced-ControlNet",
        "comfyui-frame-interpolation": "Fannovel16/ComfyUI-Frame-Interpolation",
        "efficiency-nodes-comfyui": "jags111/efficiency-nodes-comfyui",
        "comfyui-wd14-tagger": "pythongosssss/ComfyUI-WD14-Tagger",
        "comfyui-tooling-nodes": "Acly/comfyui-tooling-nodes",
        "comfyui-inpaint-nodes": "Acly/comfyui-inpaint-nodes",
        "sdxl_prompt_styler": "twri/sdxl_prompt_styler",
        "comfyui-reactor-node": "Gourieff/comfyui-reactor-node",
        "comfyui-art-venture": "sipherxyz/comfyui-art-venture",
        "comfyui-essentials": "cubiq/ComfyUI_essentials",
        "comfyui-ipadapter-plus": "cubiq/ComfyUI_IPAdapter_plus",
        "comfyui-layerdiffuse": "huchenlei/ComfyUI-layerdiffuse",
    }
    
    def __init__(self, token: str | None = None):
        self.token = token
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        if token:
            self.session.headers["Authorization"] = f"token {token}"
    
    def find_repo(self, cnr_id: str) -> dict | None:
        """Find a repository by its CNR ID or name."""
        # Check known repos first
        cnr_id_lower = cnr_id.lower()
        
        for known_id, repo_path in self.KNOWN_REPOS.items():
            if known_id.lower() == cnr_id_lower or cnr_id_lower in known_id.lower():
                return self._get_repo_info(repo_path)
        
        # Search GitHub
        return self._search_github(cnr_id)
    
    def _get_repo_info(self, repo_path: str) -> dict | None:
        """Get repository information from GitHub."""
        try:
            response = self.session.get(
                f"{self.API_BASE}/repos/{repo_path}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "name": data.get("name"),
                "full_name": data.get("full_name"),
                "description": data.get("description"),
                "html_url": data.get("html_url"),
                "clone_url": data.get("clone_url"),
                "default_branch": data.get("default_branch", "main"),
                "stars": data.get("stargazers_count", 0),
                "updated_at": data.get("updated_at"),
            }
        except requests.RequestException as e:
            console.print(f"[yellow]GitHub API error: {e}[/yellow]")
            return None
    
    def _search_github(self, query: str) -> dict | None:
        """Search GitHub for ComfyUI custom nodes."""
        search_query = f"{query} comfyui custom nodes in:name,description"
        
        try:
            response = self.session.get(
                f"{self.API_BASE}/search/repositories",
                params={"q": search_query, "per_page": 5},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            if items:
                best_match = items[0]
                return {
                    "name": best_match.get("name"),
                    "full_name": best_match.get("full_name"),
                    "description": best_match.get("description"),
                    "html_url": best_match.get("html_url"),
                    "clone_url": best_match.get("clone_url"),
                    "default_branch": best_match.get("default_branch", "main"),
                    "stars": best_match.get("stargazers_count", 0),
                    "updated_at": best_match.get("updated_at"),
                }
        except requests.RequestException as e:
            console.print(f"[yellow]GitHub search error: {e}[/yellow]")
        
        return None
    
    def search_comfyui_nodes(self, query: str, limit: int = 10) -> list[dict]:
        """Search for ComfyUI custom node repositories."""
        search_query = f"{query} comfyui in:name,description,readme"
        results = []
        
        try:
            response = self.session.get(
                f"{self.API_BASE}/search/repositories",
                params={"q": search_query, "per_page": limit, "sort": "stars"},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("items", []):
                results.append({
                    "name": item.get("name"),
                    "full_name": item.get("full_name"),
                    "description": item.get("description"),
                    "html_url": item.get("html_url"),
                    "clone_url": item.get("clone_url"),
                    "stars": item.get("stargazers_count", 0),
                })
        except requests.RequestException as e:
            console.print(f"[yellow]GitHub search error: {e}[/yellow]")
        
        return results


class ModelDatabase:
    """
    Unified model search across multiple platforms.
    Maintains a cache of known models and their download sources.
    """
    
    def __init__(
        self,
        hf_token: str | None = None,
        civitai_api_key: str | None = None,
        github_token: str | None = None,
    ):
        self.hf_search = HuggingFaceSearch(token=hf_token)
        self.civitai_search = CivitAISearch(api_key=civitai_api_key)
        self.github_search = GitHubSearch(token=github_token)
        
        # Cache for search results
        self._model_cache: dict[str, list[ModelSearchResult]] = {}
        self._node_cache: dict[str, dict] = {}
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.5  # seconds
    
    def search_model(
        self,
        filename: str,
        model_type: str | None = None,
        sources: list[str] | None = None,
        limit: int = 5,
    ) -> list[ModelSearchResult]:
        """
        Search for a model across all platforms.
        
        Args:
            filename: The model filename to search for
            model_type: Type of model (checkpoint, lora, vae, etc.)
            sources: List of sources to search (huggingface, civitai, github)
            limit: Maximum results per source
        
        Returns:
            List of ModelSearchResult sorted by relevance
        """
        cache_key = f"{filename}_{model_type}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        if sources is None:
            sources = ["huggingface", "civitai"]
        
        all_results = []
        
        # Clean up filename for better search
        search_term = self._clean_filename_for_search(filename)
        
        self._rate_limit()
        
        # Search HuggingFace
        if "huggingface" in sources:
            try:
                hf_results = self.hf_search.search_by_filename(filename, limit=limit)
                all_results.extend(hf_results)
            except Exception as e:
                console.print(f"[yellow]HuggingFace search failed: {e}[/yellow]")
        
        self._rate_limit()
        
        # Search CivitAI
        if "civitai" in sources:
            try:
                civitai_results = self.civitai_search.search_by_filename(filename, limit=limit)
                all_results.extend(civitai_results)
            except Exception as e:
                console.print(f"[yellow]CivitAI search failed: {e}[/yellow]")
        
        # Sort by score and filename similarity
        all_results.sort(
            key=lambda x: (
                x.score,
                self._filename_similarity(filename, x.filename),
            ),
            reverse=True,
        )
        
        # Cache results
        self._model_cache[cache_key] = all_results[:limit * 2]
        
        return all_results[:limit * 2]
    
    def find_best_match(
        self,
        filename: str,
        model_type: str | None = None,
    ) -> ModelSearchResult | None:
        """Find the best matching model for a filename."""
        results = self.search_model(filename, model_type=model_type, limit=5)
        
        if not results:
            return None
        
        # Check for exact filename match first
        for result in results:
            if result.filename.lower() == filename.lower():
                return result
        
        # Return highest scored result
        return results[0] if results else None
    
    def find_custom_node(self, cnr_id: str) -> dict | None:
        """Find a custom node repository by its CNR ID."""
        if cnr_id in self._node_cache:
            return self._node_cache[cnr_id]
        
        self._rate_limit()
        
        result = self.github_search.find_repo(cnr_id)
        
        if result:
            self._node_cache[cnr_id] = result
        
        return result
    
    def _clean_filename_for_search(self, filename: str) -> str:
        """Clean up a filename for better search results."""
        # Remove extension
        name = re.sub(r'\.(safetensors|ckpt|pt|pth|bin)$', '', filename, flags=re.IGNORECASE)
        
        # Replace separators with spaces
        name = re.sub(r'[_-]+', ' ', name)
        
        # Remove common suffixes that don't help search
        name = re.sub(r'\s*(fp16|fp32|bf16|fp8|e4m3fn|scaled|comfy|KJ)\s*', ' ', name, flags=re.IGNORECASE)
        
        return name.strip()
    
    def _filename_similarity(self, query: str, candidate: str) -> float:
        """Calculate similarity between filenames."""
        query_lower = query.lower()
        candidate_lower = candidate.lower()
        
        # Exact match
        if query_lower == candidate_lower:
            return 1.0
        
        # Query is substring
        if query_lower in candidate_lower:
            return 0.8
        
        # Candidate is substring
        if candidate_lower in query_lower:
            return 0.7
        
        # Word overlap
        query_words = set(re.split(r'[_\-\s.]+', query_lower))
        candidate_words = set(re.split(r'[_\-\s.]+', candidate_lower))
        
        if not query_words or not candidate_words:
            return 0.0
        
        overlap = len(query_words & candidate_words)
        total = len(query_words | candidate_words)
        
        return overlap / total if total > 0 else 0.0
    
    def _rate_limit(self):
        """Apply rate limiting between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()


# Predefined model mappings for common models
KNOWN_MODELS = {
    # LongCat models
    "LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors": {
        "url": "https://huggingface.co/Kijai/LongCat-Video_comfy/resolve/main/LongCat_TI2V_comfy_fp8_e4m3fn_scaled_KJ.safetensors",
        "source": "huggingface",
        "type": "diffusion",
        "folder": "models/diffusion_models",
    },
    "LongCat_TI2V_comfy_bf16.safetensors": {
        "url": "https://huggingface.co/Kijai/LongCat-Video_comfy/resolve/main/LongCat_TI2V_comfy_bf16.safetensors",
        "source": "huggingface",
        "type": "diffusion",
        "folder": "models/diffusion_models",
    },
    "LongCat_distill_lora_alpha64_bf16.safetensors": {
        "url": "https://huggingface.co/Kijai/LongCat-Video_comfy/resolve/main/LongCat_distill_lora_alpha64_bf16.safetensors",
        "source": "huggingface",
        "type": "lora",
        "folder": "models/loras",
    },
    
    # WanVideo / Wan 2.1 models
    "wan_2.1_vae.safetensors": {
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        "source": "huggingface",
        "type": "vae",
        "folder": "models/vae",
    },
    "umt5-xxl-enc-bf16.safetensors": {
        "url": "https://huggingface.co/ALGOTECH/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors",
        "source": "huggingface",
        "type": "clip",
        "folder": "models/clip",
    },
    
    # Common SD models
    "v1-5-pruned-emaonly.safetensors": {
        "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
        "source": "huggingface",
        "type": "checkpoint",
        "folder": "models/checkpoints",
    },
    
    # SDXL models
    "sd_xl_base_1.0.safetensors": {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "source": "huggingface",
        "type": "checkpoint",
        "folder": "models/checkpoints",
    },
    "sd_xl_refiner_1.0.safetensors": {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors",
        "source": "huggingface",
        "type": "checkpoint",
        "folder": "models/checkpoints",
    },
    
    # VAE models
    "sdxl_vae.safetensors": {
        "url": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
        "source": "huggingface",
        "type": "vae",
        "folder": "models/vae",
    },
    "vae-ft-mse-840000-ema-pruned.safetensors": {
        "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
        "source": "huggingface",
        "type": "vae",
        "folder": "models/vae",
    },
}


def get_known_model_url(filename: str) -> dict | None:
    """Get download info for a known model."""
    return KNOWN_MODELS.get(filename)
