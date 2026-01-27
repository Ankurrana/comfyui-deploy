"""
ComfyUI Manager Node Database Integration.

Fetches the official extension-node-map.json from ComfyUI Manager
to provide accurate node type → GitHub repository mapping.
"""

import json
import re
from pathlib import Path
from typing import Optional
import requests

# URL for ComfyUI Manager's extension node map
EXTENSION_NODE_MAP_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"

# Cache file for offline use
CACHE_DIR = Path.home() / ".comfyui-deploy" / "cache"
CACHE_FILE = CACHE_DIR / "extension-node-map.json"


class NodeDatabase:
    """Database for looking up custom node sources from node types."""
    
    def __init__(self, cache_ttl_hours: int = 24):
        """
        Initialize the node database.
        
        Args:
            cache_ttl_hours: How long to cache the database before refreshing
        """
        self.cache_ttl_hours = cache_ttl_hours
        self._node_to_repo: dict[str, str] = {}
        self._repo_to_nodes: dict[str, list[str]] = {}
        self._loaded = False
    
    def load(self, force_refresh: bool = False) -> bool:
        """
        Load the node database from cache or remote.
        
        Args:
            force_refresh: Force fetching from remote even if cache exists
            
        Returns:
            True if loaded successfully
        """
        if self._loaded and not force_refresh:
            return True
        
        data = None
        
        # Try cache first (unless forcing refresh)
        if not force_refresh and self._is_cache_valid():
            data = self._load_from_cache()
        
        # Fetch from remote if needed
        if data is None:
            data = self._fetch_from_remote()
            if data:
                self._save_to_cache(data)
        
        # Fall back to cache even if expired
        if data is None:
            data = self._load_from_cache()
        
        if data:
            self._build_lookup(data)
            self._loaded = True
            return True
        
        return False
    
    def get_repo_for_node_type(self, node_type: str) -> Optional[str]:
        """
        Get the GitHub repository URL for a given node type.
        
        Args:
            node_type: The ComfyUI node type (e.g., "Label (rgthree)")
            
        Returns:
            GitHub URL or None if not found
        """
        if not self._loaded:
            self.load()
        
        return self._node_to_repo.get(node_type)
    
    def get_nodes_for_repo(self, repo_url: str) -> list[str]:
        """
        Get all node types provided by a repository.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            List of node types
        """
        if not self._loaded:
            self.load()
        
        return self._repo_to_nodes.get(repo_url, [])
    
    def search_repos(self, query: str) -> list[tuple[str, list[str]]]:
        """
        Search for repositories by name or node type.
        
        Args:
            query: Search query (case-insensitive)
            
        Returns:
            List of (repo_url, [node_types]) tuples
        """
        if not self._loaded:
            self.load()
        
        results = []
        query_lower = query.lower()
        
        for repo_url, node_types in self._repo_to_nodes.items():
            # Check if query matches repo URL
            if query_lower in repo_url.lower():
                results.append((repo_url, node_types))
                continue
            
            # Check if query matches any node type
            for node_type in node_types:
                if query_lower in node_type.lower():
                    results.append((repo_url, node_types))
                    break
        
        return results
    
    def extract_cnr_id_from_url(self, github_url: str) -> str:
        """
        Extract a cnr_id-like identifier from a GitHub URL.
        
        Args:
            github_url: Full GitHub URL
            
        Returns:
            Repository name (e.g., "rgthree-comfy" from "https://github.com/rgthree/rgthree-comfy")
        """
        # Extract repo name from URL
        match = re.search(r'github\.com/[^/]+/([^/]+?)(?:\.git)?(?:/|$)', github_url)
        if match:
            return match.group(1)
        return github_url
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is not expired."""
        if not CACHE_FILE.exists():
            return False
        
        import time
        cache_age_hours = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        return cache_age_hours < self.cache_ttl_hours
    
    def _load_from_cache(self) -> Optional[dict]:
        """Load data from cache file."""
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, data: dict) -> None:
        """Save data to cache file."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _fetch_from_remote(self) -> Optional[dict]:
        """Fetch the extension node map from GitHub."""
        try:
            print("Fetching ComfyUI Manager node database...")
            response = requests.get(EXTENSION_NODE_MAP_URL, timeout=30)
            response.raise_for_status()
            data = response.json()
            print(f"  ✓ Loaded {len(data)} custom node repositories")
            return data
        except requests.RequestException as e:
            print(f"Warning: Failed to fetch node database: {e}")
            return None
    
    def _build_lookup(self, data: dict) -> None:
        """Build the reverse lookup dictionaries."""
        self._node_to_repo.clear()
        self._repo_to_nodes.clear()
        
        for repo_url, info in data.items():
            # info is [node_types_list, metadata_dict]
            if isinstance(info, list) and len(info) >= 1:
                node_types = info[0] if isinstance(info[0], list) else []
                
                self._repo_to_nodes[repo_url] = node_types
                
                for node_type in node_types:
                    # Store node type -> repo mapping
                    # Don't overwrite if already mapped (first one wins)
                    if node_type not in self._node_to_repo:
                        self._node_to_repo[node_type] = repo_url
    
    @property
    def total_nodes(self) -> int:
        """Total number of node types in the database."""
        return len(self._node_to_repo)
    
    @property
    def total_repos(self) -> int:
        """Total number of repositories in the database."""
        return len(self._repo_to_nodes)


# Global singleton instance
_node_db: Optional[NodeDatabase] = None


def get_node_database() -> NodeDatabase:
    """Get the global node database instance."""
    global _node_db
    if _node_db is None:
        _node_db = NodeDatabase()
    return _node_db


def lookup_node_repo(node_type: str) -> Optional[str]:
    """
    Quick lookup for a node type's repository.
    
    Args:
        node_type: The ComfyUI node type
        
    Returns:
        GitHub URL or None
    """
    return get_node_database().get_repo_for_node_type(node_type)


if __name__ == "__main__":
    # Test the database
    db = NodeDatabase()
    db.load()
    
    print(f"\nDatabase loaded: {db.total_repos} repos, {db.total_nodes} node types")
    
    # Test some lookups
    test_types = [
        "Label (rgthree)",
        "Fast Groups Muter (rgthree)",
        "VHS_VideoCombine",
        "WanVideoModelLoader",
        "KJNodes_SetGetWidgets",
    ]
    
    print("\nTest lookups:")
    for node_type in test_types:
        repo = db.get_repo_for_node_type(node_type)
        print(f"  {node_type}")
        print(f"    → {repo or 'NOT FOUND'}")
