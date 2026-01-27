"""
Configuration management for ComfyUI Deploy.
"""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """
    Configuration manager for ComfyUI Deploy.
    Loads settings from config file and environment variables.
    """
    
    DEFAULT_CONFIG_DIR = Path.home() / ".comfyui-deploy"
    DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"
    
    DEFAULTS = {
        "comfyui_path": None,
        "hf_token": None,
        "civitai_api_key": None,
        "github_token": None,
        "auto_download": True,
        "auto_install_nodes": True,
        "verify_checksums": True,
        "resume_downloads": True,
        "parallel_downloads": 1,
        "search_sources": ["huggingface", "civitai"],
    }
    
    ENV_MAPPINGS = {
        "COMFYUI_PATH": "comfyui_path",
        "HF_TOKEN": "hf_token",
        "HUGGING_FACE_HUB_TOKEN": "hf_token",
        "CIVITAI_API_KEY": "civitai_api_key",
        "GITHUB_TOKEN": "github_token",
    }
    
    def __init__(self, config_path: Path | str | None = None):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_FILE
        self._config: dict[str, Any] = {}
        self._load()
    
    def _load(self) -> None:
        """Load configuration from file and environment."""
        # Start with defaults
        self._config = self.DEFAULTS.copy()
        
        # Load from file if exists
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
                self._config.update(file_config)
        
        # Override with environment variables
        for env_var, config_key in self.ENV_MAPPINGS.items():
            value = os.environ.get(env_var)
            if value:
                self._config[config_key] = value
    
    def save(self) -> None:
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Don't save sensitive values from environment
        save_config = {
            k: v for k, v in self._config.items()
            if k not in ("hf_token", "civitai_api_key", "github_token") or 
            not os.environ.get(self.ENV_MAPPINGS.get(k, ""))
        }
        
        with open(self.config_path, "w") as f:
            yaml.dump(save_config, f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value
    
    def __getitem__(self, key: str) -> Any:
        return self._config.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._config[key] = value
    
    @property
    def comfyui_path(self) -> Path | None:
        """Get ComfyUI installation path."""
        path = self._config.get("comfyui_path")
        return Path(path) if path else None
    
    @comfyui_path.setter
    def comfyui_path(self, value: Path | str | None) -> None:
        self._config["comfyui_path"] = str(value) if value else None
    
    @property
    def hf_token(self) -> str | None:
        """Get HuggingFace token."""
        return self._config.get("hf_token")
    
    @property
    def civitai_api_key(self) -> str | None:
        """Get CivitAI API key."""
        return self._config.get("civitai_api_key")
    
    @property
    def github_token(self) -> str | None:
        """Get GitHub token."""
        return self._config.get("github_token")
    
    def to_dict(self) -> dict:
        """Return configuration as dictionary (masks sensitive values)."""
        result = self._config.copy()
        
        # Mask sensitive values
        for key in ("hf_token", "civitai_api_key", "github_token"):
            if result.get(key):
                result[key] = "***" + result[key][-4:] if len(result[key]) > 4 else "***"
        
        return result


# Global config instance
_config: Config | None = None


def get_config(config_path: Path | str | None = None) -> Config:
    """Get or create the global configuration instance."""
    global _config
    
    if _config is None or config_path:
        _config = Config(config_path)
    
    return _config
