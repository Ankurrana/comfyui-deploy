"""
Tests for the SmartSearch module.
"""
import pytest
from unittest.mock import patch, MagicMock
import requests

from comfyui_deploy.smart_search import (
    SmartModelSearch,
    ComfyUIManagerModels,
    CivitAISearch,
    smart_search,
)


class TestSmartModelSearch:
    """Test cases for SmartModelSearch."""
    
    @pytest.fixture
    def searcher(self):
        """Create a searcher instance."""
        return SmartModelSearch()
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests module."""
        with patch('comfyui_deploy.smart_search.requests') as mock:
            yield mock
    
    def test_find_candidate_repos_longcat(self, searcher):
        """Test pattern matching for LongCat models."""
        repos = searcher._find_candidate_repos("longcat_model.safetensors")
        
        assert "Kijai/LongCat-Video_comfy" in repos
    
    def test_find_candidate_repos_wan(self, searcher):
        """Test pattern matching for Wan models."""
        repos = searcher._find_candidate_repos("wan_2.1_vae.safetensors")
        
        assert any("Wan" in repo for repo in repos)
    
    def test_find_candidate_repos_flux(self, searcher):
        """Test pattern matching for Flux models."""
        repos = searcher._find_candidate_repos("flux_dev.safetensors")
        
        assert any("flux" in repo.lower() or "FLUX" in repo for repo in repos)
    
    def test_list_repo_files(self, searcher):
        """Test listing files from a HuggingFace repo."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = [
            {"path": "model.safetensors", "size": 1000000, "type": "file"},
            {"path": "config.json", "size": 1000, "type": "file"},
        ]
        
        # Mock the session.get method on the searcher instance
        searcher.session = MagicMock()
        searcher.session.get.return_value = mock_response
        
        # Clear cache to ensure we make the call
        searcher._repo_cache = {}
        searcher._cache_times = {}
        
        files = searcher._list_repo_files("test/repo")
        
        assert len(files) == 1  # Only .safetensors is a model file
        assert files[0]["path"] == "model.safetensors"
    
    def test_get_base_name(self, searcher):
        """Test extracting base name from filename."""
        assert searcher._get_base_name("model.safetensors") == "model"
        assert searcher._get_base_name("model_v2.ckpt") == "model_v2"
        assert searcher._get_base_name("model") == "model"


class TestComfyUIManagerModels:
    """Test cases for ComfyUIManagerModels."""
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests module."""
        with patch('comfyui_deploy.smart_search.requests') as mock:
            yield mock
    
    def test_fetch_models(self, mock_requests):
        """Test fetching model list from ComfyUI Manager."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "sd_xl_base_1.0.safetensors",
                    "type": "checkpoint",
                    "url": "https://huggingface.co/stabilityai/sdxl/resolve/main/sd_xl_base_1.0.safetensors"
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_response
        
        models = ComfyUIManagerModels.fetch_models()
        
        assert len(models) == 1
        assert models[0]["name"] == "sd_xl_base_1.0.safetensors"
    
    def test_find_model(self, mock_requests):
        """Test finding a specific model."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "sd_xl_base_1.0.safetensors",
                    "type": "checkpoint",
                    "url": "https://huggingface.co/stabilityai/sdxl/resolve/main/sd_xl_base_1.0.safetensors"
                },
                {
                    "name": "other_model.safetensors",
                    "type": "lora",
                    "url": "https://example.com/other.safetensors"
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_response
        
        result = ComfyUIManagerModels.find_model("sd_xl_base_1.0.safetensors")
        
        assert result is not None
        assert result["name"] == "sd_xl_base_1.0.safetensors"
    
    def test_find_model_not_found(self, mock_requests):
        """Test returns None when model not found."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_response
        
        result = ComfyUIManagerModels.find_model("nonexistent_model.safetensors")
        assert result is None


class TestCivitAISearch:
    """Test cases for CivitAI search."""
    
    @pytest.fixture
    def searcher(self):
        """Create a CivitAI searcher."""
        return CivitAISearch()
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests module."""
        with patch('comfyui_deploy.smart_search.requests') as mock:
            yield mock
    
    def test_search_returns_list(self, searcher, mock_requests):
        """Test that search returns a list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "items": [
                {
                    "id": 12345,
                    "name": "My Model",
                    "modelVersions": [
                        {
                            "id": 67890,
                            "files": [
                                {
                                    "name": "my_model.safetensors",
                                    "downloadUrl": "https://civitai.com/api/download/models/67890"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_response
        
        results = searcher.search("my_model")
        
        assert isinstance(results, list)
    
    def test_filename_to_search_term(self, searcher):
        """Test converting filename to search term."""
        # Should strip extension and common suffixes
        term = searcher._filename_to_search_term("my_model_v2.safetensors")
        assert "safetensors" not in term.lower()
    
    def test_is_model_file(self, searcher):
        """Test model file detection."""
        assert searcher._is_model_file("model.safetensors") is True
        assert searcher._is_model_file("model.ckpt") is True
        assert searcher._is_model_file("model.pt") is True
        assert searcher._is_model_file("config.json") is False
        assert searcher._is_model_file("readme.md") is False


class TestSmartSearchFunction:
    """Test the main smart_search convenience function."""
    
    def test_smart_search_returns_dict_or_none(self):
        """Test smart_search return type."""
        # With mocked components to avoid network calls
        with patch('comfyui_deploy.smart_search.ComfyUIManagerModels') as MockManager:
            MockManager.find_model.return_value = None
            
            with patch('comfyui_deploy.smart_search.SmartModelSearch') as MockSearch:
                instance = MagicMock()
                instance.search.return_value = None
                MockSearch.return_value = instance
                
                with patch('comfyui_deploy.smart_search.CivitAISearch') as MockCivit:
                    civit_instance = MagicMock()
                    civit_instance.find_best_match.return_value = None
                    MockCivit.return_value = civit_instance
                    
                    result = smart_search("nonexistent_model.safetensors")
                    
                    # Result should be None or dict
                    assert result is None or isinstance(result, dict)


class TestRepoPatterns:
    """Test repository pattern definitions."""
    
    def test_repo_patterns_exist(self):
        """Test that repo patterns are defined."""
        searcher = SmartModelSearch()
        assert len(searcher.REPO_PATTERNS) > 0
    
    def test_longcat_pattern_exists(self):
        """Test LongCat pattern is defined."""
        import re
        
        searcher = SmartModelSearch()
        found = False
        for pattern in searcher.REPO_PATTERNS.keys():
            if re.search(pattern, "LongCat_model", re.IGNORECASE):
                found = True
                break
        
        assert found, "LongCat pattern should exist"
    
    def test_flux_pattern_exists(self):
        """Test Flux pattern is defined."""
        import re
        
        searcher = SmartModelSearch()
        found = False
        for pattern in searcher.REPO_PATTERNS.keys():
            if re.search(pattern, "flux_dev", re.IGNORECASE):
                found = True
                break
        
        assert found, "Flux pattern should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
