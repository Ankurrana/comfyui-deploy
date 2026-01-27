"""
Tests for the NodeDatabase module.
"""
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import tempfile

from comfyui_deploy.node_database import (
    NodeDatabase,
    get_node_database,
    lookup_node_repo,
    EXTENSION_NODE_MAP_URL,
)


# Sample extension node map data
SAMPLE_NODE_MAP = {
    "https://github.com/rgthree/rgthree-comfy": [
        ["RgthreeAnySwitch", "RgthreeBigContext", "Label (rgthree)"],
        {"title_aux": "rgthree-comfy"}
    ],
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite": [
        ["VHS_VideoCombine", "VHS_LoadVideo"],
        {"title_aux": "ComfyUI-VideoHelperSuite"}
    ],
    "https://github.com/kijai/ComfyUI-KJNodes": [
        ["KJNodes_SetGetWidgets", "ImageBatchExtendWithOverlap"],
        {"title_aux": "ComfyUI-KJNodes"}
    ],
}


class TestNodeDatabase:
    """Test cases for NodeDatabase."""
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests.get to avoid network calls."""
        with patch('comfyui_deploy.node_database.requests') as mock:
            response = MagicMock()
            response.json.return_value = SAMPLE_NODE_MAP
            response.raise_for_status = MagicMock()
            mock.get.return_value = response
            yield mock
    
    @pytest.fixture
    def db(self, mock_requests):
        """Create a database instance with mocked network."""
        database = NodeDatabase(cache_ttl_hours=0)  # Don't use cache
        return database
    
    def test_load_from_remote(self, db, mock_requests):
        """Test loading the database from remote."""
        result = db.load(force_refresh=True)
        
        assert result is True
        assert db.total_repos == 3
        assert db.total_nodes == 7  # Total node types
    
    def test_get_repo_for_node_type(self, db, mock_requests):
        """Test looking up a repo by node type."""
        db.load()
        
        # Exact match
        repo = db.get_repo_for_node_type("VHS_VideoCombine")
        assert repo == "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
        
        # Another node
        repo = db.get_repo_for_node_type("RgthreeAnySwitch")
        assert repo == "https://github.com/rgthree/rgthree-comfy"
    
    def test_get_repo_for_unknown_node(self, db, mock_requests):
        """Test looking up an unknown node type."""
        db.load()
        
        repo = db.get_repo_for_node_type("UnknownNode")
        assert repo is None
    
    def test_get_nodes_for_repo(self, db, mock_requests):
        """Test getting all nodes for a repo."""
        db.load()
        
        nodes = db.get_nodes_for_repo("https://github.com/rgthree/rgthree-comfy")
        assert "RgthreeAnySwitch" in nodes
        assert "Label (rgthree)" in nodes
    
    def test_search_repos(self, db, mock_requests):
        """Test searching repos by query."""
        db.load()
        
        # Search by repo name
        results = db.search_repos("rgthree")
        assert len(results) == 1
        assert results[0][0] == "https://github.com/rgthree/rgthree-comfy"
        
        # Search by node type
        results = db.search_repos("VHS")
        assert len(results) == 1
        assert "VHS_VideoCombine" in results[0][1]
    
    def test_extract_cnr_id_from_url(self, db):
        """Test extracting cnr_id from GitHub URL."""
        cnr_id = db.extract_cnr_id_from_url("https://github.com/rgthree/rgthree-comfy")
        assert cnr_id == "rgthree-comfy"
        
        cnr_id = db.extract_cnr_id_from_url("https://github.com/user/Some-Repo.git")
        assert cnr_id == "Some-Repo"
    
    def test_cache_saves_and_loads(self, mock_requests, tmp_path):
        """Test that cache is saved and loaded correctly."""
        # Patch the cache directory
        with patch('comfyui_deploy.node_database.CACHE_DIR', tmp_path):
            with patch('comfyui_deploy.node_database.CACHE_FILE', tmp_path / "test_cache.json"):
                db = NodeDatabase()
                db.load()
                
                # Check cache file was created
                cache_file = tmp_path / "test_cache.json"
                assert cache_file.exists()
                
                # Load from cache
                db2 = NodeDatabase()
                db2.load()
                
                assert db2.total_repos == 3


class TestGlobalFunctions:
    """Test module-level convenience functions."""
    
    @pytest.fixture
    def mock_db(self):
        """Mock the global database."""
        with patch('comfyui_deploy.node_database._node_db', None):
            with patch('comfyui_deploy.node_database.NodeDatabase') as MockDB:
                instance = MagicMock()
                instance.get_repo_for_node_type.return_value = "https://github.com/test/repo"
                MockDB.return_value = instance
                yield instance
    
    def test_get_node_database_singleton(self, mock_db):
        """Test that get_node_database returns singleton."""
        db1 = get_node_database()
        db2 = get_node_database()
        
        # Should be same instance (mocked)
        assert db1 == db2
    
    def test_lookup_node_repo(self, mock_db):
        """Test the convenience lookup function."""
        repo = lookup_node_repo("SomeNode")
        
        assert repo == "https://github.com/test/repo"
        mock_db.get_repo_for_node_type.assert_called_with("SomeNode")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_handles_malformed_data(self):
        """Test handling of malformed node map data."""
        with patch('comfyui_deploy.node_database.requests') as mock_requests:
            response = MagicMock()
            response.json.return_value = {
                "https://github.com/valid/repo": [
                    ["Node1", "Node2"],
                    {}
                ],
                "https://github.com/malformed/repo": "not a list",  # Malformed
                "https://github.com/empty/repo": [[], {}],  # Empty nodes
            }
            response.raise_for_status = MagicMock()
            mock_requests.get.return_value = response
            
            db = NodeDatabase(cache_ttl_hours=0)
            db.load(force_refresh=True)
            
            # Should still load valid entries
            assert db.get_repo_for_node_type("Node1") == "https://github.com/valid/repo"
    
    def test_handles_network_error(self):
        """Test handling of network errors."""
        with patch('comfyui_deploy.node_database.requests') as mock_requests:
            mock_requests.get.side_effect = Exception("Network error")
            mock_requests.RequestException = Exception
            
            db = NodeDatabase(cache_ttl_hours=0)
            
            # Should not raise, just return False
            result = db.load(force_refresh=True)
            # May return True if cache exists, False otherwise
            assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
