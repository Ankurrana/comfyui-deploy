"""
Tests for the Downloader module including ParallelDownloader.
"""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
import tempfile
import os

from comfyui_deploy.downloader import (
    ModelDownloader,
    ParallelDownloader,
    DownloadError,
)


class TestModelDownloader:
    """Test cases for ModelDownloader."""
    
    @pytest.fixture
    def downloader(self):
        """Create a downloader instance."""
        return ModelDownloader()
    
    @pytest.fixture
    def mock_session(self):
        """Mock requests session."""
        with patch('comfyui_deploy.downloader.requests.Session') as mock:
            session = MagicMock()
            mock.return_value = session
            yield session
    
    def test_detect_source_huggingface(self, downloader):
        """Test source detection for HuggingFace URLs."""
        source = downloader._detect_source("https://huggingface.co/user/repo/resolve/main/model.safetensors")
        assert source == "huggingface"
    
    def test_detect_source_civitai(self, downloader):
        """Test source detection for CivitAI URLs."""
        source = downloader._detect_source("https://civitai.com/api/download/models/12345")
        assert source == "civitai"
    
    def test_detect_source_github(self, downloader):
        """Test source detection for GitHub URLs."""
        source = downloader._detect_source("https://github.com/user/repo/releases/download/v1/file.bin")
        assert source == "github"
    
    def test_detect_source_direct(self, downloader):
        """Test source detection for unknown URLs."""
        source = downloader._detect_source("https://example.com/file.safetensors")
        assert source == "direct"
    
    def test_extract_filename_from_url(self, downloader):
        """Test filename extraction from URL."""
        filename = downloader._extract_filename("https://example.com/path/to/model.safetensors")
        assert filename == "model.safetensors"
        
        # URL with query params
        filename = downloader._extract_filename("https://example.com/model.safetensors?download=true")
        assert filename == "model.safetensors"
    
    def test_format_size(self, downloader):
        """Test size formatting."""
        assert "B" in downloader._format_size(100)
        assert "KB" in downloader._format_size(1024)
        assert "MB" in downloader._format_size(1024 * 1024)
        assert "GB" in downloader._format_size(1024 * 1024 * 1024)
    
    def test_download_skips_existing_file(self, downloader, tmp_path):
        """Test that existing files are skipped."""
        # Create a file
        test_file = tmp_path / "existing.safetensors"
        test_file.write_text("test content")
        
        # The downloader checks if file exists, but still calls _download_direct
        # which then checks again. Let's just verify the file is returned.
        with patch.object(downloader.session, 'get') as mock_get:
            # Set up mock to simulate already exists check
            result = downloader.download(
                url="https://example.com/existing.safetensors",
                destination=tmp_path,
                filename="existing.safetensors"
            )
            
            # Should return the existing file path
            assert result == test_file


class TestParallelDownloader:
    """Test cases for ParallelDownloader."""
    
    @pytest.fixture
    def parallel_downloader(self):
        """Create a parallel downloader instance."""
        return ParallelDownloader(max_workers=2)
    
    def test_init_default_workers(self):
        """Test default worker count."""
        pd = ParallelDownloader()
        assert pd.max_workers == 4
    
    def test_init_custom_workers(self):
        """Test custom worker count."""
        pd = ParallelDownloader(max_workers=8)
        assert pd.max_workers == 8
    
    def test_download_all_empty_list(self, parallel_downloader):
        """Test downloading empty list."""
        results = parallel_downloader.download_all([])
        
        assert results["successful"] == []
        assert results["failed"] == []
        assert results["skipped"] == []
    
    def test_download_all_with_mocked_downloads(self, parallel_downloader, tmp_path):
        """Test parallel downloading with mocked downloader."""
        # Mock the internal downloader
        with patch.object(parallel_downloader.downloader, 'download') as mock_download:
            mock_download.return_value = tmp_path / "file1.safetensors"
            
            downloads = [
                {"url": "https://example.com/file1.safetensors", "destination": tmp_path},
                {"url": "https://example.com/file2.safetensors", "destination": tmp_path},
            ]
            
            results = parallel_downloader.download_all(downloads)
            
            # Should have called download twice
            assert mock_download.call_count == 2
    
    def test_download_all_handles_failures(self, parallel_downloader, tmp_path):
        """Test that failures are tracked correctly."""
        with patch.object(parallel_downloader.downloader, 'download') as mock_download:
            # First succeeds, second fails
            mock_download.side_effect = [
                tmp_path / "file1.safetensors",
                Exception("Download failed"),
            ]
            
            downloads = [
                {"url": "https://example.com/file1.safetensors", "destination": tmp_path, "filename": "file1.safetensors"},
                {"url": "https://example.com/file2.safetensors", "destination": tmp_path, "filename": "file2.safetensors"},
            ]
            
            results = parallel_downloader.download_all(downloads)
            
            assert len(results["successful"]) == 1
            assert len(results["failed"]) == 1
    
    def test_download_all_skips_no_url(self, parallel_downloader, tmp_path):
        """Test that items without URL are skipped."""
        downloads = [
            {"url": None, "destination": tmp_path, "filename": "no_url.safetensors"},
            {"destination": tmp_path, "filename": "missing_url.safetensors"},  # No url key
        ]
        
        results = parallel_downloader.download_all(downloads)
        
        assert len(results["skipped"]) == 2
        assert len(results["successful"]) == 0


class TestDownloadIntegration:
    """Integration tests (can be run with real network if needed)."""
    
    @pytest.mark.skip(reason="Requires network access")
    def test_real_download_small_file(self, tmp_path):
        """Test actual download of a small file."""
        downloader = ModelDownloader()
        
        # Use a small, stable test file
        url = "https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/requirements.txt"
        
        result = downloader.download(
            url=url,
            destination=tmp_path,
            filename="requirements.txt"
        )
        
        assert result.exists()
        assert result.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
