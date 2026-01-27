"""
Model Downloader - Downloads models from various platforms with progress tracking and resume support.
"""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download, HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

console = Console()


class DownloadError(Exception):
    """Exception raised when a download fails."""
    pass


class ModelDownloader:
    """
    Downloads models from various sources with progress tracking.
    Supports HuggingFace, CivitAI, GitHub, and direct URLs.
    """
    
    CHUNK_SIZE = 8192 * 16  # 128KB chunks
    
    def __init__(
        self,
        hf_token: str | None = None,
        civitai_api_key: str | None = None,
        verify_ssl: bool = True,
        resume: bool = True,
    ):
        self.hf_token = hf_token
        self.civitai_api_key = civitai_api_key
        self.verify_ssl = verify_ssl
        self.resume = resume
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ComfyUI-Deploy/0.1.0"
        })
    
    def download(
        self,
        url: str,
        destination: Path,
        filename: str | None = None,
        expected_sha256: str | None = None,
        source: str | None = None,
    ) -> Path:
        """
        Download a model from a URL to the destination folder.
        
        Args:
            url: The download URL
            destination: Target directory
            filename: Override filename (auto-detected if None)
            expected_sha256: Expected SHA256 hash for verification
            source: Source platform hint (huggingface, civitai, etc.)
        
        Returns:
            Path to the downloaded file
        """
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect source if not provided
        if source is None:
            source = self._detect_source(url)
        
        # Use appropriate download method
        if source == "huggingface" and HAS_HF_HUB and self._is_hf_hub_url(url):
            return self._download_hf_hub(url, destination, filename)
        elif source == "civitai":
            return self._download_civitai(url, destination, filename, expected_sha256)
        else:
            return self._download_direct(url, destination, filename, expected_sha256)
    
    def _download_direct(
        self,
        url: str,
        destination: Path,
        filename: str | None = None,
        expected_sha256: str | None = None,
    ) -> Path:
        """Download a file directly from a URL."""
        # Get filename from URL if not provided
        if filename is None:
            filename = self._extract_filename(url)
        
        target_path = destination / filename
        temp_path = destination / f".{filename}.tmp"
        
        # Check if file already exists and is complete
        if target_path.exists():
            if expected_sha256:
                if self._verify_sha256(target_path, expected_sha256):
                    console.print(f"[green]✓ {filename} already exists and verified[/green]")
                    return target_path
                else:
                    console.print(f"[yellow]! {filename} exists but hash mismatch, re-downloading[/yellow]")
            else:
                console.print(f"[green]✓ {filename} already exists[/green]")
                return target_path
        
        # Prepare headers for authentication
        headers = {}
        if "huggingface.co" in url and self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        
        # Check for resume support
        resume_pos = 0
        if self.resume and temp_path.exists():
            resume_pos = temp_path.stat().st_size
            headers["Range"] = f"bytes={resume_pos}-"
        
        try:
            response = self.session.get(
                url,
                headers=headers,
                stream=True,
                verify=self.verify_ssl,
                timeout=30,
                allow_redirects=True,
            )
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get("content-length", 0))
            if resume_pos > 0 and response.status_code == 206:
                total_size += resume_pos
                console.print(f"[cyan]Resuming download from {self._format_size(resume_pos)}[/cyan]")
            elif resume_pos > 0:
                # Server doesn't support resume, start over
                resume_pos = 0
                temp_path.unlink(missing_ok=True)
            
            # Download with progress
            mode = "ab" if resume_pos > 0 else "wb"
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Downloading {filename}", total=total_size)
                progress.update(task, completed=resume_pos)
                
                sha256_hash = hashlib.sha256()
                
                with open(temp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            sha256_hash.update(chunk)
                            progress.update(task, advance=len(chunk))
            
            # Verify hash if provided
            if expected_sha256:
                # Need to hash the full file, not just downloaded portion
                actual_hash = self._compute_sha256(temp_path)
                if actual_hash.lower() != expected_sha256.lower():
                    temp_path.unlink(missing_ok=True)
                    raise DownloadError(
                        f"SHA256 mismatch for {filename}. "
                        f"Expected: {expected_sha256}, Got: {actual_hash}"
                    )
            
            # Move to final location
            shutil.move(str(temp_path), str(target_path))
            console.print(f"[green]✓ Downloaded {filename}[/green]")
            
            return target_path
            
        except requests.RequestException as e:
            raise DownloadError(f"Failed to download {url}: {e}")
    
    def _download_hf_hub(
        self,
        url: str,
        destination: Path,
        filename: str | None = None,
    ) -> Path:
        """Download using HuggingFace Hub library."""
        if not HAS_HF_HUB:
            return self._download_direct(url, destination, filename)
        
        # Parse HuggingFace URL
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        
        # URL format: huggingface.co/{user}/{repo}/resolve/{branch}/{path}
        if len(path_parts) >= 4 and path_parts[2] == "resolve":
            repo_id = f"{path_parts[0]}/{path_parts[1]}"
            branch = path_parts[3]
            file_path = "/".join(path_parts[4:])
        else:
            # Fallback to direct download
            return self._download_direct(url, destination, filename)
        
        if filename is None:
            filename = Path(file_path).name
        
        target_path = destination / filename
        
        if target_path.exists():
            console.print(f"[green]✓ {filename} already exists[/green]")
            return target_path
        
        console.print(f"[cyan]Downloading {filename} via HuggingFace Hub...[/cyan]")
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                revision=branch,
                local_dir=str(destination),
                local_dir_use_symlinks=False,
                token=self.hf_token,
            )
            
            # HF Hub might put file in a subdirectory, move it
            downloaded_path = Path(downloaded_path)
            if downloaded_path != target_path:
                shutil.move(str(downloaded_path), str(target_path))
            
            console.print(f"[green]✓ Downloaded {filename}[/green]")
            return target_path
            
        except Exception as e:
            console.print(f"[yellow]HF Hub download failed, trying direct: {e}[/yellow]")
            return self._download_direct(url, destination, filename)
    
    def _download_civitai(
        self,
        url: str,
        destination: Path,
        filename: str | None = None,
        expected_sha256: str | None = None,
    ) -> Path:
        """Download from CivitAI with API key."""
        # Add API key if available
        if self.civitai_api_key:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}token={self.civitai_api_key}"
        
        return self._download_direct(url, destination, filename, expected_sha256)
    
    def _detect_source(self, url: str) -> str:
        """Detect the source platform from URL."""
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
    
    def _is_hf_hub_url(self, url: str) -> bool:
        """Check if URL is a HuggingFace Hub URL that can use the hub library."""
        return "huggingface.co" in url and "/resolve/" in url
    
    def _extract_filename(self, url: str) -> str:
        """Extract filename from URL."""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        
        # Get the last path component
        filename = Path(path).name
        
        # Remove query parameters that might be attached
        if "?" in filename:
            filename = filename.split("?")[0]
        
        # Handle CivitAI-style URLs
        if not filename or filename == "download":
            # Try to get from content-disposition header
            try:
                response = self.session.head(url, allow_redirects=True, timeout=10)
                cd = response.headers.get("content-disposition", "")
                if "filename=" in cd:
                    filename = cd.split("filename=")[1].strip('"\'')
            except requests.RequestException:
                pass
        
        if not filename:
            filename = "model.safetensors"
        
        return filename
    
    def _verify_sha256(self, path: Path, expected: str) -> bool:
        """Verify file SHA256 hash."""
        actual = self._compute_sha256(path)
        return actual.lower() == expected.lower()
    
    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(self.CHUNK_SIZE), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


class BatchDownloader:
    """Download multiple models with summary and error handling."""
    
    def __init__(self, downloader: ModelDownloader):
        self.downloader = downloader
        self.results: dict[str, dict] = {}
    
    def download_all(
        self,
        downloads: list[dict],
        comfyui_path: Path,
    ) -> dict:
        """
        Download all models in a list.
        
        Args:
            downloads: List of dicts with keys: url, filename, target_folder, sha256 (optional)
            comfyui_path: Base ComfyUI installation path
        
        Returns:
            Summary dict with successful, failed, and skipped downloads
        """
        comfyui_path = Path(comfyui_path)
        
        successful = []
        failed = []
        skipped = []
        
        total = len(downloads)
        
        for i, item in enumerate(downloads, 1):
            url = item.get("url")
            filename = item.get("filename")
            target_folder = item.get("target_folder", "models")
            sha256 = item.get("sha256")
            
            if not url:
                console.print(f"[yellow]Skipping {filename}: No download URL[/yellow]")
                skipped.append({"filename": filename, "reason": "No URL"})
                continue
            
            destination = comfyui_path / target_folder
            
            console.print(f"\n[bold][{i}/{total}] {filename}[/bold]")
            console.print(f"  Target: {target_folder}")
            
            try:
                path = self.downloader.download(
                    url=url,
                    destination=destination,
                    filename=filename,
                    expected_sha256=sha256,
                )
                successful.append({
                    "filename": filename,
                    "path": str(path),
                    "url": url,
                })
            except DownloadError as e:
                console.print(f"[red]✗ Failed: {e}[/red]")
                failed.append({
                    "filename": filename,
                    "url": url,
                    "error": str(e),
                })
            except Exception as e:
                console.print(f"[red]✗ Unexpected error: {e}[/red]")
                failed.append({
                    "filename": filename,
                    "url": url,
                    "error": str(e),
                })
        
        # Print summary
        console.print("\n" + "=" * 50)
        console.print("[bold]Download Summary[/bold]")
        console.print("=" * 50)
        console.print(f"[green]✓ Successful: {len(successful)}[/green]")
        console.print(f"[red]✗ Failed: {len(failed)}[/red]")
        console.print(f"[yellow]⊘ Skipped: {len(skipped)}[/yellow]")
        
        if failed:
            console.print("\n[red]Failed downloads:[/red]")
            for item in failed:
                console.print(f"  - {item['filename']}: {item['error']}")
        
        return {
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
        }
