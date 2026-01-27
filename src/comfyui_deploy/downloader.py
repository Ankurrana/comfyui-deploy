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


class ParallelDownloader:
    """
    High-performance parallel downloader for multiple files.
    
    Uses concurrent downloads with configurable workers for maximum speed.
    Supports segmented downloads for very large files.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        hf_token: str | None = None,
        civitai_api_key: str | None = None,
        segments_per_file: int = 4,
    ):
        """
        Initialize parallel downloader.
        
        Args:
            max_workers: Maximum concurrent file downloads (default: 4)
            hf_token: HuggingFace API token for gated models
            civitai_api_key: CivitAI API key
            segments_per_file: Number of segments for large file downloads (default: 4)
        """
        self.max_workers = max_workers
        self.hf_token = hf_token
        self.civitai_api_key = civitai_api_key
        self.segments_per_file = segments_per_file
        
        # Create a base downloader for single-file downloads
        self.downloader = ModelDownloader(
            hf_token=hf_token,
            civitai_api_key=civitai_api_key,
        )
    
    def download_all(
        self,
        downloads: list[dict],
        progress_callback: callable = None,
    ) -> dict:
        """
        Download multiple files in parallel.
        
        Args:
            downloads: List of dicts with keys: url, destination, filename (optional)
            progress_callback: Optional callback(completed, total, current_file)
            
        Returns:
            Dict with successful, failed, skipped lists
        """
        import concurrent.futures
        from threading import Lock
        
        successful = []
        failed = []
        skipped = []
        
        total = len(downloads)
        completed = 0
        lock = Lock()
        
        console.print(f"\n[bold cyan]⚡ Parallel Download ({self.max_workers} workers)[/bold cyan]")
        console.print(f"   {total} files to download\n")
        
        def download_one(item: dict) -> tuple[str, dict | None, str | None]:
            """Download a single file and return result."""
            url = item.get("url")
            destination = Path(item.get("destination"))
            filename = item.get("filename")
            
            if not url:
                return ("skip", {"filename": filename or "unknown", "reason": "No URL"}, None)
            
            try:
                path = self.downloader.download(
                    url=url,
                    destination=destination,
                    filename=filename,
                )
                return ("success", {"filename": path.name, "path": str(path), "url": url}, None)
            except Exception as e:
                return ("fail", {"filename": filename or url, "url": url}, str(e))
        
        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_item = {
                executor.submit(download_one, item): item 
                for item in downloads
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    status, result, error = future.result()
                    
                    with lock:
                        completed += 1
                        
                        if status == "success":
                            successful.append(result)
                            console.print(f"[green]✓ [{completed}/{total}] {result['filename']}[/green]")
                        elif status == "skip":
                            skipped.append(result)
                            console.print(f"[yellow]⊘ [{completed}/{total}] {result['filename']}: {result.get('reason', 'Skipped')}[/yellow]")
                        else:
                            result["error"] = error
                            failed.append(result)
                            console.print(f"[red]✗ [{completed}/{total}] {result['filename']}: {error}[/red]")
                        
                        if progress_callback:
                            progress_callback(completed, total, result.get('filename', ''))
                            
                except Exception as e:
                    with lock:
                        completed += 1
                        failed.append({
                            "filename": item.get("filename", "unknown"),
                            "url": item.get("url", ""),
                            "error": str(e),
                        })
                        console.print(f"[red]✗ [{completed}/{total}] Error: {e}[/red]")
        
        # Print summary
        console.print("\n" + "=" * 50)
        console.print("[bold]⚡ Parallel Download Summary[/bold]")
        console.print("=" * 50)
        console.print(f"[green]✓ Successful: {len(successful)}[/green]")
        console.print(f"[red]✗ Failed: {len(failed)}[/red]")
        console.print(f"[yellow]⊘ Skipped: {len(skipped)}[/yellow]")
        
        if failed:
            console.print("\n[red]Failed downloads:[/red]")
            for item in failed:
                console.print(f"  - {item['filename']}: {item.get('error', 'Unknown error')}")
        
        return {
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
        }
    
    def download_large_file_segmented(
        self,
        url: str,
        destination: Path,
        filename: str | None = None,
    ) -> Path:
        """
        Download a large file using multiple parallel segments.
        
        This can speed up downloads from servers that support Range requests
        by downloading multiple parts of the file simultaneously.
        
        Args:
            url: Download URL
            destination: Target directory
            filename: Output filename
            
        Returns:
            Path to downloaded file
        """
        import concurrent.futures
        import tempfile
        
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        
        # Get file info
        headers = {}
        if "huggingface.co" in url and self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        
        session = requests.Session()
        response = session.head(url, headers=headers, allow_redirects=True, timeout=30)
        
        # Check if server supports range requests
        accepts_ranges = response.headers.get("Accept-Ranges") == "bytes"
        total_size = int(response.headers.get("Content-Length", 0))
        
        if not accepts_ranges or total_size == 0 or total_size < 50 * 1024 * 1024:  # Less than 50MB
            # Fall back to regular download
            return self.downloader.download(url, destination, filename)
        
        # Extract filename
        if filename is None:
            filename = self.downloader._extract_filename(url)
        
        target_path = destination / filename
        
        # Check if already exists
        if target_path.exists() and target_path.stat().st_size == total_size:
            console.print(f"[green]✓ {filename} already exists[/green]")
            return target_path
        
        console.print(f"[cyan]⚡ Segmented download: {filename} ({self.downloader._format_size(total_size)})[/cyan]")
        console.print(f"   Using {self.segments_per_file} parallel segments")
        
        # Calculate segment ranges
        segment_size = total_size // self.segments_per_file
        segments = []
        for i in range(self.segments_per_file):
            start = i * segment_size
            end = start + segment_size - 1 if i < self.segments_per_file - 1 else total_size - 1
            segments.append((i, start, end))
        
        # Download segments in parallel
        temp_dir = tempfile.mkdtemp(prefix="comfyui_download_")
        segment_files = []
        
        def download_segment(seg_info: tuple) -> tuple[int, str]:
            seg_id, start, end = seg_info
            seg_path = Path(temp_dir) / f"segment_{seg_id}"
            
            seg_headers = headers.copy()
            seg_headers["Range"] = f"bytes={start}-{end}"
            
            response = session.get(
                url,
                headers=seg_headers,
                stream=True,
                timeout=60,
            )
            response.raise_for_status()
            
            with open(seg_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=128 * 1024):
                    f.write(chunk)
            
            return (seg_id, str(seg_path))
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.segments_per_file) as executor:
                futures = [executor.submit(download_segment, seg) for seg in segments]
                
                for future in concurrent.futures.as_completed(futures):
                    seg_id, seg_path = future.result()
                    segment_files.append((seg_id, seg_path))
                    console.print(f"  [green]✓ Segment {seg_id + 1}/{self.segments_per_file} complete[/green]")
            
            # Sort and merge segments
            segment_files.sort(key=lambda x: x[0])
            
            console.print("  [cyan]Merging segments...[/cyan]")
            with open(target_path, "wb") as outfile:
                for seg_id, seg_path in segment_files:
                    with open(seg_path, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)
            
            console.print(f"[green]✓ {filename} downloaded successfully[/green]")
            return target_path
            
        finally:
            # Cleanup temp files
            shutil.rmtree(temp_dir, ignore_errors=True)
