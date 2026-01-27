"""
Node Installer - Installs ComfyUI custom nodes from GitHub repositories.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from .model_database import GitHubSearch

console = Console()


class NodeInstallError(Exception):
    """Exception raised when node installation fails."""
    pass


class NodeInstaller:
    """
    Installs ComfyUI custom nodes from GitHub repositories.
    Handles git cloning and pip dependency installation.
    """
    
    def __init__(
        self,
        comfyui_path: Path | str,
        github_token: str | None = None,
        python_executable: str | None = None,
    ):
        self.comfyui_path = Path(comfyui_path)
        self.custom_nodes_path = self.comfyui_path / "custom_nodes"
        self.github_search = GitHubSearch(token=github_token)
        
        # Find Python executable
        if python_executable:
            self.python = python_executable
        else:
            self.python = self._find_python()
    
    def _find_python(self) -> str:
        """Find the Python executable for ComfyUI's environment."""
        # Check for common virtual environment locations
        possible_paths = [
            self.comfyui_path / "venv" / "Scripts" / "python.exe",  # Windows venv
            self.comfyui_path / "venv" / "bin" / "python",  # Linux/Mac venv
            self.comfyui_path / ".venv" / "Scripts" / "python.exe",
            self.comfyui_path / ".venv" / "bin" / "python",
            self.comfyui_path / "python_embeded" / "python.exe",  # Portable install
            self.comfyui_path.parent / "python_embeded" / "python.exe",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # Fall back to system Python
        return sys.executable
    
    def install(
        self,
        cnr_id: str,
        github_url: str | None = None,
        version: str | None = None,
    ) -> Path:
        """
        Install a custom node by its CNR ID.
        
        Args:
            cnr_id: ComfyUI Node Registry ID
            github_url: Override GitHub URL (auto-detected if None)
            version: Specific version/commit to install
        
        Returns:
            Path to the installed node directory
        """
        # Skip core nodes
        if cnr_id == "comfy-core":
            console.print(f"[dim]Skipping {cnr_id} (built-in)[/dim]")
            return self.comfyui_path
        
        # Find repository
        if github_url:
            repo_info = {"clone_url": github_url, "name": cnr_id}
        else:
            repo_info = self.github_search.find_repo(cnr_id)
        
        if not repo_info:
            raise NodeInstallError(f"Could not find repository for {cnr_id}")
        
        clone_url = repo_info.get("clone_url") or repo_info.get("html_url")
        if not clone_url:
            raise NodeInstallError(f"No clone URL found for {cnr_id}")
        
        # Ensure .git suffix
        if clone_url.endswith("/"):
            clone_url = clone_url[:-1]
        if not clone_url.endswith(".git"):
            clone_url = f"{clone_url}.git"
        
        repo_name = repo_info.get("name") or cnr_id
        node_path = self.custom_nodes_path / repo_name
        
        # Check if already installed
        if node_path.exists():
            console.print(f"[green]✓ {repo_name} already installed[/green]")
            
            # Optionally update
            if version:
                self._checkout_version(node_path, version)
            
            return node_path
        
        # Clone repository
        console.print(f"[cyan]Cloning {repo_name}...[/cyan]")
        
        try:
            self._git_clone(clone_url, node_path)
            
            if version:
                self._checkout_version(node_path, version)
            
            # Install dependencies
            self._install_dependencies(node_path)
            
            console.print(f"[green]✓ Installed {repo_name}[/green]")
            return node_path
            
        except Exception as e:
            # Clean up on failure
            if node_path.exists():
                import shutil
                shutil.rmtree(node_path, ignore_errors=True)
            raise NodeInstallError(f"Failed to install {cnr_id}: {e}")
    
    def install_from_url(self, github_url: str, name: str | None = None) -> Path:
        """Install a custom node directly from a GitHub URL."""
        if name is None:
            # Extract name from URL
            name = github_url.rstrip("/").split("/")[-1]
            if name.endswith(".git"):
                name = name[:-4]
        
        return self.install(name, github_url=github_url)
    
    def _git_clone(self, url: str, destination: Path) -> None:
        """Clone a git repository."""
        self.custom_nodes_path.mkdir(parents=True, exist_ok=True)
        
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(destination)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise NodeInstallError(f"Git clone failed: {result.stderr}")
    
    def _checkout_version(self, repo_path: Path, version: str) -> None:
        """Checkout a specific version/commit."""
        # Fetch all history for checkout
        subprocess.run(
            ["git", "fetch", "--unshallow"],
            cwd=repo_path,
            capture_output=True,
        )
        
        result = subprocess.run(
            ["git", "checkout", version],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            console.print(f"[yellow]Warning: Could not checkout version {version}[/yellow]")
    
    def _install_dependencies(self, node_path: Path) -> None:
        """Install Python dependencies for a custom node."""
        requirements_file = node_path / "requirements.txt"
        
        if not requirements_file.exists():
            return
        
        console.print(f"[cyan]Installing dependencies...[/cyan]")
        
        result = subprocess.run(
            [self.python, "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            console.print(f"[yellow]Warning: Some dependencies may have failed: {result.stderr}[/yellow]")
    
    def update(self, node_name: str) -> None:
        """Update an installed custom node to the latest version."""
        node_path = self.custom_nodes_path / node_name
        
        if not node_path.exists():
            raise NodeInstallError(f"Node {node_name} is not installed")
        
        console.print(f"[cyan]Updating {node_name}...[/cyan]")
        
        result = subprocess.run(
            ["git", "pull"],
            cwd=node_path,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise NodeInstallError(f"Git pull failed: {result.stderr}")
        
        # Reinstall dependencies
        self._install_dependencies(node_path)
        
        console.print(f"[green]✓ Updated {node_name}[/green]")
    
    def list_installed(self) -> list[dict]:
        """List all installed custom nodes."""
        nodes = []
        
        if not self.custom_nodes_path.exists():
            return nodes
        
        for item in self.custom_nodes_path.iterdir():
            if item.is_dir() and (item / ".git").exists():
                # Get git remote URL
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=item,
                    capture_output=True,
                    text=True,
                )
                
                remote_url = result.stdout.strip() if result.returncode == 0 else None
                
                # Get current commit
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=item,
                    capture_output=True,
                    text=True,
                )
                
                commit = result.stdout.strip()[:8] if result.returncode == 0 else None
                
                nodes.append({
                    "name": item.name,
                    "path": str(item),
                    "remote_url": remote_url,
                    "commit": commit,
                })
        
        return nodes
    
    def uninstall(self, node_name: str) -> None:
        """Uninstall a custom node."""
        import shutil
        
        node_path = self.custom_nodes_path / node_name
        
        if not node_path.exists():
            raise NodeInstallError(f"Node {node_name} is not installed")
        
        shutil.rmtree(node_path)
        console.print(f"[green]✓ Uninstalled {node_name}[/green]")


class BatchNodeInstaller:
    """Install multiple custom nodes with summary and error handling."""
    
    def __init__(self, installer: NodeInstaller):
        self.installer = installer
    
    def install_all(self, nodes: list[dict]) -> dict:
        """
        Install all custom nodes in a list.
        
        Args:
            nodes: List of dicts with keys: cnr_id, github_url (optional), version (optional)
        
        Returns:
            Summary dict with successful, failed, and skipped installations
        """
        successful = []
        failed = []
        skipped = []
        
        total = len(nodes)
        
        for i, node in enumerate(nodes, 1):
            cnr_id = node.get("cnr_id")
            github_url = node.get("github_url")
            version = node.get("version")
            
            if not cnr_id:
                continue
            
            if cnr_id == "comfy-core":
                skipped.append({"cnr_id": cnr_id, "reason": "Built-in"})
                continue
            
            console.print(f"\n[bold][{i}/{total}] {cnr_id}[/bold]")
            
            try:
                path = self.installer.install(
                    cnr_id=cnr_id,
                    github_url=github_url,
                    version=version,
                )
                successful.append({
                    "cnr_id": cnr_id,
                    "path": str(path),
                })
            except NodeInstallError as e:
                console.print(f"[red]✗ Failed: {e}[/red]")
                failed.append({
                    "cnr_id": cnr_id,
                    "error": str(e),
                })
            except Exception as e:
                console.print(f"[red]✗ Unexpected error: {e}[/red]")
                failed.append({
                    "cnr_id": cnr_id,
                    "error": str(e),
                })
        
        # Print summary
        console.print("\n" + "=" * 50)
        console.print("[bold]Custom Node Installation Summary[/bold]")
        console.print("=" * 50)
        console.print(f"[green]✓ Successful: {len(successful)}[/green]")
        console.print(f"[red]✗ Failed: {len(failed)}[/red]")
        console.print(f"[yellow]⊘ Skipped: {len(skipped)}[/yellow]")
        
        if failed:
            console.print("\n[red]Failed installations:[/red]")
            for item in failed:
                console.print(f"  - {item['cnr_id']}: {item['error']}")
        
        return {
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
        }
