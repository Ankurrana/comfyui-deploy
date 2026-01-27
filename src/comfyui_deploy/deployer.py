"""
Workflow Deployer - Main orchestrator for deploying ComfyUI workflows.
"""

from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .workflow_parser import WorkflowParser, WorkflowDependencies, ModelReference
from .model_database import ModelDatabase, get_known_model_url
from .smart_search import smart_search, SmartModelSearch, CivitAISearch
from .downloader import ModelDownloader, BatchDownloader
from .node_installer import NodeInstaller, BatchNodeInstaller
from .config import Config, get_config

console = Console()


class WorkflowDeployer:
    """
    Main class for deploying ComfyUI workflows.
    Coordinates parsing, model search, download, and node installation.
    """
    
    def __init__(
        self,
        comfyui_path: Path | str | None = None,
        config: Config | None = None,
    ):
        self.config = config or get_config()
        
        # Use provided path or fall back to config
        if comfyui_path:
            self.comfyui_path = Path(comfyui_path)
        elif self.config.comfyui_path:
            self.comfyui_path = self.config.comfyui_path
        else:
            self.comfyui_path = None
        
        # Initialize components
        self.parser = WorkflowParser()
        self.model_db = ModelDatabase(
            hf_token=self.config.hf_token,
            civitai_api_key=self.config.civitai_api_key,
            github_token=self.config.github_token,
        )
        
        self.downloader = ModelDownloader(
            hf_token=self.config.hf_token,
            civitai_api_key=self.config.civitai_api_key,
            resume=self.config.get("resume_downloads", True),
        )
        
        if self.comfyui_path:
            self.node_installer = NodeInstaller(
                comfyui_path=self.comfyui_path,
                github_token=self.config.github_token,
            )
        else:
            self.node_installer = None
    
    def analyze(self, workflow_path: Path | str) -> WorkflowDependencies:
        """
        Analyze a workflow and extract all dependencies.
        Also searches for download URLs for models.
        
        Returns:
            WorkflowDependencies with all found models and custom nodes
        """
        workflow_path = Path(workflow_path)
        
        console.print(f"[bold cyan]Analyzing workflow: {workflow_path.name}[/bold cyan]\n")
        
        # Parse workflow
        deps = self.parser.parse(workflow_path)
        
        # Search for model download URLs
        self._resolve_model_urls(deps)
        
        # Resolve custom node repositories
        self._resolve_node_repos(deps)
        
        return deps
    
    def _resolve_model_urls(self, deps: WorkflowDependencies) -> None:
        """Find download URLs for models that don't have them."""
        console.print("[cyan]Resolving model download URLs...[/cyan]")
        
        for model in deps.models:
            if model.download_url:
                console.print(f"  [green]✓[/green] {model.filename} (from workflow docs)")
                continue
            
            # Use smart search (ComfyUI Manager → HuggingFace → CivitAI)
            console.print(f"  [dim]Searching for {model.filename}...[/dim]")
            
            result = smart_search(
                model.filename,
                model_type=model.model_type,
                hf_token=self.config.hf_token,
                civitai_key=self.config.civitai_api_key,
            )
            
            if result:
                model.download_url = result["download_url"]
                model.source = result["source"]
                confidence = result.get("confidence", "found")
                console.print(f"  [green]✓[/green] {model.filename} ({result['source']}, {confidence})")
            else:
                console.print(f"  [yellow]![/yellow] Could not find {model.filename}")
    
    def _resolve_node_repos(self, deps: WorkflowDependencies) -> None:
        """Find GitHub URLs for custom nodes that don't have them."""
        console.print("\n[cyan]Resolving custom node repositories...[/cyan]")
        
        for node in deps.custom_nodes:
            if node.github_url:
                console.print(f"  [green]✓[/green] {node.cnr_id}")
                continue
            
            if node.cnr_id == "comfy-core":
                continue
            
            # Search for repository
            repo_info = self.model_db.github_search.find_repo(node.cnr_id)
            
            if repo_info:
                node.github_url = repo_info.get("clone_url") or repo_info.get("html_url")
                console.print(f"  [green]✓[/green] Found {node.cnr_id}: {node.github_url}")
            else:
                console.print(f"  [yellow]![/yellow] Could not find {node.cnr_id}")
    
    def print_report(self, deps: WorkflowDependencies) -> None:
        """Print a formatted report of workflow dependencies."""
        # Models table
        console.print("\n")
        console.print(Panel("[bold]Models Required[/bold]", expand=False))
        
        if deps.models:
            table = Table(show_header=True, header_style="bold")
            table.add_column("Model", style="cyan")
            table.add_column("Type")
            table.add_column("Folder")
            table.add_column("Source")
            table.add_column("URL Found", justify="center")
            
            for model in deps.models:
                url_status = "[green]✓[/green]" if model.download_url else "[red]✗[/red]"
                source = model.source or "-"
                
                table.add_row(
                    model.filename,
                    model.model_type,
                    model.target_folder,
                    source,
                    url_status,
                )
            
            console.print(table)
        else:
            console.print("[dim]No models found[/dim]")
        
        # Custom nodes table
        console.print("\n")
        console.print(Panel("[bold]Custom Nodes Required[/bold]", expand=False))
        
        if deps.custom_nodes:
            table = Table(show_header=True, header_style="bold")
            table.add_column("Node Pack", style="cyan")
            table.add_column("GitHub URL")
            table.add_column("Node Types Used")
            
            for node in deps.custom_nodes:
                if node.cnr_id == "comfy-core":
                    continue
                
                github = node.github_url or "[red]Not found[/red]"
                node_types = ", ".join(node.node_types[:3])
                if len(node.node_types) > 3:
                    node_types += f" (+{len(node.node_types) - 3} more)"
                
                table.add_row(node.cnr_id, github, node_types)
            
            console.print(table)
        else:
            console.print("[dim]No custom nodes required (uses built-in nodes only)[/dim]")
        
        # Summary
        models_with_url = sum(1 for m in deps.models if m.download_url)
        nodes_with_url = sum(1 for n in deps.custom_nodes if n.github_url and n.cnr_id != "comfy-core")
        total_nodes = sum(1 for n in deps.custom_nodes if n.cnr_id != "comfy-core")
        
        console.print("\n")
        console.print(Panel(
            f"[bold]Summary[/bold]\n"
            f"Models: {models_with_url}/{len(deps.models)} URLs found\n"
            f"Custom Nodes: {nodes_with_url}/{total_nodes} repos found",
            expand=False
        ))
    
    def export_manifest(
        self,
        deps: WorkflowDependencies,
        output_path: Path | str,
    ) -> None:
        """Export dependencies to a YAML manifest file."""
        output_path = Path(output_path)
        
        manifest = {
            "version": "1.0",
            "models": [
                {
                    "filename": m.filename,
                    "type": m.model_type,
                    "target_folder": m.target_folder,
                    "download_url": m.download_url,
                    "source": m.source,
                }
                for m in deps.models
            ],
            "custom_nodes": [
                {
                    "cnr_id": n.cnr_id,
                    "github_url": n.github_url,
                    "version": n.version,
                }
                for n in deps.custom_nodes
                if n.cnr_id != "comfy-core"
            ],
        }
        
        with open(output_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[green]✓ Exported manifest to {output_path}[/green]")
    
    def deploy(
        self,
        workflow_path: Path | str,
        download_models: bool = True,
        install_nodes: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """
        Deploy a workflow - download models and install custom nodes.
        
        Args:
            workflow_path: Path to workflow JSON file
            download_models: Whether to download models
            install_nodes: Whether to install custom nodes
            dry_run: If True, only analyze without making changes
        
        Returns:
            Summary of deployment results
        """
        if not self.comfyui_path:
            raise ValueError("ComfyUI path not configured. Set it with --comfyui-path or in config.")
        
        if not self.comfyui_path.exists():
            raise ValueError(f"ComfyUI path does not exist: {self.comfyui_path}")
        
        # Analyze workflow
        deps = self.analyze(workflow_path)
        self.print_report(deps)
        
        if dry_run:
            console.print("\n[yellow]Dry run - no changes made[/yellow]")
            return {"dry_run": True, "dependencies": self.parser.to_dict()}
        
        results = {
            "models": None,
            "custom_nodes": None,
        }
        
        # Install custom nodes first (models might depend on them)
        if install_nodes and deps.custom_nodes:
            console.print("\n" + "=" * 60)
            console.print("[bold cyan]Installing Custom Nodes[/bold cyan]")
            console.print("=" * 60)
            
            batch_installer = BatchNodeInstaller(self.node_installer)
            node_data = [
                {
                    "cnr_id": n.cnr_id,
                    "github_url": n.github_url,
                    "version": n.version,
                }
                for n in deps.custom_nodes
            ]
            results["custom_nodes"] = batch_installer.install_all(node_data)
        
        # Download models
        if download_models and deps.models:
            console.print("\n" + "=" * 60)
            console.print("[bold cyan]Downloading Models[/bold cyan]")
            console.print("=" * 60)
            
            batch_downloader = BatchDownloader(self.downloader)
            download_data = [
                {
                    "url": m.download_url,
                    "filename": m.filename,
                    "target_folder": m.target_folder,
                }
                for m in deps.models
                if m.download_url
            ]
            results["models"] = batch_downloader.download_all(download_data, self.comfyui_path)
            
            # Report models without URLs
            missing_urls = [m for m in deps.models if not m.download_url]
            if missing_urls:
                console.print("\n[yellow]Models without download URLs (manual download required):[/yellow]")
                for m in missing_urls:
                    console.print(f"  - {m.filename} ({m.target_folder})")
        
        # Final summary
        console.print("\n" + "=" * 60)
        console.print("[bold green]Deployment Complete![/bold green]")
        console.print("=" * 60)
        
        return results
    
    def deploy_from_manifest(
        self,
        manifest_path: Path | str,
        download_models: bool = True,
        install_nodes: bool = True,
    ) -> dict:
        """
        Deploy from a previously exported manifest file.
        
        Args:
            manifest_path: Path to manifest YAML file
            download_models: Whether to download models
            install_nodes: Whether to install custom nodes
        
        Returns:
            Summary of deployment results
        """
        manifest_path = Path(manifest_path)
        
        with open(manifest_path, "r") as f:
            manifest = yaml.safe_load(f)
        
        if not self.comfyui_path:
            raise ValueError("ComfyUI path not configured")
        
        results = {
            "models": None,
            "custom_nodes": None,
        }
        
        # Install custom nodes
        if install_nodes and manifest.get("custom_nodes"):
            console.print("\n[bold cyan]Installing Custom Nodes[/bold cyan]")
            
            batch_installer = BatchNodeInstaller(self.node_installer)
            results["custom_nodes"] = batch_installer.install_all(manifest["custom_nodes"])
        
        # Download models
        if download_models and manifest.get("models"):
            console.print("\n[bold cyan]Downloading Models[/bold cyan]")
            
            batch_downloader = BatchDownloader(self.downloader)
            results["models"] = batch_downloader.download_all(
                manifest["models"],
                self.comfyui_path,
            )
        
        return results


def search_model(
    query: str,
    model_type: str | None = None,
    sources: list[str] | None = None,
    limit: int = 10,
) -> None:
    """Search for a model across platforms and display results."""
    config = get_config()
    
    model_db = ModelDatabase(
        hf_token=config.hf_token,
        civitai_api_key=config.civitai_api_key,
    )
    
    console.print(f"[cyan]Searching for '{query}'...[/cyan]\n")
    
    results = model_db.search_model(
        filename=query,
        model_type=model_type,
        sources=sources,
        limit=limit,
    )
    
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim")
    table.add_column("Name")
    table.add_column("Filename")
    table.add_column("Source")
    table.add_column("Size")
    table.add_column("Score", justify="right")
    
    for i, result in enumerate(results, 1):
        table.add_row(
            str(i),
            result.name[:40] + "..." if len(result.name) > 40 else result.name,
            result.filename,
            result.source,
            result.size_str() if result.size_bytes else "-",
            f"{result.score:.0f}",
        )
    
    console.print(table)
    console.print(f"\n[dim]Found {len(results)} results[/dim]")
