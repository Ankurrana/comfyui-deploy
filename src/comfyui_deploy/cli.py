"""
CLI - Command Line Interface for ComfyUI Deploy.
"""

import sys
from pathlib import Path

import click
from rich.console import Console

from .config import Config, get_config
from .workflow_parser import WorkflowParser
from .deployer import WorkflowDeployer, search_model
from .model_database import ModelDatabase

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="comfyui-deploy")
def main():
    """
    ComfyUI Deploy - Automated workflow deployment tool.
    
    Analyze ComfyUI workflows to extract required models and custom nodes,
    then automatically download and install them.
    """
    pass


@main.command()
@click.argument("workflow", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Export manifest to file")
@click.option("--format", "output_format", type=click.Choice(["text", "yaml", "json"]), default="text")
def analyze(workflow: Path, output: Path | None, output_format: str):
    """
    Analyze a workflow and show required dependencies.
    
    Parses the workflow JSON file to extract all required models and custom nodes.
    Searches HuggingFace and CivitAI to find download URLs for models.
    """
    deployer = WorkflowDeployer()
    deps = deployer.analyze(workflow)
    
    if output:
        deployer.export_manifest(deps, output)
    else:
        deployer.print_report(deps)


@main.command()
@click.argument("workflow", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--comfyui-path", "-p",
    type=click.Path(exists=True, path_type=Path),
    help="Path to ComfyUI installation"
)
@click.option("--download-models/--no-download-models", default=True, help="Download required models")
@click.option("--install-nodes/--no-install-nodes", default=True, help="Install required custom nodes")
@click.option("--dry-run", is_flag=True, help="Analyze only, don't make changes")
@click.option("--manifest", "-m", type=click.Path(path_type=Path), help="Export manifest after analysis")
def deploy(
    workflow: Path,
    comfyui_path: Path | None,
    download_models: bool,
    install_nodes: bool,
    dry_run: bool,
    manifest: Path | None,
):
    """
    Deploy a workflow - download models and install custom nodes.
    
    Analyzes the workflow, searches for required models and custom nodes,
    then downloads/installs them to the specified ComfyUI installation.
    """
    config = get_config()
    
    if comfyui_path:
        pass
    elif config.comfyui_path:
        comfyui_path = config.comfyui_path
    else:
        console.print("[red]Error: ComfyUI path not specified.[/red]")
        console.print("Use --comfyui-path or run: comfyui-deploy config set comfyui_path /path/to/ComfyUI")
        sys.exit(1)
    
    deployer = WorkflowDeployer(comfyui_path=comfyui_path)
    
    try:
        if manifest and dry_run:
            deps = deployer.analyze(workflow)
            deployer.export_manifest(deps, manifest)
        
        results = deployer.deploy(
            workflow,
            download_models=download_models,
            install_nodes=install_nodes,
            dry_run=dry_run,
        )
        
        if manifest and not dry_run:
            deps = deployer.analyze(workflow)
            deployer.export_manifest(deps, manifest)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("manifest", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--comfyui-path", "-p",
    type=click.Path(exists=True, path_type=Path),
    help="Path to ComfyUI installation"
)
@click.option("--download-models/--no-download-models", default=True)
@click.option("--install-nodes/--no-install-nodes", default=True)
def install(
    manifest: Path,
    comfyui_path: Path | None,
    download_models: bool,
    install_nodes: bool,
):
    """
    Install from a manifest file.
    
    Deploys models and custom nodes from a previously exported manifest YAML file.
    """
    config = get_config()
    
    if not comfyui_path:
        comfyui_path = config.comfyui_path
    
    if not comfyui_path:
        console.print("[red]Error: ComfyUI path not specified.[/red]")
        sys.exit(1)
    
    deployer = WorkflowDeployer(comfyui_path=comfyui_path)
    
    try:
        deployer.deploy_from_manifest(
            manifest,
            download_models=download_models,
            install_nodes=install_nodes,
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("query")
@click.option("--type", "model_type", help="Model type (checkpoint, lora, vae, etc.)")
@click.option(
    "--source", "sources",
    multiple=True,
    type=click.Choice(["huggingface", "civitai"]),
    help="Sources to search"
)
@click.option("--limit", default=10, help="Maximum results to show")
def search(query: str, model_type: str | None, sources: tuple, limit: int):
    """
    Search for a model across platforms.
    
    Searches HuggingFace and CivitAI for models matching the query.
    """
    source_list = list(sources) if sources else None
    search_model(query, model_type=model_type, sources=source_list, limit=limit)


@main.group()
def config():
    """Manage configuration settings."""
    pass


@config.command("show")
def config_show():
    """Show current configuration."""
    cfg = get_config()
    
    console.print("[bold]Current Configuration[/bold]\n")
    
    for key, value in cfg.to_dict().items():
        if value is None:
            value = "[dim]not set[/dim]"
        console.print(f"  {key}: {value}")
    
    console.print(f"\n[dim]Config file: {cfg.config_path}[/dim]")


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a configuration value."""
    cfg = get_config()
    
    # Handle boolean values
    if value.lower() in ("true", "yes", "1"):
        value = True
    elif value.lower() in ("false", "no", "0"):
        value = False
    
    cfg.set(key, value)
    cfg.save()
    
    console.print(f"[green]✓ Set {key} = {value}[/green]")


@config.command("get")
@click.argument("key")
def config_get(key: str):
    """Get a configuration value."""
    cfg = get_config()
    value = cfg.get(key)
    
    if value is None:
        console.print(f"[dim]{key} is not set[/dim]")
    else:
        # Mask sensitive values
        if key in ("hf_token", "civitai_api_key", "github_token") and value:
            value = "***" + value[-4:] if len(value) > 4 else "***"
        console.print(f"{key} = {value}")


@config.command("init")
@click.option(
    "--comfyui-path", "-p",
    type=click.Path(path_type=Path),
    prompt="ComfyUI installation path",
    help="Path to ComfyUI installation"
)
@click.option("--hf-token", prompt=True, default="", hide_input=True, help="HuggingFace token (optional)")
@click.option("--civitai-key", prompt=True, default="", hide_input=True, help="CivitAI API key (optional)")
def config_init(comfyui_path: Path, hf_token: str, civitai_key: str):
    """Initialize configuration interactively."""
    cfg = get_config()
    
    cfg.set("comfyui_path", str(comfyui_path))
    
    if hf_token:
        cfg.set("hf_token", hf_token)
    
    if civitai_key:
        cfg.set("civitai_api_key", civitai_key)
    
    cfg.save()
    
    console.print("\n[green]✓ Configuration saved![/green]")
    console.print(f"[dim]Config file: {cfg.config_path}[/dim]")


@main.command()
@click.option(
    "--comfyui-path", "-p",
    type=click.Path(exists=True, path_type=Path),
    help="Path to ComfyUI installation"
)
def list_nodes(comfyui_path: Path | None):
    """List installed custom nodes."""
    from .node_installer import NodeInstaller
    
    config = get_config()
    
    if not comfyui_path:
        comfyui_path = config.comfyui_path
    
    if not comfyui_path:
        console.print("[red]Error: ComfyUI path not specified.[/red]")
        sys.exit(1)
    
    installer = NodeInstaller(comfyui_path)
    nodes = installer.list_installed()
    
    if not nodes:
        console.print("[dim]No custom nodes installed[/dim]")
        return
    
    from rich.table import Table
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Commit")
    table.add_column("Remote URL")
    
    for node in nodes:
        table.add_row(
            node["name"],
            node["commit"] or "-",
            node["remote_url"] or "-",
        )
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(nodes)} custom nodes[/dim]")


if __name__ == "__main__":
    main()
