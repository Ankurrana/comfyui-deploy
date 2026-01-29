"""
Setup ComfyUI with default packages and optional update functionality.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from comfyui_deploy.node_installer import NodeInstaller


def load_default_packages(config_path: str | Path | None = None) -> list[dict]:
    """Load default packages from JSON config file."""
    if config_path is None:
        # Look for default-packages.json in same directory as this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "default-packages.json"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"⚠️  Default packages config not found: {config_path}")
        return []
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("custom_nodes", [])


def update_comfyui(comfyui_path: str | Path) -> bool:
    """
    Update ComfyUI to the latest version using git pull.
    
    Args:
        comfyui_path: Path to ComfyUI installation
        
    Returns:
        True if update successful, False otherwise
    """
    comfyui = Path(comfyui_path)
    
    if not comfyui.exists():
        print(f"❌ ComfyUI path does not exist: {comfyui}")
        return False
    
    git_dir = comfyui / ".git"
    if not git_dir.exists():
        print(f"❌ ComfyUI is not a git repository. Cannot update.")
        print(f"   If using portable version, download new release manually.")
        return False
    
    print("=" * 70)
    print("UPDATING COMFYUI")
    print("=" * 70)
    
    try:
        # Stash any local changes
        print("\n📦 Stashing local changes...")
        subprocess.run(
            ["git", "stash"],
            cwd=comfyui,
            check=True,
            capture_output=True,
            text=True,
        )
        
        # Fetch latest
        print("🔄 Fetching latest changes...")
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=comfyui,
            check=True,
            capture_output=True,
            text=True,
        )
        
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=comfyui,
            check=True,
            capture_output=True,
            text=True,
        )
        branch = result.stdout.strip() or "master"
        
        # Pull latest
        print(f"⬇️  Pulling latest from {branch}...")
        result = subprocess.run(
            ["git", "pull", "origin", branch],
            cwd=comfyui,
            check=True,
            capture_output=True,
            text=True,
        )
        
        if "Already up to date" in result.stdout:
            print("✅ ComfyUI is already up to date!")
        else:
            print("✅ ComfyUI updated successfully!")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Update failed: {e}")
        if e.stderr:
            print(f"   {e.stderr}")
        return False


def install_default_packages(
    comfyui_path: str | Path,
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Install default custom node packages.
    
    Args:
        comfyui_path: Path to ComfyUI installation
        config_path: Optional path to custom default-packages.json
        dry_run: If True, only show what would be installed
        
    Returns:
        Dict with installed, skipped, and failed counts
    """
    comfyui = Path(comfyui_path)
    custom_nodes_dir = comfyui / "custom_nodes"
    
    if not custom_nodes_dir.exists():
        print(f"❌ custom_nodes directory not found: {custom_nodes_dir}")
        return {"installed": 0, "skipped": 0, "failed": 0}
    
    packages = load_default_packages(config_path)
    
    if not packages:
        print("⚠️  No default packages configured")
        return {"installed": 0, "skipped": 0, "failed": 0}
    
    print("=" * 70)
    print("DEFAULT PACKAGES")
    print("=" * 70)
    
    print(f"\n📦 Default packages to install: {len(packages)}")
    
    for pkg in packages:
        name = pkg.get("name", pkg.get("cnr_id"))
        cnr_id = pkg.get("cnr_id")
        github_url = pkg.get("github_url")
        description = pkg.get("description", "")
        
        # Check if already installed
        variations = [
            cnr_id,
            cnr_id.lower() if cnr_id else "",
            cnr_id.replace("-", "_") if cnr_id else "",
            cnr_id.replace("_", "-") if cnr_id else "",
            name,
            name.lower() if name else "",
        ]
        
        installed = any((custom_nodes_dir / v).exists() for v in variations if v)
        status = "✅" if installed else "❌"
        
        print(f"   {status} {name}")
        if description:
            print(f"      {description}")
        if github_url:
            print(f"      → {github_url}")
    
    if dry_run:
        print("\n" + "=" * 70)
        print("⚠️  DRY RUN - No changes will be made")
        print("=" * 70)
        return {"installed": 0, "skipped": 0, "failed": 0}
    
    print("\n" + "=" * 70)
    print("INSTALLING DEFAULT PACKAGES")
    print("=" * 70)
    
    installer = NodeInstaller(comfyui_path=comfyui)
    installed = 0
    skipped = 0
    failed = 0
    
    for pkg in packages:
        cnr_id = pkg.get("cnr_id")
        github_url = pkg.get("github_url")
        name = pkg.get("name", cnr_id)
        
        print(f"\n📥 {name}")
        
        try:
            path = installer.install(
                cnr_id=cnr_id,
                github_url=github_url,
            )
            if "already installed" in str(path).lower():
                skipped += 1
            else:
                installed += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n📦 Default Packages:")
    print(f"   • Installed: {installed}")
    print(f"   • Already present: {skipped}")
    print(f"   • Failed: {failed}")
    
    return {"installed": installed, "skipped": skipped, "failed": failed}


def setup_comfyui(
    comfyui_path: str,
    update: bool = False,
    install_defaults: bool = True,
    config_path: str | None = None,
    dry_run: bool = False,
):
    """
    Complete ComfyUI setup: update and install default packages.
    
    Args:
        comfyui_path: Path to ComfyUI installation
        update: Whether to update ComfyUI first
        install_defaults: Whether to install default packages
        config_path: Optional path to custom default-packages.json
        dry_run: If True, only show what would be done
    """
    comfyui = Path(comfyui_path)
    
    if not comfyui.exists():
        print(f"❌ ComfyUI path does not exist: {comfyui}")
        return
    
    # Verify this looks like a ComfyUI installation
    custom_nodes_dir = comfyui / "custom_nodes"
    models_dir = comfyui / "models"
    
    if not custom_nodes_dir.exists() or not models_dir.exists():
        print(f"❌ This doesn't look like a valid ComfyUI installation")
        print(f"   Missing: custom_nodes/ or models/ directory")
        return
    
    if update and not dry_run:
        update_comfyui(comfyui_path)
    elif update and dry_run:
        print("=" * 70)
        print("WOULD UPDATE COMFYUI (dry run)")
        print("=" * 70)
    
    if install_defaults:
        install_default_packages(comfyui_path, config_path, dry_run)
    
    if not dry_run:
        print("\n🎉 Setup complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup ComfyUI with default packages and updates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install default packages only
  python setup_comfyui.py --comfyui /workspace/ComfyUI
  
  # Update ComfyUI and install defaults
  python setup_comfyui.py --comfyui /workspace/ComfyUI --update
  
  # Only update ComfyUI (no default packages)
  python setup_comfyui.py --comfyui /workspace/ComfyUI --update --no-defaults
  
  # Dry run - see what would happen
  python setup_comfyui.py --comfyui /workspace/ComfyUI --update --dry-run
  
  # Use custom packages config
  python setup_comfyui.py --comfyui /workspace/ComfyUI --config my-packages.json
        """
    )
    parser.add_argument("--comfyui", "-c", required=True, help="Path to ComfyUI installation")
    parser.add_argument("--update", "-u", action="store_true", help="Update ComfyUI to latest version")
    parser.add_argument("--no-defaults", action="store_true", help="Skip installing default packages")
    parser.add_argument("--config", help="Path to custom default-packages.json")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    print(f"ComfyUI:  {args.comfyui}")
    print(f"Update:   {args.update}")
    print(f"Defaults: {not args.no_defaults}")
    print(f"Dry run:  {args.dry_run}")
    print()
    
    setup_comfyui(
        comfyui_path=args.comfyui,
        update=args.update,
        install_defaults=not args.no_defaults,
        config_path=args.config,
        dry_run=args.dry_run,
    )
