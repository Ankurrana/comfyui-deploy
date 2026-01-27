"""
Complete workflow deployment - Downloads models AND installs custom nodes.
"""
import sys
from pathlib import Path
from comfyui_deploy.workflow_parser import WorkflowParser
from comfyui_deploy.smart_search import smart_search
from comfyui_deploy.node_installer import NodeInstaller, BatchNodeInstaller
from comfyui_deploy.downloader import ModelDownloader, BatchDownloader, ParallelDownloader

def deploy_workflow(workflow_path: str, comfyui_path: str, dry_run: bool = False, parallel: int = 4):
    """
    Deploy a complete workflow to a ComfyUI installation.
    
    Args:
        workflow_path: Path to the workflow JSON file
        comfyui_path: Path to ComfyUI installation
        dry_run: If True, only analyze without making changes
        parallel: Number of parallel downloads (default: 4, set to 1 for sequential)
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
    
    # Parse workflow
    print("=" * 70)
    print("PARSING WORKFLOW")
    print("=" * 70)
    
    parser = WorkflowParser()
    deps = parser.parse(workflow_path)
    
    print(f"\n📦 Models required: {len(deps.models)}")
    for m in deps.models:
        target_path = comfyui / m.target_folder / m.filename
        status = "✅" if target_path.exists() else "❌"
        print(f"   {status} {m.filename} ({m.model_type})")
        print(f"      → {m.target_folder}")
    
    print(f"\n🧩 Custom nodes required: {len([n for n in deps.custom_nodes if n.cnr_id != 'comfy-core'])}")
    for n in deps.custom_nodes:
        if n.cnr_id == "comfy-core":
            continue
        
        # Check if already installed
        node_path = custom_nodes_dir / n.cnr_id
        # Try common variations of the folder name
        variations = [
            n.cnr_id,
            n.cnr_id.lower(),
            n.cnr_id.replace("-", "_"),
            n.cnr_id.replace("_", "-"),
        ]
        
        installed = any((custom_nodes_dir / v).exists() for v in variations)
        status = "✅" if installed else "❌"
        
        print(f"   {status} {n.cnr_id}")
        if n.github_url:
            print(f"      → {n.github_url}")
    
    if dry_run:
        print("\n" + "=" * 70)
        print("⚠️  DRY RUN - No changes will be made")
        print("=" * 70)
        print("\nTo actually deploy, run without --dry-run flag")
        return
    
    # ========================================
    # STEP 1: Install Custom Nodes
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 1: INSTALLING CUSTOM NODES")
    print("=" * 70)
    
    installer = NodeInstaller(comfyui_path=comfyui)
    nodes_installed = 0
    nodes_skipped = 0
    
    for node in deps.custom_nodes:
        if node.cnr_id == "comfy-core":
            continue
        
        print(f"\n📥 {node.cnr_id}")
        
        try:
            path = installer.install(
                cnr_id=node.cnr_id,
                github_url=node.github_url,
                version=node.version,
            )
            if "already installed" in str(path).lower():
                nodes_skipped += 1
            else:
                nodes_installed += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # ========================================
    # STEP 2: Download Models
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 2: DOWNLOADING MODELS")
    print("=" * 70)
    
    # First, resolve all download URLs
    print("\n🔍 Resolving model download URLs...")
    download_queue = []
    models_skipped = 0
    models_failed = 0
    
    for model in deps.models:
        target_dir = comfyui / model.target_folder
        target_file = target_dir / model.filename
        
        if target_file.exists():
            print(f"   ⏭️  {model.filename} - Already exists")
            models_skipped += 1
            continue
        
        # Find download URL if not already known
        if not model.download_url:
            result = smart_search(model.filename, model_type=model.model_type)
            
            if result:
                model.download_url = result["download_url"]
                model.source = result["source"]
                print(f"   ✅ {model.filename} - Found on {model.source}")
            else:
                print(f"   ❌ {model.filename} - Could not find download URL")
                models_failed += 1
                continue
        else:
            print(f"   ✅ {model.filename} - URL provided")
        
        download_queue.append({
            "url": model.download_url,
            "destination": target_dir,
            "filename": model.filename,
        })
    
    # Download all models (in parallel if enabled)
    models_downloaded = 0
    
    if download_queue:
        if parallel > 1:
            print(f"\n⚡ Starting parallel downloads ({parallel} workers)...")
            parallel_downloader = ParallelDownloader(max_workers=parallel)
            results = parallel_downloader.download_all(download_queue)
            models_downloaded = len(results["successful"])
            models_failed += len(results["failed"])
        else:
            print(f"\n📥 Downloading models sequentially...")
            downloader = ModelDownloader()
            for item in download_queue:
                try:
                    print(f"\n   📥 {item['filename']}")
                    path = downloader.download(
                        url=item["url"],
                        destination=item["destination"],
                        filename=item["filename"],
                    )
                    print(f"   ✅ Downloaded!")
                    models_downloaded += 1
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    models_failed += 1
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("DEPLOYMENT SUMMARY")
    print("=" * 70)
    
    print(f"\n🧩 Custom Nodes:")
    print(f"   • Installed: {nodes_installed}")
    print(f"   • Already present: {nodes_skipped}")
    
    print(f"\n📦 Models:")
    print(f"   • Downloaded: {models_downloaded}")
    print(f"   • Already present: {models_skipped}")
    print(f"   • Failed: {models_failed}")
    
    if models_failed == 0 and nodes_installed >= 0:
        print(f"\n🎉 Deployment complete! You can now load the workflow in ComfyUI.")
    else:
        print(f"\n⚠️  Deployment completed with some issues. Check the errors above.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Deploy ComfyUI workflow with all dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze workflow (dry run)
  python deploy_workflow.py --workflow my_workflow.json --comfyui /path/to/ComfyUI --dry-run
  
  # Full deployment with parallel downloads (default)
  python deploy_workflow.py --workflow my_workflow.json --comfyui /path/to/ComfyUI
  
  # Fast parallel download (8 workers)
  python deploy_workflow.py -w workflow.json -c /path/to/ComfyUI --parallel 8
  
  # Sequential download (1 at a time)
  python deploy_workflow.py -w workflow.json -c /path/to/ComfyUI --parallel 1
        """
    )
    parser.add_argument("--workflow", "-w", required=True, help="Path to workflow JSON file")
    parser.add_argument("--comfyui", "-c", required=True, help="Path to ComfyUI installation")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Analyze only, don't make changes")
    parser.add_argument("--parallel", "-p", type=int, default=4, help="Number of parallel downloads (default: 4, use 1 for sequential)")
    
    args = parser.parse_args()
    
    print(f"Workflow: {args.workflow}")
    print(f"ComfyUI:  {args.comfyui}")
    print(f"Dry run:  {args.dry_run}")
    print(f"Parallel: {args.parallel} workers")
    print()
    
    deploy_workflow(args.workflow, args.comfyui, dry_run=args.dry_run, parallel=args.parallel)
