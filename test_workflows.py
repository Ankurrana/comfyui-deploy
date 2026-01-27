"""
Test script to validate workflow model URLs without downloading.
Checks if all model download links are accessible.
"""
import os
import sys
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from comfyui_deploy.workflow_parser import WorkflowParser
from comfyui_deploy.smart_search import smart_search


def check_url(url: str, timeout: int = 10) -> tuple[str, bool, str]:
    """
    Check if a URL is accessible using HEAD request.
    Returns (url, is_valid, message)
    """
    headers = {}
    
    # Add API keys if available
    if "civitai.com" in url:
        api_key = os.environ.get("CIVITAI_API_KEY")
        if api_key:
            if "?" in url:
                url = f"{url}&token={api_key}"
            else:
                url = f"{url}?token={api_key}"
    
    if "huggingface.co" in url:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
    
    try:
        # Use HEAD request first (faster, no download)
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        
        # Some servers don't support HEAD, try GET with stream
        if response.status_code == 405:
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.close()
        
        if response.status_code == 200:
            # Try to get file size from headers
            size = response.headers.get("Content-Length", "unknown")
            if size != "unknown":
                size_mb = int(size) / (1024 * 1024)
                return (url, True, f"✅ OK ({size_mb:.1f} MB)")
            return (url, True, "✅ OK")
        else:
            return (url, False, f"❌ HTTP {response.status_code}")
    except requests.exceptions.Timeout:
        return (url, False, "❌ Timeout")
    except requests.exceptions.ConnectionError:
        return (url, False, "❌ Connection error")
    except Exception as e:
        return (url, False, f"❌ {str(e)[:50]}")


def test_workflow(workflow_path: Path, verbose: bool = False) -> dict:
    """
    Test a single workflow file.
    Returns results dict with model info and URL status.
    """
    results = {
        "workflow": str(workflow_path),
        "models": [],
        "total": 0,
        "found": 0,
        "valid": 0,
        "failed": 0,
        "not_found": 0,
    }
    
    try:
        parser = WorkflowParser()
        deps = parser.parse(str(workflow_path))
    except Exception as e:
        results["error"] = str(e)
        return results
    
    results["total"] = len(deps.models)
    
    for model in deps.models:
        model_result = {
            "filename": model.filename,
            "model_type": model.model_type,
            "url": None,
            "url_valid": False,
            "status": "",
        }
        
        # Check if model already has a URL
        if model.download_url:
            model_result["url"] = model.download_url
            model_result["source"] = "workflow"
        else:
            # Search for the model
            if verbose:
                print(f"   🔍 Searching for: {model.filename}")
            search_result = smart_search(model.filename, model.model_type)
            if search_result:
                model_result["url"] = search_result.get("download_url")
                model_result["source"] = search_result.get("source", "unknown")
        
        if model_result["url"]:
            results["found"] += 1
            # Check if URL is valid
            _, is_valid, status = check_url(model_result["url"])
            model_result["url_valid"] = is_valid
            model_result["status"] = status
            if is_valid:
                results["valid"] += 1
            else:
                results["failed"] += 1
        else:
            model_result["status"] = "❓ No URL found"
            results["not_found"] += 1
        
        results["models"].append(model_result)
    
    return results


def test_directory(directory: str, parallel: int = 4, verbose: bool = False) -> list[dict]:
    """
    Test all workflow files in a directory.
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Directory does not exist: {directory}")
        sys.exit(1)
    
    # Find all JSON files
    workflow_files = list(dir_path.glob("*.json"))
    
    if not workflow_files:
        print(f"❌ No JSON files found in: {directory}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"TESTING {len(workflow_files)} WORKFLOWS")
    print(f"{'='*70}")
    
    # Check for API keys
    civitai_key = os.environ.get("CIVITAI_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    print(f"\n🔑 API Keys:")
    print(f"   CIVITAI_API_KEY: {'✅ Set' if civitai_key else '❌ Not set'}")
    print(f"   HF_TOKEN: {'✅ Set' if hf_token else '❌ Not set'}")
    
    all_results = []
    
    for i, workflow_file in enumerate(workflow_files, 1):
        print(f"\n{'─'*70}")
        print(f"[{i}/{len(workflow_files)}] {workflow_file.name}")
        print(f"{'─'*70}")
        
        results = test_workflow(workflow_file, verbose)
        all_results.append(results)
        
        if "error" in results:
            print(f"   ❌ Error parsing: {results['error']}")
            continue
        
        if results["total"] == 0:
            print(f"   📦 No models required")
            continue
        
        print(f"\n   📦 Models: {results['total']}")
        
        for model in results["models"]:
            print(f"\n   • {model['filename']} ({model['model_type']})")
            if model["url"]:
                source = model.get("source", "")
                print(f"     URL: {model['url'][:80]}...")
                print(f"     Source: {source}")
                print(f"     Status: {model['status']}")
            else:
                print(f"     {model['status']}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    total_models = sum(r["total"] for r in all_results)
    total_found = sum(r["found"] for r in all_results)
    total_valid = sum(r["valid"] for r in all_results)
    total_failed = sum(r["failed"] for r in all_results)
    total_not_found = sum(r["not_found"] for r in all_results)
    
    print(f"\n📊 Results across {len(workflow_files)} workflows:")
    print(f"   Total models:     {total_models}")
    print(f"   URLs found:       {total_found}")
    print(f"   URLs valid:       {total_valid} ✅")
    print(f"   URLs failed:      {total_failed} ❌")
    print(f"   URLs not found:   {total_not_found} ❓")
    
    if total_failed > 0:
        print(f"\n⚠️  Failed URLs:")
        for result in all_results:
            for model in result["models"]:
                if model["url"] and not model["url_valid"]:
                    print(f"   • {model['filename']}: {model['status']}")
    
    if total_not_found > 0:
        print(f"\n⚠️  Models without URLs:")
        for result in all_results:
            for model in result["models"]:
                if not model["url"]:
                    print(f"   • {model['filename']} ({result['workflow']})")
    
    success_rate = (total_valid / total_models * 100) if total_models > 0 else 100
    print(f"\n🎯 Success rate: {success_rate:.1f}%")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test workflow model URLs without downloading"
    )
    parser.add_argument(
        "directory",
        help="Directory containing workflow JSON files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output during search"
    )
    parser.add_argument(
        "-p", "--parallel",
        type=int,
        default=4,
        help="Number of parallel URL checks (default: 4)"
    )
    
    args = parser.parse_args()
    
    results = test_directory(args.directory, parallel=args.parallel, verbose=args.verbose)
    
    # Exit with error code if any failures
    total_failed = sum(r["failed"] for r in results)
    total_not_found = sum(r["not_found"] for r in results)
    
    if total_failed > 0 or total_not_found > 0:
        sys.exit(1)
    sys.exit(0)
