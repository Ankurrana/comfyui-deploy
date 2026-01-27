"""
Generate a summary table of all workflows and their model status.
"""
import os
import sys
import requests
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from comfyui_deploy.workflow_parser import WorkflowParser
from comfyui_deploy.smart_search import smart_search


def check_url(url: str, timeout: int = 15) -> tuple[bool, str]:
    """Quick URL check."""
    headers = {}
    
    if "civitai.com" in url:
        api_key = os.environ.get("CIVITAI_API_KEY")
        if api_key:
            url = f"{url}?token={api_key}" if "?" not in url else f"{url}&token={api_key}"
    
    if "huggingface.co" in url:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
    
    try:
        response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if response.status_code == 405:
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.close()
        
        if response.status_code == 200:
            return True, "OK"
        return False, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:30]


def analyze_workflow(workflow_path: Path) -> dict:
    """Analyze a single workflow."""
    result = {
        "name": workflow_path.stem,
        "total_models": 0,
        "found": 0,
        "valid": 0,
        "missing": [],
        "failed": [],
    }
    
    try:
        parser = WorkflowParser()
        deps = parser.parse(str(workflow_path))
    except Exception as e:
        result["error"] = str(e)[:50]
        return result
    
    result["total_models"] = len(deps.models)
    
    for model in deps.models:
        url = model.download_url
        source = "workflow"
        
        if not url:
            # Search for it
            search_result = smart_search(model.filename, model.model_type)
            if search_result:
                url = search_result.get("download_url")
                source = search_result.get("source", "search")
        
        if url:
            result["found"] += 1
            is_valid, status = check_url(url)
            if is_valid:
                result["valid"] += 1
            else:
                result["failed"].append({
                    "name": model.filename,
                    "reason": status,
                    "source": source,
                })
        else:
            result["missing"].append({
                "name": model.filename,
                "type": model.model_type,
                "reason": "No URL found in any source",
            })
    
    return result


def main():
    directory = Path("video-workflows")
    workflows = sorted(directory.glob("*.json"))
    
    print("\n# Workflow Model Analysis Report\n")
    civitai_status = "Set" if os.environ.get('CIVITAI_API_KEY') else "Not Set"
    hf_status = "Set" if os.environ.get('HF_TOKEN') else "Not Set"
    print(f"**API Keys:** CIVITAI={civitai_status} | HF={hf_status}\n")
    
    # Table header
    print("| # | Workflow | Models | Found | Valid | Missing | Failed |")
    print("|---|----------|--------|-------|-------|---------|--------|")
    
    all_results = []
    
    for i, wf in enumerate(workflows, 1):
        print(f"Analyzing {wf.name}...", end="\r", file=sys.stderr)
        result = analyze_workflow(wf)
        all_results.append(result)
        
        name = result["name"][:35] + "..." if len(result["name"]) > 35 else result["name"]
        missing = len(result["missing"])
        failed = len(result["failed"])
        
        status_missing = f"MISSING: {missing}" if missing > 0 else "0"
        status_failed = f"FAILED: {failed}" if failed > 0 else "0"
        
        print(f"| {i} | {name} | {result['total_models']} | {result['found']} | {result['valid']} | {status_missing} | {status_failed} |")
    
    # Details section
    print("\n## Missing Models\n")
    any_missing = False
    for r in all_results:
        if r["missing"]:
            any_missing = True
            print(f"### {r['name']}\n")
            for m in r["missing"]:
                print(f"- **{m['name']}** ({m['type']}): {m['reason']}")
            print()
    
    if not any_missing:
        print("None! All models have URLs.\n")
    
    print("## Failed URLs\n")
    any_failed = False
    for r in all_results:
        if r["failed"]:
            any_failed = True
            print(f"### {r['name']}\n")
            for m in r["failed"]:
                print(f"- **{m['name']}**: {m['reason']} (source: {m['source']})")
            print()
    
    if not any_failed:
        print("None! All URLs are valid.\n")
    
    # Summary
    total_models = sum(r["total_models"] for r in all_results)
    total_found = sum(r["found"] for r in all_results)
    total_valid = sum(r["valid"] for r in all_results)
    total_missing = sum(len(r["missing"]) for r in all_results)
    total_failed = sum(len(r["failed"]) for r in all_results)
    
    print("## Summary\n")
    print(f"- **Total Workflows:** {len(workflows)}")
    print(f"- **Total Models:** {total_models}")
    print(f"- **URLs Found:** {total_found} ({total_found/total_models*100:.1f}%)" if total_models else "- **URLs Found:** 0")
    print(f"- **URLs Valid:** {total_valid} ({total_valid/total_models*100:.1f}%)" if total_models else "- **URLs Valid:** 0")
    print(f"- **Missing:** {total_missing}")
    print(f"- **Failed:** {total_failed}")


if __name__ == "__main__":
    main()
