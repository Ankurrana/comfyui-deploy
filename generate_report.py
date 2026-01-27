"""
Generate a markdown report of workflow model analysis.
"""
import os
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from comfyui_deploy.workflow_parser import WorkflowParser
from comfyui_deploy.smart_search import smart_search
import requests


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
                    "url": url,
                })
        else:
            result["missing"].append({
                "name": model.filename,
                "type": model.model_type,
                "reason": "No URL found in any source",
            })
    
    return result


def generate_report():
    directory = Path("video-workflows")
    workflows = sorted(directory.glob("*.json"))
    
    lines = []
    lines.append("# Workflow Model Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    civitai = "Set" if os.environ.get('CIVITAI_API_KEY') else "Not Set"
    hf = "Set" if os.environ.get('HF_TOKEN') else "Not Set"
    lines.append(f"**API Keys:** CIVITAI_API_KEY={civitai} | HF_TOKEN={hf}")
    lines.append("")
    
    # Table header
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| # | Workflow | Models | Found | Valid | Missing | Failed |")
    lines.append("|---|----------|--------|-------|-------|---------|--------|")
    
    all_results = []
    
    for i, wf in enumerate(workflows, 1):
        print(f"Analyzing [{i}/{len(workflows)}] {wf.name}...")
        result = analyze_workflow(wf)
        all_results.append(result)
        
        name = result["name"][:35] + "..." if len(result["name"]) > 35 else result["name"]
        missing = len(result["missing"])
        failed = len(result["failed"])
        
        lines.append(f"| {i} | {name} | {result['total_models']} | {result['found']} | {result['valid']} | {missing} | {failed} |")
    
    # Missing models section
    lines.append("")
    lines.append("## Missing Models (No URL Found)")
    lines.append("")
    
    any_missing = False
    for r in all_results:
        if r["missing"]:
            any_missing = True
            lines.append(f"### {r['name']}")
            lines.append("")
            lines.append("| Model | Type | Reason |")
            lines.append("|-------|------|--------|")
            for m in r["missing"]:
                lines.append(f"| {m['name']} | {m['type']} | {m['reason']} |")
            lines.append("")
    
    if not any_missing:
        lines.append("*None - all models have URLs!*")
        lines.append("")
    
    # Failed URLs section
    lines.append("## Failed URLs")
    lines.append("")
    
    any_failed = False
    for r in all_results:
        if r["failed"]:
            any_failed = True
            lines.append(f"### {r['name']}")
            lines.append("")
            lines.append("| Model | Source | Error | URL |")
            lines.append("|-------|--------|-------|-----|")
            for m in r["failed"]:
                url_short = m['url'][:50] + "..." if len(m['url']) > 50 else m['url']
                lines.append(f"| {m['name']} | {m['source']} | {m['reason']} | `{url_short}` |")
            lines.append("")
    
    if not any_failed:
        lines.append("*None - all URLs are valid!*")
        lines.append("")
    
    # Statistics
    total_models = sum(r["total_models"] for r in all_results)
    total_found = sum(r["found"] for r in all_results)
    total_valid = sum(r["valid"] for r in all_results)
    total_missing = sum(len(r["missing"]) for r in all_results)
    total_failed = sum(len(r["failed"]) for r in all_results)
    
    lines.append("## Statistics")
    lines.append("")
    lines.append(f"- **Total Workflows:** {len(workflows)}")
    lines.append(f"- **Total Models:** {total_models}")
    if total_models > 0:
        lines.append(f"- **URLs Found:** {total_found} ({total_found/total_models*100:.1f}%)")
        lines.append(f"- **URLs Valid:** {total_valid} ({total_valid/total_models*100:.1f}%)")
    lines.append(f"- **Missing:** {total_missing}")
    lines.append(f"- **Failed:** {total_failed}")
    lines.append("")
    
    # Notes
    lines.append("## Notes")
    lines.append("")
    lines.append("- **HTTP 403** errors typically mean:")
    lines.append("  - CivitAI: Model requires accepting terms on the website first")
    lines.append("  - HuggingFace: Gated model requiring license acceptance")
    lines.append("- **Missing URLs**: Model not found in ComfyUI Manager, HuggingFace, or CivitAI")
    lines.append("- **Timeout**: Server didn't respond in time (may still work)")
    lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_report()
    
    output_file = Path("workflow-analysis-report.md")
    output_file.write_text(report, encoding="utf-8")
    
    print(f"\nReport saved to: {output_file.absolute()}")
    print(report)
