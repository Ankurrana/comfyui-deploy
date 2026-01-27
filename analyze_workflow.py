"""
Analyze the LongCat workflow and find download links using smart search.
"""
from comfyui_deploy.workflow_parser import WorkflowParser
from comfyui_deploy.smart_search import smart_search

# Parse the workflow
parser = WorkflowParser()
deps = parser.parse(r"C:\Users\acer\Downloads\LongCat img2video - 5 to 30s Video - No Subgraphs.json")

# Clear any URLs found from embedded docs to test pure smart search
for model in deps.models:
    model.download_url = None
    model.source = None

print("=" * 70)
print("WORKFLOW ANALYSIS: LongCat img2video - 5 to 30s Video")
print("=" * 70)

print(f"\nFound {len(deps.models)} models in workflow:")
for m in deps.models:
    print(f"  • {m.filename} ({m.model_type}) → {m.target_folder}")

print("\n" + "=" * 70)
print("SMART SEARCH RESULTS (No local DB, no embedded docs)")
print("=" * 70)

results = []
for model in deps.models:
    print(f"\n🔍 Searching: {model.filename}")
    print(f"   Type: {model.model_type}")
    print(f"   Target: {model.target_folder}")
    
    result = smart_search(model.filename, model_type=model.model_type)
    
    if result:
        print(f"   ✅ FOUND on {result['source']} ({result['confidence']})")
        print(f"   📥 URL: {result['download_url']}")
        results.append({
            "filename": model.filename,
            "target_folder": model.target_folder,
            "download_url": result["download_url"],
            "source": result["source"],
        })
    else:
        print(f"   ❌ NOT FOUND")
        results.append({
            "filename": model.filename,
            "target_folder": model.target_folder,
            "download_url": None,
            "source": None,
        })

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

found = [r for r in results if r["download_url"]]
not_found = [r for r in results if not r["download_url"]]

print(f"\n✅ Found: {len(found)}/{len(results)} models")
print(f"❌ Not Found: {len(not_found)}/{len(results)} models")

if found:
    print("\n📦 DOWNLOAD COMMANDS:")
    print("-" * 70)
    for r in found:
        print(f"\n# {r['filename']}")
        print(f"# Target: {r['target_folder']}")
        print(f"curl -L -o \"{r['filename']}\" \"{r['download_url']}\"")

if not_found:
    print("\n⚠️  MANUAL DOWNLOAD REQUIRED:")
    print("-" * 70)
    for r in not_found:
        print(f"  • {r['filename']} → {r['target_folder']}")
