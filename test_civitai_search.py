"""Test CivitAI search integration"""
from comfyui_deploy.smart_search import CivitAISearch, smart_search

print("=" * 60)
print("Test 1: Direct CivitAI Search (NO API KEY)")
print("=" * 60)

civitai = CivitAISearch()
results = civitai.search("juggernaut", model_type="checkpoint", limit=3)

print(f"Found {len(results)} results for 'juggernaut':\n")
for r in results:
    print(f"  {r['name']}")
    print(f"    File: {r['filename']}")
    print(f"    Downloads: {r['downloads']:,}")
    print(f"    URL: {r['download_url'][:60]}...")
    print()

print("=" * 60)
print("Test 2: Smart Search with CivitAI fallback")
print("=" * 60)

# Search for something that might be on CivitAI
test_models = [
    "juggernautXL_v9.safetensors",  # Popular CivitAI model
    "realvisxl_v40.safetensors",     # Another popular one
]

for filename in test_models:
    print(f"\nSearching: {filename}")
    result = smart_search(filename, model_type="checkpoint")
    
    if result:
        print(f"  ✓ Found on {result['source']} ({result['confidence']})")
        print(f"    URL: {result['download_url'][:60]}...")
    else:
        print(f"  ✗ Not found")
