"""Test CivitAI API without API key"""
import requests

print("Testing CivitAI API (NO API KEY):")
print("=" * 60)

# Search for models - this works without API key
response = requests.get(
    "https://civitai.com/api/v1/models",
    params={"query": "SDXL", "limit": 3},
    timeout=30
)

if response.ok:
    data = response.json()
    models = data.get("items", [])
    print(f"Found {len(models)} models:\n")
    
    for model in models[:3]:
        print(f"  {model.get('name')}")
        print(f"    Type: {model.get('type')}")
        print(f"    Downloads: {model.get('stats', {}).get('downloadCount', 0)}")
        
        # Get first file from first version
        versions = model.get("modelVersions", [])
        if versions:
            files = versions[0].get("files", [])
            if files:
                print(f"    File: {files[0].get('name')}")
                print(f"    Download URL: {files[0].get('downloadUrl', 'N/A')[:60]}...")
        print()
else:
    print(f"Error: {response.status_code}")
    print(response.text[:500])
