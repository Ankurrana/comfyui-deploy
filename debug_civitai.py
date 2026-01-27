"""Debug CivitAI search for Juggernaut model."""
from comfyui_deploy.smart_search import CivitAISearch
import os
import requests
from dotenv import load_dotenv
load_dotenv()

civitai = CivitAISearch(api_key=os.environ.get('CIVITAI_API_KEY'))

# Test the search term conversion
filename = 'Juggernaut_X_RunDiffusion.safetensors'
search_term = civitai._filename_to_search_term(filename)
print(f'Filename: "{filename}"')
print(f'Search term: "{search_term}"')
print()

# Search with our code
print("=== Search via our CivitAISearch class ===")
results = civitai.search(filename, model_type='checkpoint', limit=10)
print(f'Results found: {len(results)}')
for r in results:
    print(f'  - {r["filename"]} (model: {r["name"]})')

print()

# Try direct API call with different search terms
print("=== Direct API test with 'Juggernaut X RunDiffusion' ===")
response = requests.get(
    "https://civitai.com/api/v1/models",
    params={"query": "Juggernaut X RunDiffusion", "types": "Checkpoint", "limit": 5},
    timeout=30
)
data = response.json()
print(f'API returned {len(data.get("items", []))} models')
for m in data.get("items", [])[:5]:
    print(f'  Model: {m["name"]} (ID: {m["id"]})')
    for v in m.get("modelVersions", [])[:1]:
        for f in v.get("files", [])[:2]:
            print(f'    - File: {f["name"]}')

print()

# Try just "Juggernaut"
print("=== Direct API test with just 'Juggernaut' ===")
response = requests.get(
    "https://civitai.com/api/v1/models",
    params={"query": "Juggernaut", "types": "Checkpoint", "limit": 5},
    timeout=30
)
data = response.json()
print(f'API returned {len(data.get("items", []))} models')
for m in data.get("items", [])[:5]:
    print(f'  Model: {m["name"]} (ID: {m["id"]})')
