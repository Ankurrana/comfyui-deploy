"""Debug CivitAI search terms."""
from comfyui_deploy.smart_search import CivitAISearch
import os
from dotenv import load_dotenv
load_dotenv()

civitai = CivitAISearch(api_key=os.environ.get('CIVITAI_API_KEY'))

filename = 'Juggernaut_X_RunDiffusion.safetensors'

# Check what search terms we generate
terms = civitai._filename_to_search_terms(filename)
print(f'Filename: "{filename}"')
print(f'Search terms generated: {terms}')
print()

# Try each term individually
import requests
for term in terms:
    print(f'Trying "{term}"...')
    response = requests.get(
        "https://civitai.com/api/v1/models",
        params={"query": term, "types": "Checkpoint", "limit": 3},
        timeout=30
    )
    data = response.json()
    items = data.get("items", [])
    print(f'  Results: {len(items)}')
    for m in items[:3]:
        print(f'    - {m["name"]} (ID: {m["id"]})')
    print()

# Now try "juggernautXL" which we know works
print('Trying "juggernautXL" directly...')
response = requests.get(
    "https://civitai.com/api/v1/models",
    params={"query": "juggernautXL", "types": "Checkpoint", "limit": 5},
    timeout=30
)
data = response.json()
items = data.get("items", [])
print(f'Results: {len(items)}')
for m in items:
    print(f'  - {m["name"]} (ID: {m["id"]})')
