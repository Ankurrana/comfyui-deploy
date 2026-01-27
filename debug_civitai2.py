"""Debug CivitAI search further."""
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# The model you mentioned: https://civitai.com/models/133005?modelVersionId=456194
# Let's fetch it directly by ID
model_id = 133005
version_id = 456194

print(f"=== Fetching model {model_id} directly ===")
response = requests.get(
    f"https://civitai.com/api/v1/models/{model_id}",
    timeout=30
)

if response.status_code == 200:
    data = response.json()
    print(f"Model name: {data['name']}")
    print(f"Type: {data['type']}")
    print(f"Tags: {data.get('tags', [])}")
    print()
    
    # Find the specific version
    for version in data.get("modelVersions", []):
        print(f"  Version: {version['name']} (ID: {version['id']})")
        for f in version.get("files", []):
            print(f"    - {f['name']}")
            if version['id'] == version_id:
                print(f"      ^ THIS IS THE VERSION YOU WANT")
                print(f"      Download URL: {f.get('downloadUrl', 'N/A')}")
else:
    print(f"Error: {response.status_code}")

print()

# Now let's see what search terms would find this
print("=== Testing various search terms ===")
search_terms = [
    "Juggernaut XL",
    "juggernautXL",
    "Juggernaut",
    "SDXL Juggernaut",
    "Juggernaut_X",
]

for term in search_terms:
    response = requests.get(
        "https://civitai.com/api/v1/models",
        params={"query": term, "types": "Checkpoint", "limit": 3},
        timeout=30
    )
    data = response.json()
    items = data.get("items", [])
    print(f'"{term}": {len(items)} results')
    if items:
        for m in items[:2]:
            print(f"  - {m['name']} (ID: {m['id']})")
