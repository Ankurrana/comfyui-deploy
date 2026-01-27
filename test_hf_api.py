"""
Test script for HuggingFace file listing API
"""
import requests

# Test: List files in Kijai's LongCat repo directly
print('Listing files in Kijai/LongCat-Video_comfy repo:')
print('=' * 60)

response = requests.get(
    'https://huggingface.co/api/models/Kijai/LongCat-Video_comfy/tree/main',
    timeout=30
)

if response.ok:
    files = response.json()
    safetensor_files = [f for f in files if f.get('path', '').endswith('.safetensors')]
    print(f'Found {len(safetensor_files)} safetensor files:')
    for f in safetensor_files:
        print(f"  - {f['path']}")
else:
    print(f'Error: {response.status_code}')
