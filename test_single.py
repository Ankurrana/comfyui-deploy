"""Quick single workflow test."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from comfyui_deploy.workflow_parser import WorkflowParser
from comfyui_deploy.smart_search import smart_search
import requests

def check_url(url, timeout=15):
    headers = {}
    if 'civitai.com' in url:
        key = os.environ.get('CIVITAI_API_KEY')
        if key:
            url = f'{url}?token={key}' if '?' not in url else f'{url}&token={key}'
    if 'huggingface.co' in url:
        token = os.environ.get('HF_TOKEN')
        if token:
            headers['Authorization'] = f'Bearer {token}'
    try:
        r = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code == 405:
            r = requests.get(url, headers=headers, timeout=timeout, stream=True)
            r.close()
        return r.status_code == 200, f'HTTP {r.status_code}'
    except Exception as e:
        return False, str(e)[:30]

wf = 'video-workflows/EP27 SDXL Image to Digital Painting with Lora and Control Net and UPSCALER.json'
parser = WorkflowParser()
deps = parser.parse(wf)

print(f'\nWorkflow: EP27 SDXL Image to Digital Painting')
print(f'Models found: {len(deps.models)}\n')
print('='*80)

for m in deps.models:
    print(f'\nModel: {m.filename}')
    print(f'  Type: {m.model_type}')
    print(f'  Target: {m.target_folder}')
    
    url = m.download_url
    source = 'embedded in workflow'
    
    if not url:
        print('  URL: Not in workflow, searching...')
        result = smart_search(m.filename, m.model_type)
        if result:
            url = result.get('download_url')
            source = result.get('source', 'search')
        else:
            print('  STATUS: NOT FOUND - No URL in any source')
            continue
    
    print(f'  URL: {url}')
    print(f'  Source: {source}')
    
    valid, status = check_url(url)
    status_text = "VALID" if valid else "FAILED"
    print(f'  Status: {status} - {status_text}')
