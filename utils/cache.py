import hashlib
import json
import os

CACHE_DIR = "cache"

os.makedirs(CACHE_DIR,exist_ok=True)

def get_cache_key(data):
    return hashlib.md5(json.dumps(data,sort_keys=True).encode()).hexdigest()

def load_cache(key):
    path = f"{CACHE_DIR}\\{key}.json"
    if os.path.exists(path):
        with open(path,'r') as f:
            return json.load(f)
    return None

def save_cache(key,data):
    path = f"{CACHE_DIR}\\{key}.json"
    with open(path, 'w') as f:
        json.dump(data,f)