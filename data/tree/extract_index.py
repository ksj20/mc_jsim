import json, itertools, sys
from pathlib import Path

def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            yield full
            yield from walk(v, full)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk(item, prefix)

all_keys = set()
# for p in Path("json_dir").glob("*.json"):          # loop over many files
all_keys |= set(walk(json.load(open(sys.argv[1]))))

print("\n".join(sorted(all_keys)))
