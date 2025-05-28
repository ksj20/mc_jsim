import json, math, random, hashlib, difflib
from typing import Any, List, Tuple, Iterator
import pandas as pd

from pprint import pprint as pp

# [('identical',
#   {'items': [3, 5, 8],
#    'meta': {'tags': ['json', 'similarity'], 'views': 1200},
#    'user': {'id': 1, 'name': 'Lee', 'premium': False}},
#   {'items': [3, 5, 8],
#    'meta': {'tags': ['json', 'similarity'], 'views': 1200},
#    'user': {'id': 1, 'name': 'Lee', 'premium': False}}),
#  ('numeric field modified',
#   {'items': [3, 5, 8],
#    'meta': {'tags': ['json', 'similarity'], 'views': 1200},
#    'user': {'id': 1, 'name': 'Lee', 'premium': False}},
#   {'items': [3, 5, 8],
#    'meta': {'tags': ['json', 'similarity'], 'views': 1800},
#    'user': {'id': 1, 'name': 'Lee', 'premium': False}}),
#  ('new field added',
#   {'items': [3, 5, 8],
#    'meta': {'tags': ['json', 'similarity'], 'views': 1200},
#    'user': {'id': 1, 'name': 'Lee', 'premium': False}},
#   {'items': [3, 5, 8],
#    'meta': {'tags': ['json', 'similarity'], 'views': 1200},
#    'user': {'email': 'lee@example.com',
#             'id': 1,
#             'name': 'Lee',
#             'premium': False}}),
#  ('array order changed',
#   {'items': [3, 5, 8],
#    'meta': {'tags': ['json', 'similarity'], 'views': 1200},
#    'user': {'id': 1, 'name': 'Lee', 'premium': False}},
#   {'items': [8, 5, 3],
#    'meta': {'tags': ['json', 'similarity'], 'views': 1200},
#    'user': {'id': 1, 'name': 'Lee', 'premium': False}}),
#  ('different schema',
#   {'items': [3, 5, 8],
#    'meta': {'tags': ['json', 'similarity'], 'views': 1200},
#    'user': {'id': 1, 'name': 'Lee', 'premium': False}},
#   {'basket': [11, 22],
#    'profile': {'nickname': 'Lee', 'uid': 'A99'},
#    'status': 'active'})]

# ---------- lightweight MC-JSim implementation (no external deps) ----------
_IDX = "<idx>"
_PRIM_BUCKETS = 20
_STRING_HASH_BITS = 16

def _numeric_bucket(x: float, b: int = _PRIM_BUCKETS) -> str:
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    mag = int(min(b - 1, max(0, math.floor(math.log10(abs(x))))))
    return f"{sign}1e{mag:+d}"

def _hash_string(s: str, bits: int = _STRING_HASH_BITS) -> str:
    digest = hashlib.md5(s.encode("utf8")).digest()
    val = int.from_bytes(digest, "big") >> (128 - bits)
    return f"str_{val:0{bits // 4}x}"

def _encode_content(v: Any) -> str:
    if isinstance(v, dict):
        return "obj"
    if isinstance(v, list):
        return "arr"
    if isinstance(v, bool):
        return str(v).lower()
    if v is None:
        return "null"
    # if isinstance(v, (int, float)):
    #     return _numeric_bucket(float(v))
    if isinstance(v, (int, float)):
        return f"num_{v:.2f}"
    if isinstance(v, str):
        return _hash_string(v)
    return "unk"

def _iter_path_tokens(node: Any, prefix: Tuple[str, ...] = ()) -> Iterator[Tuple[str, str]]:
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _iter_path_tokens(v, prefix + (k,))
    elif isinstance(node, list):
        # for v in node:
        #     yield from _iter_path_tokens(v, prefix + (_IDX,))
        for (i, v) in enumerate(node):
            yield from _iter_path_tokens(v, prefix + (str(i),))
    else:
        yield "/".join(prefix), _encode_content(node)

def fingerprint(doc: Any) -> List[str]:
    return ["§".join(t) for t in _iter_path_tokens(doc)]

# simple deterministic MinHash using md5 + different salts
def minhash(tokens: List[str], num_perm: int = 128) -> List[int]:
    mins = [2**128-1] * num_perm
    for token in tokens:
        for i in range(num_perm):
            seed = f"salt{i}"
            hval = int(hashlib.md5((seed + token).encode("utf8")).hexdigest(), 16)
            if hval < mins[i]:
                mins[i] = hval
    return mins

def jaccard_minhash(mins1: List[int], mins2: List[int]) -> float:
    equal = sum(1 for a, b in zip(mins1, mins2) if a == b)
    return equal / len(mins1)

def _lev_distance_ratio(a: str, b: str) -> float:
    sm = difflib.SequenceMatcher(None, a, b)
    return 1 - sm.ratio()

def _primitive_divergence(x: Any, y: Any) -> float:
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(x - y) / (abs(x) + abs(y) + 1e-12)
    if isinstance(x, str) and isinstance(y, str):
        return _lev_distance_ratio(x, y)
    return 0.0 if x == y else 1.0

def content_divergence(j1: Any, j2: Any) -> float:
    def flatmap(j):
        return {p: v for p, v in _iter_path_tokens(j)}
    m1, m2 = flatmap(j1), flatmap(j2)
    shared = m1.keys() & m2.keys()
    if not shared:
        return 1.0
    return sum(_primitive_divergence(m1[p], m2[p]) for p in shared) / len(shared)

def mcjsim(j1: Any, j2: Any, num_perm: int = 128, alpha: float = 0.7) -> float:
    fp1, fp2 = fingerprint(j1), fingerprint(j2)
    mh1, mh2 = minhash(fp1, num_perm), minhash(fp2, num_perm)
    jaccard_hat = jaccard_minhash(mh1, mh2)
    div = content_divergence(j1, j2)
    return alpha * jaccard_hat + (1 - alpha) * (1 - div)

# ---------- experiments ----------
baseline = {
    "user": {"id": 1, "name": "Lee", "premium": False},
    "items": [3, 5, 8],
    "meta": {"views": 1200, "tags": ["json", "similarity"]}
}

experiments = []

# 1. identical
experiments.append(("identical", baseline, baseline))

# 2. numeric change
doc2 = json.loads(json.dumps(baseline))
doc2["meta"]["views"] = 1800
experiments.append(("numeric field modified", baseline, doc2))

# 3. added field
doc3 = json.loads(json.dumps(baseline))
doc3["user"]["email"] = "lee@example.com"
experiments.append(("new field added", baseline, doc3))

# 4. array reordering
doc4 = json.loads(json.dumps(baseline))
doc4["items"] = [8, 5, 3]
experiments.append(("array order changed", baseline, doc4))

# 5. structural divergence
doc5 = {
    "profile": {"uid": "A99", "nickname": "Lee"},
    "basket": [11, 22],
    "status": "active"
}
experiments.append(("different schema", baseline, doc5))

pp(experiments)

results = []
for label, a, b in experiments:
    sim = mcjsim(a, b)
    results.append({"experiment": label, "similarity": round(sim, 3)})

df = pd.DataFrame(results)
print("MC-JSim experiments", df)


