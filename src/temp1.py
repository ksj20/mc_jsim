import json, math, random, hashlib, difflib
from typing import Any, List, Tuple, Iterator
import pandas as pd

from pprint import pprint as pp

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
    "user": {
        "id": "USR001",
        "username": "johndoe",
        "email": "john.doe@example.com",
        "is_active": True,
        "profile": {
            "first_name": "John",
            "last_name": "Doe",
            "age": 30,
            "address": {
                "street": "123 Main St",
                "city": "Anytown",
                "zip_code": "12345",
                "country": "USA"
            },
            "interests": ["reading", "hiking", "cooking"]
        },
        "preferences": {
            "newsletter": True,
            "notifications": {
                "email": True,
                "sms": False
            }
        }
    },
    "products": [
        {
            "product_id": "PROD001",
            "name": "Laptop Pro",
            "category": "Electronics",
            "price": 1200.00,
            "specifications": {
                "cpu": "Intel i7",
                "ram_gb": 16,
                "storage_gb": 512,
                "os": "Windows"
            },
            "reviews": [
                {"user_id": "USR002", "rating": 5, "comment": "Excellent product!"},
                {"user_id": "USR003", "rating": 4, "comment": "Good value for money."}
            ]
        },
        {
            "product_id": "PROD002",
            "name": "Mechanical Keyboard",
            "category": "Accessories",
            "price": 75.50,
            "specifications": {
                "layout": "US",
                "backlit": True,
                "switches": "Blue"
            },
            "reviews": []
        }
    ],
    "order_history": [
        {"order_id": "ORD001", "date": "2023-01-15", "total": 1275.50},
        {"order_id": "ORD002", "date": "2023-03-20", "total": 50.00}
    ],
    "settings": {
        "theme": "dark",
        "language": "en-US",
        "privacy_level": "high"
    }
}

experiments = []

# 1. Identical
experiments.append(("identical", baseline, baseline))

# 2. Numeric field modified (price of Laptop Pro)
doc2 = json.loads(json.dumps(baseline))
doc2["products"][0]["price"] = 1150.00
experiments.append(("numeric field modified (price)", baseline, doc2))

# 3. String field modified (user's email)
doc3 = json.loads(json.dumps(baseline))
doc3["user"]["email"] = "john.doe.new@example.com"
experiments.append(("string field modified (email)", baseline, doc3))

# 4. New field added (user's phone number)
doc4 = json.loads(json.dumps(baseline))
doc4["user"]["profile"]["phone"] = "555-123-4567"
experiments.append(("new field added (phone)", baseline, doc4))

# 5. Field removed (user's age)
doc5 = json.loads(json.dumps(baseline))
del doc5["user"]["profile"]["age"]
experiments.append(("field removed (age)", baseline, doc5))

# 6. Array reordering (products)
doc6 = json.loads(json.dumps(baseline))
doc6["products"] = [doc6["products"][1], doc6["products"][0]]
experiments.append(("array order changed (products)", baseline, doc6))

# 7. Item added to array (new interest for user)
doc7 = json.loads(json.dumps(baseline))
doc7["user"]["profile"]["interests"].append("gardening")
experiments.append(("item added to array (interest)", baseline, doc7))

# 8. Nested numeric change (RAM in Laptop Pro)
doc8 = json.loads(json.dumps(baseline))
doc8["products"][0]["specifications"]["ram_gb"] = 32
experiments.append(("nested numeric field modified (RAM)", baseline, doc8))

# 9. Nested string change (city in address)
doc9 = json.loads(json.dumps(baseline))
doc9["user"]["profile"]["address"]["city"] = "Newtown"
experiments.append(("nested string field modified (city)", baseline, doc9)
)
# 10. Boolean field changed (newsletter preference)
doc10 = json.loads(json.dumps(baseline))
doc10["user"]["preferences"]["newsletter"] = False
experiments.append(("boolean field changed (newsletter)", baseline, doc10))

# 11. Array item modified (rating of a product review)
doc11 = json.loads(json.dumps(baseline))
doc11["products"][0]["reviews"][0]["rating"] = 3
experiments.append(("array item modified (review rating)", baseline, doc11))

# 12. Complete structural divergence
doc12 = {
    "report_id": "REP001",
    "generation_date": "2024-05-27",
    "summary": "This is a summary report.",
    "metrics": {
        "total_users": 1500,
        "active_users": 1200
    }
}
experiments.append(("different schema", baseline, doc12))

# 13. Null field added
doc13 = json.loads(json.dumps(baseline))
doc13["user"]["last_login"] = None
experiments.append(("null field added", baseline, doc13))

# 14. Empty list added
doc14 = json.loads(json.dumps(baseline))
doc14["user"]["past_orders"] = []
experiments.append(("empty list added", baseline, doc14))

# 15. Empty object added
doc15 = json.loads(json.dumps(baseline))
doc15["user"]["additional_info"] = {}
experiments.append(("empty object added", baseline, doc15))


pp(experiments)

results = []
for label, a, b in experiments:
    sim = mcjsim(a, b)
    results.append({"experiment": label, "similarity": round(sim, 3)})

df = pd.DataFrame(results)
print("MC-JSim experiments", df)