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
    # Handle type mismatch for primitives
    if type(x) != type(y):
        return 1.0 # Max divergence if types are fundamentally different

    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        max_val = max(abs(x), abs(y))
        if max_val == 0:
            return 0.0
        return abs(x - y) / max_val
    if isinstance(x, str) and isinstance(y, str):
        return _lev_distance_ratio(x, y)
    
    # For bool and None, exact match is 0, mismatch is 1
    return 0.0 if x == y else 1.0

def _iter_path_value_pairs(node: Any, prefix: Tuple[str, ...] = ()) -> Iterator[Tuple[str, Any]]:
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _iter_path_value_pairs(v, prefix + (k,))
    elif isinstance(node, list):
        # We handle arrays by iterating through indexed elements.
        # This makes array reordering result in path changes for content_divergence.
        for (i, v) in enumerate(node):
            yield from _iter_path_value_pairs(v, prefix + (str(i),))
    else:
        yield "/".join(prefix), node

def content_divergence_with_values(j1: Any, j2: Any) -> float:
    m1 = {p: v for p, v in _iter_path_value_pairs(j1)}
    m2 = {p: v for p, v in _iter_path_value_pairs(j2)}
    
    shared_paths = m1.keys() & m2.keys()
    
    if not shared_paths:
        return 1.0 # No common paths, maximum divergence from content perspective

    total_divergence = 0.0
    for p in shared_paths:
        total_divergence += _primitive_divergence(m1[p], m2[p])
    
    return total_divergence / len(shared_paths)

def custom_structural_divergence(j1: Any, j2: Any) -> float:
    """
    Calculates structural divergence based on common paths and type consistency.
    This avoids explicit JSON Patch operations.
    
    - Penalizes paths present in one document but not the other.
    - Penalizes type mismatches at shared paths (dict vs. list vs. primitive).
    - Array order changes will naturally lead to different paths here due to indexing.
    """
    
    paths1_with_types = {p: type(v) for p, v in _iter_path_value_pairs(j1)}
    paths2_with_types = {p: type(v) for p, v in _iter_path_value_pairs(j2)}

    all_paths = paths1_with_types.keys() | paths2_with_types.keys()
    
    if not all_paths: # Both empty documents
        return 0.0

    divergence_score = 0.0
    
    for path in all_paths:
        val1_exists = path in paths1_with_types
        val2_exists = path in paths2_with_types

        if val1_exists and val2_exists:
            # Path exists in both, check type consistency
            if paths1_with_types[path] != paths2_with_types[path]:
                # Significant penalty for type mismatch at the same path
                divergence_score += 1.0
            # Else, types match, 0 divergence from structural perspective here
        elif val1_exists != val2_exists:
            # Path exists in one but not the other (added/removed)
            divergence_score += 1.0 # Penalty for presence/absence
            
    # Normalize by the total number of unique paths
    # This gives a value between 0 (identical structure) and 1 (completely different paths)
    return divergence_score / len(all_paths)


def mcjsim_v2(j1: Any, j2: Any, num_perm: int = 128, alpha: float = 0.5, beta: float = 0.3) -> float:
    """
    MC-JSim algorithm with:
    1. MinHash Jaccard Similarity (path-content fingerprints)
    2. Content Similarity (primitive value divergence at shared paths)
    3. Custom Structural Similarity (path presence/absence and type consistency)
    
    Args:
        j1, j2: The JSON documents to compare.
        num_perm: Number of permutations for MinHash.
        alpha: Weight for Jaccard similarity (structural fingerprinting).
        beta: Weight for Content Similarity (primitive value divergence).
              The remaining weight (1 - alpha - beta) will be for the custom structural similarity.
    """
    
    total_weight = alpha + beta
    if total_weight > 1.0:
        raise ValueError("Alpha and Beta weights combined cannot exceed 1.0")

    # 1. MinHash Jaccard Similarity (based on path-content fingerprints)
    # This gives a quick overall sense of shared elements, including encoded content.
    fp1, fp2 = fingerprint(j1), fingerprint(j2)
    mh1, mh2 = minhash(fp1, num_perm), minhash(fp2, num_perm)
    jaccard_hat = jaccard_minhash(mh1, mh2)

    # 2. Content Similarity (primitive value similarity at shared paths)
    # This focuses on how similar the *values* are for paths that exist in both.
    content_div = content_divergence_with_values(j1, j2)
    content_sim = 1.0 - content_div # Convert divergence to similarity

    # 3. Custom Structural Similarity (path presence/absence and type consistency)
    # This explicitly addresses structural differences like added/removed paths
    # and type changes at common paths, without relying on a full diff algorithm.
    custom_struct_div = custom_structural_divergence(j1, j2)
    custom_struct_sim = 1.0 - custom_struct_div # Convert divergence to similarity

    # Combine the similarities with weighted averages
    gamma = 1.0 - alpha - beta
    
    combined_similarity = (alpha * jaccard_hat) + \
                          (beta * content_sim) + \
                          (gamma * custom_struct_sim)
                          
    return combined_similarity


# ---------- experiments (using the larger JSON data) ----------
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
experiments.append(("nested string field modified (city)", baseline, doc9))

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

# 16. Complex change: multiple fields modified, added, and removed
doc16 = json.loads(json.dumps(baseline))
doc16["user"]["profile"]["age"] = 31 # Modified
doc16["user"]["profile"]["address"]["country"] = "Canada" # Modified
doc16["products"].pop(1) # Removed one product
doc16["order_history"][0]["total"] = 1300.00 # Modified
doc16["settings"]["new_setting"] = "value" # Added new setting
experiments.append(("complex mixed changes", baseline, doc16))

# 17. Change type of a node
doc17 = json.loads(json.dumps(baseline))
doc17["user"]["is_active"] = "yes" # Boolean to string
experiments.append(("field type changed (is_active)", baseline, doc17))

# 18. Change object to array
doc18 = json.loads(json.dumps(baseline))
doc18["user"]["preferences"]["notifications"] = ["email_on", "sms_off"] # Object to array
experiments.append(("object to array change", baseline, doc18))

# 19. Array element replaced with different type
doc19 = json.loads(json.dumps(baseline))
doc19["user"]["profile"]["interests"][0] = {"activity": "reading", "level": "high"} # String to object
experiments.append(("array element type changed", baseline, doc19))


# pp(experiments)

results = []
# Using new mcjsim_v2
for label, a, b in experiments:
    # Adjust weights as needed. 0.4 for MinHash, 0.4 for content, 0.2 for custom structural
    sim = mcjsim_v2(a, b, alpha=0.4, beta=0.4)
    results.append({"experiment": label, "similarity": round(sim, 3)})

df = pd.DataFrame(results)
print("MC-JSim experiments (with Custom Structural Component)\n", df)