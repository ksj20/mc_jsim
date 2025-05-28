import json, math, random, hashlib, difflib
from typing import Any, List, Tuple, Iterator
import pandas as pd
import numpy as np # For distance matrix
from sklearn.cluster import AgglomerativeClustering # For clustering

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
    """Generates a list of unique path-content tokens for a JSON document."""
    return ["§".join(t) for t in _iter_path_tokens(doc)]

# simple deterministic MinHash using md5 + different salts
def minhash(tokens: List[str], num_perm: int = 128) -> List[int]:
    """
    Generates a MinHash signature for a list of tokens.
    Uses num_perm different hash functions (simulated by salts)
    to find the minimum hash value for each permutation.
    """
    mins = [2**128-1] * num_perm # Initialize with a very large number
    for token in tokens:
        for i in range(num_perm):
            seed = f"salt{i}" # Deterministic "salt" for each permutation
            # Compute hash value for the token with the current salt
            hval = int(hashlib.md5((seed + token).encode("utf8")).hexdigest(), 16)
            if hval < mins[i]:
                mins[i] = hval # Update minimum if current hash is smaller
    return mins

def jaccard_minhash(mins1: List[int], mins2: List[int]) -> float:
    """
    Estimates Jaccard similarity from two MinHash signatures.
    The estimate is the proportion of matching minimum hash values.
    """
    if not mins1 or not mins2:
        return 0.0 # Handle empty signatures
    equal = sum(1 for a, b in zip(mins1, mins2) if a == b)
    return equal / len(mins1) if len(mins1) > 0 else 0.0

def jaccard_similarity_exact(set1_elements: List[str], set2_elements: List[str]) -> float:
    """
    Calculates the exact Jaccard similarity between two sets of elements.
    J(A, B) = |A intersect B| / |A union B|
    """
    set1 = set(set1_elements)
    set2 = set(set2_elements)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 1.0 # Both sets are empty, considered identical
    return intersection / union

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
    """Iterates over path-value pairs in a JSON document, yielding actual values."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _iter_path_value_pairs(v, prefix + (k,))
    elif isinstance(node, list):
        for (i, v) in enumerate(node):
            yield from _iter_path_value_pairs(v, prefix + (str(i),))
    else:
        yield "/".join(prefix), node

def content_divergence_with_values(j1: Any, j2: Any) -> float:
    """Calculates content divergence by comparing primitive values at shared paths."""
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
    - Penalizes paths present in one document but not the other.
    - Penalizes type mismatches at shared paths (dict vs. list vs. primitive).
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
        elif val1_exists != val2_exists:
            # Path exists in one but not the other (added/removed)
            divergence_score += 1.0
            
    # Normalize by the total number of unique paths
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
    fp1, fp2 = fingerprint(j1), fingerprint(j2)
    mh1, mh2 = minhash(fp1, num_perm), minhash(fp2, num_perm)
    jaccard_hat = jaccard_minhash(mh1, mh2)

    # 2. Content Similarity (primitive value similarity at shared paths)
    content_div = content_divergence_with_values(j1, j2)
    content_sim = 1.0 - content_div # Convert divergence to similarity

    # 3. Custom Structural Similarity (path presence/absence and type consistency)
    custom_struct_div = custom_structural_divergence(j1, j2)
    custom_struct_sim = 1.0 - custom_struct_div # Convert divergence to similarity

    # Combine the similarities with weighted averages
    gamma = 1.0 - alpha - beta
    
    combined_similarity = (alpha * jaccard_hat) + \
                          (beta * content_sim) + \
                          (gamma * custom_struct_sim)
                          
    return combined_similarity


# ---------- Experiment JSON Data ----------
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

# --- Existing Experiments ---
experiments.append(("identical", baseline, baseline))
doc2 = json.loads(json.dumps(baseline)); doc2["products"][0]["price"] = 1150.00
experiments.append(("numeric field modified (price)", baseline, doc2))
doc3 = json.loads(json.dumps(baseline)); doc3["user"]["email"] = "john.doe.new@example.com"
experiments.append(("string field modified (email)", baseline, doc3))
doc4 = json.loads(json.dumps(baseline)); doc4["user"]["profile"]["phone"] = "555-123-4567"
experiments.append(("new field added (phone)", baseline, doc4))
doc5 = json.loads(json.dumps(baseline)); del doc5["user"]["profile"]["age"]
experiments.append(("field removed (age)", baseline, doc5))
doc6 = json.loads(json.dumps(baseline)); doc6["products"] = [doc6["products"][1], doc6["products"][0]]
experiments.append(("array order changed (products)", baseline, doc6))
doc7 = json.loads(json.dumps(baseline)); doc7["user"]["profile"]["interests"].append("gardening")
experiments.append(("item added to array (interest)", baseline, doc7))
doc8 = json.loads(json.dumps(baseline)); doc8["products"][0]["specifications"]["ram_gb"] = 32
experiments.append(("nested numeric field modified (RAM)", baseline, doc8))
doc9 = json.loads(json.dumps(baseline)); doc9["user"]["profile"]["address"]["city"] = "Newtown"
experiments.append(("nested string field modified (city)", baseline, doc9))
doc10 = json.loads(json.dumps(baseline)); doc10["user"]["preferences"]["newsletter"] = False
experiments.append(("boolean field changed (newsletter)", baseline, doc10))
doc11 = json.loads(json.dumps(baseline)); doc11["products"][0]["reviews"][0]["rating"] = 3
experiments.append(("array item modified (review rating)", baseline, doc11))
doc12 = {"report_id": "REP001", "generation_date": "2024-05-27", "summary": "This is a summary report.", "metrics": {"total_users": 1500, "active_users": 1200}}
experiments.append(("different schema", baseline, doc12))
doc13 = json.loads(json.dumps(baseline)); doc13["user"]["last_login"] = None
experiments.append(("null field added", baseline, doc13))
doc14 = json.loads(json.dumps(baseline)); doc14["user"]["past_orders"] = []
experiments.append(("empty list added", baseline, doc14))
doc15 = json.loads(json.dumps(baseline)); doc15["user"]["additional_info"] = {}
experiments.append(("empty object added", baseline, doc15))
doc16 = json.loads(json.dumps(baseline))
doc16["user"]["profile"]["age"] = 31
doc16["user"]["profile"]["address"]["country"] = "Canada"
doc16["products"].pop(1)
doc16["order_history"][0]["total"] = 1300.00
doc16["settings"]["new_setting"] = "value"
experiments.append(("complex mixed changes", baseline, doc16))
doc17 = json.loads(json.dumps(baseline)); doc17["user"]["is_active"] = "yes"
experiments.append(("field type changed (is_active)", baseline, doc17))
doc18 = json.loads(json.dumps(baseline)); doc18["user"]["preferences"]["notifications"] = ["email_on", "sms_off"]
experiments.append(("object to array change", baseline, doc18))
doc19 = json.loads(json.dumps(baseline)); doc19["user"]["profile"]["interests"][0] = {"activity": "reading", "level": "high"}
experiments.append(("array element type changed", baseline, doc19))

# --- NEW EXPERIMENTS ---

# 20. Change value within an array (not reorder)
doc20 = json.loads(json.dumps(baseline))
doc20["user"]["profile"]["interests"][1] = "painting" # hiking to painting
experiments.append(("array item value changed (interests)", baseline, doc20))

# 21. Change a key name
doc21 = json.loads(json.dumps(baseline))
doc21["user"]["profile"]["full_name"] = doc21["user"]["profile"].pop("first_name") + " " + doc21["user"]["profile"].pop("last_name")
experiments.append(("key name changed (profile name)", baseline, doc21))

# 22. Add/remove elements from a nested array (reviews)
doc22 = json.loads(json.dumps(baseline))
doc22["products"][0]["reviews"].append({"user_id": "USR004", "rating": 5, "comment": "Fast shipping!"}) # Add review
doc22["products"][1]["reviews"].append({"user_id": "USR005", "rating": 3, "comment": "Okay keyboard."}) # Add review to empty
experiments.append(("nested array items added", baseline, doc22))

# 23. Deeply nested modification (OS of Laptop Pro)
doc23 = json.loads(json.dumps(baseline))
doc23["products"][0]["specifications"]["os"] = "Linux"
experiments.append(("deeply nested modification (OS)", baseline, doc23))

# 24. Minor text change in a deep field
doc24 = json.loads(json.dumps(baseline))
doc24["products"][0]["reviews"][0]["comment"] = "Excellent product! Highly recommended."
experiments.append(("minor text change in deep field", baseline, doc24))

# 25. Significant numeric change (total order amount)
doc25 = json.loads(json.dumps(baseline))
doc25["order_history"][0]["total"] = 50.00 # From 1275.50 to 50.00
experiments.append(("significant numeric change", baseline, doc25))

# 26. Multiple small, distributed changes
doc26 = json.loads(json.dumps(baseline))
doc26["user"]["is_active"] = False # bool
doc26["user"]["profile"]["city"] = "Othertown" # string
doc26["products"][0]["price"] = 1201.00 # float
doc26["settings"]["theme"] = "light" # string
experiments.append(("multiple small distributed changes", baseline, doc26))


# pp(experiments)

results = []
for label, a, b in experiments:
    fp_a = fingerprint(a)
    fp_b = fingerprint(b)

    jaccard_exact = jaccard_similarity_exact(fp_a, fp_b)
    
    mh_a = minhash(fp_a)
    mh_b = minhash(fp_b)
    jaccard_minhash_val = jaccard_minhash(mh_a, mh_b)

    sim = mcjsim_v2(a, b, alpha=0.4, beta=0.4) # Adjust weights as needed

    results.append({
        "experiment": label,
        "jaccard_exact": round(jaccard_exact, 3),
        "jaccard_minhash": round(jaccard_minhash_val, 3),
        "mcjsim_v2": round(sim, 3)
    })

df = pd.DataFrame(results)
print("MC-JSim experiments (with Exact Jaccard Comparison)\n", df)

print("\n" + "="*50 + "\n")

# ---------- Clustering Experiment ----------

def perform_json_clustering(
    json_docs: List[Tuple[str, Any]], 
    n_clusters: int = 3, 
    sim_func=mcjsim_v2,
    sim_func_kwargs: dict = None
) -> None:
    """
    Performs clustering on a list of JSON documents based on their similarity.

    Args:
        json_docs: A list of tuples, where each tuple is (document_name, json_object).
        n_clusters: The number of clusters to form.
        sim_func: The similarity function to use (e.g., mcjsim_v2).
        sim_func_kwargs: Keyword arguments to pass to the similarity function.
    """
    if sim_func_kwargs is None:
        sim_func_kwargs = {}

    num_docs = len(json_docs)
    if num_docs < 2:
        print("Not enough documents to cluster.")
        return

    # 1. Calculate pairwise similarity matrix
    similarity_matrix = np.zeros((num_docs, num_docs))
    for i in range(num_docs):
        for j in range(num_docs):
            if i == j:
                similarity_matrix[i, j] = 1.0 # Document is identical to itself
            elif i < j:
                # Calculate similarity only once per pair (matrix is symmetric)
                sim_val = sim_func(json_docs[i][1], json_docs[j][1], **sim_func_kwargs)
                similarity_matrix[i, j] = sim_val
                similarity_matrix[j, i] = sim_val

    # 2. Convert similarity matrix to a distance matrix
    # Distance = 1 - Similarity. Ensure distances are non-negative.
    distance_matrix = 1 - similarity_matrix
    distance_matrix[distance_matrix < 0] = 0 # Clamp negative values if any due to float precision

    print("\nPairwise Similarity Matrix:")
    print(pd.DataFrame(similarity_matrix, 
                       index=[name for name, _ in json_docs],
                       columns=[name for name, _ in json_docs]).round(3))
    print("\nPairwise Distance Matrix:")
    print(pd.DataFrame(distance_matrix, 
                       index=[name for name, _ in json_docs],
                       columns=[name for name, _ in json_docs]).round(3))

    # 3. Perform clustering using AgglomerativeClustering
    # We use affinity='precomputed' because we provide the distance matrix.
    # linkage='average' is a common choice for agglomerative clustering.
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    cluster_labels = agg_clustering.fit_predict(distance_matrix)

    print(f"\nClustering Results (n_clusters={n_clusters}):")
    for i, (doc_name, _) in enumerate(json_docs):
        print(f"Document '{doc_name}' -> Cluster {cluster_labels[i]}")

    # Optional: Print documents per cluster
    clusters_grouped = {i: [] for i in range(n_clusters)}
    for i, (doc_name, _) in enumerate(json_docs):
        clusters_grouped[cluster_labels[i]].append(doc_name)
    
    print("\nDocuments per Cluster:")
    for cluster_id, doc_names in clusters_grouped.items():
        print(f"Cluster {cluster_id}: {', '.join(doc_names)}")


# --- JSON Documents for Clustering ---
# Define a set of documents that should logically cluster
cluster_docs = [
    ("Doc A (Baseline)", baseline),
    ("Doc B (User Mod)", json.loads(json.dumps(baseline))),
    ("Doc C (Prod Mod)", json.loads(json.dumps(baseline))),
    ("Doc D (Minor changes)", json.loads(json.dumps(baseline))),
    ("Doc E (Different Schema 1)", {
        "blog_post": {
            "id": "BP001",
            "title": "Introduction to Data Science",
            "author": {"name": "Alice", "email": "alice@example.com"},
            "tags": ["data science", "ai", "machine learning"],
            "publish_date": "2024-05-01",
            "content_summary": "A beginner's guide to data science concepts."
        },
        "comments": [
            {"comment_id": "C001", "user": "Reader1", "text": "Very informative!"}
        ]
    }),
    ("Doc F (Different Schema 2)", {
        "blog_post": {
            "id": "BP002",
            "title": "Advanced Python Tips",
            "author": {"name": "Bob", "email": "bob@example.com"},
            "tags": ["python", "programming", "best practices"],
            "publish_date": "2024-05-10",
            "content_summary": "Tips and tricks for experienced Python developers."
        },
        "comments": []
    }),
    ("Doc G (User Profile A)", {
        "profile_id": "P001",
        "username": "user_a",
        "email": "user.a@domain.com",
        "settings": {"dark_mode": True, "language": "en"}
    }),
    ("Doc H (User Profile B)", {
        "profile_id": "P002",
        "username": "user_b",
        "email": "user.b@domain.com",
        "settings": {"dark_mode": False, "language": "fr"}
    }),
    ("Doc I (Prod Mod Major)", json.loads(json.dumps(baseline))),
]

# Make changes to clustering docs
cluster_docs[1][1]["user"]["profile"]["age"] = 35 # Doc B: user mod
cluster_docs[1][1]["user"]["profile"]["interests"].append("coding")

cluster_docs[2][1]["products"][0]["price"] = 999.99 # Doc C: product mod
cluster_docs[2][1]["products"][0]["specifications"]["os"] = "MacOS"

cluster_docs[3][1]["user"]["is_active"] = False # Doc D: minor changes
cluster_docs[3][1]["settings"]["theme"] = "light"
cluster_docs[3][1]["products"][1]["price"] = 70.00

cluster_docs[8][1]["products"] = [] # Doc I: Major product modification (empty products list)
cluster_docs[8][1]["order_history"].pop(0)

# --- Add new documents for clustering experiment ---

# Doc J: Very minor change from Baseline
doc_j = json.loads(json.dumps(baseline))
doc_j["user"]["preferences"]["notifications"]["sms"] = True
cluster_docs.append(("Doc J (Baseline Minor Mod)", doc_j))

# Doc K: Another minor change from Baseline
doc_k = json.loads(json.dumps(baseline))
doc_k["user"]["profile"]["age"] = 31
cluster_docs.append(("Doc K (Baseline Another Minor Mod)", doc_k))

# Doc L: Similar to Doc E (Blog Post) but different content
doc_l = {
    "blog_post": {
        "id": "BP003",
        "title": "Getting Started with Machine Learning",
        "author": {"name": "Alice", "email": "alice@example.com"},
        "tags": ["machine learning", "ai", "data science"],
        "publish_date": "2024-05-15",
        "content_summary": "An introductory guide to machine learning algorithms."
    },
    "comments": [
        {"comment_id": "C002", "user": "MLFan", "text": "Great explanation!"}
    ]
}
cluster_docs.append(("Doc L (Similar to E)", doc_l))

# Doc M: Similar to Doc G (User Profile) but different user
doc_m = {
    "profile_id": "P003",
    "username": "user_c",
    "email": "user.c@domain.com",
    "settings": {"dark_mode": True, "language": "es"}
}
cluster_docs.append(("Doc M (Similar to G)", doc_m))

print("Running Clustering Experiment...")
perform_json_clustering(
    cluster_docs, 
    n_clusters=3,
    sim_func_kwargs={"alpha": 0.4, "beta": 0.4} # Pass weights to mcjsim_v2
)
print("\n" + "="*50 + "\n")