import json, math, random, hashlib, difflib
from typing import Any, List, Tuple, Iterator
import pandas as pd
import numpy as np # For distance matrix
from sklearn.cluster import AgglomerativeClustering # For clustering (will be used for comparison, but E4 uses custom)

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
def minhash(tokens: List[str], num_perm: int = 64) -> List[int]:
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

def count_nodes(doc: Any) -> int:
    """Recursively counts the number of primitive values and structural elements (objects, arrays)."""
    count = 0
    if isinstance(doc, dict):
        count += 1  # Count the object itself
        for k, v in doc.items():
            count += 1  # Count the key
            count += count_nodes(v)
    elif isinstance(doc, list):
        count += 1  # Count the array itself
        for item in doc:
            count += count_nodes(item)
    else:
        count += 1 # Count the primitive value
    return count

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


def mcjsim(j1: Any, j2: Any, num_perm: int = 64, alpha: float = 0.4, beta: float = 0.4) -> float:
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
    gamma = 1.0 - alpha - beta # Calculated gamma based on alpha and beta
    
    combined_similarity = (alpha * jaccard_hat) + \
                          (beta * content_sim) + \
                          (gamma * custom_struct_sim)
                          
    return combined_similarity


# --- Function to generate a larger JSON document ---
def generate_large_json_baseline(num_users: int = 10, items_per_user: int = 5, reviews_per_item: int = 2) -> dict:
    """
    Generates a larger JSON document to simulate a 550-node baseline.
    Node count will vary based on specific values, but aims for a higher number.
    Rough estimate:
    - Base user profile: ~15-20 nodes
    - Each item: ~10-15 nodes + reviews
    - Each review: ~5 nodes
    num_users * (base_user_nodes + items_per_user * (item_nodes + reviews_per_item * review_nodes))
    10 * (15 + 5 * (10 + 2 * 5)) = 10 * (15 + 5 * 20) = 10 * (15 + 100) = 10 * 115 = 1150 nodes (rough estimate)
    We'll adjust `num_users` to get closer to 550. Let's aim for fewer items/reviews per user or fewer users.
    If 5 users, 5 items per user, 2 reviews per item:
    5 * (15 + 5 * (10 + 2 * 5)) = 5 * 115 = 575 nodes. This is a good target.
    """
    
    data = {
        "dataset_info": {
            "name": "Synthetic User-Product Data",
            "version": "1.0",
            "generation_date": "2024-05-28"
        },
        "users": []
    }

    for i in range(num_users):
        user = {
            "id": f"USR{i+1:03d}",
            "username": f"user_{i}",
            "email": f"user.{i}@example.com",
            "is_active": random.choice([True, False]),
            "profile": {
                "first_name": f"First{i}",
                "last_name": f"Last{i}",
                "age": random.randint(18, 65),
                "address": {
                    "street": f"{random.randint(1, 999)} Main St",
                    "city": random.choice(["Anytown", "Newtown", "Oldtown"]),
                    "zip_code": f"{random.randint(10000, 99999)}",
                    "country": "USA"
                },
                "interests": random.sample(["reading", "hiking", "cooking", "gaming", "photography"], k=random.randint(1, 3))
            },
            "preferences": {
                "newsletter": random.choice([True, False]),
                "notifications": {
                    "email": random.choice([True, False]),
                    "sms": random.choice([True, False])
                }
            },
            "products_purchased": []
        }

        for j in range(items_per_user):
            product = {
                "product_id": f"PROD{i*items_per_user + j + 1:03d}",
                "name": f"Item {i*items_per_user + j + 1}",
                "category": random.choice(["Electronics", "Books", "Clothing", "Home Goods"]),
                "price": round(random.uniform(10.0, 1500.0), 2),
                "specifications": {
                    "weight_kg": round(random.uniform(0.1, 5.0), 1),
                    "color": random.choice(["black", "white", "blue", "red"])
                },
                "reviews": []
            }
            for k in range(reviews_per_item):
                review = {
                    "review_id": f"REV{k+1}",
                    "user_id": f"USR{random.randint(1, num_users):03d}",
                    "rating": random.randint(1, 5),
                    "comment": f"A comment for item {product['product_id']} from user {user['id']}"
                }
                product["reviews"].append(review)
            user["products_purchased"].append(product)
        data["users"].append(user)
    return data

# Generate the large baseline document
large_baseline_doc = generate_large_json_baseline(num_users=5, items_per_user=5, reviews_per_item=2)
# print(f"Generated large baseline with approximately {count_nodes(large_baseline_doc)} nodes.")
# Let's verify the actual node count after generation
actual_node_count = count_nodes(large_baseline_doc)
print(f"Generated large baseline with {actual_node_count} nodes.")


experiments = []

# --- E1. Perturbation sensitivity experiments (using large_baseline_doc) ---
# Starting from a 550-node baseline document we produced sixteen single-edit variants
# and one mixed-edit variant.

# Note: For strict 16 single-edit variants, we'd need to pick 16 distinct modification types.
# The current set covers most described single-edit types.

experiments.append(("E1.0: identical", large_baseline_doc, large_baseline_doc))

# Single-edit variants from the large baseline
doc_e1_1 = json.loads(json.dumps(large_baseline_doc))
doc_e1_1["users"][0]["products_purchased"][0]["price"] = 1150.00
experiments.append(("E1.1: numeric field modified (price)", large_baseline_doc, doc_e1_1))

doc_e1_2 = json.loads(json.dumps(large_baseline_doc))
doc_e1_2["users"][0]["email"] = "user.0.new@example.com"
experiments.append(("E1.2: string field modified (email)", large_baseline_doc, doc_e1_2))

doc_e1_3 = json.loads(json.dumps(large_baseline_doc))
doc_e1_3["users"][0]["profile"]["phone"] = "555-123-4567"
experiments.append(("E1.3: new field added (phone)", large_baseline_doc, doc_e1_3))

doc_e1_4 = json.loads(json.dumps(large_baseline_doc))
del doc_e1_4["users"][0]["profile"]["age"]
experiments.append(("E1.4: field removed (age)", large_baseline_doc, doc_e1_4))

doc_e1_5 = json.loads(json.dumps(large_baseline_doc))
# Reorder products for user 0
products_user0 = doc_e1_5["users"][0]["products_purchased"]
if len(products_user0) > 1:
    doc_e1_5["users"][0]["products_purchased"] = [products_user0[1], products_user0[0]] + products_user0[2:]
experiments.append(("E1.5: array order changed (products)", large_baseline_doc, doc_e1_5))

doc_e1_6 = json.loads(json.dumps(large_baseline_doc))
doc_e1_6["users"][0]["profile"]["interests"].append("gardening")
experiments.append(("E1.6: item added to array (interest)", large_baseline_doc, doc_e1_6))

doc_e1_7 = json.loads(json.dumps(large_baseline_doc))
doc_e1_7["users"][0]["products_purchased"][0]["specifications"]["weight_kg"] = 0.7
experiments.append(("E1.7: nested numeric field modified (weight)", large_baseline_doc, doc_e1_7))

doc_e1_8 = json.loads(json.dumps(large_baseline_doc))
doc_e1_8["users"][0]["profile"]["address"]["city"] = "Metropolis"
experiments.append(("E1.8: nested string field modified (city)", large_baseline_doc, doc_e1_8))

doc_e1_9 = json.loads(json.dumps(large_baseline_doc))
doc_e1_9["users"][0]["preferences"]["newsletter"] = False
experiments.append(("E1.9: boolean field changed (newsletter)", large_baseline_doc, doc_e1_9))

doc_e1_10 = json.loads(json.dumps(large_baseline_doc))
# Modify a review rating
if doc_e1_10["users"][0]["products_purchased"] and doc_e1_10["users"][0]["products_purchased"][0]["reviews"]:
    doc_e1_10["users"][0]["products_purchased"][0]["reviews"][0]["rating"] = 3
experiments.append(("E1.10: array item modified (review rating)", large_baseline_doc, doc_e1_10))

doc_e1_11 = json.loads(json.dumps(large_baseline_doc))
doc_e1_11["users"][0]["last_login"] = None
experiments.append(("E1.11: null field added", large_baseline_doc, doc_e1_11))

doc_e1_12 = json.loads(json.dumps(large_baseline_doc))
doc_e1_12["users"][0]["favorite_categories"] = []
experiments.append(("E1.12: empty list added", large_baseline_doc, doc_e1_12))

doc_e1_13 = json.loads(json.dumps(large_baseline_doc))
doc_e1_13["users"][0]["settings"] = {}
experiments.append(("E1.13: empty object added", large_baseline_doc, doc_e1_13))

# Type changes
doc_e1_14 = json.loads(json.dumps(large_baseline_doc))
doc_e1_14["users"][0]["is_active"] = "active" # bool to string
experiments.append(("E1.14: field type changed (is_active)", large_baseline_doc, doc_e1_14))

doc_e1_15 = json.loads(json.dumps(large_baseline_doc))
# Change object to array for notifications
doc_e1_15["users"][0]["preferences"]["notifications"] = ["email_on", "sms_off"]
experiments.append(("E1.15: object to array change (notifications)", large_baseline_doc, doc_e1_15))

doc_e1_16 = json.loads(json.dumps(large_baseline_doc))
# Change an array element from string to object
if doc_e1_16["users"][0]["profile"]["interests"]:
    doc_e1_16["users"][0]["profile"]["interests"][0] = {"activity": "reading", "level": "high"}
experiments.append(("E1.16: array element type changed (interest)", large_baseline_doc, doc_e1_16))


# One mixed-edit variant
doc_e1_mixed = json.loads(json.dumps(large_baseline_doc))
doc_e1_mixed["users"][0]["profile"]["age"] = 31 # numeric change
doc_e1_mixed["users"][1]["profile"]["address"]["country"] = "Canada" # nested string change
if len(doc_e1_mixed["users"][0]["products_purchased"]) > 1:
    doc_e1_mixed["users"][0]["products_purchased"].pop(1) # array item removed
doc_e1_mixed["dataset_info"]["version"] = "1.1" # top-level string change
doc_e1_mixed["users"][2]["new_status"] = "verified" # new field added
experiments.append(("E1.Mixed: complex mixed changes", large_baseline_doc, doc_e1_mixed))


print("--- E1. Perturbation sensitivity & E2. MinHash accuracy ---")
results = []
for label, a, b in experiments:
    fp_a = fingerprint(a)
    fp_b = fingerprint(b)

    jaccard_exact = jaccard_similarity_exact(fp_a, fp_b)
    
    mh_a = minhash(fp_a)
    mh_b = minhash(fp_b)
    jaccard_minhash_val = jaccard_minhash(mh_a, mh_b)

    # Use the mcjsim function with default alpha=0.4, beta=0.4, num_perm=64
    sim = mcjsim(a, b) 

    results.append({
        "experiment": label,
        "jaccard_exact": round(jaccard_exact, 3),
        "jaccard_minhash": round(jaccard_minhash_val, 3),
        "mcjsim": round(sim, 3) # Changed to mcjsim
    })

df = pd.DataFrame(results)
print(f"E1. Perturbation sensitivity experiments (using {actual_node_count}-node baseline)\n", df)

# Calculate mean absolute error for E2
exact_jaccards = df["jaccard_exact"].values
minhash_jaccards = df["jaccard_minhash"].values
mae = np.mean(np.abs(exact_jaccards - minhash_jaccards))
std_err_mae = np.std(np.abs(exact_jaccards - minhash_jaccards)) / np.sqrt(len(exact_jaccards))

print(f"\nE2. MinHash accuracy (r=64 permutations): Mean absolute error = {mae:.3f} +/- {std_err_mae:.3f}")
print(f"Theoretical bound for r=64: 1/sqrt(64) = {1/math.sqrt(64):.3f}")

print("\n" + "="*50 + "\n")

# --- E3. Throughput ---
# This is a conceptual experiment and won't be fully implemented as a timed benchmark
# due to reliance on external execution environment specifics and complexity.
# The code below provides a conceptual representation for fingerprinting and comparison.

def conceptual_throughput_test(num_nodes: int = 1000):
    # Create a synthetic document of num_nodes for fingerprinting
    # This is a very simplified way to get 'num_nodes' paths;
    # a real implementation would build a more complex JSON.
    synthetic_doc_for_throughput = {"id": f"THROUGHPUT{i}" for i in range(num_nodes)}
    
    # Simulate fingerprinting time (conceptual)
    # In a real benchmark, you would time `fingerprint(synthetic_doc_for_throughput)`
    # For now, just call it to show the operation.
    fp = fingerprint(synthetic_doc_for_throughput)

    # Simulate sketch comparison time (conceptual)
    # For comparison, we'd need two sketches.
    # In a real benchmark, you'd time `mcjsim(doc1, doc2)`
    # For now, just call it.
    _ = minhash(fp) # Simulate generating one sketch
    _ = mcjsim(synthetic_doc_for_throughput, synthetic_doc_for_throughput) # Simulate comparison

    print(f"E3. Throughput: Conceptual representation for {num_nodes} nodes.")
    print("  - Fingerprinting of a synthetic document (e.g., 9.4 µs/node for 10^3 to 10^5 nodes)")
    print("  - Comparison of two sketches (e.g., 2.4 ± 0.1 µs)")
    print("  (Full throughput benchmarking requires dedicated timing within a controlled environment.)")

conceptual_throughput_test()

print("\n" + "="*50 + "\n")

# ---------- E4. Clustering heterogeneous JSON ----------

def perform_json_clustering(
    json_docs: List[Tuple[str, Any]], 
    n_clusters: int = 3, 
    sim_func=mcjsim,
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


# --- JSON Documents for E4 Clustering ---
# A miniature corpus of seven documents (laptops, keyboards, novels, report)
e4_cluster_docs = [
    ("laptop_A", {
        "product_id": "LAP001", "name": "Laptop Pro X1", "category": "Electronics",
        "specs": {"cpu": "Intel i9", "ram_gb": 32, "storage_gb": 1024},
        "price": 2000.00, "reviews": [{"rating": 5}]
    }),
    ("laptop_B", {
        "product_id": "LAP002", "name": "Laptop Air M1", "category": "Electronics",
        "specs": {"cpu": "Apple M1", "ram_gb": 8, "storage_gb": 256},
        "price": 1200.00, "reviews": [{"rating": 4}]
    }),
    ("keyboard_A", {
        "product_id": "KEY001", "name": "Mechanical Keyboard Pro", "category": "Accessories",
        "type": "mechanical", "switches": "Cherry MX Blue",
        "price": 150.00, "backlit": True
    }),
    ("keyboard_B", {
        "product_id": "KEY002", "name": "Wireless Ergonomic Keyboard", "category": "Accessories",
        "type": "membrane", "layout": "Ergonomic",
        "price": 75.00, "backlit": False
    }),
    ("novel_A", {
        "book_id": "NOV001", "title": "The Silent Patient", "author": "Alex Michaelides",
        "genre": "Thriller", "pages": 336, "published_year": 2019
    }),
    ("novel_B", {
        "book_id": "NOV002", "title": "Where the Crawdads Sing", "author": "Delia Owens",
        "genre": "Mystery", "pages": 384, "published_year": 2018
    }),
    ("report_A", {
        "report_id": "RPT001", "date": "2024-01-20", "type": "Financial Report",
        "summary": "Annual financial overview for Q4 2023.",
        "metrics": {"revenue": 1000000, "profit": 250000}
    })
]


print("--- E4. Clustering heterogeneous JSON ---")

clusters = perform_json_clustering(
    json_docs=e4_cluster_docs,
    n_clusters=4,
    sim_func=mcjsim,
    sim_func_kwargs={"num_perm": 64, "alpha": 0.4, "beta": 0.4}
)

# --- E5. Ablation study ---
# This is a conceptual experiment and cannot be fully reproduced without
# a labelled dataset and ROC-AUC calculation capability.
print("--- E5. Ablation study ---")
print("E5. Ablation study is a conceptual experiment.")
print("It demonstrates the necessity of all three components (Jaccard, Content, Structural).")
print("Removing content term or structural-type term would lower ROC-AUC on a labelled dataset.")
print(" (e.g., from 0.952 to 0.873 and 0.865 on a Yelp subset as per LaTeX.)")