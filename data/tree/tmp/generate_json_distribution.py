import random
from typing import Any, Dict, List, Optional
from fractions import Fraction
import uuid

# I will use this file for the generation of indexes - 2025.05.20

class RandomDataGenerator:
    def generate(self) -> Any:
        raise NotImplementedError

# (Your primitive generators would go here.)

class JSONIndexGenerator:
    """
    Generates a nested blueprint of dicts & lists, where each leaf is:
      { "__type_distribution__": { "int": w1, "List[int]": w2, ... } }

    - max_depth limits nesting: at depth>=max_depth all nodes are leaves
    - max_width caps number of entries per object/array
    - seed=None → fresh randomness
    - seed=int  → reproducible
    - leaf_type_weights: Optional[Dict[str, float]] where keys match type strings
    """
    def __init__(self,
                 max_depth: int = 1,
                 max_width: int = 1,
                 primitive_type_names: Optional[List[str]] = None,
                 leaf_type_weights: Optional[Dict[str, float]] = None,
                 seed: Optional[int] = None):
        self.max_depth = max_depth
        self.max_width = max_width
        self._rand     = random.Random(seed)

        # default primitive names
        self.types = primitive_type_names or [
            "int", "float", "Fraction", "str", "UUID", "word", "None"
        ]

        # build default weights: each simple & list form = weight 1.0
        if leaf_type_weights:
            # use user‐provided weights directly
            self.leaf_dist = dict(leaf_type_weights)
        else:
            dist: Dict[str, float] = {}
            for t in self.types:
                dist[t] = 1.0
                dist[f"List[{t}]"] = 1.0
            # normalize to sum=1
            total = sum(dist.values())
            self.leaf_dist = {k: v/total for k, v in dist.items()}

    def generate_index(self, depth: int = 0) -> Any:
        # if at or beyond max_depth, produce a leaf
        if depth >= self.max_depth:
            return {"__type_distribution__": dict(self.leaf_dist)}

        # otherwise choose node type (object/array at root; include leaf deeper)
        choices = (["object", "array"]
                   if depth == 0
                   else ["object", "array", "leaf"])
        node = self._rand.choice(choices)

        if node == "object":
            obj: Dict[str, Any] = {}
            for _ in range(self._rand.randint(1, self.max_width)):
                key = f"key_{self._rand.randrange(1_000_000)}"
                obj[key] = self.generate_index(depth + 1)
            return obj

        if node == "array":
            # one‐schema list
            elem = self.generate_index(depth + 1)
            return [elem]

        # leaf → embed distribution
        return {"__type_distribution__": dict(self.leaf_dist)}

    def extract_key_hierarchy(self, node: Any) -> Any:
        """
        - dict → return { key: extract_key_hierarchy(value) }
        - list → return [ extract_key_hierarchy(first_element) ]
        - leaf (the dict with "__type_distribution__") → return None
        """
        if isinstance(node, dict):
            # leaf dict has only the distribution key
            if set(node.keys()) == {"__type_distribution__"}:
                return None
            return {k: self.extract_key_hierarchy(v) for k, v in node.items()}

        if isinstance(node, list):
            if not node:
                return []
            return [self.extract_key_hierarchy(node[0])]

        return None


# —— Example usage —— #
if __name__ == "__main__":
    idx = JSONIndexGenerator(max_depth=2, max_width=1, seed=None)
    blueprint = idx.generate_index()
    
    hierarchy = idx.extract_key_hierarchy(blueprint)
    import pprint
    print("Full blueprint:")
    pprint.pprint(blueprint, width=60, indent=2)
    print("\nKey hierarchy:")
    pprint.pprint(hierarchy, width=60, indent=2)

