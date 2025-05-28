import random
import uuid
from fractions import Fraction
from typing import Any, Dict, List, Union, Optional, Type, Tuple

class JSONIndexGenerator:
    """
    Generates a nested “index” (blueprint) of dicts and lists, whose leaves
    are type‐hints like Union[int, List[float]].
    """
    def __init__(self,
                 max_depth: int = 4,
                 max_width: int = 4,
                 seed: Optional[int] = None):
        self.max_depth = max_depth
        self.max_width = max_width
        # own RNG: seed=None → time/OS entropy; seed=int → reproducible
        self._rand = random.Random(seed)

        # our pool of primitive Python types
        self._prims: List[Type] = [
            int, float, Fraction, str, uuid.UUID
        ]

    def _make_type_hint(self) -> Any:
        """Create a Union[primitive, List[other_primitive]]"""
        t1 = self._rand.choice(self._prims)
        t2 = self._rand.choice(self._prims)
        # typing.List and typing.Union are subscriptable
        return Union[t1, List[t2]]

    def generate_index(self, depth: int = 0) -> Any:
        # At the root, force object or array
        choices = ["object", "array"] if depth == 0 else ["object", "array", "leaf"]
        node_type = self._rand.choice(choices)

        if node_type == "object":
            obj: Dict[str, Any] = {}
            for _ in range(self._rand.randint(1, self.max_width)):
                # generate a dummy key
                key = f"key_{self._rand.randrange(1_000_000)}"
                obj[key] = self.generate_index(depth + 1)
            return obj

        elif node_type == "array":
            # list of exactly one element‐schema
            element_schema = self.generate_index(depth + 1)
            return [element_schema]

        else:  # leaf placeholder → a type hint
            return self._make_type_hint()

# —— Example usage —— #
if __name__ == "__main__":
    idx_gen = JSONIndexGenerator(max_depth=3, max_width=3, seed=None)
    blueprint = idx_gen.generate_index()
    import pprint
    pprint.pprint(blueprint, indent=3)
