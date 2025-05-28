import random
import uuid
import json
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, List, Tuple, Dict, Optional

# ——— Reproducibility for everything except UUID ———
# SEED = 42
# random.seed(SEED)

random.seed(0)

class RandomDataGenerator(ABC):
    @abstractmethod
    def generate(self) -> Any:
        pass

# —— Primitive generators —— #

class IntGenerator(RandomDataGenerator):
    def __init__(self, low: int = 0, high: int = 100):
        self.low, self.high = low, high
    def generate(self) -> int:
        return random.randint(self.low, self.high)

class FloatGenerator(RandomDataGenerator):
    def __init__(self, low: float = 0.0, high: float = 1.0):
        self.low, self.high = low, high
    def generate(self) -> float:
        return random.uniform(self.low, self.high)

class RationalGenerator(RandomDataGenerator):
    def __init__(self, num_range: Tuple[int, int] = (0, 10), den_range: Tuple[int, int] = (1, 10)):
        self.n_low, self.n_high = num_range
        self.d_low, self.d_high = den_range
    def generate(self) -> Fraction:
        n = random.randint(self.n_low, self.n_high)
        d = random.randint(self.d_low, self.d_high)
        return Fraction(n, d)

class BinaryStringGenerator(RandomDataGenerator):
    def __init__(self, length: int = 8):
        self.length = length
    def generate(self) -> str:
        return ''.join(random.choice('01') for _ in range(self.length))

class UUIDGenerator(RandomDataGenerator):
    def generate(self) -> uuid.UUID:
        return uuid.uuid4()

class WordGenerator(RandomDataGenerator):
    def __init__(self,
                 syllable_range: Tuple[int, int] = (2, 3),
                 consonants: List[str] = list("bcdfghjklmnpqrstvwxyz"),
                 vowels: List[str] = list("aeiou")):
        self.min_syl, self.max_syl = syllable_range
        self.consonants = consonants
        self.vowels = vowels
    def generate(self) -> str:
        return "".join(
            random.choice(self.consonants) +
            random.choice(self.vowels) +
            random.choice(self.consonants)
            for _ in range(random.randint(self.min_syl, self.max_syl))
        )

# —— Step 1: Index (blueprint) generator —— #

class JSONIndexGenerator:
    """
    Generates a nested structure of dicts, lists, and None leaves.
    Ensures the root is never None.
    """
    def __init__(self,
                 max_depth: int = 3,
                 max_width: int = 4,
                 key_gen: RandomDataGenerator = WordGenerator()):
        self.max_depth = max_depth
        self.max_width = max_width
        self.key_gen = key_gen

    def generate_index(self, depth: int = 0) -> Any:
        # Root must be object or array
        if depth == 0:
            node_type = random.choice(["object", "array"])
        else:
            node_type = random.choice(["object", "array", "leaf"])

        if node_type == "object":
            obj: Dict[str, Any] = {}
            for _ in range(random.randint(1, self.max_width)):
                key = self.key_gen.generate()
                obj[key] = self.generate_index(depth + 1)
            return obj

        elif node_type == "array":
            arr: List[Any] = []
            for _ in range(random.randint(1, self.max_width)):
                arr.append(self.generate_index(depth + 1))
            return arr

        else:  # leaf placeholder
            return None

# —— Step 2: Fill blueprint with actual values —— #

class JSONFromIndexGenerator:
    def __init__(self,
                 index_blueprint: Any,
                 primitive_gens: Optional[List[RandomDataGenerator]] = None):
        self.index = index_blueprint
        self.primitive_gens = primitive_gens or [
            IntGenerator(), FloatGenerator(), RationalGenerator(),
            BinaryStringGenerator(), UUIDGenerator(), WordGenerator()
        ]
        # sentinel to detect “no arg passed”
        self._SENTINEL = object()

    def fill(self, node: Any = None) -> Any:
        # If the caller didn't pass a node, start from the root blueprint.
        if node is None and getattr(self, '_calling_fill', False) is False:
            # mark that we're now inside fill so we don't confuse root None with leaf None
            self._calling_fill = True
            try:
                return self.fill(self.index)
            finally:
                self._calling_fill = False

        # Now: node is either:
        #  - None       → a leaf placeholder
        #  - dict/list → recurse
        #  - anything else (shouldn't happen) → return as-is

        if node is None:
            return random.choice(self.primitive_gens).generate()

        if isinstance(node, dict):
            return {k: self.fill(v) for k, v in node.items()}

        if isinstance(node, list):
            return [self.fill(v) for v in node]

        return node

# … after you build `blueprint` as before …
if __name__ == "__main__":
    idx_gen = JSONIndexGenerator(max_depth=3, max_width=3)
    blueprint = idx_gen.generate_index()

    filler = JSONFromIndexGenerator(blueprint)
    filled = filler.fill()   # now correctly fills the entire structure

    print("Blueprint:")
    print(json.dumps(blueprint, indent=2, default=str))
    print("\nFilled JSON:")
    print(json.dumps(filled, indent=2, default=str))
