import random
import uuid
import json
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, List, Tuple, Dict

# ——— Reproducibility for everything except UUID ———
SEED = 123
random.seed(SEED)

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
        syllables = random.randint(self.min_syl, self.max_syl)
        return "".join(
            random.choice(self.consonants) +
            random.choice(self.vowels) +
            random.choice(self.consonants)
            for _ in range(syllables)
        )

# —— Complex JSON generator —— #

class ComplexJSONGenerator(RandomDataGenerator):
    def __init__(self,
                 max_depth: int = 3,
                 obj_branches: int = 2,
                 arr_branches: int = 2,
                 prim_branches: int = 2,
                 key_gen: RandomDataGenerator = WordGenerator(),
                 primitive_gens: List[RandomDataGenerator] = None):
        self.max_depth     = max_depth
        self.obj_branches  = obj_branches
        self.arr_branches  = arr_branches
        self.prim_branches = prim_branches
        self.key_gen       = key_gen
        self.primitive_gens = primitive_gens or [
            IntGenerator(), FloatGenerator(), RationalGenerator(),
            BinaryStringGenerator(), UUIDGenerator(), WordGenerator()
        ]

    def generate(self, depth: int = 0) -> Any:
        # Base case: at max depth, return a single primitive
        if depth >= self.max_depth:
            return random.choice(self.primitive_gens).generate()

        # 1) Build an object with nested objects
        result: Dict[str, Any] = {}
        for i in range(self.obj_branches):
            key = self.key_gen.generate() + "_obj" + str(i)
            result[key] = self.generate(depth + 1)

        # 2) Add some arrays whose elements are complex JSON
        for i in range(self.arr_branches):
            arr = []
            for _ in range(self.arr_branches):
                arr.append(self.generate(depth + 1))
            result[self.key_gen.generate() + "_arr" + str(i)] = arr

        # 3) Finally sprinkle in a few primitives
        for i in range(self.prim_branches):
            result[self.key_gen.generate() + "_val" + str(i)] = \
                random.choice(self.primitive_gens).generate()

        return result

# —— Example usage —— #

if __name__ == "__main__":
    gen = ComplexJSONGenerator(
        max_depth    = 3,
        obj_branches = 2,
        arr_branches = 2,
        prim_branches= 2
    )

    # Generate and pretty-print
    sample = gen.generate()
    print(json.dumps(sample, default=str, indent=2))
