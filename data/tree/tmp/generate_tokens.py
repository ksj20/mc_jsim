import random

consonants = list("bcdfghjklmnpqrstvwxyz")
vowels = list("aeiou")
def random_word():
    syllables = random.randint(2,3)  # 2 or 3 syllables
    word = ""
    for _ in range(syllables):
        c1 = random.choice(consonants)
        v = random.choice(vowels)
        c2 = random.choice(consonants)
        word += c1+v+c2
    return word

random.seed(42)
words=set()
while len(words)<1000:
    w=random_word()
    # ensure not existing and not accidentally a real offensive word - we'll just ensure unique
    words.add(w)

word_list = sorted(words)
# len(word_list[:20]), word_list[:20]

# lines=[]
# for i in range(0,1000,20):
#     lines.append(", ".join(word_list[i:i+20]))

output_text = "\n".join(word_list)
print(output_text)