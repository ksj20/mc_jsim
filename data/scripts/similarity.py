import re, sys, difflib, pathlib, textwrap

def strip_cpp(text: str) -> str:
    # remove comments, spaces, blank lines
    text = re.sub(r'//.*?$|/\*.*?\*/', '', text, flags=re.S|re.M)
    text = re.sub(r'\s+', ' ', text)
    return text

code1 = strip_cpp(pathlib.Path(sys.argv[1]).read_text())
code2 = strip_cpp(pathlib.Path(sys.argv[2]).read_text())
bshow = sys.argv[3]

indent = "      "          # six spaces
width  = 200                # wrap column

if bshow == "true":
    print(f"{sys.argv[1]}:")
    print(textwrap.fill(
            code1,
            width=width,
            initial_indent=indent,
            subsequent_indent=indent
    ))
    print(f"{sys.argv[2]}:")
    print(textwrap.fill(
            code2,
            width=width,
            initial_indent=indent,
            subsequent_indent=indent
    ))

ratio = difflib.SequenceMatcher(None, code1, code2).ratio()
print(f"{ratio*100:.2f}% similarity")