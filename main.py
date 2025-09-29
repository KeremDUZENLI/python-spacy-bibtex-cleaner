import re
from pathlib import Path
import spacy


INPUT = Path("data/references.bib")
OUTPUT = Path("data/references.cleaned.bib")

SPACY_MODEL = "en_core_web_sm"
ENTITY_LABELS = {
    "PERSON",
    "GPE",
    "LOC",
    "ORG",
    "NORP",
    "WORK_OF_ART",
    "EVENT",
    "PRODUCT",
}
_ACRONYM_TOKEN_RE = re.compile(r"\b([A-Z0-9]{2,}|(?=\w*\d)\w+)\b")
_TITLE_FIELD_RE = re.compile(r"(title\s*=\s*\{)(.*?)(\})", re.DOTALL | re.IGNORECASE)


def load_model(name):
    try:
        return spacy.load(name, disable=["tagger", "parser"])
    except Exception as e:
        raise SystemExit(
            f"Failed to load model {name}: {e}\nInstall: python -m spacy download {name}"
        )


def split_braces(text):
    return re.split(r"(\{.*?\})", text, flags=re.DOTALL)


def brace_entities(nlp, text):
    doc = nlp(text)
    spans = [
        (ent.start_char, ent.end_char)
        for ent in doc.ents
        if ent.label_ in ENTITY_LABELS
    ]
    if not spans:
        return text
    spans.sort()
    merged = []
    for a, b in spans:
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
    out = []
    last = 0
    for a, b in merged:
        out.append(text[last:a])
        out.append("{" + text[a:b] + "}")
        last = b
    out.append(text[last:])
    return "".join(out)


def brace_entities_preserve_existing(nlp, title):
    parts = split_braces(title)
    out = []
    for p in parts:
        if p.startswith("{") and p.endswith("}"):
            out.append(p)
        else:
            out.append(brace_entities(nlp, p))
    return "".join(out)


def protect_acronyms_preserve_braces(chunk):
    parts = split_braces(chunk)
    out = []
    for p in parts:
        if p.startswith("{") and p.endswith("}"):
            out.append(p)
        else:

            def repl(m):
                tok = m.group(1)
                if re.fullmatch(r"[A-Z0-9]{2,}", tok) or re.search(r"\d", tok):
                    return "{" + tok + "}"
                return tok

            out.append(_ACRONYM_TOKEN_RE.sub(repl, p))
    return "".join(out)


def sentence_case_preserve_braces(text):
    parts = split_braces(text)
    out = []
    is_first_part = True
    for p in parts:
        if p.startswith("{") and p.endswith("}"):
            out.append(p)
        else:
            s = p.lower()
            if is_first_part and re.search(r"[a-zA-Z]", s):
                s = re.sub(
                    r"^([^A-Za-z0-9]*)([A-Za-z])",
                    lambda m: m.group(1) + m.group(2).upper(),
                    s,
                    count=1,
                )
                is_first_part = False

            s = re.sub(
                r"(:\s*)([A-Za-z])", lambda m: m.group(1) + m.group(2).upper(), s
            )
            out.append(s)
    return "".join(out)


def process_title(nlp, raw):
    step1 = brace_entities_preserve_existing(nlp, raw)
    step2 = protect_acronyms_preserve_braces(step1)
    step3 = sentence_case_preserve_braces(step2)
    return step3, (step3 != raw)


def clean_file(nlp, input_path, output_path):
    text = input_path.read_text(encoding="utf-8")
    total = 0
    changed = 0

    def replace(m):
        nonlocal total, changed
        total += 1
        prefix, content, suffix = m.group(1), m.group(2), m.group(3)
        cleaned, did_change = process_title(nlp, content)
        if did_change:
            changed += 1
        return prefix + cleaned + suffix

    new_text = _TITLE_FIELD_RE.sub(replace, text)
    output_path.write_text(new_text, encoding="utf-8")
    return total, changed


nlp = load_model(SPACY_MODEL)
total, changed = clean_file(nlp, INPUT, OUTPUT)
print("Processed titles:", total)
print("Titles changed:", changed)
print("Cleaned file written to:", OUTPUT)
