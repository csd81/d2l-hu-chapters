#!/usr/bin/env python3
"""
Terminology linter for d2l-hu-chapters.

Scans prose text only — fenced code blocks (``` / ~~~) and inline
backtick spans are skipped.  Reports every occurrence of a forbidden
form together with the canonical replacement.

Usage
-----
    python scripts/check_terminology.py            # scan all chapter_*/**/*.md
    python scripts/check_terminology.py file.md    # scan specific files
    python scripts/check_terminology.py --fix      # auto-fix in-place (use with care)

Exit codes: 0 = clean, 1 = violations found.
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Denylist: (regex_pattern, canonical_form, re_flags)
# Patterns are matched against prose lines only (code stripped).
# Add new entries here when TERMINOLOGY.md gains a new "nem:" ruling.
# ---------------------------------------------------------------------------
DENYLIST: list[tuple[str, str, int]] = [
    # Optimization
    (r"gradiens ereszkedés",             "gradienscsökkenés",                   re.I),
    (r"gradient descent",                "gradienscsökkenés",                   re.I),
    (r"gradiens módszer",                "gradienscsökkenés",                   re.I),
    (r"sztochasztikus gradiens módszer", "sztochasztikus gradienscsökkenés",    re.I),
    (r"sztochasztikus gradiens descent", "sztochasztikus gradienscsökkenés",    re.I),
    (r"tanulási sebesség",               "tanulási ráta",                       re.I),
    (r"learning rate",                   "tanulási ráta",                       re.I),
    (r"batch size",                      "batch méret",                         re.I),
    # Attention / Transformer
    (r"figyelem mechanizmus",            "figyelemmechanizmus",                 re.I),
    (r"figyelmi mechanizmus",            "figyelemmechanizmus",                 re.I),
    (r"Transzformer",                    "Transformer",                         0),
    (r"enkóder",                         "kódoló",                              re.I),
    (r"dekóder",                         "dekódoló",                            re.I),
    # Datasets
    (r"tanítási halmaz",                 "tanítóhalmaz",                        re.I),
    (r"tanítási adathalmaz",             "tanítóhalmaz",                        re.I),
    (r"tanítókészlet",                   "tanítóhalmaz",                        re.I),
    (r"tesztkészlet",                    "teszthalmaz",                         re.I),
    (r"tesztelési halmaz",               "teszthalmaz",                         re.I),
    (r"tesztelési készlet",              "teszthalmaz",                         re.I),
    (r"tesztadathalmaz",                 "teszthalmaz",                         re.I),
    (r"validációs adathalmaz",           "validációs halmaz",                   re.I),
    (r"érvényesítési halmaz",            "validációs halmaz",                   re.I),
    (r"érvényesítési készlet",           "validációs halmaz",                   re.I),
    # Neural network / architecture
    (r"minibatch(?![-\u2011]sgd)",       "mini-batch",                          re.I),
    (r"rétegnormalizálás",               "rétegnormalizáció",                   re.I),
    (r"batch normalizálás",              "batchnormalizáció",                   re.I),
    (r"batch normalizáció",              "batchnormalizáció",                   re.I),
    (r"batch norm\b",                    "batchnormalizáció",                   re.I),
    (r"\bbackpropagation\b",             "visszaterjesztés",                    re.I),
    (r"visszafelé irányú terjesztés",    "visszaterjesztés",                    re.I),
    (r"előre-terjedés",                  "előreterjesztés",                     re.I),
    (r"előre irányú terjesztés",         "előreterjesztés",                     re.I),
    (r"előre irányú számítás",           "előreterjesztés",                     re.I),
    (r"előre irányú folyamat",           "előreterjesztés",                     re.I),
    (r"forward propagáció",              "előreterjesztés",                     re.I),
    (r"előrepasszolás",                  "előreterjesztés",                     re.I),
    (r"előre irányú menet",              "előremenet",                          re.I),
    (r"forward propagation",             "előreterjesztés",                     re.I),
    (r"objektumfelismerés",              "objektumdetektálás",                  re.I),
]

# Patterns that are always OK regardless of context (suppress false positives)
ALLOWLIST_RE = re.compile(
    r"(:label:|:ref:|:numref:|#\s*)"   # cross-reference directives and headings IDs
)


def iter_prose_lines(text: str):
    """Yield (1-based lineno, prose_text) skipping fenced blocks and inline code."""
    inline = re.compile(r"`[^`\n]+`")
    fence  = re.compile(r"^(`{3,}|~{3,})")
    in_fence = False
    for i, line in enumerate(text.splitlines(), 1):
        if fence.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        yield i, inline.sub("", line)


def check_file(path: Path) -> list[tuple[int, str, str]]:
    """Return list of (lineno, matched_text, canonical) violations."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [(0, str(exc), "")]

    hits: list[tuple[int, str, str]] = []
    for lineno, prose in iter_prose_lines(text):
        if ALLOWLIST_RE.search(prose):
            continue
        for pattern, canonical, flags in DENYLIST:
            for m in re.finditer(pattern, prose, flags):
                hits.append((lineno, m.group(), canonical))
    return hits


def fix_file(path: Path) -> int:
    """Apply all replacements in-place. Returns number of substitutions made."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0

    lines = text.splitlines(keepends=True)
    fence = re.compile(r"^(`{3,}|~{3,})")
    inline = re.compile(r"`[^`\n]+`")
    in_fence = False
    total = 0

    result = []
    for line in lines:
        bare = line.rstrip("\n")
        if fence.match(bare):
            in_fence = not in_fence
            result.append(line)
            continue
        if in_fence:
            result.append(line)
            continue

        new_line = line
        for pattern, canonical, flags in DENYLIST:
            # Only replace in the prose portions (preserve inline code spans)
            parts = inline.split(new_line)
            codes = inline.findall(new_line)
            replaced_parts = []
            for part in parts:
                new_part, n = re.subn(pattern, canonical, part, flags=flags)
                total += n
                replaced_parts.append(new_part)
            # Reassemble
            merged = ""
            for j, part in enumerate(replaced_parts):
                merged += part
                if j < len(codes):
                    merged += codes[j]
            new_line = merged
        result.append(new_line)

    path.write_text("".join(result), encoding="utf-8")
    return total


def collect_paths(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(p) for p in argv if not p.startswith("--")]
    return sorted(
        set(Path(".").glob("chapter_*/**/*.md"))
        | set(Path(".").glob("chapter_*/*.md"))
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    fix_mode = "--fix" in argv
    paths = collect_paths([a for a in argv if a != "--fix"])

    total_violations = 0
    for path in paths:
        if fix_mode:
            n = fix_file(path)
            if n:
                print(f"fixed {n} violation(s) in {path}")
                total_violations += n
        else:
            hits = check_file(path)
            for lineno, found, canonical in hits:
                print(f"{path}:{lineno}: '{found}' → '{canonical}'")
                total_violations += 1

    if fix_mode:
        if total_violations:
            print(f"\nFixed {total_violations} violation(s) total.")
        else:
            print("Nothing to fix.")
        return 0

    if total_violations:
        print(f"\n{total_violations} violation(s) found. Run with --fix to auto-correct.",
              file=sys.stderr)
        return 1

    print("OK – no terminology violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
