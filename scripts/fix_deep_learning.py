#!/usr/bin/env python3
"""One-shot migration: replace 'deep learning' forms with mélytanulás/mélytanulási.

Context rules:
  - with attached suffix (no hyphen, no space): strip suffix, map to back-vowel mélytanulás form
  - with hyphen-suffix: same
  - before bare noun (adjective context):  → mélytanulási
  - otherwise (standalone, before possessive-noun, before connector): → mélytanulás
"""

import re
import sys
from pathlib import Path

# Suffix forms with direct attachment (no space, no hyphen)
# Each entry: (regex that matches whole 'deep learning...' token, replacement)
SUFFIX_FORMS = [
    # Ablative
    (r'deep learningt[oó]l\b',          'mélytanulástól'),
    # Instrumental (gel form from the wrong stem)
    (r'deep learninggel\b',             'mélytanulással'),
    # Locative
    (r'deep learningb[ae]n\b',          'mélytanulásban'),
    # Sublative
    (r'deep learningr[ae]\b',           'mélytanulásra'),
    # Allative
    (r'deep learning(?:h[oe]z|hez)\b',  'mélytanuláshoz'),
    # Dative
    (r'deep learningn[ae]k\b',          'mélytanulásnak'),
    # Accusative
    (r'deep learninget\b',              'mélytanulást'),
    # Terminative
    (r'deep learningig\b',              'mélytanulásig'),
    # Adessive / "beli" locative adjective
    (r'deep learningbeli\b',            'mélytanulásbeli'),
]

# Hyphen-attached suffix forms
HYPHEN_SUFFIX_FORMS = [
    (r'deep learning-b[ae]n\b',         'mélytanulásban'),
    (r'deep learning-r[ae]\b',          'mélytanulásra'),
    (r'deep learning-h[oe]z\b',         'mélytanuláshoz'),
    (r'deep learning-n[ae]k\b',         'mélytanulásnak'),
    (r'deep learning-t[oó]l\b',         'mélytanulástól'),
    (r'deep learning-beli\b',           'mélytanulásbeli'),
    (r'deep learning-es\b',             'mélytanulási'),
    (r'deep learning-nak\b',            'mélytanulásnak'),
]

# "deep learning NOUN" → mélytanulási NOUN (adjective context)
# These are bare nouns that follow deep learning as a modifier.
ADJECTIVE_NOUNS = (
    r'keretrendszer|könyvtár|modell|algoritmus|szerver|számítás|módszer|'
    r'rendszer|feladat|probléma|kutatás|architektúra|keret|eszköz|megközelítés|'
    r'műveletek|összetevő'
)
ADJECTIVE_PATTERN = re.compile(
    r'deep learning (' + ADJECTIVE_NOUNS + r')',
    re.I,
)

# After all the above, remaining bare "deep learning" → mélytanulás
BARE = re.compile(r'deep learning\b', re.I)


def fix_line(line: str) -> str:
    for pattern, repl in SUFFIX_FORMS + HYPHEN_SUFFIX_FORMS:
        line = re.sub(pattern, repl, line, flags=re.I)
    line = ADJECTIVE_PATTERN.sub(r'mélytanulási \1', line)
    line = BARE.sub('mélytanulás', line)
    return line


def process_file(path: Path) -> int:
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    inline = re.compile(r'`[^`\n]+`')
    fence = re.compile(r'^(`{3,}|~{3,})')
    in_fence = False
    result = []
    changes = 0

    for line in lines:
        bare = line.rstrip('\n')
        if fence.match(bare):
            in_fence = not in_fence
            result.append(line)
            continue
        if in_fence:
            result.append(line)
            continue

        # Process prose only; preserve inline code spans
        parts = inline.split(line)
        codes = inline.findall(line)
        new_parts = []
        for part in parts:
            fixed = fix_line(part)
            if fixed != part:
                changes += 1
            new_parts.append(fixed)
        merged = ''
        for j, part in enumerate(new_parts):
            merged += part
            if j < len(codes):
                merged += codes[j]
        result.append(merged)

    if changes:
        path.write_text(''.join(result), encoding='utf-8')
    return changes


def main():
    paths = sorted(
        set(Path('.').glob('chapter_*/**/*.md'))
        | set(Path('.').glob('chapter_*/*.md'))
    )
    total = 0
    for p in paths:
        n = process_file(p)
        if n:
            print(f'fixed {n} in {p}')
            total += n
    print(f'\nTotal: {total} replacements')


if __name__ == '__main__':
    main()
