#!/usr/bin/env python3
"""Inlines the project's quoted includes into one self-contained translation unit.

Library Checker (like most judges) accepts a single source file, so the solution's
`#include "engine/..."` lines are replaced with the header contents, recursively. Every
header of the library carries `#pragma once`, so each one is inlined at most once;
system includes (`<...>`) are left untouched.

With --strip-comments, comments are removed by the compiler's own preprocessor in
"preprocessed" mode (which keeps every directive and macro intact and respects string
literals), then blank lines and repeated system includes are dropped, and so is every
space that does not separate two identifiers or two operator characters (string and
character literals and preprocessor directives are kept verbatim). Judges' source fields
choke on the library's ~65 KB otherwise.

Usage: bundle.py SOURCE [-I DIR]... [-o OUTPUT] [--strip-comments] [--cxx g++]
"""

import argparse
import os
import re
import subprocess
import sys

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')
PRAGMA_ONCE_RE = re.compile(r'^\s*#\s*pragma\s+once\b')


def resolve(name, current_dir, include_dirs):
    for directory in [current_dir, *include_dirs]:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    raise FileNotFoundError(f'cannot resolve #include "{name}" (from {current_dir})')


def inline(path, include_dirs, seen, out):
    seen.add(path)
    current_dir = os.path.dirname(path)
    with open(path, encoding='utf-8') as source:
        for line in source:
            if PRAGMA_ONCE_RE.match(line):
                continue
            match = INCLUDE_RE.match(line)
            if not match:
                out.append(line)
                continue
            target = resolve(match.group(1), current_dir, include_dirs)
            if target not in seen:
                out.append(f'// ---- begin {os.path.relpath(target, include_dirs[0])} ----\n')
                inline(target, include_dirs, seen, out)
                out.append(f'// ---- end {os.path.relpath(target, include_dirs[0])} ----\n')
    if out and not out[-1].endswith('\n'):
        out.append('\n')


WORD_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_$')
OPERATOR_CHARS = set('+-*/%&|^<>=!:.~?')


def starts_number(out):
    """Whether the word token being emitted at the end of `out` is a numeric literal."""
    text = ''.join(out[-32:])
    start = len(text)
    while start > 0 and text[start - 1] in WORD_CHARS:
        start -= 1
    return start < len(text) and text[start].isdigit()


def compact_line(line):
    """Drops whitespace that is not needed to keep adjacent tokens apart."""
    if line.lstrip().startswith('#'):
        return line.strip()
    out = []
    pending_space = False
    i = 0
    while i < len(line):
        char = line[i]
        if char in ' \t':
            pending_space = True
            i += 1
            continue
        if pending_space and out:
            prev = out[-1][-1]
            if (prev in WORD_CHARS and char in WORD_CHARS) or (prev in OPERATOR_CHARS and char in OPERATOR_CHARS):
                out.append(' ')
        pending_space = False
        if char == "'" and starts_number(out):
            out.append(char)  # a digit separator, as in 1'000'000
            i += 1
            continue
        if char in '"\'':
            # Copy a string or character literal verbatim, honouring escapes.
            end = i + 1
            while line[end] != char:
                end += 2 if line[end] == '\\' else 1
            out.append(line[i:end + 1])
            i = end + 1
            continue
        out.append(char)
        i += 1
    return ''.join(out)


def strip_comments(text, cxx):
    stripped = subprocess.run(
        [cxx, '-fpreprocessed', '-dD', '-E', '-P', '-x', 'c++', '-'],
        input=text, capture_output=True, text=True, check=True,
    ).stdout
    seen_system_includes = set()
    lines = []
    for line in stripped.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        if re.match(r'^\s*#\s*include\s*<', line):
            key = line.replace(' ', '')
            if key in seen_system_includes:
                continue
            seen_system_includes.add(key)
        lines.append(compact_line(line))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('-I', dest='include_dirs', action='append', default=[])
    parser.add_argument('-o', dest='output')
    parser.add_argument('--strip-comments', action='store_true')
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()

    include_dirs = [os.path.realpath(d) for d in args.include_dirs] or [os.getcwd()]
    out = []
    inline(os.path.realpath(args.source), include_dirs, set(), out)
    text = ''.join(out)
    if args.strip_comments:
        text = strip_comments(text, args.cxx)
    if args.output:
        with open(args.output, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == '__main__':
    main()
