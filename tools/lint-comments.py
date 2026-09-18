#!/usr/bin/env python3
"""Linter for launcher scripts and patch file comments.

The rules:
1. Plain words only. Banned: pin, seam, gate, knob, lever, tax, vintage,
   load-bearing (and endings like pinned). Shard and sharding are fine.
2. No semicolon anywhere in a launcher script, code included. In patch
   files the semicolon is only flagged inside comments, because changing
   patch code would break the diff.
3. A multi-line comment block shorter than 80 characters in total is a
   wall stub. Flag it and write it as one line.
4. A line that carries less than 30% of the 100 character line and then
   ends with a dot is a bad leftover. That text goes to the next line.
5. A sentence-ending dot must sit between column 40 and column 60 of the
   comment (counted from the # sign). Outside that window the text is
   folded to the next line, or the next sentence is pulled up.
6. Do not chain lists with commas and "and". Use bullet points.
6b. Do not use "not X, but Y" or "X, not Y" phrasing. State what is
    true instead.
6c. No jargon metaphors: floor, ceiling, wall, buy are banned.
6d. No vague "edge it" comparisons. Name what beats what.
6e. In docs, a prose line with 4 or more figures becomes a table or
    bullet points. Table rows and headings are exempt.
7. Long walls of text with no bullet points get a hint to use a diagram
   (lifecycle, dataflow) or bullet points instead.

Usage:
    python3 tools/lint-comments.py            # launchers + patches
    python3 tools/lint-comments.py --fix      # rewrite launcher comments
    python3 tools/lint-comments.py file...    # check these files only

Patch files are never rewritten, even with --fix. Their line counts must
stay exact or the diff stops applying. In patches only added (+) lines
are problems. Context lines mirror the base image, so their findings are
hints that need a regenerated patch. In patch code comments, gate/up is
the model's tensor name and is allowed. Exit code 0 when clean, 1 when
problems were found. Hints never fail the run.
"""
import re
import sys
import pathlib

BANNED = re.compile(
    r"\b(pin|pins|pinned|pinning|seam|seams|gate|gates|gated|knob|knobs|"
    r"lever|levers|tax|taxes|vintage|load-bearing|provenance|"
    r"floors?|ceilings?|walls?|buys?|bought|buying)\b",
    re.I,
)
FIX_WORDS = {
    "pin": "set", "pins": "sets", "pinned": "set", "pinning": "setting",
    "seam": "join", "seams": "joins",
    "gate": "check", "gates": "checks", "gated": "checked",
    "knob": "setting", "knobs": "settings",
    "lever": "change", "levers": "changes",
    "tax": "cost", "taxes": "costs",
    "vintage": "setup",
    "load-bearing": "important",
    "provenance": "credit",
    "floor": "bottom limit", "floors": "bottom limits",
    "ceiling": "top limit", "ceilings": "top limits",
    "wall": "limit", "walls": "limits",
    "buy": "get", "buys": "gets", "bought": "got", "buying": "getting",
}
EMDASH = ("\u2014", "\u2013")  # em dash and en dash are banned
NOT_BUT = re.compile(r"\bnot\b.*,\s*but\b", re.I)  # 'not X, but Y' phrasing
COMMA_NOT = re.compile(r",\s*not\s")  # 'X, not Y' tag-on negation
EDGE_IT = re.compile(r"\bedge(s|d)?\s+(it|them|this|that)\b|\bedging\s+(it|them)\b",
                     re.I)
FIGURE = re.compile(r"(?<![\w.=\-/+])\d+(?:\.\d+)?\b(?![\w])")
def is_comma_and_chain(text):
    """A comma list joined by 'and' instead of bullet points.

    Commas inside numbers (1,186,598) do not count."""
    cleaned = re.sub(r"\d,\d", "", text)
    return "," in cleaned and re.search(r"\band\b", cleaned) is not None
WALL_MIN = 80          # a multi-line comment block must total at least this
FOLD_WIDTH = 100       # the 100 character line the dot window is measured on
DOT_MIN = 40           # sentence dot must land in this window ...
DOT_MAX = 60           # ... or be folded to the next line
STUB = 30              # under 30% of the line with a dot = leftover
SEP = re.compile(r"#\s*=+\s*$")
BULLET = re.compile(r"^[-*•]\s+")


def comment_of(line):
    """Return (is_full_comment, inline_comment_or_None)."""
    s = line.strip()
    if s.startswith("#") and not s.startswith("#!"):
        return True, None
    idx = line.rfind(" #")
    if idx != -1 and "#" not in line[idx + 2:]:
        return False, line[idx + 2:].strip().rstrip("\\").strip()
    return False, None


def word_issues(text, where, report, semi, allow_gate=False, emdash=False):
    if semi and ";" in text:
        report.append(f"{where}: semicolon in comment: {text[:70]}")
    if emdash and any(d in text for d in EMDASH):
        report.append(f"{where}: em dash is banned, use a comma or colon: "
                      f"{text[:70]}")
    m = BANNED.search(text)
    if m and allow_gate and m.group(0).lower().startswith("gate"):
        m = None  # in patch code, gate/up is the model's tensor name, not a metaphor
    if m:
        report.append(f"{where}: plain words only, found '{m.group(0)}': {text[:70]}")
    if is_comma_and_chain(text):
        report.append(f"{where}: chained list with commas and 'and', use bullet "
                      f"points: {text[:70]}")
    if NOT_BUT.search(text):
        report.append(f"{where}: 'not X, but Y' phrasing, reword as a plain "
                      f"statement: {text[:70]}")
    if COMMA_NOT.search(text):
        report.append(f"{where}: 'X, not Y' tag-on negation, state the true "
                      f"thing instead: {text[:70]}")
    if EDGE_IT.search(text):
        report.append(f"{where}: 'edge it' phrasing is vague, name the "
                      f"comparison: {text[:70]}")


def dot_col(prefix, words):
    """1-based column of the final dot if the last word ends a sentence."""
    if not words or not words[-1].endswith("."):
        return None
    return len(prefix) + len(" ".join(words))


def reflow_block(indent, lines):
    """Rewrap one comment block so every dot lands in the 40-60 window."""
    paras, cur = [], []
    for ln in lines:
        body = ln.strip()[1:].strip()
        if body == "":
            if cur:
                paras.append(cur)
                cur = []
            paras.append(None)
        else:
            cur.append(ln)
    if cur:
        paras.append(cur)

    out = []
    for para in paras:
        if para is None:
            out.append(indent + "#")
            continue
        words = []
        for ln in para:
            words += ln.strip()[1:].strip().split()
        if not any(w.endswith(".") for w in words):
            out += para  # nothing to place, leave untouched
            continue
        buf, pend = [], []
        for w in words:
            buf.append(w)
            col = dot_col("# ", buf)
            if col is None:
                if len("# " + " ".join(buf)) > FOLD_WIDTH:
                    pend2 = []
                    while buf and len("# " + " ".join(buf)) > FOLD_WIDTH:
                        pend2.insert(0, buf.pop())
                    out.append(indent + "# " + " ".join(buf))
                    buf = pend2
                continue
            if DOT_MIN <= col <= DOT_MAX:
                out.append(indent + "# " + " ".join(buf))
                buf, pend = [], []
            elif col < DOT_MIN:
                continue  # too short, pull the next sentence up
            else:  # too long, fold trailing words to the next line
                while buf and len("# " + " ".join(buf)) > DOT_MAX:
                    pend.insert(0, buf.pop())
                if buf:
                    out.append(indent + "# " + " ".join(buf))
                buf, pend = pend, []
        if buf:
            col = dot_col("# ", buf)
            if col is not None and col < DOT_MIN and out:
                # short tail: pull words back from the previous line until
                # the tail dot lands inside the window
                prev_words = out[-1][len(indent) + 2:].split()
                moved = []
                while prev_words and len(prev_words) > 1:
                    moved.insert(0, prev_words.pop())
                    buf2 = moved + buf
                    col2 = dot_col("# ", buf2)
                    if col2 and DOT_MIN <= col2 <= DOT_MAX:
                        out[-1] = indent + "# " + " ".join(prev_words)
                        buf = buf2
                        break
                    if col2 and col2 > DOT_MAX:
                        prev_words.append(moved.pop(0))
                        break
            if buf:
                while len("# " + " ".join(buf)) > FOLD_WIDTH:
                    pend2 = []
                    while len("# " + " ".join(buf)) > FOLD_WIDTH:
                        pend2.insert(0, buf.pop())
                    out.append(indent + "# " + " ".join(buf))
                    buf = pend2
                out.append(indent + "# " + " ".join(buf))
    return out


def collapse_wall(indent, lines):
    """A multi-line block under 80 chars total becomes one line."""
    bodies = [ln.strip()[1:].strip() for ln in lines]
    text = " ".join(" ".join(b.split()) for b in bodies)
    if len(lines) >= 2 and len(text) < WALL_MIN:
        line = indent + "# " + text
        if text.endswith(".") and not (DOT_MIN <= len(line) <= DOT_MAX):
            line = line[:-1].rstrip()
        return [line]
    return lines


def parse_blocks(lines):
    """Yield (start_index, block_lines) for contiguous full-comment lines."""
    i, n = 0, len(lines)
    while i < n:
        full, _ = comment_of(lines[i])
        if full and not SEP.match(lines[i].strip()):
            j = i + 1
            while j < n:
                f2, _ = comment_of(lines[j])
                if f2 and not SEP.match(lines[j].strip()):
                    j += 1
                else:
                    break
            yield i, lines[i:j]
            i = j
        else:
            i += 1


def fix_file(path, report):
    changed = 0
    lines = path.read_text().splitlines()
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        full, inline = comment_of(line)
        if full and not SEP.match(line.strip()):
            j = i + 1
            while j < n:
                f2, _ = comment_of(lines[j])
                if f2 and not SEP.match(lines[j].strip()):
                    j += 1
                else:
                    break
            block = lines[i:j]
            indent = " " * (len(line) - len(line.lstrip()))
            new_block = reflow_block(indent, block)
            new_block = collapse_wall(indent, new_block)
            if new_block != block:
                changed += 1
            out += new_block
            i = j
            continue
        if inline:
            where = f"{path}:{i+1}"
            before = len(report)
            word_issues(inline, where, report, semi=False)
            if len(report) > before and BANNED.search(inline):
                line = line[:line.rfind(" #") + 2] + " " + \
                    BANNED.sub(lambda m: FIX_WORDS.get(m.group(0).lower(), m.group(0)), inline)
                changed += 1
        out.append(line)
        i += 1
    if changed:
        path.write_text("\n".join(out) + "\n")
    return changed


def check_blocks(path, lines, report, hints, patch):
    for start, block in parse_blocks(lines):
        if patch and all(not _patch_body(ln) for ln in block):
            continue
        bodies = [(_patch_body(ln) if patch else ln.strip()[1:].strip()) for ln in block]
        text_lines = [b for b in bodies if b]
        total = len(" ".join(text_lines))
        where = f"{path}:{start+1}"
        # patch context lines mirror the base image, so their findings are
        # hints. Fixing them would need a regenerated patch.
        def sink(kind_line):
            return hints if (patch and kind_line[:1] == " ") else report
        if len(text_lines) >= 2 and total < WALL_MIN:
            sink(block[0]).append(f"{where}: wall of text is only {total} chars, "
                                  f"write it as one line")
        if len(text_lines) >= 5 and total >= 320 and \
                not any(BULLET.match(b) for b in text_lines):
            hints.append(f"{where}: long wall of text ({len(text_lines)} lines) - "
                         f"consider a lifecycle or dataflow diagram, or bullet points")
        for k, ln in enumerate(block):
            where2 = f"{path}:{start+k+1}"
            body = bodies[k]
            if not body:
                continue
            dest = sink(ln)
            word_issues(body, where2, dest, semi=False, allow_gate=patch,
                        emdash=patch)
            if body.endswith(".") and not body.endswith("..."):
                col = len("# " + body) if not patch else len(body)
                if STUB < col < DOT_MIN:
                    dest.append(f"{where2}: dot at column {col}, less than {DOT_MIN} "
                                f"(short leftover, fold to next line): {body[:70]}")
                elif col > DOT_MAX:
                    dest.append(f"{where2}: dot at column {col}, outside "
                                f"{DOT_MIN}-{DOT_MAX}: {body[:70]}")
            if re.match(r"^\d+\.", body):
                dest.append(f"{where2}: comment starts with a number, an "
                            f"orphaned wrap of the previous line, merge it "
                            f"upward: {body[:70]}")


def _patch_body(line):
    """Comment text of a diff line that carries added or context code."""
    content = line[1:] if line[:1] in "+ " else line
    s = content.strip()
    if s.startswith("#") and not s.startswith("#!"):
        return s
    idx = content.rfind(" #")
    if idx != -1 and "#" not in content[idx + 2:]:
        return content[idx + 2:].strip().rstrip("\\").strip()
    return ""


def check_launcher(path, report, hints):
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines, 1):
        if ";" in line:
            report.append(f"{path}:{i}: semicolon is banned outright, column "
                          f"{line.index(';')+1}: {line.strip()[:70]}")
        if any(d in line for d in EMDASH):
            report.append(f"{path}:{i}: em dash is banned outright: "
                          f"{line.strip()[:70]}")
    check_blocks(path, lines, report, hints, patch=False)


def figure_count(line):
    """Standalone numbers on a line, thousands commas stripped.
    Identifiers like 4bpw, MTP-3, DCP=2, 0.26.1 do not count."""
    return len(FIGURE.findall(re.sub(r"\d,\d", "", line)))


def check_docs(path, report, hints):
    """Markdown docs: banned words, dashes, semicolons in prose lines.
    Fenced code blocks are skipped."""
    in_fence = False
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if ";" in line:
            report.append(f"{path}:{i}: semicolon is banned outright: "
                          f"{line.strip()[:70]}")
        if any(d in line for d in EMDASH):
            report.append(f"{path}:{i}: em dash is banned, use a comma or "
                          f"colon: {line.strip()[:70]}")
        word_issues(line, f"{path}:{i}", report, semi=False)
        stripped = line.strip()
        if not stripped.startswith(("|", "#")) and figure_count(line) >= 4:
            report.append(f"{path}:{i}: {figure_count(line)} figures in one "
                          f"line, use a table or bullet points: "
                          f"{stripped[:70]}")


def check_patch(path, report, hints):
    lines = path.read_text().splitlines()
    keep = []
    for ln in lines:
        if ln[:1] in "+ ":  # added or context lines = the live file state
            keep.append(ln)
        else:
            keep.append("")  # break blocks at diff structure lines
    check_blocks(path, keep, report, hints, patch=True)


def default_files(root):
    files = []
    for pat in ["serve_vllm_GLM53f-exl3-*.jj.r38.sh",
                "internal/serve-glm53f-exl3-*-local.sh",
                "patches/*.patch",
                "README.md"]:
        files += sorted(g for g in root.glob(pat) if g.is_file())
    return files


def main():
    args = sys.argv[1:]
    fix = "--fix" in args
    args = [a for a in args if a != "--fix"]
    root = pathlib.Path(__file__).resolve().parent.parent
    files = [pathlib.Path(a) for a in args] or default_files(root)
    report, hints = [], []
    for f in files:
        if f.suffix == ".patch":
            if fix:
                print(f"{f}: patch files are never rewritten, the diff must "
                      f"keep applying")
            check_patch(f, report, hints)
        elif f.suffix == ".md":
            if fix:
                print(f"{f}: docs are never rewritten by the linter")
            check_docs(f, report, hints)
        else:
            if fix:
                n = fix_file(f, report)
                if n:
                    print(f"{f}: reflowed {n} comment block(s)")
            check_launcher(f, report, hints)
    for h in hints:
        print("hint:", h)
    if report:
        print(f"\n{len(report)} problem(s):")
        for r in report:
            print(" ", r)
        sys.exit(1)
    print(f"clean: {len(files)} file(s), all rules pass")


if __name__ == "__main__":
    main()
