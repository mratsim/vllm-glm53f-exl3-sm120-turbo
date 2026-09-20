#!/usr/bin/env python3
"""Linter for launcher scripts and patch file comments.

The rules:
1. Plain words only. Two blocklists merge here: the local list (pin, seam,
   gate, knob, lever, tax, vintage, load-bearing) and the Tattletale
   writing-docs blocklist (historical and temporal words, campaign terms,
   test-theater metaphors). Model-architecture names (gate.weight, gated,
   sigmoid gate) and compound terms of art (power envelope, GPU wave,
   cryptographic digest) stay exempt.
2. No semicolon anywhere in a launcher script, code included. In patch
   files the semicolon is only flagged inside comments, because changing
   patch code would break the diff.
3. No narration of the journey: temporal words (currently, as of, no
   longer, legacy, previously), design justification (because, instead
   of, rather than, which is why), and pipeline artifact IDs (SLOP-002,
   boot #12, iter-3) are banned. State the present contract instead.
4. A multi-line comment block shorter than 80 characters in total is a
   wall stub. Flag it and write it as one line.
5. A line that carries less than 30% of the 100 character line and then
   ends with a dot is a bad leftover. That text goes to the next line.
6. A sentence-ending dot must be beyond column 40 of the comment (counted
   from the # sign) on a prose line capped at 140 characters, the
   Tattletale PROSE_CAP. Past the cap the text folds to the next line, or
   the next sentence is pulled up. Reflow fills lines up to the cap;
   hanging-indent structure (patch listings, layout tables, bullets)
   survives the reflow as units.
7. Do not chain lists with commas and "and". Use bullet points.
7b. Do not use "not X, but Y" or "X, not Y" phrasing. State what is
    true instead.
7c. No jargon metaphors: floor, ceiling, wall, buy are banned.
7d. No vague "edge it" comparisons. Name what beats what.
7e. In docs, a prose line with 4 or more figures becomes a table or
    bullet points. Table rows and headings are exempt.
8. Long walls of text with no bullet points get a hint to use a diagram
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
    r"floors?|ceilings?|walls?|buys?|bought|buying|"
    r"lands?|landed|"
    r"sits?|sitting|\bsat\b)\b",
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
    "land": "arrive", "lands": "arrives", "landed": "applied",
    "sit": "stay", "sits": "stays", "sitting": "staying", "sat": "stayed",
}

# Tattletale writing-docs blocklist merge. Each entry:
# (pattern, exemption regex or None, message).
# Historical and temporal words (legacy, currently, as of, ...) live in
# the TEMPORAL pattern below; design justification lives in
# DESIGN_NARRATION. Compound terms of art stay allowed: "power envelope"
# (EE term), "wave" in GPU kernel context, "gate" names in model
# architectures (sigmoid gate, gate.weight, gated, GLU), cryptographic
# digest.
TATTLETALE = [
    (r"\bdraw\b|\bdraws\b",
     None,
     "use read, take, or use"),
    (r"\bbite\b|\bbites\b",
     None,
     "use chunk, step, or case"),
    (r"\bmissions?\b",
     None,
     "use the module's real name or path"),
    (r"\bdigests?\b",
     r"\bsha|hash|checksum|blake|md5|content.digest|RepoDigests",
     "use summary or report (the cryptographic sense stays exempt)"),
    (r"\bmutations?\b",
     None,
     "use change or variation"),
    (r"\bRED\b|\bGREEN\b",
     None,
     "state the invariant in present tense"),
    (r"\boracles?\b",
     None,
     "use reference implementation"),
    (r"\bprobes?\b|\bprobed\b|\bprobing\b",
     r"\bprobe_\b",
     "use test or check"),
    (r"\bdeviation class(?:es)?\b",
     None,
     "describe the actual difference"),
    (r"\btail[- ]mass\b",
     None,
     "use tail probability"),
    (r"\bbatter(?:y|ies)\b",
     None,
     "use checks or suite"),
    (r"\bdonors?\b",
     None,
     "use recorded family or recorded source"),
    (r"\btripwires?\b",
     None,
     "use guard or check"),
    (r"\btwin\b|\btwins\b",
     None,
     "use paired launcher or local variant"),
    (r"\blaws?\b",
     None,
     "use rule, rule set, or contract"),
    (r"\bcensus\b|\bcensuses\b",
     None,
     "use counts or per-element counts"),
    (r"\bconvicts?\b|\bacquits?\b",
     None,
     "state what the comparison shows"),
    (r"\breceipts?\b",
     None,
     "cite the command and its output that prove the claim"),
    (r"\bpostures?\b",
     None,
     "use build variant, configuration, or name the flags"),
    (r"\brungs?\b",
     None,
     "name the tier directly"),
    (r"\bsubstrates?\b",
     None,
     "use base, foundation, or name the component"),
    (r"\bincumbents?\b",
     None,
     "name the existing recording directly"),
]
TATTLETALE = [(re.compile(p, re.I),
               re.compile(e, re.I) if e else None, m) for p, e, m in TATTLETALE]

TEMPORAL = re.compile(
    r"\b(?:currently|previously|formerly|historically|legacy|outdated|"
    r"obsolete|no longer|used to)\b|\bas of \b"
    r"|\bonce (?:the|this|that|it|we|both)\b|\bnow that\b"
    r"|\bonce .+ lands\b|\bwill (?:be|change) .+ soon\b",
    re.I,
)
DESIGN_NARRATION = re.compile(
    r"\bbecause\b|\binstead of\b|\brather than\b|\bwhich is why\b",
    re.I,
)
ARTIFACT_REF = re.compile(
    r"\b(?:SLOP|QA|JIRA|TODO)-?\d{2,}\b|\bboot #\d+\b|\biter-\d+\b",
    re.I,
)

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
FOLD_WIDTH = 140       # the Tattletale PROSE_CAP, reflow fills to this
DOT_MIN = 40           # a final dot at column 40 or less is a short leftover
DOT_MAX = 140          # ... and inside the 140 character prose line
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


GATE_ARCH = re.compile(
    r"sigmoid|gated|gate\.weight|_gate|gate_|gating (?:network|mechanism|"
    r"layer)|swiglu|\bglu\b|\bweight|gate/up|gates/up",
    re.I,
)


def word_issues(text, where, report, semi, allow_gate=False, emdash=False):
    if semi and ";" in text:
        report.append(f"{where}: semicolon in comment: {text[:70]}")
    if emdash and any(d in text for d in EMDASH):
        report.append(f"{where}: em dash is banned, use a comma or colon: "
                      f"{text[:70]}")
    m = BANNED.search(text)
    if m and allow_gate and GATE_ARCH.search(text) \
            and m.group(0).lower().startswith("gate"):
        m = None  # gate/up, sigmoid gate: the model's tensor names
    if m:
        report.append(f"{where}: plain words only, found '{m.group(0)}': {text[:70]}")
    for pattern, exempt, message in TATTLETALE:
        t = pattern.search(text)
        if t and exempt and exempt.search(text):
            continue
        if t:
            report.append(f"{where}: {message} (found '{t.group(0)}'): "
                          f"{text[:70]}")
    t = TEMPORAL.search(text)
    if t:
        report.append(f"{where}: temporal or historical prose, state the "
                      f"present contract (found '{t.group(0)}'): {text[:70]}")
    t = DESIGN_NARRATION.search(text)
    if t:
        report.append(f"{where}: design justification prose, state the "
                      f"contract instead (found '{t.group(0)}'): {text[:70]}")
    t = ARTIFACT_REF.search(text)
    if t:
        report.append(f"{where}: pipeline artifact reference, name the "
                      f"object directly (found '{t.group(0)}'): {text[:70]}")
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


UNIT_KEY = re.compile(r"^[A-Za-z0-9_.\-]+\s{2,}\S|^- |^\* |^\u2022 ")


def unit_split(para):
    """Split a comment paragraph into hanging-indent units.

    A unit starts on a flush line, a bullet, a key-like token (word then
    two or more spaces, the patch-listing and layout-table shape), or a
    dedent. Every other indented line continues the current unit. Units
    wrap independently so structure survives the reflow.
    """
    units, unit, prev_lead = [], [], 0
    for ln in para:
        raw = ln.strip()[1:]
        lead = max(0, len(raw) - len(raw.lstrip()) - 1)
        if not unit or lead < prev_lead or UNIT_KEY.match(raw.strip()):
            if unit:
                units.append(unit)
            unit = [ln]
        else:
            unit.append(ln)
        prev_lead = lead
    if unit:
        units.append(unit)
    return units


def wrap_unit(indent, unit, out, deco=False):
    """Rewrap one unit: fill lines to the 140 cap, keep final dots beyond column 40
    DOT_MIN. The first line keeps the unit lead, continuations keep the
    continuation indent found in the original text. deco re-adds the
    "---" line marker on every emitted line. Key-like units keep their
    column alignment across the reflow."""
    first_raw = unit[0].strip()[1:]
    lead0 = max(0, len(first_raw) - len(first_raw.lstrip()) - 1)
    cont = lead0
    for ln in unit[1:]:
        raw = ln.strip()[1:]
        l = max(0, len(raw) - len(raw.lstrip()) - 1)
        if l > lead0:
            cont = l
            break
    words = []
    for ln in unit:
        words += ln.strip()[1:].strip().split()
    if not words:
        return

    # key-like first token plus its column padding, kept out of the word
    # stream so joins and pull-ups cannot collapse it
    key_m = re.match(r"^(\S+)(\s{2,})(?=\S)", first_raw.strip())
    key_text = None
    if key_m:
        key_text = key_m.group(1) + " " * len(key_m.group(2))
        words = words[1:]
        if not words:
            out.append(indent + prefix_of(lead0, deco) + " " + key_text)
            return

    def prefix_of(lead):
        return "#" + " " * lead + (" ---" if deco else "")

    def jtext(ws, first):
        if first and key_text:
            return key_text + " ".join(ws)
        return " ".join(ws)

    def col(lead, ws, first):
        return len(prefix_of(lead)) + 1 + len(jtext(ws, first))

    start_len = len(out)
    used_first = False
    last_prefix = None
    buf = []
    for w in words:
        lead = lead0 if not used_first else cont
        first = not used_first
        if buf and col(lead, buf + [w], first) > FOLD_WIDTH:
            # a parenthetical is one semantic unit: never split it across
            # lines, close the line before the opening parenthesis
            text = jtext(buf, first)
            open_paren = text.count("(") > text.count(")")
            k = max((i for i, x in enumerate(buf) if "(" in x), default=-1) \
                if open_paren else -1
            if k > 0:
                rest = buf[k:]
                buf = buf[:k]
                last_prefix = prefix_of(lead)
                out.append(indent + prefix_of(lead) + " " + jtext(buf, first))
                used_first = True
                buf = rest
                # fall through: the paren group continues on the new line
            else:
                last_prefix = prefix_of(lead)
                out.append(indent + prefix_of(lead) + " " + jtext(buf, first))
                used_first = True
                buf = [w]
                continue
        buf.append(w)
        if buf[-1].endswith(".") and not buf[-1].endswith("..."):
            c = col(lead, buf, first)
            if c >= DOT_MIN:
                last_prefix = prefix_of(lead)
                out.append(indent + prefix_of(lead) + " " + jtext(buf, first))
                used_first = True
                buf = []
            # c < DOT_MIN: too short, keep pulling the next words up
    if buf:
        lead = lead0 if not used_first else cont
        first = not used_first
        c = col(lead, buf, first)
        # the previous line is only safe to rewrite when it is not the
        # key line of this unit
        prev_rewritable = len(out) > start_len + 1 or not key_text
        if c < DOT_MIN and len(out) > start_len and last_prefix \
                and prev_rewritable:
            # short tail: pull words back from the previous line of this
            # unit until the tail dot is beyond DOT_MIN
            prev_body = out[-1][len(indent):].split("#", 1)[1].strip()
            if prev_body.startswith("--- "):
                prev_body = prev_body[4:]
            prev_words = prev_body.split()
            moved = []
            while prev_words and len(prev_words) > 1:
                moved.insert(0, prev_words.pop())
                buf2 = moved + buf
                c2 = col(lead, buf2, False)
                if DOT_MIN <= c2 <= DOT_MAX:
                    out[-1] = indent + last_prefix + " " + " ".join(prev_words)
                    buf = buf2
                    break
                if c2 > DOT_MAX:
                    prev_words.append(moved.pop(0))
                    break
        moved = []
        while buf and col(lead, buf, first) > FOLD_WIDTH:
            moved.insert(0, buf.pop())
        if prev_rewritable and len(out) > start_len and buf and len(buf) <= 2 \
                and not any(w.endswith(".") for w in buf):
            # no 1-2 word stub lines: pull trailing words from the previous
            # line of this unit until the tail carries 3+ words
            prev_body = out[-1][len(indent):].split("#", 1)[1].strip()
            if prev_body.startswith("--- "):
                prev_body = prev_body[4:]
            prev_words = prev_body.split()
            pulled = []
            while len(buf) + len(pulled) < 3 and len(prev_words) > 3:
                pulled.insert(0, prev_words.pop())
            if pulled:
                out[-1] = indent + last_prefix + " " + " ".join(prev_words)
                buf = pulled + buf
        if buf:
            out.append(indent + prefix_of(lead) + " " + jtext(buf, first))
        buf = moved
        if buf:
            out.append(indent + prefix_of(cont) + " " + " ".join(buf))


def reflow_block(indent, lines):
    """Rewrap one comment block so every final dot is beyond DOT_MIN inside the
    140 character prose line (the Tattletale PROSE_CAP). Hanging-indent
    structure survives as units (see unit_split)."""
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
        # decorated blocks: every line carries the "---" marker. Strip it,
        # reflow the prose, wrap_unit re-adds it on every emitted line.
        deco = all(ln.strip()[1:].strip().startswith("--- ") for ln in para)
        if deco:
            para = [indent + "# " + ln.strip()[1:].strip()[4:] for ln in para]
        for unit in unit_split(para):
            wrap_unit(indent, unit, out, deco=deco)
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
                # dot column counts the # sign plus the comment's own
                # hanging indent, the same convention the reflow uses
                if not patch:
                    raw = ln.strip()[1:]
                    lead = max(0, len(raw) - len(raw.lstrip()) - 1)
                    col = len("# ") + lead + len(body)
                else:
                    col = len(body)
                if STUB < col < DOT_MIN:
                    dest.append(f"{where2}: dot at column {col}, less than {DOT_MIN} "
                                f"(short leftover, fold to next line): {body[:70]}")
                elif col > DOT_MAX:
                    dest.append(f"{where2}: dot at column {col}, outside "
                                f"{DOT_MIN}-{DOT_MAX}: {body[:70]}")
            if re.match(r"^\d{1,2}\.\s+\S", body):
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


def _md_prose_bodies(lines):
    """Per line, the prose body of a markdown paragraph, or None.

    Fences, headings, table rows, list items and blank lines end a
    paragraph. Blockquote markers are stripped so a quoted paragraph
    counts as one paragraph."""
    out, in_fence = [], False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(None)
            continue
        if in_fence:
            out.append(None)
            continue
        s = line.strip()
        if s.startswith(">"):
            s = s[1:].strip()
        if not s or s.startswith(("#", "|", "-", "*")) \
                or re.match(r"^\d+\.\s", s):
            out.append(None)
        else:
            out.append(s)
    return out


def check_docs(path, report, hints):
    """Markdown docs: banned words, dashes, semicolons in prose lines.
    Fenced code blocks are skipped."""
    bodies = _md_prose_bodies(path.read_text().splitlines())
    for i in range(len(bodies) - 1):
        body, nxt = bodies[i], bodies[i + 1]
        if body and nxt and body.endswith(".") and not body.endswith("..."):
            report.append(
                f"{path}:{i+1}: sentence-closing dot at the line end while "
                f"the paragraph continues on the next line, pull the next "
                f"sentence up or close the paragraph there: {body[:70]}")
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


def check_dockerfile(path, report, hints):
    """Dockerfile comment blocks: banned words, dashes, semicolons in the
    comment text, dot window, reflow quality. RUN code lines keep their
    shell semantics, so the whole-line launcher scans do not apply."""
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("#") and not s.startswith("#!"):
            if any(d in s for d in EMDASH):
                report.append(f"{path}:{i}: em dash is banned, use a comma "
                              f"or colon: {s[:70]}")
            if ";" in s:
                report.append(f"{path}:{i}: semicolon in comment: {s[:70]}")
    check_blocks(path, lines, report, hints, patch=False)


def default_files(root):
    files = []
    for pat in ["serve_vllm_GLM53f-exl3-*.kk.beta.sh",
                "internal/serve-glm53f-exl3-*-local.sh",
                "patches/*.patch",
                "README.md",
                "Dockerfile"]:
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
        elif f.name == "Dockerfile":
            if fix:
                n = fix_file(f, report)
                if n:
                    print(f"{f}: reflowed {n} comment block(s)")
            check_dockerfile(f, report, hints)
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
