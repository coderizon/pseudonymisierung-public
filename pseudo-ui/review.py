"""
review.py — Freigabeschritt für pseudo-ui.

Die Pipeline (`pseudo`) liefert pro Durchlauf findings.json (Kandidaten und
Beschreibungen) und pseudonymized.txt (Text mit [Ersetzungen]).  Positionen
der Ersetzungen liefert sie nicht.  Dieses Modul

  1. baut aus findings.json die Liste der Einträge (ein Eintrag pro Original),
  2. ordnet jede [Ersetzung] im Pipeline-Text einem Eintrag zu — per Abgleich
     mit dem Originaltext, sonst über die Kontext-Schnipsel der Findings,
  3. wendet Projekt-Korrekturen aus früheren Freigaben an,
  4. erzeugt nach der Freigabe den Endtext neu (pseudonymized.txt); der
     unveränderte Pipeline-Text bleibt als pseudonymized.auto.txt erhalten.

Nur Standardbibliothek.  Für PDF wird `pdftotext` benutzt, falls vorhanden.
"""

import json
import os
import re
import subprocess
import threading
import time
import zipfile
from xml.etree import ElementTree as ET

STATE_DIR = os.path.abspath(os.environ.get("PSEUDO_UI_STATE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ui")))
PROJECTS_DIR = os.path.join(STATE_DIR, "projects")
REVIEW_FILE = "review.json"
AUTO_FILE = "pseudonymized.auto.txt"
FINAL_FILE = "pseudonymized.txt"
FINDINGS_FILE = "findings.json"

TOKEN_RE = re.compile(r"\[([^\[\]\n]{1,160})\]")
PERSON_LABEL_RE = re.compile(r"^PERSON_([A-Z]+|\d+)$")

CATEGORIES = [
    "PERSON", "ORG_CLIENT", "ORG_PARTNER", "ORG_UNIT", "LOCATION", "PRODUCT",
    "TECHNOLOGY", "PROCESS_ARTIFACT", "STRATEGY", "BUSINESS_METRIC", "CONTRACT",
    "FUNCTION_TITLE", "INDUSTRY", "NEGATIVE_KNOWHOW", "TEMPORAL", "CONTACT", "SONSTIGE",
]
CATEGORY_LABELS = {
    "PERSON": "Personen",
    "ORG_CLIENT": "Unternehmen (Kunden, Markt)",
    "ORG_PARTNER": "Partner und Lieferanten",
    "ORG_UNIT": "Organisationseinheiten",
    "LOCATION": "Orte",
    "PRODUCT": "Produkte",
    "TECHNOLOGY": "Technologien",
    "PROCESS_ARTIFACT": "Prozesse und Artefakte",
    "STRATEGY": "Strategien",
    "BUSINESS_METRIC": "Kennzahlen",
    "CONTRACT": "Verträge",
    "FUNCTION_TITLE": "Funktionen und Titel",
    "INDUSTRY": "Branchen",
    "NEGATIVE_KNOWHOW": "Negatives Know-how",
    "TEMPORAL": "Zeitangaben",
    "CONTACT": "Kontaktdaten",
    "SONSTIGE": "Sonstige",
}
CONTACT_REPLACEMENT = "KONTAKTDATEN ENTFERNT"

_lock = threading.Lock()


# ------------------------------------------------------------------ utils ---

def norm_key(text):
    t = re.sub(r"[\s.\-_/·'’\"„“]+", "", str(text).casefold())
    return t or str(text).casefold()


def _ws_norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ------------------------------------------------------------ source text ---

def source_text(path):
    """Text des Eingabedokuments, so nah wie möglich an der Pipeline-Ingestion."""
    if not path or not os.path.isfile(path):
        return None
    ext = path.rsplit(".", 1)[-1].lower()
    try:
        if ext in ("txt", "md"):
            return _read(path)
        if ext == "docx":
            return _docx_text(path)
        if ext == "xlsx":
            return _xlsx_text(path)
        if ext == "pdf":
            r = subprocess.run(["pdftotext", "-enc", "UTF-8", path, "-"],
                               capture_output=True, timeout=120)
            return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None
    except Exception:
        return None
    return None


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paras = []
    for p in root.iter(_W + "p"):
        buf = []
        for el in p.iter():
            if el.tag == _W + "t":
                buf.append(el.text or "")
            elif el.tag == _W + "tab":
                buf.append("\t")
            elif el.tag in (_W + "br", _W + "cr"):
                buf.append("\n")
        paras.append("".join(buf))
    return "\n".join(paras)


_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _xlsx_text(path):
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(_S + "si"):
                shared.append("".join(t.text or "" for t in si.iter(_S + "t")))
        lines = []
        for name in sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet")):
            for row in ET.fromstring(z.read(name)).iter(_S + "row"):
                cells = []
                for c in row.iter(_S + "c"):
                    v = c.find(_S + "v")
                    if v is None or v.text is None:
                        cells.append("")
                        continue
                    if c.get("t") == "s":
                        cells.append(shared[int(v.text)] if int(v.text) < len(shared) else "")
                    else:
                        cells.append(v.text)
                lines.append("\t".join(cells))
    return "\n".join(lines)


# --------------------------------------------------------------- findings ---

def _entries_from_findings(findings):
    """Ein Eintrag je Original (Schlüssel = normalisierter Text)."""
    rejected = {(c.get("text", ""), c.get("category", "")) for c in findings.get("pass2_rejected", [])}
    final = [c for c in findings.get("pass1_candidates", []) if (c.get("text", ""), c.get("category", "")) not in rejected]
    final += findings.get("pass2_added", [])
    desc = {}
    for d in findings.get("klasse1_descriptions", []) + findings.get("klasse2_descriptions", []):
        desc[(d.get("original", ""), d.get("category", ""))] = d.get("description", "")

    ents = {}
    order = []

    def add(c, rejected_only=False):
        text = str(c.get("text", "")).strip()
        cat = str(c.get("category", "") or "SONSTIGE")
        if not text:
            return
        key = norm_key(text)
        d = desc.get((text, cat))
        if key not in ents:
            ents[key] = {
                "key": key, "text": text, "category": cat, "suggested": d,
                "context": c.get("context", ""), "categories": [cat],
                "rejected": rejected_only, "count": 0, "source": "pipeline",
            }
            order.append(key)
        else:
            e = ents[key]
            if cat not in e["categories"]:
                e["categories"].append(cat)
            # Kategorie mit Beschreibung gewinnt; verworfene Kandidaten ändern nichts
            if not rejected_only and d and not e["suggested"]:
                e["suggested"], e["category"] = d, cat
            if not rejected_only and e["rejected"]:
                e["rejected"], e["category"] = False, cat
            if not e["context"] and c.get("context"):
                e["context"] = c["context"]

    for c in final:
        add(c)
    for c in findings.get("pass2_rejected", []):
        add(c, rejected_only=True)

    for e in ents.values():
        if e["category"] == "CONTACT" and not e["suggested"]:
            e["suggested"] = CONTACT_REPLACEMENT
        e["replacement"] = e["suggested"] or ""
        e["action"] = "keep" if e["rejected"] else "replace"
    return [ents[k] for k in order]


# -------------------------------------------------------------- alignment ---

def _lookahead(orig, i, auto, j, n=14):
    """Vergleicht die nächsten n Nicht-Leerzeichen; Abbruch am nächsten Token im Pipeline-Text.
    Rückgabe: (Anzahl verglichener Zeichen, alles gleich?)."""
    cnt = 0
    while cnt < n:
        while j < len(auto) and auto[j].isspace():
            j += 1
        while i < len(orig) and orig[i].isspace():
            i += 1
        if j >= len(auto) or auto[j] == "[":
            return cnt, True
        if i >= len(orig) or orig[i] != auto[j]:
            return cnt, False
        cnt += 1
        i += 1
        j += 1
    return cnt, True


def _resync(orig, i, auto, j, known, max_skip=400):
    """Länge des Originalabschnitts, der dem Token an auto[j-len:j] entspricht."""
    for s in known:
        if orig.startswith(s, i):
            cnt, ok = _lookahead(orig, i + len(s), auto, j)
            if ok:
                return len(s)
    lo = orig[i:i + 400].casefold()
    for s in known:
        if lo.startswith(s.casefold()):
            cnt, ok = _lookahead(orig, i + len(s), auto, j)
            if ok:
                return len(s)
    best = None
    for k in range(1, min(max_skip, len(orig) - i) + 1):
        cnt, ok = _lookahead(orig, i + k, auto, j)
        if ok and cnt >= 3:
            # Wortgrenze bevorzugen
            if i + k >= len(orig) or not orig[i + k - 1].isalnum() or not orig[i + k].isalnum():
                return k
            best = best or k
    return best


def align(orig, auto, known):
    """Ordnet jedem [Token] im Pipeline-Text den Originalabschnitt zu. None bei Fehlschlag."""
    known = sorted(set(k for k in known if k), key=len, reverse=True)
    tokens = []
    i = j = 0
    n, m = len(orig), len(auto)
    while j < m:
        c = auto[j]
        if c == "[":
            mt = TOKEN_RE.match(auto, j)
            if mt and not orig.startswith(mt.group(0), i):
                tok = mt.group(0)
                after = j + len(tok)
                k = _resync(orig, i, auto, after, known)
                if k is None:
                    return None
                tokens.append({"pos": j, "len": len(tok), "surface": orig[i:i + k]})
                i += k
                j = after
                continue
        if i < n and orig[i] == c:
            i += 1
            j += 1
            continue
        if c.isspace():
            j += 1
            continue
        if i < n and orig[i].isspace():
            i += 1
            continue
        # kleiner Versatz (Zeichen von der Ingestion normalisiert): beide Seiten leicht verschieben
        fixed = False
        for di in range(0, 6):
            for dj in range(0, 6):
                if di == dj == 0:
                    continue
                cnt, ok = _lookahead(orig, i + di, auto, j + dj)
                if ok and cnt >= 6:
                    i += di
                    j += dj
                    fixed = True
                    break
            if fixed:
                break
        if not fixed:
            return None
    return tokens


def _assign_aligned(tokens, auto, ents):
    """Token → Eintrag über den Originalabschnitt."""
    by_key = {e["key"]: e for e in ents}
    for t in tokens:
        desc = auto[t["pos"] + 1:t["pos"] + t["len"] - 1]
        surface = t["surface"].strip()
        key = norm_key(surface)
        e = by_key.get(key)
        if not e:
            for cand in ents:
                if cand["suggested"] == desc and (key in cand["key"] or cand["key"] in key):
                    e = cand
                    break
        if not e:
            for cand in ents:
                if key and (key in cand["key"] or cand["key"] in key) and len(key) >= 3:
                    e = cand
                    break
        if not e:
            cat = next((c["category"] for c in ents if c["suggested"] == desc), None)
            if not cat:
                cat = "PERSON" if PERSON_LABEL_RE.match(desc) else "SONSTIGE"
            e = {"key": key or f"tok{t['pos']}", "text": surface, "category": cat, "suggested": desc,
                 "context": "", "categories": [cat], "rejected": False, "count": 0,
                 "source": "pipeline", "replacement": desc, "action": "replace"}
            ents.append(e)
            by_key[e["key"]] = e
        if not e["suggested"]:
            e["suggested"] = desc
            e["replacement"] = desc
        if e["rejected"]:
            e["rejected"], e["action"] = False, "replace"
        e["count"] += 1
        t["key"] = e["key"]
        t["desc"] = desc
    return tokens


def _mask_texts(s, texts):
    for t in texts:
        s = s.replace(t, "\x00")
    return s


def _assign_by_context(auto, ents):
    """Ohne Originaltext: Token → Eintrag über Beschreibung, Kontext-Schnipsel und Reihenfolge."""
    tokens = []
    ambiguous = 0
    texts = sorted({e["text"] for e in ents}, key=len, reverse=True)
    persons = [e for e in ents if e["category"] == "PERSON" and not e["rejected"]]
    # Personen-Kennungen vergibt die Pipeline in Reihenfolge des ersten Auftretens; die Kandidaten
    # stehen in findings.json in Dokumentreihenfolge. Passt beides zusammen, ist die Zuordnung eindeutig.
    labels = []
    for mt in TOKEN_RE.finditer(auto):
        if PERSON_LABEL_RE.match(mt.group(1)) and mt.group(1) not in labels:
            labels.append(mt.group(1))
    by_order = {}
    if labels and len(labels) == len(persons) and labels == sorted(labels, key=_label_value):
        by_order = dict(zip(labels, persons))

    for mt in TOKEN_RE.finditer(auto):
        desc = mt.group(1)
        pos = mt.start()
        t = {"pos": pos, "len": len(mt.group(0)), "surface": None, "desc": desc}
        cands = [e for e in ents if e["suggested"] == desc]
        if PERSON_LABEL_RE.match(desc) and not cands:
            cands = [e for e in persons if not e["suggested"]]
        pick = None
        if len(cands) == 1:
            pick = cands[0]
        elif len(cands) > 1:
            before = auto[max(0, pos - 120):pos]
            before = re.sub(r"^[^\[\]]*\]", "\x00", before)
            before = _ws_norm(_mask_texts(TOKEN_RE.sub("\x00", before), texts))
            after = auto[mt.end():mt.end() + 120]
            after = re.sub(r"\[[^\[\]]*$", "\x00", after)
            after = _ws_norm(_mask_texts(TOKEN_RE.sub("\x00", after), texts))
            scored = []
            for e in cands:
                ctx = e.get("context") or ""
                idx = ctx.find(e["text"])
                if idx < 0:
                    continue
                pre = _ws_norm(_mask_texts(ctx[:idx], texts))[-30:]
                suf = _ws_norm(_mask_texts(ctx[idx + len(e["text"]):], texts))[:30]
                scored.append((_common_suffix(before, pre) + _common_prefix(after, suf), e))
            scored.sort(key=lambda x: -x[0])
            if scored and scored[0][0] >= 4 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                pick = scored[0][1]
            elif desc in by_order and by_order[desc] in cands:
                pick = by_order[desc]
        if not pick and cands and desc not in by_order:
            same = [m for m in TOKEN_RE.finditer(auto) if m.group(1) == desc]
            if len(same) == len(cands):
                pick = cands[[m.start() for m in same].index(pos)]
                t["guess"] = True
        if pick:
            if not pick["suggested"]:
                pick["suggested"] = desc
                pick["replacement"] = desc
            pick["count"] += 1
            t["key"] = pick["key"]
        elif cands:
            t["key"] = None
            t["ambiguous"] = [e["key"] for e in cands]
            for e in cands:
                e["amb"] = e.get("amb", 0) + 1
            ambiguous += 1
        else:
            t["key"] = None
        tokens.append(t)
    return tokens, ambiguous


def _label_value(label):
    s = label[7:]
    if s.isdigit():
        return int(s)
    v = 0
    for ch in s:
        v = v * 26 + (ord(ch) - 64)
    return v


def _common_suffix(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def _common_prefix(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


# ------------------------------------------------------- project dictionary ---

def project_path(project):
    if not project or not re.match(r"^[A-Za-z0-9._-]{1,64}$", project):
        return None
    return os.path.join(PROJECTS_DIR, project + ".json")


def load_project(project):
    p = project_path(project)
    return (_load_json(p, None) or {"entries": {}}) if p else {"entries": {}}


def apply_project(ents, project):
    entries = load_project(project).get("entries", {})
    n = 0
    for e in ents:
        d = entries.get(e["key"])
        if not d:
            continue
        changed = False
        if d.get("category") and d["category"] != e["category"]:
            e["category"], changed = d["category"], True
        if d.get("action") in ("keep", "replace") and d["action"] != e["action"]:
            e["action"], changed = d["action"], True
        # Personen-Kennungen vergibt die Pipeline projektweit selbst; nur Beschreibungen übernehmen
        if e["category"] != "PERSON" and d.get("replacement") and d["replacement"] != e["replacement"]:
            e["replacement"], changed = d["replacement"], True
        if changed:
            e["source"] = "projekt"
            n += 1
    return n


def save_project(project, ents):
    p = project_path(project)
    if not p:
        return
    data = load_project(project)
    entries = data.setdefault("entries", {})
    now = time.time()
    for e in ents:
        if e.get("source") == "auto" or not e.get("text"):
            continue
        rec = {"text": e["text"], "category": e["category"], "action": e["action"], "updated": now}
        if e["category"] != "PERSON":
            rec["replacement"] = e["replacement"]
        entries[e["key"]] = rec
    data["updated"] = now
    _save_json(p, data)


# ---------------------------------------------------------------- similar ---

def _mark_similar(ents):
    keys = [e["key"] for e in ents]
    last = {}
    for e in ents:
        if e["category"] == "PERSON":
            w = e["text"].split()
            if w:
                last.setdefault(norm_key(w[-1]), []).append(e["key"])
    for e in ents:
        sim = []
        k = e["key"]
        for o in keys:
            if o == k:
                continue
            if len(k) >= 3 and len(o) >= 3 and (k.startswith(o) or o.startswith(k) or k.endswith(o) or o.endswith(k)):
                sim.append(o)
        if e["category"] == "PERSON":
            w = e["text"].split()
            if w:
                for o in last.get(norm_key(w[-1]), []):
                    if o != k and o not in sim:
                        sim.append(o)
        e["similar"] = sim[:4]


# ---------------------------------------------------------------- prepare ---

def prepare(dirpath, project=None, input_path=None, force=False, inputs_dir=None):
    """Erzeugt review.json für einen Ergebnisordner (oder liefert die vorhandene)."""
    with _lock:
        rpath = os.path.join(dirpath, REVIEW_FILE)
        existing = _load_json(rpath)
        if existing and not force:
            return existing
        findings = _load_json(os.path.join(dirpath, FINDINGS_FILE))
        if not findings:
            return None
        auto_path = os.path.join(dirpath, AUTO_FILE)
        final_path = os.path.join(dirpath, FINAL_FILE)
        if not os.path.exists(auto_path):
            if not os.path.exists(final_path):
                return None
            os.replace(final_path, auto_path)
            _write(final_path, _read(auto_path))
        auto = _read(auto_path)

        if input_path is None and inputs_dir and findings.get("file"):
            cand = os.path.join(inputs_dir, os.path.basename(str(findings["file"])))
            if os.path.isfile(cand):
                input_path = cand
        ents = _entries_from_findings(findings)
        warnings = []
        mode = "context"
        tokens = None
        orig = source_text(input_path)
        if orig is not None:
            known = [e["text"] for e in ents]
            tokens = align(orig, auto, known)
            if tokens is not None:
                tokens = _assign_aligned(tokens, auto, ents)
                mode = "aligned"
            else:
                warnings.append("Abgleich mit dem Originaldokument nicht möglich, Zuordnung über Kontext.")
        if tokens is None:
            tokens, amb = _assign_by_context(auto, ents)
            guessed = sum(1 for t in tokens if t.get("guess"))
            if guessed:
                warnings.append(f"{guessed} Ersetzungen wurden nach Reihenfolge zugeordnet (Originaldokument nicht mehr vorhanden). "
                                "Bitte Vorschau prüfen.")
            if amb:
                warnings.append(f"{amb} Ersetzungen konnten keinem Eintrag eindeutig zugeordnet werden "
                                "(gleiche Beschreibung bei mehreren Einträgen). Änderungen gelten dort nur, "
                                "wenn alle betroffenen Einträge gleich geändert werden.")
        # Einträge ohne Ersetzung im Text hat die Pipeline bewusst nicht ersetzt: so lassen
        for e in ents:
            if e["count"] == 0 and not e.get("amb") and not e["rejected"]:
                e["action"] = "keep"
        _mark_similar(ents)
        if project is None:
            project = (existing or {}).get("project") or "default"
        applied = apply_project(ents, project)
        review = {
            "version": 1, "status": "pending", "mode": mode, "prepared": time.time(),
            "approved": None, "project": project, "file": findings.get("file", ""),
            "model": findings.get("model", ""), "warnings": warnings,
            "project_applied": applied, "entities": ents, "tokens": tokens,
        }
        _save_json(rpath, review)
        return review


def load(dirpath):
    return _load_json(os.path.join(dirpath, REVIEW_FILE))


def summary(dirpath):
    r = load(dirpath)
    if not r:
        return None
    ents = r.get("entities", [])
    return {"status": r.get("status"), "approved": r.get("approved"), "mode": r.get("mode"),
            "entities": len([e for e in ents if not e.get("rejected") or e.get("action") == "replace"]),
            "project": r.get("project")}


def public(review):
    """Sicht für die Oberfläche (ohne Token-Positionen)."""
    r = dict(review)
    r.pop("tokens", None)
    r["categories"] = [{"id": c, "label": CATEGORY_LABELS.get(c, c)} for c in CATEGORIES]
    r["descriptions"] = sorted({e["replacement"] for e in r["entities"] if e.get("replacement") and e["category"] != "PERSON"})
    return r


# ----------------------------------------------------------------- render ---

def _next_person_label(ents):
    labels = [e["replacement"] for e in ents if e["category"] == "PERSON" and PERSON_LABEL_RE.match(e.get("replacement") or "")]
    nums = [int(l[7:]) for l in labels if l[7:].isdigit()]
    if nums and not any(l[7:].isalpha() for l in labels):
        return f"PERSON_{max(nums) + 1}"
    alphas = [l[7:] for l in labels if l[7:].isalpha()]

    def val(s):
        v = 0
        for ch in s:
            v = v * 26 + (ord(ch) - 64)
        return v

    def name(v):
        s = ""
        while v > 0:
            v, r = divmod(v - 1, 26)
            s = chr(65 + r) + s
        return s

    return "PERSON_" + name(max([val(a) for a in alphas] + [0]) + 1)


def _apply_edits(review, edits, added, project=None):
    by_key = {e["key"]: e for e in review["entities"]}
    for ed in edits or []:
        e = by_key.get(str(ed.get("key", "")))
        if not e:
            continue
        cat = str(ed.get("category") or e["category"])
        e["category"] = cat if cat in CATEGORIES else e["category"]
        e["action"] = "keep" if ed.get("action") == "keep" else "replace"
        rep = str(ed.get("replacement") or "").strip()
        if e["category"] == "CONTACT" and not rep:
            rep = CONTACT_REPLACEMENT
        e["replacement"] = rep
        if e["source"] == "pipeline" and (e["replacement"] != (e["suggested"] or "") or e["action"] != ("keep" if e["rejected"] else "replace")):
            e["source"] = "manuell"
    for ad in added or []:
        text = str(ad.get("text") or "").strip()
        if not text:
            continue
        key = norm_key(text)
        cat = str(ad.get("category") or "SONSTIGE")
        cat = cat if cat in CATEGORIES else "SONSTIGE"
        rep = str(ad.get("replacement") or "").strip()
        if key in by_key:
            e = by_key[key]
            e["category"], e["action"] = cat, "replace"
            if rep:
                e["replacement"] = rep
            e["source"] = "manuell"
            continue
        e = {"key": key, "text": text, "category": cat, "suggested": None, "context": "",
             "categories": [cat], "rejected": False, "count": 0, "source": "manuell",
             "replacement": rep, "action": "replace", "similar": []}
        review["entities"].append(e)
        by_key[key] = e
    for e in review["entities"]:
        if e["action"] == "replace" and not e["replacement"]:
            if e["category"] == "PERSON":
                e["replacement"] = _next_person_label(review["entities"])
            elif e["category"] == "CONTACT":
                e["replacement"] = CONTACT_REPLACEMENT
    if project:
        review["project"] = project


class RestnamenFehler(ValueError):
    """Eine Freigabe darf bestätigte, zu ersetzende Namen nicht stehen lassen."""


def _namensmuster(text):
    return r"(?<!\w)" + r"[ \t]+".join(re.escape(t) for t in text.split()) + r"(?!\w)"


def restnamen(review, text):
    reste = []
    behalten = [m.span() for e in review["entities"]
                if e["category"] == "PERSON" and e["action"] == "keep" and e["text"].strip()
                for m in re.finditer(_namensmuster(e["text"]), text, re.IGNORECASE)]
    for e in review["entities"]:
        if e["category"] != "PERSON" or e["action"] != "replace" or not e["text"].strip():
            continue
        anzahl = sum(1 for m in re.finditer(_namensmuster(e["text"]), text, re.IGNORECASE)
                     if not any(a <= m.start() and m.end() <= b for a, b in behalten))
        if anzahl:
            reste.append({"text": e["text"], "anzahl": anzahl})
    return reste


def render(review, auto):
    """Endtext aus Pipeline-Text, Token-Zuordnung und Einträgen."""
    by_key = {e["key"]: e for e in review["entities"]}
    tokens = sorted(review.get("tokens") or [], key=lambda t: t["pos"])
    out = []
    pos = 0
    stats = {"replaced": 0, "restored": 0, "unchanged": 0, "searched": 0}

    # Bestätigte Personen auch dann vervollständigen, wenn erst ein Teil der
    # Fundstellen ersetzt wurde. Andere Kategorien behalten ihre Kontextauswahl.
    search = [e for e in review["entities"]
              if e["action"] == "replace" and e["replacement"]
              and (e["count"] == 0 or e["category"] == "PERSON") and len(e["text"]) >= 2]
    # Längere, ausdrücklich beibehaltene Namen nicht über einen Teilnamen ersetzen.
    search += [e for e in review["entities"] if e["category"] == "PERSON"
               and e["action"] == "keep" and len(e["text"]) >= 2]
    search.sort(key=lambda e: -len(e["text"]))
    muster = re.compile("|".join(
        f"(?P<n{i}>{_namensmuster(e['text']) if e['category'] == 'PERSON' else r'(?<!\w)' + re.escape(e['text']) + r'(?!\w)'})"
        for i, e in enumerate(search)
    ), re.IGNORECASE) if search else None

    def segment(s):
        if muster is None or not s:
            return s
        def ersetzen(m):
            e = search[int(m.lastgroup[1:])]
            if e["action"] == "keep":
                return m.group()
            stats["searched"] += 1
            return "[" + e["replacement"] + "]"
        # Ein Durchgang verhindert, dass neue Kennungen erneut ersetzt werden.
        # Bereits vorhandene Klammerausdrücke bleiben intakt; Restnamen darin
        # werden anschließend gemeldet und verhindern eine blinde Freigabe.
        teile = []
        cursor = 0
        for t in TOKEN_RE.finditer(s):
            teile.append(muster.sub(ersetzen, s[cursor:t.start()]))
            teile.append(t.group())
            cursor = t.end()
        teile.append(muster.sub(ersetzen, s[cursor:]))
        return "".join(teile)

    for t in tokens:
        out.append(segment(auto[pos:t["pos"]]))
        tok = auto[t["pos"]:t["pos"] + t["len"]]
        desc = t.get("desc") or tok[1:-1]
        e = by_key.get(t.get("key") or "")
        cands = [by_key[k] for k in t.get("ambiguous") or [] if k in by_key]
        if not e and cands:
            decisions = {(c["action"], c["replacement"] if c["action"] == "replace" else "") for c in cands}
            if len(decisions) == 1:
                e = cands[0]
        if e is None:
            out.append(tok)
            stats["unchanged"] += 1
        elif e["action"] == "keep":
            out.append(t.get("surface") or e["text"])
            stats["restored"] += 1
        else:
            rep = e["replacement"] or desc
            out.append("[" + rep + "]")
            stats["replaced"] += 1
        pos = t["pos"] + t["len"]
    out.append(segment(auto[pos:]))
    text = "".join(out)
    stats["restnamen"] = restnamen(review, text)
    return text, stats


def preview(dirpath, edits, added, project=None):
    review = load(dirpath)
    if not review:
        return None, None
    _apply_edits(review, edits, added, project)
    auto = _read(os.path.join(dirpath, AUTO_FILE))
    return render(review, auto)


def approve(dirpath, edits, added, project=None):
    with _lock:
        review = load(dirpath)
        if not review:
            return None, None
        _apply_edits(review, edits, added, project)
        auto = _read(os.path.join(dirpath, AUTO_FILE))
        text, stats = render(review, auto)
        if stats["restnamen"]:
            namen = ", ".join(f"{e['text']} ({e['anzahl']}×)" for e in stats["restnamen"])
            raise RestnamenFehler(f"Freigabe gestoppt: Zu ersetzende Namen stehen noch im Text: {namen}. Bitte Vorschau und Ersetzungen prüfen.")
        _write(os.path.join(dirpath, FINAL_FILE), text)
        review["namenspruefung_version"] = 1
        review["status"] = "approved"
        review["approved"] = time.time()
        review["stats"] = stats
        _save_json(os.path.join(dirpath, REVIEW_FILE), review)
        try:
            save_project(review.get("project"), review["entities"])
        except OSError:
            pass
        return review, stats


def reset(dirpath, project=None, input_path=None, inputs_dir=None):
    """Zurück zum Vorschlag der Pipeline (Projekt-Korrekturen werden erneut angewendet)."""
    with _lock:
        auto_path = os.path.join(dirpath, AUTO_FILE)
        if os.path.exists(auto_path):
            _write(os.path.join(dirpath, FINAL_FILE), _read(auto_path))
        old = load(dirpath) or {}
        try:
            os.remove(os.path.join(dirpath, REVIEW_FILE))
        except OSError:
            pass
    return prepare(dirpath, project or old.get("project"), input_path, inputs_dir=inputs_dir)
