#!/usr/bin/env python3
"""
pseudo-ui — local web front-end for the pseudonymisation engine.

Wraps the `pseudo` CLI: upload documents to data/ui/inputs, start runs, follow the
log live, browse and download results from data/ui/outputs.  Standard library only.

    python3 app.py                      # serve on 127.0.0.1:8789
    python3 app.py set-password         # set / change the UI password
    python3 app.py generate-password    # set a random password and print it
"""

import getpass
import hashlib
import hmac
import html
import io
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.abspath(os.environ.get("PSEUDO_UI_STATE_DIR", os.path.join(os.path.dirname(APP_DIR), "data", "ui")))
INPUTS = os.path.join(STATE_DIR, "inputs")
OUTPUTS = os.path.join(STATE_DIR, "outputs")
STATIC = os.path.join(APP_DIR, "static")
LOGS = os.path.join(STATE_DIR, "logs")
CONFIG_PATH = os.path.join(STATE_DIR, "config.json")
PSEUDO_BIN = os.environ.get("PSEUDO_BIN", os.path.join(os.path.dirname(APP_DIR), "scripts", "pseudo"))

HOSTS = [h.strip() for h in os.environ.get("PSEUDO_UI_HOST", "127.0.0.1").split(",") if h.strip()]
PORT = int(os.environ.get("PSEUDO_UI_PORT", "8789"))

ALLOWED_EXT = {"pdf", "docx", "xlsx", "md", "txt"}
MAX_UPLOAD = 200 * 1024 * 1024
SESSION_TTL = 12 * 3600
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW = 15 * 60
PROJECT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
SAFE_NAME_RE = re.compile(r"[^\w .()\-\[\]äöüÄÖÜß]", re.UNICODE)


# ---------------------------------------------------------------- config ---

def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, CONFIG_PATH)


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), 300_000
    ).hex()
    return salt, digest


def set_password(password):
    cfg = load_config()
    salt, digest = hash_password(password)
    cfg["salt"] = salt
    cfg["hash"] = digest
    cfg.setdefault("secret", secrets.token_hex(32))
    save_config(cfg)


def verify_password(password):
    cfg = load_config()
    if "hash" not in cfg:
        return False
    _, digest = hash_password(password, cfg["salt"])
    return hmac.compare_digest(digest, cfg["hash"])


def secret_key():
    cfg = load_config()
    if "secret" not in cfg:
        cfg["secret"] = secrets.token_hex(32)
        save_config(cfg)
    return cfg["secret"].encode()


# -------------------------------------------------------------- sessions ---

def make_session():
    expires = int(time.time()) + SESSION_TTL
    nonce = secrets.token_hex(16)
    payload = f"{expires}.{nonce}"
    sig = hmac.new(secret_key(), payload.encode(), "sha256").hexdigest()
    return f"{payload}.{sig}"


def valid_session(token):
    if not token:
        return False
    parts = token.split(".")
    if len(parts) != 3:
        return False
    payload = ".".join(parts[:2])
    sig = hmac.new(secret_key(), payload.encode(), "sha256").hexdigest()
    if not hmac.compare_digest(sig, parts[2]):
        return False
    try:
        return int(parts[0]) > time.time()
    except ValueError:
        return False


_login_failures = {}
_login_lock = threading.Lock()


def login_blocked(ip):
    with _login_lock:
        now = time.time()
        attempts = [t for t in _login_failures.get(ip, []) if now - t < LOGIN_WINDOW]
        _login_failures[ip] = attempts
        return len(attempts) >= LOGIN_MAX_FAILURES


def login_failed(ip):
    with _login_lock:
        _login_failures.setdefault(ip, []).append(time.time())


# ---------------------------------------------------------------- runner ---

class Runner:
    """Runs one `pseudo` job at a time and keeps its log on disk."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.job = None  # dict: file, project, log, started, ended, rc

    def start(self, filename, project):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return False, "Es läuft bereits ein Durchlauf."
            path = os.path.join(INPUTS, filename)
            if not os.path.isfile(path):
                return False, "Datei nicht gefunden."
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(LOGS, f"{stamp}.log")
            log = open(log_path, "wb")
            cmd = [PSEUDO_BIN, path, "--project", project]
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, cwd=INPUTS,
                )
            except OSError as e:
                log.close()
                return False, f"Start fehlgeschlagen: {e}"
            self.job = {
                "file": filename, "project": project, "log": log_path,
                "started": time.time(), "ended": None, "rc": None,
            }
            threading.Thread(target=self._wait, args=(self.proc, log), daemon=True).start()
            return True, "Durchlauf gestartet."

    def _wait(self, proc, log):
        rc = proc.wait()
        log.close()
        with self.lock:
            if self.job and self.job["rc"] is None:
                self.job["rc"] = rc
                self.job["ended"] = time.time()
            job = dict(self.job) if self.job else None
        if rc == 0 and job:
            try:
                stem = os.path.splitext(job["file"])[0]
                outs = [o for o in list_outputs() if o["source"] == stem]
                if outs:
                    name = outs[0]["name"]
                    review.prepare(os.path.join(OUTPUTS, name), job["project"],
                                   os.path.join(INPUTS, job["file"]))
                    with self.lock:
                        if self.job and self.job["started"] == job["started"]:
                            self.job["output"] = name
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"review prepare failed: {e}\n")

    def status(self):
        with self.lock:
            job = dict(self.job) if self.job else None
        running = bool(self.proc and self.proc.poll() is None)
        info = {"running": running, "job": None}
        if job:
            info["job"] = {
                "file": job["file"], "project": job["project"],
                "started": job["started"], "ended": job["ended"], "rc": job["rc"],
                "elapsed": (job["ended"] or time.time()) - job["started"],
                "output": job.get("output"),
            }
            info["progress"] = parse_progress(self.read_log(0)[0])
        return info

    def read_log(self, offset):
        with self.lock:
            job = self.job
        if not job:
            return "", 0
        try:
            with open(job["log"], "rb") as f:
                f.seek(offset)
                data = f.read()
                return data.decode("utf-8", "replace"), offset + len(data)
        except FileNotFoundError:
            return "", 0


STEP_RE = re.compile(r"=== Step (\d+)/(\d+): (.+?) ===")
CHUNK_RE = re.compile(r"chunk (\d+)/(\d+)")


def parse_progress(text):
    steps = STEP_RE.findall(text)
    if not steps:
        return {"step": 0, "steps": 6, "label": "Container wird gestartet", "chunk": None}
    n, total, label = steps[-1]
    tail = text[text.rfind("=== Step"):]
    chunks = CHUNK_RE.findall(tail)
    chunk = None
    if chunks:
        c, ct = chunks[-1]
        chunk = {"done": int(c), "total": int(ct)}
    return {"step": int(n), "steps": int(total), "label": label, "chunk": chunk}


runner = Runner()


# ------------------------------------------------------------- listings ---

def safe_join(base, name):
    name = unquote(name)
    if not name or "/" in name or name in (".", "..") or name.startswith("."):
        return None
    path = os.path.realpath(os.path.join(base, name))
    if os.path.dirname(path) != os.path.realpath(base):
        return None
    return path


def list_inputs():
    items = []
    if not os.path.isdir(INPUTS):
        return items
    for name in sorted(os.listdir(INPUTS)):
        path = os.path.join(INPUTS, name)
        if not os.path.isfile(path) or name.startswith("."):
            continue
        st = os.stat(path)
        items.append({"name": name, "size": st.st_size, "mtime": st.st_mtime,
                      "ok": name.rsplit(".", 1)[-1].lower() in ALLOWED_EXT})
    return items


OUT_RE = re.compile(r"^(.*)_ollama_(.+?)_(\d{8}_\d{6})$")


def list_outputs():
    items = []
    if not os.path.isdir(OUTPUTS):
        return items
    for name in os.listdir(OUTPUTS):
        path = os.path.join(OUTPUTS, name)
        if not os.path.isdir(path) or name.startswith("."):
            continue
        m = OUT_RE.match(name)
        source, model, stamp = (m.groups() if m else (name, "", ""))
        try:
            when = datetime.strptime(stamp, "%Y%m%d_%H%M%S").timestamp()
        except ValueError:
            when = os.stat(path).st_mtime
        item = {"name": name, "source": source, "model": model.replace("-", ":", 1),
                "time": when, "files": sorted(f for f in os.listdir(path) if not f.endswith(".tmp")),
                "total": None, "entities": None, "review": review.summary(path)}
        try:
            with open(os.path.join(path, "timing.md")) as f:
                mt = re.search(r"^Total:\s*([\d.]+)s", f.read(), re.M)
                if mt:
                    item["total"] = float(mt.group(1))
        except OSError:
            pass
        try:
            with open(os.path.join(path, "findings.json")) as f:
                item["entities"] = json.load(f).get("summary", {}).get("final_count")
        except (OSError, ValueError):
            pass
        items.append(item)
    items.sort(key=lambda i: i["time"], reverse=True)
    return items


# ------------------------------------------------------------ multipart ---

def parse_multipart(body, content_type):
    """Tiny multipart/form-data parser. Returns (fields, files)."""
    m = re.search(r'boundary="?([^";]+)"?', content_type)
    if not m:
        raise ValueError("boundary missing")
    boundary = b"--" + m.group(1).encode()
    fields, files = {}, {}
    for part in body.split(boundary)[1:]:
        if part in (b"--\r\n", b"--", b"\r\n"):
            continue
        part = part.removeprefix(b"\r\n")
        head, _, data = part.partition(b"\r\n\r\n")
        data = data.removesuffix(b"\r\n")
        headers = head.decode("utf-8", "replace")
        disp = re.search(r'name="([^"]*)"', headers)
        if not disp:
            continue
        name = disp.group(1)
        fn = re.search(r'filename="([^"]*)"', headers)
        if fn:
            files[name] = (fn.group(1), data)
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


def clean_filename(raw):
    raw = os.path.basename(raw.replace("\\", "/"))
    raw = SAFE_NAME_RE.sub("_", raw).strip(" .")
    return raw[:180] if raw else ""


# ------------------------------------------------------------ templates ---

def read_static(name):
    with open(os.path.join(STATIC, name), "rb") as f:
        return f.read()


def logo_html():
    svg = os.path.join(STATIC, "logo.svg")
    if os.path.exists(svg):
        return '<img class="logo" src="/static/logo.svg" alt="Lokale Pseudonymisierung">'
    return ('<div class="logo-text"><span class="box">LOKAL</span>'
            '<span class="slash"></span><span class="box">PSEUDO</span></div>')


def page(body, title="Pseudonymisierung"):
    css = read_static("style.css").decode()
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Lokale Pseudonymisierung</title>
<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
<style>{css}</style></head>
<body>{body}</body></html>"""


def login_page(error=""):
    err = f'<p class="error">{html.escape(error)}</p>' if error else ""
    return page(f"""
<main class="login">
  <header class="brand">{logo_html()}</header>
  <h1>Pseudonymisierung</h1>
  <form method="post" action="/login">
    <input type="password" name="password" placeholder="Passwort" autofocus autocomplete="current-password" required>
    <button class="pill primary" type="submit">Anmelden</button>
  </form>
  {err}
</main>""", "Anmelden")


def app_page():
    js = read_static("app.js").decode()
    return page(f"""
<div class="top-wrap"><header class="top">
  <div class="brand">{logo_html()}</div>
  <nav>
    <button type="button" class="info-btn" id="info-open" aria-label="So funktioniert es" title="So funktioniert es">i</button>
    <form method="post" action="/logout"><button class="pill small" type="submit">Abmelden</button></form>
  </nav>
</header></div>

<div class="modal" id="info" hidden>
  <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="info-title">
    <div class="modal-head">
      <div class="modal-tabs" role="tablist">
        <button type="button" class="mtab active" data-mtab="info" role="tab">Info</button>
        <button type="button" class="mtab" data-mtab="hardware" role="tab">Hardware</button>
        <button type="button" class="mtab" data-mtab="modell" role="tab">Modell</button>
      </div>
      <button type="button" class="modal-close" id="info-close" aria-label="Schließen">×</button>
    </div>

    <div class="mpanels">
    <div class="mpanel active" data-mpanel="info">
      <h2 id="info-title">So funktioniert es</h2>
      <ol class="howto">
        <li><span class="num">1</span><div><strong>Daten</strong><p>Dokumente hochladen: PDF, DOCX, XLSX, MD oder TXT, max. 200 MB pro Datei.</p></div></li>
        <li><span class="num">2</span><div><strong>Starten</strong><p>Ein Dokument pro Durchlauf, typisch 3 bis 15 Minuten. Der Projektname ist optional: Dokumente mit gleichem Projektnamen bekommen dieselben Pseudonyme, z. B. bleibt eine Person über mehrere Interviews PERSON_12.</p></div></li>
        <li><span class="num">3</span><div><strong>Prüfen</strong><p>Das System zeigt alle erkannten Kandidaten mit Kategorie und Ersetzung. Beschreibungen korrigieren, Einträge zusammenführen oder ausnehmen, dann freigeben. Korrekturen werden pro Projekt gemerkt und beim nächsten Dokument vorgeschlagen.</p></div></li>
        <li><span class="num">4</span><div><strong>Ergebnisse</strong><p>Jeder Durchlauf hat einen eigenen Ordner mit dem freigegebenen Text, dem unveränderten Systemvorschlag, Findings und Timing.</p></div></li>
      </ol>
    </div>

    <div class="mpanel" data-mpanel="hardware">
      <div class="mhero"><img src="/static/hardware.png" alt="NVIDIA DGX Spark" class="mart mart-wide"><div><h2>NVIDIA DGX Spark</h2><p class="muted">Die Pipeline läuft vollständig auf diesem Gerät. Keine Daten verlassen das Haus.</p></div></div>
      <dl class="specs">
        <dt>Chip</dt><dd>NVIDIA GB10 Grace Blackwell Superchip</dd>
        <dt>CPU</dt><dd>20 Arm-Kerne, 10× Cortex-X925 und 10× Cortex-A725</dd>
        <dt>Speicher</dt><dd>128 GB Unified Memory, gemeinsam für CPU und GPU</dd>
        <dt>Festplatte</dt><dd>4 TB NVMe</dd>
        <dt>System</dt><dd>Linux, NVIDIA-Treiber 580</dd>
      </dl>
    </div>

    <div class="mpanel" data-mpanel="modell">
      <div class="mhero"><img src="/static/model.svg" alt="" class="mart"><div><h2>Gemma 4, 26B</h2><p class="muted">Offenes Sprachmodell von Google, betrieben lokal über Ollama.</p></div></div>
      <dl class="specs">
        <dt>Parameter</dt><dd>25,8 Milliarden</dd>
        <dt>Quantisierung</dt><dd>Q4_K_M, rund 18 GB im Speicher</dd>
        <dt>Kontext</dt><dd>bis zu 262.144 Token</dd>
        <dt>Aufgaben</dt><dd>Entitätserkennung in zwei Durchgängen, Generalisierung von Orten, Firmen, Kennzahlen und Funktionen, Pseudonymisierung</dd>
        <dt>Datenschutz</dt><dd>Kein Cloud-Dienst beteiligt. Modell und Dokumente bleiben auf dem DGX Spark.</dd>
      </dl>
    </div>
    </div>
  </div>
</div>

<main class="wrap">
  <section class="hero">
    <h1>Dokumente pseudonymisieren.</h1>
  </section>

  <nav class="steps" aria-label="Schritte">
    <button type="button" class="step" data-tab="daten"><span class="num">1</span>Daten</button>
    <button type="button" class="step" data-tab="starten"><span class="num">2</span>Starten<span class="dot" id="run-dot" hidden></span></button>
    <button type="button" class="step" data-tab="pruefen"><span class="num">3</span>Prüfen<span class="badge" id="rv-count" hidden></span></button>
    <button type="button" class="step" data-tab="ergebnisse"><span class="num">4</span>Ergebnisse</button>
  </nav>

  <section class="panel" data-panel="daten" hidden>
    <form id="upload" class="drop" enctype="multipart/form-data">
      <input type="file" id="file" name="file" accept=".pdf,.docx,.xlsx,.md,.txt" multiple hidden>
      <p><strong>Datei hierher ziehen</strong> oder <button type="button" class="link" id="pick">auswählen</button></p>
      <p id="upload-status" class="muted"></p>
    </form>
    <details class="fold" id="inputs-fold">
      <summary>Hochgeladene Dokumente <span class="badge" id="in-count-2" hidden></span></summary>
      <ul id="inputs" class="list"></ul>
    </details>
    <div class="panel-foot"><span class="spacer"></span><button type="button" class="pill primary" data-goto="starten">Weiter: Starten</button></div>
  </section>

  <section class="panel" data-panel="starten" hidden>
    <form id="run">
      <div class="fields">
        <label>Dokument
          <select id="input-select" name="file" required></select></label>
        <label>Projektname <span class="muted">(optional)</span>
          <input type="text" id="project" name="project" placeholder="default" pattern="[A-Za-z0-9._\\-]{{1,64}}"></label>
      </div>
      <div class="panel-foot" id="run-foot">
        <p id="run-msg" class="muted" style="margin:0"></p>
        <span class="spacer"></span>
        <button class="pill primary" type="submit" id="run-btn">Durchlauf starten</button>
      </div>
    </form>
    <div class="statusbox" id="statusbox" hidden>
      <div id="status" class="status"></div>
      <div class="progress-head"><span id="progress-line" class="muted"></span><span id="pct" class="pct"></span></div>
      <div class="bar" id="progress"><div id="bar-fill"></div></div>
      <details class="fold" id="logbox"><summary>Protokoll</summary><pre id="log" class="log"></pre></details>
      <p id="status-next" hidden><button type="button" class="pill primary" data-goto="pruefen">Weiter: Prüfen</button></p>
    </div>
  </section>

  <section class="panel" data-panel="pruefen" hidden>
    <div class="panel-head">
      <div><h2>Vorschlag prüfen</h2><p class="sub muted" id="rv-sub">Alle erkannten Kandidaten mit Kategorie und Ersetzung. Anpassen, dann freigeben.</p></div>
      <label class="rv-pick">Durchlauf<select id="rv-dir"></select></label>
    </div>
    <p id="rv-empty" class="muted" hidden>Noch kein Durchlauf zum Prüfen.</p>
    <div id="rv-body" hidden>
      <div class="rv-tools">
        <input type="search" id="rv-filter" placeholder="Filtern …" autocomplete="off">
        <label class="rv-inline">Projekt<input type="text" id="rv-project" pattern="[A-Za-z0-9._\\-]{{1,64}}" placeholder="default"></label>
        <span id="rv-info" class="muted"></span>
      </div>
      <div id="rv-warn" class="warn" hidden></div>
      <div id="rv-groups"></div>
      <details class="fold" id="rv-addbox"><summary>Eintrag hinzufügen</summary>
        <div class="rv-add">
          <input type="text" id="add-text" placeholder="Text im Dokument">
          <select id="add-cat"></select>
          <input type="text" id="add-rep" list="rv-descs" placeholder="Ersetzung">
          <button type="button" class="pill small" id="add-btn">Hinzufügen</button>
        </div>
      </details>
      <details class="fold" id="rv-prevbox"><summary>Vorschau des Endtexts</summary><pre id="rv-preview" class="preview"></pre></details>
      <div class="panel-foot">
        <p id="rv-msg" class="muted" style="margin:0"></p>
        <span class="spacer"></span>
        <button type="button" class="pill ghost" id="rv-reset">Zurücksetzen</button>
        <button type="button" class="pill" id="rv-prev">Vorschau</button>
        <button type="button" class="pill primary" id="rv-approve">Freigeben</button>
      </div>
      <p id="rv-done" hidden><button type="button" class="pill" data-goto="ergebnisse">Zu den Ergebnissen</button></p>
    </div>
    <datalist id="rv-descs"></datalist>
  </section>

  <section class="panel" data-panel="ergebnisse" hidden>
    <ul id="outputs" class="list results"></ul>
  </section>
</main>

<footer class="foot muted">Lokale Pseudonymisierung · Pseudonymisierung (PII) und Generalisierung (Geschäftsgeheimnisse) mit Freigabe vor der Weitergabe.</footer>
<script>{js}</script>""")


# -------------------------------------------------------------- handler ---

class Handler(BaseHTTPRequestHandler):
    server_version = "pseudo-ui/1.0"

    # ---- helpers
    def client_ip(self):
        fwd = self.headers.get("X-Forwarded-For")
        return fwd.split(",")[0].strip() if fwd else self.client_address[0]

    def cookie(self, name):
        c = SimpleCookie(self.headers.get("Cookie", ""))
        return c[name].value if name in c else None

    def authed(self):
        return valid_session(self.cookie("session"))

    def is_https(self):
        return self.headers.get("X-Forwarded-Proto", "").lower() == "https"

    def session_cookie(self, token, max_age):
        secure = "; Secure" if self.is_https() else ""
        return f"session={token}; Path=/; Max-Age={max_age}; HttpOnly{secure}; SameSite=Strict"

    def send(self, status, body, ctype="text/html; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, status=200):
        self.send(status, json.dumps(obj), "application/json")

    def redirect(self, location, extra=None):
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def read_body(self, limit=MAX_UPLOAD + 65536):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > limit:
            raise ValueError("too large")
        return self.rfile.read(length)

    def same_origin(self):
        origin = self.headers.get("Origin") or self.headers.get("Referer")
        if not origin or origin == "null":
            return True
        host = self.headers.get("Host", "").lower()
        onet = urlsplit(origin).netloc.lower()
        ok = onet == host or onet in {f"{h}:{PORT}" for h in HOSTS}
        if not ok:
            sys.stderr.write(f"origin check failed: Origin={self.headers.get('Origin')!r} Referer={self.headers.get('Referer')!r} Host={self.headers.get('Host')!r}\n")
        return ok

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s %s\n" % (self.client_ip(), self.log_date_time_string(), fmt % args))

    # ---- routing
    def do_GET(self):
        url = urlsplit(self.path)
        path = url.path
        if path.startswith("/static/"):
            return self.serve_static(path[len("/static/"):])
        if path == "/login":
            return self.send(200, login_page())
        if not self.authed():
            if path.startswith("/api/"):
                return self.send_json({"error": "unauthorized"}, 401)
            return self.redirect("/login")
        if path == "/":
            return self.send(200, app_page())
        if path == "/api/state":
            return self.send_json({"inputs": list_inputs(), "outputs": list_outputs(),
                                   "status": runner.status()})
        if path == "/api/log":
            q = parse_qs(url.query)
            offset = int(q.get("offset", ["0"])[0] or 0)
            text, new_offset = runner.read_log(offset)
            return self.send_json({"text": text, "offset": new_offset,
                                   "status": runner.status()})
        if path == "/api/review":
            return self.handle_review_get(parse_qs(url.query))
        if path.startswith("/outputs/"):
            return self.serve_output(path[len("/outputs/"):])
        self.send(404, page("<main class='wrap'><h1>Nicht gefunden.</h1></main>"))

    def do_POST(self):
        path = urlsplit(self.path).path
        if not self.same_origin():
            return self.send_json({"error": "cross-origin request rejected"}, 403)
        if path == "/login":
            return self.handle_login()
        if not self.authed():
            return self.send_json({"error": "unauthorized"}, 401)
        if path == "/logout":
            return self.redirect("/login", {"Set-Cookie": self.session_cookie("", 0)})
        if path == "/api/upload":
            return self.handle_upload()
        if path == "/api/run":
            return self.handle_run()
        if path == "/api/delete-input":
            return self.handle_delete_input()
        if path in ("/api/review/preview", "/api/review/approve", "/api/review/reset"):
            return self.handle_review_post(path.rsplit("/", 1)[-1])
        self.send_json({"error": "not found"}, 404)

    # ---- actions
    def handle_login(self):
        ip = self.client_ip()
        if login_blocked(ip):
            return self.send(429, login_page("Zu viele Fehlversuche. Bitte in 15 Minuten erneut versuchen."))
        body = self.read_body(64 * 1024).decode("utf-8", "replace")
        password = parse_qs(body).get("password", [""])[0]
        time.sleep(0.3)
        if not verify_password(password):
            login_failed(ip)
            return self.send(401, login_page("Passwort falsch."))
        token = make_session()
        cookie = self.session_cookie(token, SESSION_TTL)
        self.redirect("/", {"Set-Cookie": cookie})

    def handle_upload(self):
        try:
            body = self.read_body()
        except ValueError:
            return self.send_json({"error": "Datei zu groß (max. 200 MB)."}, 413)
        try:
            _, files = parse_multipart(body, self.headers.get("Content-Type", ""))
        except ValueError as e:
            return self.send_json({"error": str(e)}, 400)
        saved, errors = [], []
        for _, (raw_name, data) in files.items():
            name = clean_filename(raw_name)
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if not name or ext not in ALLOWED_EXT:
                errors.append(f"{raw_name}: Format nicht unterstützt")
                continue
            if not data.strip():
                errors.append(f"{raw_name}: Datei ist leer (0 Byte). Bitte erst speichern, dann hochladen.")
                continue
            os.makedirs(INPUTS, exist_ok=True)
            with open(os.path.join(INPUTS, name), "wb") as f:
                f.write(data)
            saved.append(name)
        self.send_json({"saved": saved, "errors": errors})

    def handle_run(self):
        try:
            data = json.loads(self.read_body(64 * 1024) or b"{}")
        except ValueError:
            return self.send_json({"error": "bad json"}, 400)
        filename = str(data.get("file", ""))
        project = str(data.get("project") or "default").strip()
        if not safe_join(INPUTS, filename):
            return self.send_json({"error": "Ungültiger Dateiname."}, 400)
        if not PROJECT_RE.match(project):
            return self.send_json({"error": "Projektname: nur Buchstaben, Ziffern, . _ - (max. 64 Zeichen)."}, 400)
        ok, msg = runner.start(filename, project)
        self.send_json({"ok": ok, "message": msg}, 200 if ok else 409)

    def handle_delete_input(self):
        try:
            data = json.loads(self.read_body(64 * 1024) or b"{}")
        except ValueError:
            return self.send_json({"error": "bad json"}, 400)
        path = safe_join(INPUTS, str(data.get("file", "")))
        if not path or not os.path.isfile(path):
            return self.send_json({"error": "Datei nicht gefunden."}, 404)
        st = runner.status()
        if st["running"] and st["job"]["file"] == os.path.basename(path):
            return self.send_json({"error": "Datei wird gerade verarbeitet."}, 409)
        os.remove(path)
        self.send_json({"ok": True})

    # ---- review
    def review_dir(self, name):
        path = safe_join(OUTPUTS, str(name or ""))
        return path if path and os.path.isdir(path) else None

    def handle_review_get(self, q):
        dirpath = self.review_dir(q.get("dir", [""])[0])
        if not dirpath:
            return self.send_json({"error": "Ergebnis nicht gefunden."}, 404)
        try:
            r = review.load(dirpath) or review.prepare(dirpath, None, None, inputs_dir=INPUTS)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"review prepare failed: {e}\n")
            return self.send_json({"error": f"Vorbereitung fehlgeschlagen: {e}"}, 500)
        if not r:
            return self.send_json({"error": "Keine Findings in diesem Ergebnis."}, 404)
        if r.get("status") == "approved" and not r.get("namenspruefung_version"):
            r = dict(r)
            r["warnings"] = list(r.get("warnings", [])) + [
                "Diese Freigabe stammt von vor der Namensprüfung. Bitte Vorschau öffnen "
                "und erneut prüfen; fehlende Namensstellen werden jetzt ergänzt."
            ]
        self.send_json(review.public(r))

    def handle_review_post(self, action):
        try:
            data = json.loads(self.read_body(4 * 1024 * 1024) or b"{}")
        except ValueError:
            return self.send_json({"error": "bad json"}, 400)
        dirpath = self.review_dir(data.get("dir"))
        if not dirpath:
            return self.send_json({"error": "Ergebnis nicht gefunden."}, 404)
        project = str(data.get("project") or "").strip() or None
        if project and not PROJECT_RE.match(project):
            return self.send_json({"error": "Projektname: nur Buchstaben, Ziffern, . _ - (max. 64 Zeichen)."}, 400)
        edits = data.get("edits") if isinstance(data.get("edits"), list) else []
        added = data.get("added") if isinstance(data.get("added"), list) else []
        try:
            if action == "reset":
                r = review.reset(dirpath, project, None, inputs_dir=INPUTS)
                return self.send_json(review.public(r) if r else {"error": "Zurücksetzen fehlgeschlagen."})
            if action == "preview":
                text, stats = review.preview(dirpath, edits, added, project)
                if text is None:
                    return self.send_json({"error": "Keine Freigabedaten."}, 404)
                return self.send_json({"text": text, "stats": stats})
            r, stats = review.approve(dirpath, edits, added, project)
            if not r:
                return self.send_json({"error": "Keine Freigabedaten."}, 404)
            out = review.public(r)
            out["stats"] = stats
            return self.send_json(out)
        except review.RestnamenFehler as e:
            return self.send_json({"error": str(e)}, 422)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"review {action} failed: {e}\n")
            return self.send_json({"error": f"Fehler: {e}"}, 500)

    # ---- files
    def serve_static(self, name):
        path = safe_join(STATIC, name)
        if not path or not os.path.isfile(path):
            return self.send(404, "not found", "text/plain")
        ctype = {"css": "text/css", "js": "application/javascript",
                 "svg": "image/svg+xml", "png": "image/png"}.get(name.rsplit(".", 1)[-1], "application/octet-stream")
        self.send(200, read_static(name), ctype, {"Cache-Control": "public, max-age=3600"})

    def serve_output(self, rest):
        rest = unquote(rest)
        if rest.endswith(".zip") and "/" not in rest:
            dirpath = safe_join(OUTPUTS, rest[:-4])
            if not dirpath or not os.path.isdir(dirpath):
                return self.send(404, "not found", "text/plain")
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for fn in sorted(os.listdir(dirpath)):
                    z.write(os.path.join(dirpath, fn), fn)
            return self.send(200, buf.getvalue(), "application/zip",
                             {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(rest)}"})
        parts = rest.split("/")
        if len(parts) != 2:
            return self.send(404, "not found", "text/plain")
        dirpath = safe_join(OUTPUTS, parts[0])
        filepath = dirpath and safe_join(dirpath, parts[1])
        if not filepath or not os.path.isfile(filepath):
            return self.send(404, "not found", "text/plain")
        ctype = {"json": "application/json", "md": "text/markdown; charset=utf-8",
                 "txt": "text/plain; charset=utf-8"}.get(parts[1].rsplit(".", 1)[-1], "application/octet-stream")
        with open(filepath, "rb") as f:
            data = f.read()
        disposition = "inline" if ctype.startswith(("text/", "application/json")) else "attachment"
        self.send(200, data, ctype, {"Content-Disposition": f"{disposition}; filename*=UTF-8''{quote(parts[1])}"})


# ----------------------------------------------------------------- main ---

def main(argv):
    if len(argv) > 1 and argv[1] == "set-password":
        pw = getpass.getpass("Neues Passwort: ")
        if len(pw) < 8:
            sys.exit("Mindestens 8 Zeichen.")
        if pw != getpass.getpass("Wiederholen: "):
            sys.exit("Passwörter stimmen nicht überein.")
        set_password(pw)
        print("Passwort gesetzt.")
        return
    if len(argv) > 1 and argv[1] == "generate-password":
        pw = secrets.token_urlsafe(12)
        set_password(pw)
        print(pw)
        return
    if "hash" not in load_config():
        sys.exit("Kein Passwort gesetzt. Bitte zuerst: python3 app.py set-password")
    os.makedirs(LOGS, exist_ok=True)
    servers = []
    for host in HOSTS:
        srv = ThreadingHTTPServer((host, PORT), Handler)
        srv.daemon_threads = True
        servers.append(srv)
        print(f"pseudo-ui listening on http://{host}:{PORT}", flush=True)
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv)
