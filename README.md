# Lokale Pseudonymisierung

Öffentliche Codekopie der Dokumenten-Engine und der Review-Weboberfläche.
Enthalten sind `app/`, `config/`, `pseudo-ui/`, Tests und ein ausdrücklich
synthetischer Testkorpus. Angebote, Kundendokumente, Betriebsdaten, Zugangsdaten
und die private Git-Historie sind nicht Bestandteil dieser Veröffentlichung.

## Funktionen

PDF, DOCX, XLSX, Markdown und Text lokal verarbeiten; erkannte Angaben prüfen,
Ersetzungen korrigieren und Ergebnisse freigeben. Die Qualität hängt vom Modell
und Dokument ab. Eine erfolgreiche technische Verarbeitung ersetzt keine
inhaltliche Prüfung der Ergebnisse.

## Native Einrichtung auf macOS

Voraussetzungen: Git, Python 3.12, natives Ollama sowie Poppler (`pdftotext`)
für die PDF-Prüfung in der neuen Oberfläche. Unter Apple Silicon kann natives
Ollama die GPU nutzen; Intel-Macs verwenden CPU-Inferenz. Aktuelle Anforderungen:
https://docs.ollama.com/macos

```sh
git clone https://github.com/coderizon/pseudonymisierung-public.git
cd pseudonymisierung-public
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Vor einem Modell-Download Architektur, RAM und freien Speicher prüfen.
Der übernommene Engine-Standard ist `gemma4:26b`; er ist nicht für jedes MacBook
geeignet. `OLLAMA_MODEL` in `.env` auf ein tatsächlich verfügbares, passendes
lokales Ollama-Modell setzen und dieses mit `ollama pull <modell>` bereitstellen.
Bei Modellwechseln die Ergebnisqualität anhand synthetischer Beispiele prüfen.
Ollama lokal starten; `ollama list` und `http://127.0.0.1:11434/api/tags` prüfen.

```sh
# Einmalig ein eigenes lokales UI-Passwort setzen:
./scripts/start-ui.sh set-password

# Review-UI, in einem eigenen Terminal laufen lassen:
./scripts/start-ui.sh
# http://127.0.0.1:8789

# Optional zusätzlich Engine-API und ursprüngliche Oberfläche:
./scripts/start-engine.sh
# http://127.0.0.1:8000 und /health
```

Die Review-UI ruft die Engine über `scripts/pseudo` direkt in derselben Python-
Umgebung auf. Dafür sind kein Linux-Dienstkonto, kein `sudo` und kein `systemd`
erforderlich. Sie benötigt nicht zusätzlich den API-Server auf Port 8000.

## Lokale Daten und Konfiguration

`data/engine/` enthält Uploads, Ergebnisse und verschlüsselte Zuordnungen der
Engine; `data/ui/` enthält UI-Eingaben, Ergebnisse, Projektgedächtnis, Logs und
Login-Konfiguration. Diese Verzeichnisse und `.env` sind gitignored. Vorhandene
private Schlüssel oder Betriebsdaten eines anderen Systems nicht übernehmen.

Umgebungsvariablen: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `DATA_DIR`,
`PSEUDO_UI_STATE_DIR`, `PSEUDO_UI_HOST`, `PSEUDO_UI_PORT`, `PSEUDO_PYTHON` und
optional `PSEUDO_BIN`. Die Startskripte lesen `.env` aus dem Projektordner.
Server standardmäßig nur an `127.0.0.1` binden.

## Prüfung

```sh
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest pseudo-ui/tests -q
```

Danach einen synthetischen Text über die UI hochladen, verarbeiten, die
Ersetzungen prüfen, freigeben und herunterladen. PDF, DOCX und XLSX mit eigenen
synthetischen Dateien testen. Modellantworten und Hardwarebeschleunigung müssen
auf dem Ziel-Mac geprüft werden; diese Veröffentlichung behauptet keinen bereits
ausgeführten Mac-End-to-End-Test.

## Docker

`Dockerfile` und `docker-compose.yml` sind als alternative Infrastruktur
enthalten. Compose startet einen eigenen Ollama-Dienst und lädt standardmäßig
das dort konfigurierte Modell. Für den Mac zuerst den nativen Weg oben verwenden
und Compose nicht ungeprüft starten.
