"""CLI interface for the pseudonymization pipeline.

Usage (inside container):
    python -m app run <file> --project <id>
    python -m app mappings <project_id> [--format csv]
    python -m app health
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import sys
from pathlib import Path

import typer

import app.config as _cfg
from app.config import OLLAMA_BASE_URL, OUTPUT_DIR

cli = typer.Typer(
    name="pseudo",
    help="Pseudonymisierungs-Pipeline: Lokale Pseudonymisierung und Generalisierung von Mandantenunterlagen.",
    no_args_is_help=True,
)


def _print_status(msg: str) -> None:
    sys.stderr.write(f"  {msg}\n")
    sys.stderr.flush()


@cli.command()
def run(
    file: Path = typer.Argument(..., help="Pfad zur Eingabedatei", exists=True, readable=True),
    project: str = typer.Option("default", "--project", "-p", help="Projekt-/Mandanten-ID"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Ausgabedatei (Standard: stdout)"),
    auto_output: bool = typer.Option(False, "--auto-output", "-O", help="Ausgabe in <OUTPUT_DIR>/<projekt>/<dateiname>"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Alle Artefakte in outputs/pseudonymization/ speichern"),
    model: str | None = typer.Option(None, "--model", "-m", help="Modell (Standard: OLLAMA_MODEL aus Umgebung)"),
    chunk_size: int | None = typer.Option(None, "--chunk-size", "-c", help="Max Chunk-Groesse in Zeichen (Standard: 2000)"),
    max_chunks: int | None = typer.Option(
        None, "--max-chunks",
        help="Nur die ersten N Chunks verarbeiten (Entwickler-Flag fuer schnelle Iteration bei Fehleranalyse).",
    ),
) -> None:
    """Dokument pseudonymisieren."""
    from app.orchestrator import process_document
    from app.pseudonymizer import set_chunk_size

    if model:
        _cfg.set_model(model)
    if chunk_size:
        set_chunk_size(chunk_size)

    data = file.read_bytes()
    filename = file.name

    if auto_output and not verbose:
        out_dir = Path(OUTPUT_DIR) / project
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / filename

    _print_status(f"Verarbeite: {filename}")
    _print_status(f"Projekt: {project}")
    _print_status(f"Modell: {_cfg.OLLAMA_MODEL}")
    if max_chunks is not None:
        _print_status(f"--max-chunks aktiv: erste {max_chunks} Chunks")
    _print_status("")

    result = asyncio.run(
        process_document(
            data, filename, project,
            max_chunks=max_chunks,
        )
    )

    for step in result.steps:
        icon = {"done": "+", "failed": "x", "skipped": "-"}.get(step.status.value, ".")
        time_str = f" ({step.duration_seconds}s)" if step.duration_seconds > 0 else ""
        _print_status(f"[{icon}] {step.label}{time_str}")
        if step.detail:
            _print_status(f"    {step.detail}")

    _print_status("")

    if result.warnings:
        for w in result.warnings:
            _print_status(f"WARNUNG: {w}")
        _print_status("")

    _print_status(
        f"Ergebnis: {result.entities_found} Entitaeten erkannt, "
        f"{result.entities_pseudonymized} ersetzt, "
        f"{result.processing_time_seconds}s Gesamtzeit"
    )

    if result.status.value != "success":
        _print_status("Verarbeitung fehlgeschlagen. Kein Ergebnis zur Freigabe erzeugt.")
        raise typer.Exit(code=1)

    if verbose:
        _write_verbose_output(file, project, result)
    elif output:
        output.write_text(result.final_text, encoding="utf-8")
        _print_status(f"Output geschrieben: {output}")
    else:
        sys.stdout.write(result.final_text)
        if not result.final_text.endswith("\n"):
            sys.stdout.write("\n")


def _write_verbose_output(file: Path, project: str, result: "ProcessingResult") -> None:
    """Write all pipeline artifacts to outputs/pseudonymization/<project>/<file>_<model>/."""
    from app.orchestrator import ProcessingResult  # noqa: F811

    stem = Path(file.name).stem
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "ollama_" + _cfg.OLLAMA_MODEL.replace(":", "-").replace("/", "-") + "_" + timestamp
    out_dir = Path(OUTPUT_DIR) / "pseudonymization" / project / f"{stem}_{model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # findings.json
    pass1_keys = {(e["text"].lower(), e["category"]) for e in result.pass1_candidates}
    pass2_keys = {(e["text"].lower(), e["category"]) for e in result.entities}
    kept = pass1_keys & pass2_keys
    rejected = pass1_keys - pass2_keys
    added = pass2_keys - pass1_keys

    pass2_rejected = [e for e in result.pass1_candidates if (e["text"].lower(), e["category"]) in rejected]
    pass2_added = [e for e in result.entities if (e["text"].lower(), e["category"]) in added]

    # Build Klasse 1 descriptions list for findings output
    k1_descriptions_list = []
    if result.klasse1_descriptions:
        # Reverse-map: find the original entity text and category for each description
        entity_by_lower: dict[str, dict] = {}
        for e in result.entities:
            key = e["text"].lower()
            if key not in entity_by_lower:
                entity_by_lower[key] = e
        for original_lower, description in sorted(result.klasse1_descriptions.items()):
            entity = entity_by_lower.get(original_lower, {})
            k1_descriptions_list.append({
                "original": entity.get("text", original_lower),
                "category": entity.get("category", ""),
                "description": description,
            })

    # Build Klasse 2 descriptions list for findings output
    k2_descriptions_list = []
    if result.klasse2_descriptions:
        if not entity_by_lower:
            for e in result.entities:
                key = e["text"].lower()
                if key not in entity_by_lower:
                    entity_by_lower[key] = e
        for original_lower, description in sorted(result.klasse2_descriptions.items()):
            entity = entity_by_lower.get(original_lower, {})
            k2_descriptions_list.append({
                "original": entity.get("text", original_lower),
                "category": entity.get("category", ""),
                "description": description,
            })

    findings = {
        "file": file.name,
        "model": _cfg.OLLAMA_MODEL,
        "summary": {
            "pass1_count": len(result.pass1_candidates),
            "pass2_kept": len(kept),
            "pass2_rejected": len(rejected),
            "pass2_added": len(added),
            "final_count": len(result.entities),
            "klasse1_descriptions": len(k1_descriptions_list),
            "klasse2_descriptions": len(k2_descriptions_list),
        },
        "pass1_candidates": result.pass1_candidates,
        "pass2_rejected": pass2_rejected,
        "pass2_added": pass2_added,
        "klasse1_descriptions": k1_descriptions_list,
        "klasse2_descriptions": k2_descriptions_list,
        "warnings": result.warnings,
    }
    (out_dir / "findings.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    # pseudonymized.txt
    text = result.pre_generalization_text or result.pseudonymized_text
    if text:
        (out_dir / "pseudonymized.txt").write_text(text, encoding="utf-8")

    # timing.md
    timing_lines = [
        f"# Timing: {file.name}",
        f"",
        f"Model: {_cfg.OLLAMA_MODEL}",
        f"Total: {result.processing_time_seconds}s",
        f"",
    ]

    for step in result.steps:
        icon = {"done": "+", "failed": "x", "skipped": "-"}.get(step.status.value, ".")
        timing_lines.append(f"[{icon}] {step.label}: {step.duration_seconds}s")
        if step.detail:
            timing_lines.append(f"    {step.detail}")

    if k1_descriptions_list:
        timing_lines.append("")
        timing_lines.append("## Klasse 1: Typ-erhaltende Generalisierung")
        timing_lines.append("")
        timing_lines.append("| Kategorie | Original | Beschreibung |")
        timing_lines.append("|-----------|----------|--------------|")
        for item in k1_descriptions_list:
            timing_lines.append(
                f"| {item['category']} | {item['original']} | {item['description']} |"
            )

    if result.pass1_chunk_timings:
        timing_lines.append("")
        timing_lines.append("## NER Pass 1: Per-Chunk Timing")
        timing_lines.append("")
        timing_lines.append("| Chunk | Chars | Duration | Entities |")
        timing_lines.append("|-------|-------|----------|----------|")
        for ct in result.pass1_chunk_timings:
            timing_lines.append(
                f"| {ct['chunk']} | {ct['chars']:,} | {ct['duration']}s | {ct['entities']} |"
            )
        total_p1 = sum(ct["duration"] for ct in result.pass1_chunk_timings)
        timing_lines.append(f"| **Total** | | **{total_p1:.2f}s** | |")

    if result.pass2_chunk_timings:
        timing_lines.append("")
        timing_lines.append("## Verification Pass 2: Per-Chunk Timing")
        timing_lines.append("")
        timing_lines.append("| Chunk | Chars | Candidates | Duration | Verified |")
        timing_lines.append("|-------|-------|------------|----------|----------|")
        for ct in result.pass2_chunk_timings:
            timing_lines.append(
                f"| {ct['chunk']} | {ct['chars']:,} | {ct['candidates']} | {ct['duration']}s | {ct['verified']} |"
            )
        total_p2 = sum(ct["duration"] for ct in result.pass2_chunk_timings)
        timing_lines.append(f"| **Total** | | | **{total_p2:.2f}s** | |")

    timing_lines.append("")
    (out_dir / "timing.md").write_text("\n".join(timing_lines) + "\n", encoding="utf-8")

    _print_status(f"Verbose-Ausgabe geschrieben: {out_dir}/")
    _print_status("  findings.json, pseudonymized.txt, timing.md")


@cli.command()
def mappings(
    project_id: str = typer.Argument(..., help="Projekt-/Mandanten-ID"),
    format: str = typer.Option("json", "--format", "-f", help="Ausgabeformat: json oder csv"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Ausgabedatei (Standard: stdout)"),
) -> None:
    """Mapping-Tabelle eines Projekts exportieren."""
    from app.mapping_store import MappingStore

    store = MappingStore(project_id)
    all_mappings = store.get_all_mappings()
    store.close()

    if not all_mappings:
        _print_status(f"Keine Mappings fuer Projekt '{project_id}' gefunden.")
        raise typer.Exit(0)

    if format == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["category", "original", "pseudonym", "context"])
        writer.writeheader()
        for m in all_mappings:
            writer.writerow(m)
        text = buf.getvalue()
    else:
        text = json.dumps(all_mappings, ensure_ascii=False, indent=2)

    if output:
        output.write_text(text, encoding="utf-8")
        _print_status(f"Mappings geschrieben: {output} ({len(all_mappings)} Eintraege)")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


@cli.command()
def evaluate(
    corpus: Path | None = typer.Option(
        None, "--corpus", "-c",
        help="Pfad zum Testkorpus-Verzeichnis (Standard: data/test_corpus)",
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Ergebnisse in Datei schreiben"),
    json_output: bool = typer.Option(False, "--json", help="Ausgabe als JSON"),
    project: str = typer.Option("eval", "--project", "-p", help="Projekt-ID fuer die Evaluation"),
) -> None:
    """Evaluation Harness: Pipeline gegen annotierten Testkorpus auswerten."""
    from app.evaluation import (
        ANNOTATIONS_DIR,
        DOCUMENTS_DIR,
        CorpusResult,
        DocumentResult,
        aggregate_corpus,
        evaluate_document_entities,
        format_report,
        format_report_json,
        load_ground_truth,
    )
    from app.orchestrator import process_document

    annotations_dir = Path(corpus) / "annotations" if corpus else ANNOTATIONS_DIR
    documents_dir = Path(corpus) / "documents" if corpus else DOCUMENTS_DIR

    _print_status("Lade Ground-Truth-Annotationen...")
    gt_docs = load_ground_truth(annotations_dir)
    _print_status(f"{len(gt_docs)} Dokumente geladen")
    _print_status("")

    doc_results: list[DocumentResult] = []

    for gt_doc in gt_docs:
        doc_path = documents_dir / gt_doc.document
        if not doc_path.exists():
            _print_status(f"WARNUNG: Dokument nicht gefunden: {doc_path}")
            continue

        _print_status(f"Verarbeite: {gt_doc.document}")
        data = doc_path.read_bytes()

        result = asyncio.run(
            process_document(data, gt_doc.document, project)
        )

        for step in result.steps:
            icon = {"done": "+", "failed": "x", "skipped": "-"}.get(step.status.value, ".")
            time_str = f" ({step.duration_seconds}s)" if step.duration_seconds > 0 else ""
            _print_status(f"  [{icon}] {step.label}{time_str}")

        doc_eval = evaluate_document_entities(result.entities, gt_doc.entities)
        doc_eval.document = gt_doc.document
        doc_results.append(doc_eval)

        _print_status(
            f"  => TP={doc_eval.overall.tp} FP={doc_eval.overall.fp} FN={doc_eval.overall.fn}"
            f"  P={doc_eval.overall.precision:.1%} R={doc_eval.overall.recall:.1%}"
        )
        _print_status("")

    corpus_result = aggregate_corpus(doc_results)

    if json_output:
        import json
        text = json.dumps(format_report_json(corpus_result), ensure_ascii=False, indent=2)
    else:
        text = format_report(corpus_result)

    if output:
        output.write_text(text, encoding="utf-8")
        _print_status(f"Ergebnisse geschrieben: {output}")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


@cli.command()
def estimate(
    file: Path = typer.Argument(..., help="Pfad zur Eingabedatei", exists=True, readable=True),
    show_text: bool = typer.Option(True, "--show-text/--no-text", help="Chunk-Text anzeigen oder ausblenden"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Ergebnisse in outputs/estimation/ speichern"),
    project: str = typer.Option("default", "--project", "-p", help="Projekt-ID (fuer Ausgabeverzeichnis)"),
    model: str | None = typer.Option(None, "--model", "-m", help="Modell (Standard: OLLAMA_MODEL aus Umgebung)"),
    chunk_size: int | None = typer.Option(None, "--chunk-size", "-c", help="Max Chunk-Groesse in Zeichen (Standard: 2000)"),
    max_chunks: int | None = typer.Option(
        None, "--max-chunks",
        help="Nur die ersten N Chunks schaetzen (Entwickler-Flag fuer schnelle Iteration bei Fehleranalyse).",
    ),
) -> None:
    """Token-Verbrauch fuer NER-Erkennung schaetzen (ohne Inferenz)."""
    import time as time_mod

    if model:
        _cfg.set_model(model)

    from app.pseudonymizer import set_chunk_size
    if chunk_size:
        set_chunk_size(chunk_size)

    import app.pseudonymizer as _pseudo
    from app.category_config import get_config
    from app.ingestion import ingest_document
    from app.llm_client import count_chat_tokens, count_tokens, get_model_context_length
    from app.pseudonymizer import chunk_text

    config = get_config()
    data = file.read_bytes()
    doc = ingest_document(data, file.name)
    full_text = doc.full_text

    if not full_text.strip():
        _print_status(f"Keine Textinhalte in {file.name}")
        raise typer.Exit(1)

    ner_prompt = config.build_ner_prompt()
    verification_prompt = config.build_verification_prompt()
    chunks = chunk_text(full_text, _pseudo.MAX_CHUNK_SIZE)
    if max_chunks is not None and max_chunks >= 0:
        chunks = chunks[:max_chunks]

    async def _run() -> None:
        estimation_start = time_mod.time()
        context_length = await get_model_context_length()
        full_text_tokens = await count_tokens(full_text)

        def pct(n: int) -> str:
            return f"{n / context_length:.1%}"

        def bar(n: int, width: int = 30) -> str:
            filled = round(n / context_length * width)
            return "[" + "#" * filled + "." * (width - filled) + "]"

        lines: list[str] = []

        def out(msg: str = "") -> None:
            _print_status(msg)
            lines.append(msg)

        out(f"Datei: {file.name}")
        out(f"Format: {doc.format}")
        out(f"Modell: {_cfg.OLLAMA_MODEL}")
        out(f"Kontextfenster: {context_length:,} Tokens")
        out(f"Chunks: {len(chunks)} (max {_pseudo.MAX_CHUNK_SIZE} Zeichen)")
        out("")

        out(f"Gesamter Text: {full_text_tokens:,} Tokens ({pct(full_text_tokens)})")
        out("")

        out("=== NER Pass 1: Chunks ===")
        ner_total_input = 0
        for i, chunk in enumerate(chunks, 1):
            messages = [
                {"role": "system", "content": ner_prompt},
                {"role": "user", "content": f"Analysiere folgenden Text:\n\n{chunk.text}"},
            ]
            request_tokens = await count_chat_tokens(messages)
            ner_total_input += request_tokens

            out(f"--- Chunk {i}/{len(chunks)} ({len(chunk.text)} Zeichen) ---")
            out(f"  Request total: {request_tokens:>6,} Tokens  {pct(request_tokens):>6}  {bar(request_tokens)}")
            out(f"  Frei fuer Antwort: {context_length - request_tokens:>6,} Tokens  {pct(context_length - request_tokens):>6}")
            if show_text:
                out(f"  Text:")
                for line in chunk.text.splitlines():
                    out(f"    | {line}")
            out("")

        out("=== Verifikation Pass 2: Chunks ===")
        out("  (User-Message enthaelt zusaetzlich Kandidaten-JSON; hier mit leerer Liste geschaetzt)")
        out("")
        ver_total_input = 0
        for i, chunk in enumerate(chunks, 1):
            messages = [
                {"role": "system", "content": verification_prompt},
                {"role": "user", "content": (
                    f"Textabschnitt:\n\n{chunk.text}\n\n"
                    "Kandidaten-Entitaeten:\n[]"
                )},
            ]
            request_tokens = await count_chat_tokens(messages)
            ver_total_input += request_tokens

            out(f"  Chunk {i}/{len(chunks)}: {request_tokens:>6,} Tokens  {pct(request_tokens):>6}  {bar(request_tokens)}")

        estimation_duration = round(time_mod.time() - estimation_start, 2)

        out("")
        out("=== Zusammenfassung ===")
        out(f"  Kontextfenster:              {context_length:>8,} Tokens")
        out(f"  Gesamter Text:               {full_text_tokens:>8,} Tokens  {pct(full_text_tokens):>6}")
        out(f"  NER Pass 1 gesamt:           {ner_total_input:>8,} Tokens  (Summe aller {len(chunks)} Requests)")
        out(f"  Verifikation Pass 2 gesamt:  {ver_total_input:>8,} Tokens  (ohne Kandidaten-JSON)")
        out(f"  Pipeline gesamt (Input):     {ner_total_input + ver_total_input:>8,} Tokens")
        out(f"  Schaetzungsdauer:            {estimation_duration:>8.1f}s")

        if verbose:
            stem = Path(file.name).stem
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_tag = "ollama_" + _cfg.OLLAMA_MODEL.replace(":", "-").replace("/", "-") + "_" + timestamp
            out_dir = Path(OUTPUT_DIR) / "estimation" / project
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}_{model_tag}.md"
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            _print_status("")
            _print_status(f"Ergebnisse geschrieben: {out_path}")

    asyncio.run(_run())


@cli.command()
def chunks(
    file: Path = typer.Argument(..., help="Pfad zur Eingabedatei", exists=True, readable=True),
    chunk_size: int | None = typer.Option(None, "--chunk-size", "-c", help="Max Chunk-Groesse in Zeichen (Standard: 2000)"),
    max_chunks: int | None = typer.Option(
        None, "--max-chunks",
        help="Nur die ersten N Chunks anzeigen (Entwickler-Flag fuer schnelle Iteration bei Fehleranalyse).",
    ),
) -> None:
    """Chunking-Ergebnis fuer eine Datei anzeigen."""
    import app.pseudonymizer as _pseudo
    from app.ingestion import ingest_document
    from app.pseudonymizer import chunk_text, set_chunk_size

    if chunk_size:
        set_chunk_size(chunk_size)

    data = file.read_bytes()
    doc = ingest_document(data, file.name)
    full_text = doc.full_text

    if not full_text.strip():
        _print_status(f"Keine Textinhalte in {file.name}")
        raise typer.Exit(1)

    result = chunk_text(full_text, _pseudo.MAX_CHUNK_SIZE)
    if max_chunks is not None and max_chunks >= 0:
        result = result[:max_chunks]

    _print_status(f"Datei: {file.name}")
    _print_status(f"Format: {doc.format}")
    _print_status(f"Gesamtlaenge: {len(full_text):,} Zeichen")
    _print_status(f"Chunks: {len(result)} (max {_pseudo.MAX_CHUNK_SIZE} Zeichen)")
    _print_status("")

    for i, chunk in enumerate(result, 1):
        _print_status(f"--- Chunk {i}/{len(result)} ({len(chunk.text)} Zeichen) ---")
        for line in chunk.text.splitlines():
            _print_status(f"  | {line}")
        _print_status("")


@cli.command()
def reset(
    project: str | None = typer.Option(None, "--project", "-p", help="Nur dieses Projekt zuruecksetzen"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Ohne Rueckfrage ausfuehren"),
) -> None:
    """Datenbank und Ausgaben zuruecksetzen."""
    import shutil
    import sqlite3

    from app.config import DB_PATH

    out_dir = Path(OUTPUT_DIR)

    if project:
        target_desc = f"Projekt '{project}'"
        db_action = f"Mappings fuer Projekt '{project}' loeschen"
        out_target = out_dir / project
        out_action = f"Ausgaben loeschen: {out_target}" if out_target.exists() else None
    else:
        target_desc = "ALLE Projekte"
        db_action = f"Gesamte Datenbank loeschen: {DB_PATH}"
        out_action = f"Alle Ausgaben loeschen: {out_dir}" if out_dir.exists() else None

    _print_status(f"Reset: {target_desc}")
    _print_status(f"  - {db_action}")
    if out_action:
        _print_status(f"  - {out_action}")
    _print_status("")

    if not yes:
        confirm = typer.confirm("Fortfahren?")
        if not confirm:
            _print_status("Abgebrochen.")
            raise typer.Exit(0)

    if project:
        conn = sqlite3.connect(DB_PATH)
        # Delete occurrence variants for this project
        conn.execute(
            "DELETE FROM occurrence_variants WHERE mapping_rowid IN "
            "(SELECT rowid FROM mappings WHERE project_id = ?)",
            (project,),
        )
        conn.execute("DELETE FROM mappings WHERE project_id = ?", (project,))
        conn.commit()
        deleted = conn.total_changes
        conn.close()
        _print_status(f"  {deleted} Datensaetze geloescht")
        if (out_dir / project).exists():
            shutil.rmtree(out_dir / project)
            _print_status(f"  Ausgaben geloescht: {out_dir / project}")
    else:
        db_path = Path(DB_PATH)
        if db_path.exists():
            db_path.unlink()
            _print_status("  Datenbank geloescht")
        if out_dir.exists():
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            _print_status("  Ausgaben geloescht")

    _print_status("")
    _print_status("Reset abgeschlossen.")


@cli.command()
def health() -> None:
    """Systemstatus pruefen (Ollama + Modell)."""
    from app.llm_client import check_model_ready

    _print_status(f"Ollama: {OLLAMA_BASE_URL}")
    _print_status(f"Modell: {_cfg.OLLAMA_MODEL}")

    ready = asyncio.run(check_model_ready())

    if ready:
        _print_status("Status: bereit")
        raise typer.Exit(0)
    else:
        _print_status("Status: NICHT bereit (Modell nicht verfuegbar)")
        raise typer.Exit(1)
