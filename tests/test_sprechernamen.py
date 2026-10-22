"""Bestätigte Sprecher bleiben über Abschnittsgrenzen konsistent."""

from unittest.mock import AsyncMock, patch

import pytest

from app.orchestrator import JobStatus, process_document
from app.pseudonymizer import DetectedEntity, EntityOccurrence, detect_entities


@pytest.mark.asyncio
async def test_wiederkehrende_sprecher_trotz_verwerfung_im_ersten_abschnitt():
    text = "00:00:02 Max Muster\nHallo.\n00:00:24 Max Muster\nJa.\n00:01:03 Max Muster\nDanke."
    person = DetectedEntity("Max Muster", "PERSON", "")
    letzter = text.rfind(person.text)
    with (
        patch("app.orchestrator.warm_up", new=AsyncMock()),
        patch("app.orchestrator.detect_entities", new=AsyncMock(return_value=[person])),
        patch("app.orchestrator.verify_occurrences", new=AsyncMock(return_value=[EntityOccurrence(person, letzter, letzter + len(person.text))])),
        patch("app.orchestrator.compute_klasse1_descriptions", new=AsyncMock(return_value={})),
        patch("app.orchestrator.compute_klasse2_descriptions", new=AsyncMock(return_value={})),
    ):
        r = await process_document(text.encode(), "gespraech.txt", "sprecher-test")
    assert r.status == JobStatus.SUCCESS
    assert "Max Muster" not in r.final_text
    assert r.final_text.count("[PERSON_A]") == 3
    assert "00:00:02" in r.final_text and "00:01:03" in r.final_text


def test_sprecherergaenzung_beachtet_kontext_und_verschiedene_namen():
    from app.sprecher import ergaenze_sprecher
    text = "00:00:02 Winter\nIm Winter ist es kalt.\n00:00:24 WINTER\nJa.\n00:00:30 Max Muster\nAntwort.\n00:00:40 Max Mustermann\nAntwort."
    person = DetectedEntity("Winter", "PERSON", "")
    start = text.index("Winter")
    original = [EntityOccurrence(person, start, start + 6)]
    result = ergaenze_sprecher(text, original)
    assert [(o.start, o.end) for o in result] == [(start, start + 6), (text.index("WINTER"), text.index("WINTER") + 6)]
    assert len(original) == 1


def test_doppelpunkt_und_zeitstempel_sind_sprecherformen():
    from app.sprecher import ergaenze_sprecher
    text = "Max Muster: Hallo.\nMax  Muster: Danke.\nMax Mustermann: Nein."
    person = DetectedEntity("Max Muster", "PERSON", "")
    result = ergaenze_sprecher(text, [EntityOccurrence(person, 0, 10)])
    assert len(result) == 2
    assert [text[o.start:o.end] for o in result] == ["Max Muster", "Max  Muster"]


@pytest.mark.asyncio
async def test_erkennungsfehler_liefert_keinen_stillen_teilerfolg():
    with patch("app.pseudonymizer.chat_json", new=AsyncMock(return_value={"entities": "kaputt"})):
        with pytest.raises(RuntimeError, match="unvollständig"):
            await detect_entities("Max Muster spricht.")


@pytest.mark.asyncio
async def test_eindeutige_listenantwort_und_wiederholung():
    antwort = [{"text": "Max Muster", "category": "PERSON", "context": ""}]
    fake = AsyncMock(side_effect=[ValueError("kaputte Antwort"), antwort])
    with patch("app.pseudonymizer.chat_json", new=fake):
        result = await detect_entities("Max Muster spricht.")
    assert [e.text for e in result] == ["Max Muster"]
    assert fake.await_count == 2


def test_cli_fehler_erzeugt_keinen_erfolgreichen_export(tmp_path):
    from typer.testing import CliRunner
    from app.cli import cli
    from app.orchestrator import ProcessingResult
    datei = tmp_path / "test.txt"
    datei.write_text("Max Muster")
    ausgabe = tmp_path / "ergebnis.txt"
    result = ProcessingResult(status=JobStatus.FAILED, original_filename="test.txt", warnings=["Erkennung unvollständig"])
    with patch("app.orchestrator.process_document", new=AsyncMock(return_value=result)):
        run = CliRunner().invoke(cli, ["run", str(datei), "--output", str(ausgabe)])
    assert run.exit_code != 0
    assert not ausgabe.exists()
