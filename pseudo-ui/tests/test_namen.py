"""Namenskonsistenz beim Prüfen, ohne Modellaufruf oder echte Kundendaten."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("review_namentest", Path(__file__).parents[1] / "review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def fall(auto="[PERSON_I] sagt Hallo. Max Muster antwortet."):
    e = {"key": "maxmuster", "text": "Max Muster", "category": "PERSON", "replacement": "PERSON_I", "suggested": "PERSON_I", "action": "replace", "count": 1, "source": "pipeline", "rejected": False}
    t = {"key": e["key"], "pos": 0, "len": 10, "desc": "PERSON_I", "surface": "Max Muster"}
    return {"version": 1, "status": "pending", "project": "test", "entities": [e], "tokens": [t]}, auto


def test_bereits_teilweise_ersetzter_name_wird_vervollstaendigt():
    r, auto = fall()
    text, stats = review.render(r, auto)
    assert text == "[PERSON_I] sagt Hallo. [PERSON_I] antwortet."
    assert stats["searched"] == 1


def test_wortgrenzen_schreibweise_und_bestehende_kennungen():
    r, auto = fall("[PERSON_I] MAX MUSTER Max  Muster Max Mustermann [Max Muster].")
    text, stats = review.render(r, auto)
    assert text == "[PERSON_I] [PERSON_I] [PERSON_I] Max Mustermann [Max Muster]."
    assert stats["restnamen"] == [{"text": "Max Muster", "anzahl": 1}]


def test_explizit_behaltene_person_bleibt_unveraendert():
    r, auto = fall()
    r["entities"][0]["action"] = "keep"
    text, stats = review.render(r, auto)
    assert text == "Max Muster sagt Hallo. Max Muster antwortet."
    assert stats["restnamen"] == []


def test_keine_kaskade_und_keine_automatische_namenszusammenfuehrung():
    r, auto = fall("[PERSON_I] Max Muster und Erika Beispiel.")
    e = copy.deepcopy(r["entities"][0])
    e.update(key="erikabeispiel", text="Erika Beispiel", replacement="PERSON_118", count=0)
    r["entities"].append(e)
    text, stats = review.render(r, auto)
    assert text == "[PERSON_I] [PERSON_I] und [PERSON_118]."
    assert "[[" not in text


def test_restname_stoppt_freigabe_ohne_dateien_zu_aendern(tmp_path, monkeypatch):
    r, auto = fall("[PERSON_I] [Max Muster]")
    (tmp_path / "review.json").write_text(json.dumps(r))
    (tmp_path / "pseudonymized.auto.txt").write_text(auto)
    (tmp_path / "pseudonymized.txt").write_text("BISHERIGE FREIGABE")
    monkeypatch.setattr(review, "PROJECTS_DIR", str(tmp_path / "projekte"))
    before = (tmp_path / "review.json").read_bytes()
    with pytest.raises(review.RestnamenFehler, match="Freigabe"):
        review.approve(str(tmp_path), [], [])
    assert (tmp_path / "pseudonymized.txt").read_text() == "BISHERIGE FREIGABE"
    assert (tmp_path / "review.json").read_bytes() == before
    assert not (tmp_path / "projekte").exists()


def test_alte_vorschau_aendert_nicht_das_original(tmp_path):
    r, auto = fall()
    (tmp_path / "review.json").write_text(json.dumps(r))
    (tmp_path / "pseudonymized.auto.txt").write_text(auto)
    (tmp_path / "pseudonymized.txt").write_text(auto)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    text, stats = review.preview(str(tmp_path), [], [])
    assert "Max Muster" not in text
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_kurzer_name_ueberschreibt_keine_bewusste_ausnahme():
    r, _ = fall()
    r["entities"][0].update(key="max", text="Max")
    r["tokens"][0].update(key="max", surface="Max")
    e = copy.deepcopy(r["entities"][0])
    e.update(key="maxmustermann", text="Max Mustermann", action="keep", count=0)
    r["entities"].append(e)
    text, stats = review.render(r, "[PERSON_I] Max Mustermann. Max.")
    assert text == "[PERSON_I] Max Mustermann. [PERSON_I]."
    assert stats["restnamen"] == []
