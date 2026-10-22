# Synthetischer Testkorpus (TASK-005)

Testkorpus fuer die Evaluation der NER-Pipeline und der Pseudonymisierungs-/Generalisierungstransformationen. Alle Dokumente sind synthetisch (LLM-generiert mit realistischen Strukturen). Die Entitaeten sind fiktiv.

## Dokumente

| # | Datei | Typ | Sprache | Beschreibung |
|---|-------|-----|---------|--------------|
| 1 | `strategiesitzung_q4.txt` | Sitzungsprotokoll | DE | Strategiesitzung mit Finanzbericht, M&A-Planung, Personalthemen |
| 2 | `workshop_digitalisierung.txt` | Workshop-Protokoll | DE | Digitalisierung Kernprozesse: Ist-Analyse, Loesungsdesign, Budgetbeschluesse |
| 3 | `org_chart_restrukturierung.txt` | Strategiepapier / Org-Chart | DE | Organisatorische Neuaufstellung mit Standortstrategie und M&A-Pipeline |
| 4 | `projekt_status_uebersicht.txt` | Tabellarische Uebersicht | DE | 5 Beratungsmandate mit Detailberichten, Risiken, Meilensteinen |
| 5 | `produkt_spezifikation_ml.txt` | Produktspezifikation | DE | ML-basierte Workforce-Planning-Loesung: Architektur, Kennzahlen, Referenzen |
| 6 | `interview_transkript_beratung.txt` | Interview-Transkript | DE | Gespraech mit Geschaeftsfuehrer eines Medtech-Startups ueber Strategie und Organisation |
| 7 | `consulting_engagement_summary.txt` | Engagement Summary | EN | Aerospace MES-Transformation: Finanzen, Risikoregister, Cross-Site-Koordination |

## Annotation-Schema

Jede Annotation ist eine JSON-Datei in `annotations/` mit identischem Dateinamen (`.json` statt `.txt`).

```json
{
  "document": "dateiname.txt",
  "language": "de|en",
  "description": "Kurzbeschreibung des Dokuments",
  "entities": [
    {
      "text": "exakter Text wie im Dokument",
      "category": "CATEGORY_NAME",
      "context": "Kontextsatz, in dem die Entitaet vorkommt"
    }
  ]
}
```

Das Schema entspricht der `DetectedEntity`-Datenklasse in `app/pseudonymizer.py`. Jedes Vorkommen einer Entitaet wird einzeln annotiert (auch Wiederholungen).

## Kategorie-Abdeckung

Jede Kategorie muss in mindestens 2 Dokumenten vorkommen.

| Kategorie | Klasse | Doc 1 | Doc 2 | Doc 3 | Doc 4 | Doc 5 | Doc 6 | Doc 7 | Gesamt |
|-----------|--------|-------|-------|-------|-------|-------|-------|-------|--------|
| PERSON | 1 | x | x | x | x | x | x | x | 7 |
| CONTACT | 1 | x | | | | x | x | x | 4 |
| ORG_CLIENT | 1 | | | x | x | | x | x | 4 |
| ORG_UNIT | 1 | | x | x | | | x | | 3 |
| LOCATION | 1 | x | x | x | x | x | x | x | 7 |
| FUNCTION_TITLE | 2 | | x | x | x | x | x | x | 6 |
| ORG_PARTNER | 2 | x | | x | x | x | x | x | 6 |
| INDUSTRY | 2 | | | x | x | x | x | x | 5 |
| PRODUCT | 2 | | x | | x | x | x | x | 5 |
| PROCESS_ARTIFACT | 2 | | x | | x | x | | x | 4 |
| BUSINESS_METRIC | 2 | x | x | x | x | x | x | x | 7 |
| TEMPORAL | 2 | x | x | x | x | x | x | x | 7 |
| CONTRACT | 2 | | x | x | x | | x | x | 5 |
| TECHNOLOGY | 2 | | x | | x | x | | x | 4 |
| STRATEGY | 2 | x | | x | | | x | | 3 |
| NEGATIVE_KNOWHOW | 2 | | | x | | | x | x | 3 |
| QUASI_ID | 2 | | | | | | x | | 1 |

QUASI_ID erscheint nur in Doc 6, da es ein Post-Processing-Flag ist und keine direkt erkannte Entitaet. Die Annotation dort illustriert eine identifizierende Attributkombination (Branche + Region + Groesse + Nische).

## Verwendung

Der Korpus dient als Ground Truth fuer:

1. **TASK-008** (Evaluation Harness): Precision/Recall/F1 pro Kategorie und Klasse
2. **TASK-007** (Two-Pass NER): Validierung der Extraktionsqualitaet
3. **TASK-010/011** (Transformationen): Pruefung der Generalisierungsqualitaet
4. **TASK-013** (Quality Iteration): Iterative Verbesserung gegen Zielwerte

Zielwerte: >90% Recall (Klasse 1), >85% Recall (Klasse 2), >80% Precision (beide Klassen).

## Statistik

- 7 Dokumente (6 DE, 1 EN)
- 16 von 17 Kategorien direkt annotiert (QUASI_ID als Flag)
- Gesamtzahl annotierter Entitaeten: siehe einzelne JSON-Dateien
