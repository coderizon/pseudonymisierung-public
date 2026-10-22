"""Wiederkehrende Sprecherzeilen mit bereits bestätigten Personen ergänzen.

Keine neue Personenerkennung und keine Zusammenführung verschiedener Namen.
Fließtext bleibt der Kontextprüfung vorbehalten; Zeitstempel bleiben erhalten.
"""
import re

from app.pseudonymizer import EntityOccurrence

ZEIT = r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?"


def ergaenze_sprecher(text, occurrences):
    namen = {}
    for occ in occurrences:
        if occ.entity.category == "PERSON":
            namen.setdefault(occ.entity.text.casefold(), occ.entity)
    result = list(occurrences)
    for person in namen.values():
        if not person.text.strip() or "\n" in person.text:
            continue
        name = r"[ \t]+".join(re.escape(teil) for teil in person.text.split())
        # Zeit vor Name, Name mit Zeit dahinter oder Name: Redebeitrag.
        muster = re.compile(
            rf"^[ \t]*(?:{ZEIT}[ \t]+(?P<vor>{name})[ \t]*:?[ \t]*$"
            rf"|(?P<nach>{name})[ \t]+{ZEIT}[ \t]*:?[ \t]*$"
            rf"|(?P<rede>{name})[ \t]*:(?!\d))",
            re.MULTILINE | re.IGNORECASE,
        )
        stellen = [m.span(m.lastgroup) for m in muster.finditer(text)]
        if len(stellen) < 2:
            continue
        for start, end in stellen:
            konflikt = [o for o in result if o.start < end and start < o.end]
            if konflikt:
                if all(o.start == start and o.end == end and o.entity.category == "PERSON"
                       and o.entity.text.casefold() == person.text.casefold() for o in konflikt):
                    continue
                raise RuntimeError("Widersprüchliche Erkennung einer Sprecherzeile; keine sichere Ersetzung möglich.")
            result.append(EntityOccurrence(person, start, end))
    return result
