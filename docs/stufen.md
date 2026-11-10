# Die Stufen im Einzelnen

Jede Stufe beantwortet eine Frage. Hier steht je Stufe, was sie tut, welche Parameter sie hat und wie ihr Ergebnis zu lesen ist. Die Standardwerte der Parameter stehen in `DEFAULT_STAGES` in `src/styrke/config.py`; abweichende Werte gehören in den Abschnitt `stages` der Testmatrix.

Für alle Stufen gilt:

- Vor jeder Stufe startet styrke die Engine neu, damit der Cache leer ist, und schickt drei kurze Anfragen zum Aufwärmen.
- Jede Anfrage steht einzeln in `requests.jsonl`. Die Kurzauswertung einer Stufe (`summary_sN.json`) ist nur ein erster Blick; alle Kennzahlen lassen sich aus den Rohdaten neu rechnen.
- Zielwerte setzt keine Stufe. Ob eine Antwortzeit gut genug ist, entscheidet die spätere Auswertung.

## Stufe 0, Machbarkeit

**Frage:** Lädt die Kombination aus Modell und Engine, wie lange dauert das, und welche Promptlänge verarbeitet sie?

**Ablauf:** eine Sinnprobe mit kurzer Antwort, eine Probe mit eingeschaltetem Nachdenken, die Eichung der Promptlänge, dann Prompts steigender Länge (`context_probes`) bis zum ersten Fehlschlag.

**Ergebnis:** `feasibility.json`. Lädt die Zelle nicht, steht dort der Grund. Spätere Stufen lassen Promptlängen aus, die hier gescheitert sind.

## Stufe 1, Einzelstrom

**Frage:** Wie schnell ist eine einzelne Anfrage ohne Konkurrenz, je nach Promptlänge?

**Ablauf:** je Promptlänge (`lengths`) und Denkmodus (`thinking`) mehrere Anfragen nacheinander (`reps`).

**Ergebnis:** Zeit bis zum ersten Token, Tokens pro Sekunde beim Schreiben der Antwort und das Einlesetempo des Prompts. Das sind die Bestwerte der Zelle; unter Last wird es langsamer.

## Stufe 2, Parallelität

**Frage:** Wie viel Durchsatz liefert die Zelle bei immer mehr gleichzeitigen Anfragen, und ab wann lohnt mehr nicht?

**Ablauf:** Je Laststufe (`levels`) schicken N Arbeiter ununterbrochen kurze Anfragen. Nach einer Anlaufzeit folgen Messfenster (`window_s`, `windows`). Wächst der Gesamtdurchsatz gegenüber dem bisherigen Bestwert um weniger als 5 % (`saturation_gain`), gilt die Zelle als gesättigt; es folgen noch zwei Laststufen, dann endet die Stufe.

**Ergebnis:** je Laststufe der Gesamtdurchsatz, das Tempo je Anfrage und die Zeit bis zum ersten Token. Steigt die Zeit bis zum ersten Token sprunghaft, stehen Anfragen in der Warteschlange der Engine.

## Stufe 3, KV-Füllstand

**Frage:** Arbeitet die Zelle schlechter, wenn ihr KV-Pool voll ist, und behält sie dabei wartende Gespräche im Cache?

**Ablauf:** Der Pool wird schrittweise mit gehaltenen Gesprächen belegt (`fills`, als Anteil der Poolgröße). Ein gehaltenes Gespräch ist ein langer Prompt (`session_tokens`), dessen Zwischenstand die Engine nach der Antwort im Cache behält. Je Füllstand läuft dann eine Kurzform der Stufe 2 mit frischen kurzen Anfragen (`levels`). Danach bekommt ein Teil der gehaltenen Gespräche eine Folgefrage (`probe_sessions`).

**Ergebnis:** Durchsatz und Antwortzeiten je Füllstand, dazu der Anteil der gehaltenen Gespräche, die nach der Last noch im Cache lagen. Zwei Urteile stehen nebeneinander:

- `retained_by_ttft`: nach Antwortzeit. Kam die Folgefrage in höchstens der halben Zeit des kalten Prompts, lag das Gespräch noch im Cache. Dieses Urteil braucht keine Angaben der Engine und ist deshalb zwischen Engines vergleichbar.
- `retained_by_cache_report`: nach Angabe der Engine, soweit sie Cache-Treffer meldet.

**Zu beachten:**

- Die Poolgröße kommt bei FreeToken aus der Startoption `kv-reserve-tokens`. Ist sie nicht bekannt, lässt styrke die Stufe aus; dann `stages.s3.pool_tokens` setzen.
- Die Füllstände bauen aufeinander auf, ohne Neustart dazwischen. Im Pool liegen deshalb auch die Reste der kurzen Anfragen. Wie voll er laut Engine war, steht in `engine_metrics_after_fill` und `engine_metrics_after_load`.
- Behält eine Engine den Zwischenstand beendeter Anfragen gar nicht, belegen gehaltene Gespräche nichts. Dann unterscheiden sich die Füllstände nicht, und Stufe 4 zeigt bei Folgefragen keinen Gewinn.

## Stufe 4, Präfix-Cache

**Frage:** Wie viel schneller ist eine Folgefrage als ein neuer Prompt gleicher Länge, und wie lange hält dieser Vorteil?

**Ablauf:** Je Promptlänge (`lengths`) beginnt ein Gespräch kalt. Sofort danach kommt eine Folgefrage; die Engine kennt ihren Anfang schon und muss nur den neuen Teil einlesen. Dazu zwei Zusatzmessungen:

- nach Pause: eine weitere Folgefrage nach `pause_s` Sekunden Ruhe (einmal je Promptlänge);
- nach Verdrängung: Zwischen zwei Folgefragen schiebt styrke mehr fremden Text durch die Engine, als der KV-Pool fasst (`evict_factor` mal Poolgröße).

**Ergebnis:** Zeit bis zum ersten Token kalt und bei der Folgefrage, der Faktor dazwischen und, soweit die Engine es meldet, der Anteil des Prompts aus dem Cache. Nach der Verdrängung sollte die Folgefrage wieder so lange brauchen wie der kalte Prompt; tut sie es nicht, verdrängt die Engine anders als angenommen.

## Stufe 5, Störwirkung

**Frage:** Was passiert mit laufenden Chats, wenn jemand ein sehr großes Dokument einreicht?

**Ablauf:** Im Hintergrund laufen ununterbrochen kurze Chats (`chats`). Nach einer Ruhephase (`baseline_s`) kommt eine einzelne große, kalte Anfrage dazu (`big_lengths`). Verglichen werden vier Abschnitte: vor der großen Anfrage, während die Engine ihren Prompt einliest, während sie ihre Antwort schreibt, und danach.

**Ergebnis:** je Abschnitt der Durchsatz der Chats, die Abstände zwischen zwei Tokens derselben Antwort, die längste Zeit ganz ohne Token (`silence_max_s`) und die Zeit bis zum ersten Token für Chats, die in dem Abschnitt begonnen haben. `slowdown_prefill_pct` ist der Einbruch des Durchsatzes beim Einlesen gegenüber vorher.

**Zu beachten:** Lässt die Engine nur so viele Anfragen gleichzeitig zu, wie Chats laufen, wartet die große Anfrage erst auf einen freien Platz und verdrängt dann einen Chat in die Warteschlange. Das Ergebnis hängt also von der Variante ab.

## Stufe 6, Szenario

**Frage:** Welche Last trägt die Zelle im gemischten Betrieb?

**Ablauf:** Vorgegeben ist die Ankunftsrate in Modellaufrufen pro Sekunde (`rates`), nicht die Zahl gleichzeitiger Anfragen. Gespräche treffen zufällig ein (Poisson-Ankünfte) und gehören zu einer Sitzungsart:

| Sitzungsart | Anteil | erster Prompt | Züge | Besonderheit |
| --- | --- | --- | --- | --- |
| `kurzer_chat` | 60 % | 200 bis 1.500 Tokens | 4 | Folgefragen nach im Mittel 30 s |
| `langer_chat` | 25 % | 4.000 bis 24.000 Tokens | 4 | Folgefragen mit bis zu 1.000 Tokens neuem Text |
| `dokumentarbeit` | 10 % | 64.000 bis 250.000 Tokens | 3 | Fragen an ein großes Dokument |
| `agentenlauf` | 5 % | 3.000 bis 8.000 Tokens | 6 | Werkzeugschleife: Züge ohne Pause, jeder bringt bis zu 3.000 Tokens Werkzeugergebnis mit |

Die Anteile stehen in `load_model.session_mix` der Testmatrix, die Form der Sitzungsarten in `stages.s6.sessions`. Jeder Zug schickt wie eine Chat-Oberfläche den ganzen Verlauf mit; Folgefragen können deshalb den Präfix-Cache treffen. Die Länge des ersten Prompts wird so gezogen, dass kurze häufiger sind als lange.

Jede Rate läuft `duration_s` lang (15 Minuten); ausgewertet wird das Fenster nach der Anlaufzeit (`warmup_s`). Ein Abschnitt gilt als nicht getragen, wenn

- der Rückstau wächst: Im letzten Viertel des Fensters laufen im Mittel mehr als 1,5-mal so viele Aufrufe wie im ersten, plus 4;
- mehr als 10 % der Aufrufe bis zum Ende des Abschnitts nicht fertig werden;
- oder mehr als 5 % der Aufrufe in einen Fehler laufen.

Nach dem ersten nicht getragenen Abschnitt folgt noch einer zur Bestätigung, dann endet die Stufe.

**Ergebnis:** je Rate die Antwortzeiten (gesamt, je Sitzungsart, erste Züge und Folgefragen getrennt), die mittlere Bearbeitungszeit eines Aufrufs, der Durchsatz und die Zahl gleichzeitig laufender Aufrufe. `max_sustained_rate` ist die höchste Rate ohne Überlast; `users_supported` rechnet sie mit der Formel des Lastmodells in Berechtigte je Nutzungsprofil um.

**Zu beachten:**

- „Getragen" heißt nur: Die Engine kommt hinterher. Ob die Antwortzeiten dabei annehmbar sind, ist eine zweite Frage; dafür stehen sie je Rate in der Tabelle.
- Die Werkzeugschleife ist nachgestellt: Die Werkzeugergebnisse sind Text aus dem Korpus, echte Werkzeugaufrufe macht das Modell hier nicht. Für die Last zählt, dass der Kontext Zug um Zug wächst.
- Das Nachdenken ist ausgeschaltet (`thinking: off`). Nutzt der spätere Betrieb das Nachdenken, schreibt das Modell je Aufruf deutlich mehr Tokens, und die tragbare Rate liegt niedriger. Dann `stages.s6.thinking: on` setzen.
- Abweichung vom Plan: Dort stehen zwei weitere Abschnitte nach der Überlast. Bei vorgegebener Ankunftsrate zeigt ein überlasteter Abschnitt nur eine wachsende Warteschlange; einer genügt zur Bestätigung (`extra_levels_after_overload`).

## Stufe 7, Dauerlauf

**Frage:** Bleibt die Zelle über längere Zeit stabil?

**Ablauf:** Der gemischte Betrieb der Stufe 6 läuft 60 Minuten mit 70 % der dort getragenen Rate (`share_of_sustained`). Die Rate stammt aus `summary_s6.json` desselben Laufs; ohne sie muss `stages.s7.rate` gesetzt sein.

**Ergebnis:** Antwortzeiten und Durchsatz in Zeitscheiben von fünf Minuten, die Veränderung vom ersten zum letzten Drittel in Prozent, alle Fehler, dazu der Verlauf von GPU-Speicher und freiem Arbeitsspeicher. Ein stetig sinkender freier Arbeitsspeicher oder stetig steigende Antwortzeiten deuten auf ein Problem, das im kurzen Test nicht auffällt.

## Stufe 8, Qualität

**Frage:** Wie gut sind die Antworten?

**Ablauf:** Jede Frage des Fragensatzes `quality/fragen.yaml` geht einmal ohne und einmal mit Nachdenken an das Modell. Die 26 Fragen verteilen sich auf sieben Kategorien:

| Kategorie | Fragen | Prüfung |
| --- | --- | --- |
| `firmenwissen` | 5 | feste Begriffe, dazu von Hand |
| `ehrlichkeit` | 4 | von Hand: Gibt das Modell zu, was es nicht wissen kann, statt etwas zu erfinden? |
| `deutsch` | 4 | teils feste Begriffe, dazu von Hand; fremde Schriftzeichen werden gezählt |
| `denken` | 6 | eindeutige Lösung, automatisch |
| `langer_kontext` | 2 | eine Angabe in einem Text von 60.000 und 180.000 Tokens wiederfinden, automatisch |
| `werkzeuge` | 4 | richtiges Werkzeug mit richtigen Angaben, automatisch |
| `struktur` | 1 | gültiges JSON mit den verlangten Schlüsseln und Werten, automatisch |

Anders als in den Leistungsstufen bleiben die Antworten erhalten: `answers/s8.jsonl` enthält je Frage die Antwort, die Referenzantwort und das Ergebnis der Prüfung.

**Bewertung von Hand, blind:**

```bash
.venv/bin/styrke bewertung export --out bewertung            # neuester Lauf je Zelle und Variante
.venv/bin/styrke bewertung import bewertung/bogen.csv bewertung/schluessel.json
```

`export` schreibt `bogen.csv` (Frage, Referenzantwort, Antwort, leere Spalte für 0 bis 3 Punkte) mit den Antworten aller Läufe in gemischter Reihenfolge und ohne Herkunft, dazu `anleitung.md` für die bewertende Person. `schluessel.json` ordnet die Zeilen den Läufen zu und bleibt beim Auftraggeber. `import` rechnet aus dem ausgefüllten Bogen mittlere Punkte je Lauf und Kategorie.

**Zu beachten:**

- Die Referenzantworten zum Firmenwissen stammen aus öffentlichen Quellen, die im Fragensatz genannt sind. Sie sollten vor der Bewertung von jemandem aus dem Unternehmen gegengelesen werden.
- Die Werkzeuge sind nachgebaut: Geprüft wird, ob das Modell das richtige Werkzeug verlangt; ausgeführt wird nichts. Ob eine Engine Werkzeugaufrufe überhaupt erkennt, hängt von ihren Startoptionen ab (`engine_args` der Modelle in der Testmatrix). Lehnt sie ab, steht der Fehler im Ergebnis.
- Jede Frage läuft einmal. Bei den eingestellten Sampling-Werten kann dieselbe Frage beim nächsten Mal anders ausgehen; knappe Unterschiede zwischen zwei Modellen sagen deshalb wenig.
- Eine automatische Prüfung auf feste Begriffe ist streng in eine Richtung: Sie erkennt fehlende Angaben, aber keine zusätzlichen falschen. Dafür ist die Bewertung von Hand da.
