# styrke

styrke misst, wie gut ein Sprachmodell mit einer bestimmten Inferenz-Engine auf einer gegebenen Maschine arbeitet: Antwortzeiten, Durchsatz unter Last, Verhalten bei langen Prompts. Modelle, Engines und Konfigurationen stehen in einer Testmatrix; jede Kombination durchläuft dieselben Stufen, und alle Messwerte landen als Rohdaten je Anfrage auf der Platte.

**Stand:** Alle Teile des Plans sind gebaut: die Testmatrix mit sieben Zellen, die Adapter für FreeToken, vLLM und SGLang, die Stufen 0 bis 8, der Betrachter für laufende Tests, der Bericht, die Blindbewertung und der Mitschreiber. An einer echten Engine gestartet wurde bisher nur FreeToken mit den Stufen 0 bis 2; alles andere ist gegen eine nachgebaute Engine geprüft, die Adapter für vLLM und SGLang nur gegen deren Dokumentation.

| Stufe | Frage | Stand | Dauer, grob |
| --- | --- | --- | --- |
| 0 Machbarkeit | Lädt die Zelle, wie lange, welcher Kontext geht? | vorhanden | Ladezeit plus wenige Minuten |
| 1 Einzelstrom | Wie schnell ist eine einzelne Anfrage, je nach Promptlänge? | vorhanden | 10 bis 30 Minuten |
| 2 Parallelität | Wie viel Durchsatz bei steigender Dauerlast, bis zur Sättigung? | vorhanden | 20 Minuten |
| 3 KV-Füllstand | Schadet ein voller Cache, und behält er wartende Gespräche? | vorhanden | 30 Minuten |
| 4 Präfix-Cache | Wie viel schneller sind Folgefragen, und wie lange? | vorhanden | 30 Minuten |
| 5 Störwirkung | Was tut eine große Anfrage den laufenden Chats an? | vorhanden | 10 Minuten |
| 6 Szenario | Welche Last trägt die Zelle im gemischten Betrieb? | vorhanden | bis zu 2 Stunden |
| 7 Dauerlauf | Bleibt sie über eine Stunde stabil? | vorhanden | 65 Minuten |
| 8 Qualität | Wie gut sind die Antworten? | vorhanden | 10 bis 30 Minuten |

Die Dauern sind aus den Parametern geschätzt, nicht gemessen; sie hängen vom Einlesetempo der Engine ab. Dazu kommt vor jeder Stufe die Ladezeit des Modells. Was jede Stufe genau tut und wie ihr Ergebnis zu lesen ist, steht in [docs/stufen.md](docs/stufen.md).

## Einrichten

Voraussetzungen: Linux mit Python 3.10 oder neuer, Docker mit NVIDIA Container Toolkit, eine NVIDIA-GPU.

```bash
apt install -y python3-venv git
git clone https://github.com/marcoderso/styrke /opt/styrke
cd /opt/styrke
python3 -m venv .venv
.venv/bin/pip install -e .

cp local.example.yaml local.yaml      # Pfade der Maschine eintragen
.venv/bin/styrke matrix               # zeigt Zellen und Varianten
.venv/bin/styrke corpus fetch         # lädt rund 40 MB deutsche Wikipedia-Artikel als Prompt-Material
```

`local.yaml` überschreibt einzelne Werte der Testmatrix und bleibt auf der Maschine (steht in `.gitignore`). Wichtig sind `host.models_dir` (dort liegen die Modellordner) und `host.cache_dir`.

Für einen Messlauf braucht jede Engine ihr Image. Das für FreeToken wird gebaut (dauert einige Minuten):

```bash
docker build -t styrke/freetoken:0.2.1 engines/freetoken
```

Liegt schon ein passendes Image auf der Maschine, genügt ein Eintrag in `local.yaml`; die Vorlage enthält ihn auskommentiert:

```yaml
engines:
  freetoken:
    image: <name>:<tag>
```

Die Images für vLLM und SGLang werden gezogen, die Modelle einmal heruntergeladen:

```bash
docker pull vllm/vllm-openai:latest
docker pull lmsysorg/sglang:latest
hf download Qwen/Qwen3.8-27B-FP8 --local-dir <models_dir>/Qwen3.8-27B-FP8
```

Beide Images stehen in der Testmatrix auf `latest`, weil die Modelle neu sind. Für eine Messreihe das Image einmal ziehen und danach nicht erneuern; welches benutzt wurde, steht als `image_id` im Manifest jedes Laufs. Fehlt vor einem Lauf ein Image oder ein Modell, nennt styrke das passende Kommando.

`corpus fetch` lässt sich fortsetzen: Was schon im Ordner `corpus/` liegt, zählt mit und wird nicht erneut geladen. Antwortet die Quelle zeitweise nicht, wartet der Abruf mit wachsenden Pausen und gibt erst nach gut zehn Minuten auf; derselbe Aufruf macht später dort weiter. Ein Teilbestand ist bereits nutzbar, `styrke corpus info` zeigt seine Größe.

Ist die Quelle gar nicht erreichbar: eigene deutsche Textdateien (`*.txt`, zusammen mindestens einige Megabyte) in den Ordner `corpus/` legen. Ohne Korpus greift ein eingebauter Mini-Korpus, der nur für Funktionstests taugt.

## Messen

Alle Kommandos im Ordner des Repositorys ausführen.

**Funktionstest** gegen eine schon laufende Engine, ohne sie anzuhalten. Er fährt alle Stufen mit winzigen Parametern, dauert wenige Minuten und zeigt, ob alles zusammenspielt; die Zahlen sind keine Messung:

```bash
.venv/bin/styrke run --cell flashnext-freetoken --endpoint http://127.0.0.1:8000 --smoke
```

**Messlauf.** styrke startet die Engine selbst als Container, jede Stufe mit frisch gestarteter Engine. Die GPU muss dafür frei sein; ein anderer Dienst auf der Karte ist vorher zu stoppen. Ohne `--stages` laufen alle Stufen, das dauert mehrere Stunden. Deshalb in `tmux` starten:

```bash
tmux new -s styrke
.venv/bin/styrke run --cell flashnext-freetoken --variant basis                 # alle Stufen
.venv/bin/styrke run --cell flashnext-freetoken --variant basis --stages 0-2    # nur die ersten drei
```

Danach:

```bash
.venv/bin/styrke show runs/<laufordner>                               # Kurzauswertung
.venv/bin/styrke run --cell flashnext-freetoken --resume runs/<laufordner>   # fortsetzen
```

`--resume` überspringt abgeschlossene Stufen und fährt die übrigen im selben Laufordner. So lässt sich ein abgebrochener Lauf fortsetzen und ein Lauf mit den Stufen 0 bis 2 später um die Stufen 3 bis 7 ergänzen. Eine abgebrochene Stufe beginnt dabei von vorn.

**Gesamtlauf über alle Zellen.** Ein Aufruf misst alle sieben Zellen nacheinander und schreibt am Ende den Bericht:

```bash
tmux new -s styrke
.venv/bin/styrke all --vorbereiten
```

`--vorbereiten` zieht fehlende Images und lädt fehlende Modelle herunter, bevor die Messung beginnt; ohne den Schalter nennt styrke nur, was fehlt. Je Zelle läuft die erste Variante, mit `--varianten alle` jede. Vorhandene Läufe werden fortgesetzt und fertige übersprungen: Nach einem Abbruch genügt derselbe Aufruf noch einmal. Mit allen Stufen dauert eine Zelle etwa fünf Stunden, der Gesamtlauf also gut einen Tag; `--stages 0-2` liefert in wenigen Stunden einen ersten Vergleich, den ein späterer Aufruf ohne `--stages` vervollständigt. Mit `--cells a,b` lässt sich die Auswahl einschränken.

**Zusehen.** In einem zweiten Fenster zeigt der Betrachter den Stand des neuesten Laufs: Stufen mit Dauer und Fehlern, Durchsatz und Antwortzeiten der letzten 30 Sekunden, GPU, Arbeitsspeicher und das Protokoll. Er liest nur die Dateien des Laufordners, lässt sich jederzeit beenden und funktioniert auch für beendete Läufe.

```bash
.venv/bin/styrke watch                    # folgt dem jeweils neuesten Lauf; beenden mit Strg+C
.venv/bin/styrke watch runs/<laufordner> --once
```

Nützliche Schalter von `styrke run`:

| Schalter | Wirkung |
| --- | --- |
| `--variant ID` | Variante der Zelle; ohne Angabe die erste |
| `--stages 0,1` oder `--stages 3-7` | nur diese Stufen; ohne Angabe alle |
| `--endpoint URL` | an eine laufende Engine anhängen statt selbst zu starten |
| `--smoke` | kleine Parameter für einen schnellen Funktionstest |
| `--resume ORDNER` | abgeschlossene Stufen überspringen |
| `--redo` | mit `--resume`: die gewählten Stufen wiederholen, auch wenn sie abgeschlossen sind; die Auswertung nimmt den letzten Anlauf |
| `--allow-busy-gpu` | auch starten, wenn die GPU schon belegt ist |

Vor dem Start prüft styrke die Vorbereitung: Image vorhanden, Modell vorhanden, GPU frei. Fehlt etwas, nennt styrke alle Punkte auf einmal, legt keinen Laufordner an und endet mit dem Rückgabewert 3.

Ein Fehlstart der Engine ist dagegen ein Ergebnis: Lädt eine Zelle nicht, steht der Grund in `feasibility.json`, und der Rückgabewert ist 2.

Scheitert eine einzelne Stufe, laufen die übrigen trotzdem; ein langer Lauf über Nacht geht also nicht an einem Fehler in der Mitte verloren. Am Ende nennt styrke die nicht abgeschlossenen Stufen, `--resume` holt sie nach. Nach zwei Engine-Fehlern in Folge endet der Lauf. Ein Fehler in styrke selbst steht mit allen Einzelheiten in `stage_errors.log` im Laufordner; der Rückgabewert ist dann 1.

## Bericht

```bash
.venv/bin/styrke report --out bericht.html
.venv/bin/styrke report --ttft-p95 2 --kosten-monat 9000 --out bericht.html
.venv/bin/styrke report runs/<lauf-a> runs/<lauf-b> --out vergleich.html
```

Der Bericht ist eine einzelne HTML-Datei ohne äußere Abhängigkeiten, die sich weitergeben und im Browser öffnen lässt. Ohne Angabe von Laufordnern nimmt er je Zelle und Variante den neuesten Lauf unter `runs/`; Funktionstests bleiben weg. Er enthält:

- **Kennzahlen für die Entscheidung:** je Lauf Ladezeit, Tempo einer einzelnen Anfrage, höchster Durchsatz, Gewinn bei Folgefragen, Einbruch bei einer großen Anfrage, die getragene Rate und die Zahl der Berechtigten, die sie bedient.
- **Diagramme:** Durchsatz über die Zahl gleichzeitiger Anfragen, Zeit bis zum ersten Token über die Promptlänge und über die Ankunftsrate.
- **Alle Stufen je Lauf** als Tabellen, zum Aufklappen.

Zielwerte und Kosten kommen nachträglich dazu; gemessen wird dafür nichts neu:

| Schalter | Wirkung |
| --- | --- |
| `--ttft-p95 S` | nur Raten zählen, bei denen 95 % der Aufrufe ihr erstes Token nach höchstens S Sekunden haben |
| `--dauer-p95 S` | ebenso für die Gesamtdauer eines Aufrufs |
| `--kosten-monat EUR` | Monatskosten der Maschine; daraus Kosten je Berechtigtem und je Million Ausgabe-Tokens |
| `--profil NAME` | Nutzungsprofil des Lastmodells für die Zahl der Berechtigten (Standard: `mittel`) |

## Testmatrix

`matrix.yaml` hat fünf Abschnitte:

- `host`: Pfade und Adresse der Maschine.
- `engines`: je Engine das Image, den Port im Container und Grundoptionen.
- `models`: je Modell das Repository auf Hugging Face, die Kontextlänge und wie sich das Nachdenken (Thinking) schalten lässt.
- `cells`: welches Modell auf welcher Engine läuft, mit Varianten. Die Angaben unter `args` einer Variante werden zu Optionen der Engine: `max-running-requests: 8` wird zu `--max-running-requests 8`.
- `stages`: Parameter der Stufen, soweit sie von den Standardwerten abweichen sollen (Standardwerte: `DEFAULT_STAGES` in `src/styrke/config.py`).

Vor dem Start gleicht styrke die Optionen mit der Hilfe des Images ab. Eine unbekannte Option führt zu einer klaren Fehlermeldung statt zu einem Container, der sofort wieder endet; bei FreeToken wird eine umbenannte Option (`moe-strategy` und `moe-backend`) passend gewählt. Eine Option mit Fragezeichen am Ende (`"enable-metrics?": true`) ist ein Wunsch: Kennt das Image sie nicht, entfällt sie, und das Manifest nennt sie unter `dropped_flags`.

Optionen, die ein Modell bei einer Engine braucht, stehen am Modell unter `engine_args`, etwa der Reasoning-Parser der Qwen-Modelle. Die Reihenfolge, in der sich Angaben überschreiben: Adapter, `base_args` der Engine, `engine_args` des Modells, `args` der Variante.

## Rohdaten

Jeder Lauf schreibt einen Ordner unter `runs/`:

```text
manifest.json     Hardware, Image, Modell, alle Startoptionen und Stufenparameter
requests.jsonl    eine Zeile je Anfrage, mit Zeitstempeln der Antwortstücke
samples.jsonl     GPU, Rechner und Engine in festen Abständen
events.jsonl      Ereignisse (Stufen, Laststufen, Engine-Start)
run.log           die Ausgabe von styrke run, für den Betrachter
engine.log        Ausgabe der Engine
feasibility.json  Ergebnis der Stufe 0
calibration.json  Eichung der Promptlänge
summary_sN.json   Kurzauswertung der Stufe N (1 bis 8)
stage_errors.log  nur wenn eine Stufe an einem Fehler in styrke selbst gescheitert ist
state.json        abgeschlossene Stufen
answers/          Antworttexte der Sinnprobe und der Qualitätsstufe (s8.jsonl)
```

Die Felder beschreibt [docs/rohdaten.md](docs/rohdaten.md). Antworttexte werden bei den Leistungsstufen nicht gespeichert. Zielwerte wie eine Höchstzeit bis zum ersten Token lassen sich nachträglich auf die Rohdaten anwenden; neu gemessen werden muss dafür nichts.

## Mitschreiber

Der Mitschreiber ist ein durchreichender Proxy für den Testbetrieb. Er sitzt zwischen den Nutzern und der Engine, reicht jede Anfrage unverändert weiter und hält je Anfrage nur Längen und Zeiten fest. Prompts, Antworten, Adressen und Nutzernamen werden nicht gespeichert; Absender erscheinen als unumkehrbare Kennung.

```bash
# Die Engine lauscht auf 8001, der Mitschreiber übernimmt die bisherige Adresse 8000.
.venv/bin/styrke mitschreiber serve --listen 127.0.0.1:8000 --upstream http://127.0.0.1:8001 --out usage

.venv/bin/styrke mitschreiber report --out usage --berechtigte 400
```

Die Auswertung nennt Verteilungen von Prompt- und Antwortlängen und schätzt für die Spitzenstunde die Größen des Lastmodells. Zwei Hinweise:

- **Tokenzahlen.** Bei gestreamten Antworten liefert die Engine Zählwerte nur, wenn der Aufrufer sie anfordert. `--inject-usage` fordert sie an (der Aufrufer bekommt dann ein zusätzliches letztes Stück mit den Zählwerten); `--tokenizer <modellordner>/tokenizer.json` zählt stattdessen lokal, ohne die Anfrage zu verändern.
- **Nutzer unterscheiden.** Kommen alle Anfragen vom selben Backend, sieht der Mitschreiber nur einen Absender. Einzelne Nutzer lassen sich dann nur zählen, wenn das Backend das Feld `user` mitschickt.

Eine Vorlage für den Betrieb als Dienst liegt in `deploy/styrke-mitschreiber.service`.

## Lastmodell

```bash
.venv/bin/styrke lastmodell
```

rechnet aus der Zahl der Berechtigten und drei Nutzungsprofilen (Abschnitt `load_model` der Testmatrix) die Zahl gleichzeitiger Anfragen in der Spitzenstunde. Die Formel steht in `src/styrke/loadmodel.py`.

## Entwicklung

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Die Tests brauchen weder GPU noch Docker. Sie laufen gegen eine Attrappe (`tests/mockserver.py`), die sich in den gemessenen Punkten wie eine Engine verhält: begrenztes Einlesetempo, Tokens mit einer Rate, eine Höchstzahl gleichzeitiger Anfragen mit Warteschlange, ein Präfix-Cache mit begrenztem Pool und Verdrängung.

Alle Stufen von Hand gegen die Attrappe, in zwei Fenstern:

```bash
.venv/bin/python tests/mockserver.py
.venv/bin/styrke run --cell flashnext-freetoken --endpoint http://127.0.0.1:18999 --smoke
```
