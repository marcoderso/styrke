# styrke

styrke misst, wie gut ein Sprachmodell mit einer bestimmten Inferenz-Engine auf einer gegebenen Maschine arbeitet: Antwortzeiten, Durchsatz unter Last, Verhalten bei langen Prompts. Modelle, Engines und Konfigurationen stehen in einer Testmatrix; jede Kombination durchläuft dieselben Stufen, und alle Messwerte landen als Rohdaten je Anfrage auf der Platte.

**Stand: Schritt 1 von 7.** Vorhanden sind das Gerüst, die Testmatrix, die Rohdatenablage, der Adapter für FreeToken, der Lastgenerator mit den Stufen 0 bis 2 und der Mitschreiber. Die Adapter für vLLM und SGLang, die Stufen 3 bis 8, die TUI und der Bericht folgen.

| Stufe | Zweck | Stand |
| --- | --- | --- |
| 0 Machbarkeit | Lädt die Zelle, wie lange, welcher Kontext geht? | vorhanden |
| 1 Einzelstrom | Grundwerte ohne Konkurrenz, über die Promptlänge | vorhanden |
| 2 Parallelität | Verhalten unter steigender Dauerlast, bis zur Sättigung | vorhanden |
| 3 KV-Füllstand | Einfluss eines vollen Cache | folgt |
| 4 Präfix-Cache | Nutzen des Cache bei Folgefragen | folgt |
| 5 Störwirkung | Was eine große Anfrage den anderen antut | folgt |
| 6 Szenario | Tragfähigkeit im gemischten Betrieb | folgt |
| 7 Dauerlauf | Stabilität | folgt |
| 8 Qualität | Antwortgüte | folgt |

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

Falls `corpus fetch` die Quelle nicht erreicht: eigene deutsche Textdateien (`*.txt`, zusammen mindestens einige Megabyte) in den Ordner `corpus/` legen. Ohne Korpus greift ein eingebauter Mini-Korpus, der nur für Funktionstests taugt.

## Messen

Alle Kommandos im Ordner des Repositorys ausführen.

**Funktionstest** gegen eine schon laufende Engine, ohne sie anzuhalten. Dauert etwa eine Minute und zeigt, ob alles zusammenspielt; die Zahlen sind keine Messung:

```bash
.venv/bin/styrke run --cell flashnext-freetoken --endpoint http://127.0.0.1:8000 --smoke
```

**Messlauf.** styrke startet die Engine selbst als Container, jede Stufe mit frisch gestarteter Engine. Die GPU muss dafür frei sein; ein anderer Dienst auf der Karte ist vorher zu stoppen. Der Lauf dauert bei einem großen Modell ein bis zwei Stunden, deshalb in `tmux` starten:

```bash
tmux new -s styrke
.venv/bin/styrke run --cell flashnext-freetoken --variant basis
```

Danach:

```bash
.venv/bin/styrke show runs/<laufordner>                               # Kurzauswertung
.venv/bin/styrke run --cell flashnext-freetoken --resume runs/<laufordner>   # nach Abbruch fortsetzen
```

Nützliche Schalter von `styrke run`:

| Schalter | Wirkung |
| --- | --- |
| `--variant ID` | Variante der Zelle; ohne Angabe die erste |
| `--stages 0,1` | nur diese Stufen |
| `--endpoint URL` | an eine laufende Engine anhängen statt selbst zu starten |
| `--smoke` | kleine Parameter für einen schnellen Funktionstest |
| `--resume ORDNER` | abgeschlossene Stufen überspringen |
| `--allow-busy-gpu` | auch starten, wenn die GPU schon belegt ist |

Ein Fehlstart ist ein Ergebnis: Lädt eine Zelle nicht, steht der Grund in `feasibility.json`, und der Rückgabewert ist 2.

## Testmatrix

`matrix.yaml` hat fünf Abschnitte:

- `host`: Pfade und Adresse der Maschine.
- `engines`: je Engine das Image, den Port im Container und Grundoptionen.
- `models`: je Modell das Repository auf Hugging Face, die Kontextlänge und wie sich das Nachdenken (Thinking) schalten lässt.
- `cells`: welches Modell auf welcher Engine läuft, mit Varianten. Die Angaben unter `args` einer Variante werden zu Optionen der Engine: `max-running-requests: 8` wird zu `--max-running-requests 8`.
- `stages`: Parameter der Stufen, soweit sie von den Standardwerten abweichen sollen (Standardwerte: `DEFAULT_STAGES` in `src/styrke/config.py`).

Beim FreeToken-Adapter gleicht styrke die Optionen vor dem Start mit `ft serve --help` des Images ab. Eine umbenannte Option (`moe-strategy` und `moe-backend`) wird passend gewählt, eine unbekannte führt zu einer klaren Fehlermeldung statt zu einem Container, der sofort wieder endet.

## Rohdaten

Jeder Lauf schreibt einen Ordner unter `runs/`:

```text
manifest.json     Hardware, Image, Modell, alle Startoptionen und Stufenparameter
requests.jsonl    eine Zeile je Anfrage, mit Zeitstempeln der Antwortstücke
samples.jsonl     GPU, Rechner und Engine in festen Abständen
events.jsonl      Ereignisse (Stufen, Laststufen, Engine-Start)
engine.log        Ausgabe der Engine
feasibility.json  Ergebnis der Stufe 0
calibration.json  Eichung der Promptlänge
summary_s1.json   Kurzauswertung der Stufe 1
summary_s2.json   Kurzauswertung der Stufe 2
state.json        abgeschlossene Stufen
answers/          Antworttexte, wo sie ausdrücklich aufbewahrt werden
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

Die Tests brauchen weder GPU noch Docker. Sie laufen gegen eine Attrappe (`tests/mockserver.py`), die sich in den gemessenen Punkten wie eine Engine verhält: begrenztes Einlesetempo, Tokens mit einer Rate, eine Höchstzahl gleichzeitiger Anfragen mit Warteschlange.
