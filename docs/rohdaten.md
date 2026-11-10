# Rohdaten

Alle Dateien eines Laufs sind zeilenweises JSON (eine Zeile, ein Datensatz) oder einfache JSON-Dateien. Zeiten ohne weitere Angabe sind Sekunden.

## requests.jsonl

Eine Zeile je Anfrage, auch für fehlgeschlagene und abgebrochene.

| Feld | Bedeutung |
| --- | --- |
| `id` | Kennung der Anfrage |
| `ts_submit` | Zeitpunkt des Absendens (Unixzeit) |
| `stage` | Stufe, in der die Anfrage entstand |
| `attempt` | der wievielte Anlauf der Stufe; nach einem Abbruch und Fortsetzen größer als 1. Für Auswertungen je Stufe den höchsten Wert nehmen |
| `what` | Zweck der Anfrage, soweit sie keine gewöhnliche Lastanfrage ist; die Werte stehen unter der Tabelle |
| `thinking` | `off` oder `on` |
| `prompt_target` | angestrebte Promptlänge in Tokens |
| `prompt_local` | Promptlänge nach eigener Zählung |
| `level`, `worker`, `seq` | Dauerlast der Stufen 2, 3 und 5: Zahl paralleler Arbeiter, Nummer des Arbeiters, laufende Nummer |
| `rep` | Stufen 1, 4 und 5: Wiederholung |
| `fill` | Stufe 3: Füllstand des KV-Pools als Anteil, zu dem die Anfrage gehört |
| `session`, `turn` | Gespräche der Stufen 3, 4, 6 und 7: Kennung des Gesprächs und Nummer des Zugs (0 ist der erste, kalte Zug) |
| `kind`, `rate` | Stufen 6 und 7: Sitzungsart und vorgegebene Ankunftsrate in Aufrufen pro Sekunde |
| `part`, `idle_s` | Stufe 4: `eviction` für den Teil zur Verdrängung; Ruhezeit vor der Anfrage |
| `max_tokens` | Obergrenze der Antwortlänge |
| `ok` | Antwort vollständig erhalten |
| `http_status`, `error` | Statuscode und Fehlertext; `abgebrochen` heißt, styrke hat die Anfrage am Ende einer Laststufe beendet |
| `ttft_s` | Zeit bis zum ersten Stück der Antwort, gleich welcher Art |
| `ttfr_s` | Zeit bis zum ersten Stück Nachdenken |
| `ttfc_s` | Zeit bis zum ersten Stück der eigentlichen Antwort |
| `total_s` | Gesamtdauer |
| `n_chunks` | Zahl der Antwortstücke |
| `chunk_ms` | Zeitpunkt jedes Stücks in Millisekunden ab dem Absenden |
| `chunk_kinds` | je Stück ein Zeichen: `r` Nachdenken, `c` Antwort, `t` Werkzeugaufruf |
| `chars_reasoning`, `chars_content` | Zeichenzahl von Nachdenken und Antwort |
| `prompt_tokens`, `completion_tokens` | Zählwerte der Engine |
| `cached_tokens` | Prompt-Tokens, die aus dem Präfix-Cache kamen, soweit die Engine das meldet |
| `reasoning_tokens` | Tokens des Nachdenkens, soweit gemeldet |
| `usage` | der Zählblock der Engine, unverändert |
| `usage_source` | woher `completion_tokens` stammt: `stream` (Engine), `tokenizer` (lokal gezählt), `chunks` (Zahl der Stücke als Näherung) |
| `finish_reason` | Grund für das Ende laut Engine |
| `decode_tps` | Tokens pro Sekunde in der Erzeugungsphase: `(completion_tokens - 1) / (total_s - ttft_s)` |

Werte von `what`:

| Wert | Stufe | Bedeutung |
| --- | --- | --- |
| `warmup` | alle | Aufwärmen nach dem Start der Engine; zählt in keiner Auswertung |
| `sanity`, `sanity_thinking`, `context_probe` | 0 | Sinnprobe ohne und mit Nachdenken, Kontextprobe |
| `fill`, `probe` | 3 | erster Zug eines gehaltenen Gesprächs; Folgefrage nach der Last |
| `cold`, `followup`, `after_pause` | 4 | erster Zug, Folgefrage sofort, Folgefrage nach Ruhezeit |
| `before_eviction`, `flood`, `after_eviction` | 4 | Folgefrage vor der Verdrängung, fremder Text, Folgefrage danach |
| `background`, `big` | 5 | laufender Chat; die große Anfrage |

Bei Gesprächen ist `prompt_target` die Länge des ersten Prompts; die tatsächliche Länge jedes Zugs steht in `prompt_tokens`. Die Antworten des Modells, die ein Gespräch im nächsten Zug mitschickt, stehen nicht in den Rohdaten.

Ein Stück kann mehrere Tokens enthalten. Für den Durchsatz je Zeitfenster verteilt styrke die Tokenzahl einer Anfrage gleichmäßig auf ihre Stücke.

## samples.jsonl

Messreihen, unterschieden durch `kind`.

| `kind` | Felder |
| --- | --- |
| `gpu` | `gpu` (Index), `util_pct`, `mem_used_mib`, `mem_total_mib`, `power_w`, `temp_c`; im Sekundentakt |
| `host` | `load1`, `mem_total_gib`, `mem_available_gib`, `styrke_cpu_s` (CPU-Zeit von styrke selbst, um den eigenen Einfluss zu sehen) |
| `engine` | Kennzahlen der Engine, unverändert übernommen; bei FreeToken `v1_stats` und `v1_cache_status` |
| `load` | Stufen 6 und 7, im Sekundentakt: `rate`, `inflight` (Aufrufe, die gerade an der Engine laufen), `sessions` (offene Gespräche) |

## events.jsonl

`run_start`, `stage_start`, `engine_start` (mit dem vollständigen Docker-Kommando), `level_start`, `level_end`, `stage_end`, `run_end`, jeweils mit `ts`. Dazu aus einzelnen Stufen: `fill_start`, `fill_done` (Stufe 3), `flood_start`, `flood_end` (Stufe 4), `injection_start`, `injection_end` (Stufe 5) und `worker_error`, falls ein Lastarbeiter unerwartet endet. Bei `level_start` ist `level` in den Stufen 2 und 3 die Zahl paralleler Anfragen, in Stufe 6 die Ankunftsrate.

## feasibility.json

Ergebnis der Stufe 0: `loaded`, `load_time_s`, Antwort von `/v1/models`, Ergebnis der Sinnprobe, ob Zählwerte im Stream kommen, ob sich das Nachdenken schalten lässt, die Eichung, die Kontextproben mit `max_context_ok`, der Zustand der GPU und die Kennzahlen der Engine nach dem Laden. Lädt die Zelle nicht, stehen dort `loaded: false` und der Grund.

## calibration.json

So trifft styrke die angestrebte Promptlänge: Mit dem Tokenizer des Modells wird lokal gezählt und nur der Aufschlag des Chat-Templates an der Engine gemessen (`overhead_tokens`). Ohne Tokenizer wird zusätzlich das Verhältnis Zeichen pro Token gemessen (`chars_per_token`). `check_deviation_pct` nennt die Abweichung einer Kontrollanfrage.

## Kurzauswertungen (summary_sN.json)

So zeigt `styrke show` sie an. Sie sind ein erster Blick; alles darin lässt sich aus `requests.jsonl` neu rechnen. Was die Stufen tun, steht in [stufen.md](stufen.md).

| Datei | Inhalt |
| --- | --- |
| `summary_s1.json` | Mediane je Denkmodus und Promptlänge; `skipped` nennt ausgelassene Längen |
| `summary_s2.json` | je Laststufe der Durchsatz je Messfenster, Antwortzeiten, Fehler; `saturated_at` und der Grund für das Ende |
| `summary_s3.json` | `fills`: je Füllstand die gehaltenen Gespräche, die Kennzahlen der Engine und der Anteil, der nach der Last noch im Cache lag (`retained_by_ttft`, `retained_by_cache_report`). `rows`: je Füllstand und Laststufe dieselben Felder wie in Stufe 2 |
| `summary_s4.json` | `rows`: je Promptlänge die Zeit bis zum ersten Token kalt, bei der Folgefrage und nach der Pause, der Faktor (`speedup`) und der Anteil aus dem Cache. `eviction`: dieselbe Folgefrage vor und nach der Verdrängung |
| `summary_s5.json` | je großer Anfrage ihre eigenen Zeiten und für die Abschnitte `baseline`, `prefill`, `decode`, `recovery` der Durchsatz der Chats (`tps`), die Abstände zwischen Tokens (`gap_*_ms`), die längste Zeit ohne Token (`silence_max_s`) und die Zeit bis zum ersten Token neu begonnener Chats |
| `summary_s6.json` | je Rate: `achieved_rate` (tatsächlich eingetroffene Aufrufe pro Sekunde), Antwortzeiten gesamt, je Sitzungsart (`by_kind`) und getrennt nach erstem Zug und Folgefrage (`by_turn`), `service_mean_s`, `inflight_*`, `overload` (Gründe; leer heißt getragen). Dazu `max_sustained_rate` und `users_supported` je Nutzungsprofil |
| `summary_s7.json` | `overall` wie eine Zeile der Stufe 6, `buckets` je Zeitscheibe, die Veränderung vom ersten zum letzten Drittel (`*_drift_pct`) und `resources` mit GPU-Speicher und freiem Arbeitsspeicher am Anfang und am Ende |

Steht in einer Kurzauswertung `skipped`, wurde die Stufe ausgelassen; der Wert nennt den Grund.

## Mitschrift (usage/usage-JJJJ-MM-TT.jsonl)

Eine Zeile je Anfrage an die Engine, ohne Inhalte.

| Feld | Bedeutung |
| --- | --- |
| `ts` | Zeitpunkt der Anfrage (Unixzeit) |
| `client`, `user` | unumkehrbare Kennungen von Absenderadresse und Feld `user` |
| `method`, `path`, `status` | Art der Anfrage und Statuscode |
| `stream` | gestreamte Antwort verlangt |
| `model` | angefragter Modellname |
| `n_messages`, `prompt_chars`, `system_chars` | Zahl der Nachrichten, Zeichenzahl des gesamten Prompts und des System-Prompts |
| `n_images`, `n_tools`, `tools_chars` | Zahl der Bilder und Werkzeuge, Größe der Werkzeugbeschreibungen |
| `max_tokens`, `temperature`, `top_p`, `top_k`, `enable_thinking` | mitgeschickte Einstellungen |
| `prompt_tokens`, `completion_tokens`, `cached_tokens`, `reasoning_tokens` | Zählwerte, soweit verfügbar |
| `tokens_source` | `usage` (Engine) oder `tokenizer` (lokal gezählt) |
| `out_chars`, `reasoning_chars` | Zeichenzahl von Antwort und Nachdenken |
| `tool_call`, `finish_reason` | Werkzeugaufruf in der Antwort, Grund für das Ende |
| `ttfb_s`, `ttft_s`, `duration_s` | Zeit bis zur Antwort der Engine, bis zum ersten Stück Text, Gesamtdauer |
| `aborted`, `error` | Aufrufer hat abgebrochen; Fehler beim Weiterreichen |
