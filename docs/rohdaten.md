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
| `what` | Zweck bei Sonderanfragen: `warmup`, `sanity`, `sanity_thinking`, `context_probe` |
| `thinking` | `off` oder `on` |
| `prompt_target` | angestrebte Promptlänge in Tokens |
| `prompt_local` | Promptlänge nach eigener Zählung |
| `level`, `worker`, `seq` | Stufe 2: Zahl paralleler Arbeiter, Nummer des Arbeiters, laufende Nummer |
| `rep` | Stufe 1: Wiederholung |
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

Ein Stück kann mehrere Tokens enthalten. Für den Durchsatz je Zeitfenster verteilt styrke die Tokenzahl einer Anfrage gleichmäßig auf ihre Stücke.

## samples.jsonl

Messreihen, unterschieden durch `kind`.

| `kind` | Felder |
| --- | --- |
| `gpu` | `gpu` (Index), `util_pct`, `mem_used_mib`, `mem_total_mib`, `power_w`, `temp_c`; im Sekundentakt |
| `host` | `load1`, `mem_total_gib`, `mem_available_gib`, `styrke_cpu_s` (CPU-Zeit von styrke selbst, um den eigenen Einfluss zu sehen) |
| `engine` | Kennzahlen der Engine, unverändert übernommen; bei FreeToken `v1_stats` und `v1_cache_status` |

## events.jsonl

`run_start`, `stage_start`, `engine_start` (mit dem vollständigen Docker-Kommando), `level_start`, `level_end`, `stage_end`, `run_end`, jeweils mit `ts`.

## feasibility.json

Ergebnis der Stufe 0: `loaded`, `load_time_s`, Antwort von `/v1/models`, Ergebnis der Sinnprobe, ob Zählwerte im Stream kommen, ob sich das Nachdenken schalten lässt, die Eichung, die Kontextproben mit `max_context_ok`, der Zustand der GPU und die Kennzahlen der Engine nach dem Laden. Lädt die Zelle nicht, stehen dort `loaded: false` und der Grund.

## calibration.json

So trifft styrke die angestrebte Promptlänge: Mit dem Tokenizer des Modells wird lokal gezählt und nur der Aufschlag des Chat-Templates an der Engine gemessen (`overhead_tokens`). Ohne Tokenizer wird zusätzlich das Verhältnis Zeichen pro Token gemessen (`chars_per_token`). `check_deviation_pct` nennt die Abweichung einer Kontrollanfrage.

## summary_s1.json und summary_s2.json

Kurzauswertungen, wie `styrke show` sie anzeigt. Stufe 1: Mediane je Denkmodus und Promptlänge. Stufe 2: je Laststufe der Durchsatz je Messfenster, Antwortzeiten, Fehler, dazu `saturated_at` und der Grund für das Ende der Stufe.

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
