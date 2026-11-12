"""Die Bewertungsseite: eine einzelne HTML-Datei, in der man Antworten je Frage in eine Rangfolge bringt.

Die Seite braucht kein Netz und keinen Server. Sie merkt sich den Stand im
Browser und gibt das Ergebnis als JSON-Datei aus, die ``styrke bewertung
import`` einliest. Von welchem Modell eine Antwort stammt, steht nicht in der
Seite; das weiß nur die Schlüsseldatei.
"""

from __future__ import annotations

import json
from typing import Any

PAGE = r"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Antworten bewerten</title>
<style>
:root{--bg:#f6f5f1;--card:#fff;--ink:#1f1e1d;--ink2:#5f5e58;--line:#dcdbd3;--accent:#1f5fbf;--ok:#1a7f4b;--soft:#ecebe5}
@media (prefers-color-scheme:dark){:root{--bg:#191918;--card:#242423;--ink:#f0efec;--ink2:#b9b8ae;--line:#3b3b38;--accent:#7fb0ff;--ok:#5fc58e;--soft:#2e2e2c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header strong{margin-right:auto}main{max-width:860px;margin:0 auto;padding:16px 16px 80px}
input[type=text]{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);width:190px}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}@media (prefers-color-scheme:dark){button.primary{color:#111}}
button:disabled{opacity:.45;cursor:default}
.dots{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 16px}.dots button{width:34px;height:34px;padding:0;border-radius:50%;font-size:13px}
.dots .done{background:var(--ok);border-color:var(--ok);color:#fff}.dots .here{outline:3px solid var(--accent);outline-offset:2px}
h1{font-size:21px;margin:8px 0 6px;white-space:pre-wrap}.muted{color:var(--ink2);font-size:14px}
.ref{background:var(--soft);border-radius:10px;padding:10px 14px;margin:10px 0 16px;white-space:pre-wrap}
.ref b{display:block;font-size:13px;color:var(--ink2);font-weight:600;margin-bottom:2px}
.edge{font-size:13px;color:var(--ink2);text-align:center;margin:6px 0}
ol{list-style:none;margin:0;padding:0}
li.card{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:8px 0;display:flex;gap:10px;padding:10px 12px;cursor:grab}
li.card.drag{opacity:.4}li.card.over{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent)}
.rank{flex:none;width:34px;height:34px;border-radius:50%;background:var(--soft);display:flex;align-items:center;justify-content:center;font-weight:700}
.text{flex:1;min-width:0;white-space:pre-wrap;overflow-wrap:anywhere;max-height:15em;overflow:hidden;position:relative}
.text.open{max-height:none}.more{display:block;margin-top:4px;font-size:14px;color:var(--accent);background:none;border:none;padding:0}
.moves{flex:none;display:flex;flex-direction:column;gap:6px}.moves button{width:40px;height:40px;padding:0;font-size:18px}
nav{display:flex;gap:10px;justify-content:space-between;margin-top:18px;flex-wrap:wrap}
.note{background:var(--soft);border-radius:10px;padding:12px 14px;margin:14px 0}
</style></head><body>
<header><strong>Antworten bewerten</strong>
<label class="muted">Dein Name <input type="text" id="rater" placeholder="Vorname genügt"></label>
<span class="muted" id="count"></span><button class="primary" id="export">Ergebnis speichern</button></header>
<main>
<div class="note" id="intro">Zu jeder Frage siehst du mehrere Antworten. Bring sie in eine Reihenfolge: die beste nach oben, die schlechteste nach unten. Ziehe die Karten mit der Maus oder nutze die Pfeile. Maßstab ist die Referenzantwort; der Wortlaut darf abweichen. Fremde Schriftzeichen oder Wörter in anderen Sprachen mitten im Text sind ein Mangel. Wenn du fertig bist, klicke oben auf „Ergebnis speichern“ und gib die Datei zurück.</div>
<div class="dots" id="dots"></div>
<div class="muted" id="where"></div><h1 id="question"></h1>
<div class="ref"><b>Referenzantwort (Maßstab)</b><span id="reference"></span></div>
<div class="edge">▲ beste Antwort</div><ol id="cards"></ol><div class="edge">▼ schlechteste Antwort</div>
<nav><button id="prev">Zurück</button><button class="primary" id="next">Reihenfolge bestätigen und weiter</button></nav>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const KEY = 'styrke-bewertung-' + data.id;
let state = {rater: '', order: {}, done: {}, at: 0};
try { const saved = JSON.parse(localStorage.getItem(KEY)); if (saved && saved.order) state = saved; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
const $ = id => document.getElementById(id);
const q = () => data.questions[state.at];
const order = () => { const k = q().id; if (!state.order[k]) state.order[k] = q().cards.map((_, i) => i); return state.order[k]; };

function move(from, to) {
  const o = order(); if (to < 0 || to >= o.length || from === to) return;
  o.splice(to, 0, o.splice(from, 1)[0]); delete state.done[q().id]; save(); render();
}
function render() {
  const question = q(), o = order();
  $('where').textContent = 'Frage ' + (state.at + 1) + ' von ' + data.questions.length + (state.done[question.id] ? ' · bestätigt' : '');
  $('question').textContent = question.frage; $('reference').textContent = question.referenz || '–';
  const list = $('cards'); list.textContent = '';
  o.forEach((cardIndex, pos) => {
    const li = document.createElement('li'); li.className = 'card'; li.draggable = true; li.dataset.pos = pos;
    const rank = document.createElement('div'); rank.className = 'rank'; rank.textContent = pos + 1;
    const wrap = document.createElement('div'); wrap.style.flex = '1'; wrap.style.minWidth = '0';
    const text = document.createElement('div'); text.className = 'text'; text.textContent = question.cards[cardIndex].text || '(keine Antwort)';
    wrap.appendChild(text);
    const moves = document.createElement('div'); moves.className = 'moves';
    [['▲', pos - 1, 'eine Stelle nach oben'], ['▼', pos + 1, 'eine Stelle nach unten']].forEach(([label, to, title]) => {
      const b = document.createElement('button'); b.textContent = label; b.title = title; b.setAttribute('aria-label', title);
      b.disabled = to < 0 || to >= o.length; b.onclick = () => move(pos, to); moves.appendChild(b);
    });
    li.append(rank, wrap, moves); list.appendChild(li);
    if (text.scrollHeight > text.clientHeight + 4) {
      const more = document.createElement('button'); more.className = 'more'; more.textContent = 'ganz anzeigen';
      more.onclick = () => { text.classList.toggle('open'); more.textContent = text.classList.contains('open') ? 'einklappen' : 'ganz anzeigen'; };
      wrap.appendChild(more);
    }
    li.addEventListener('dragstart', e => { li.classList.add('drag'); e.dataTransfer.setData('text/plain', pos); e.dataTransfer.effectAllowed = 'move'; });
    li.addEventListener('dragend', () => li.classList.remove('drag'));
    li.addEventListener('dragover', e => { e.preventDefault(); li.classList.add('over'); });
    li.addEventListener('dragleave', () => li.classList.remove('over'));
    li.addEventListener('drop', e => { e.preventDefault(); move(parseInt(e.dataTransfer.getData('text/plain'), 10), pos); });
  });
  const dots = $('dots'); dots.textContent = '';
  data.questions.forEach((item, i) => {
    const b = document.createElement('button'); b.textContent = i + 1; b.title = item.frage.slice(0, 80);
    if (state.done[item.id]) b.classList.add('done'); if (i === state.at) b.classList.add('here');
    b.onclick = () => { state.at = i; save(); render(); scrollTo(0, 0); }; dots.appendChild(b);
  });
  const done = data.questions.filter(item => state.done[item.id]).length;
  $('count').textContent = done + ' von ' + data.questions.length + ' bestätigt';
  $('prev').disabled = state.at === 0;
  $('next').textContent = state.at === data.questions.length - 1 ? 'Reihenfolge bestätigen' : 'Reihenfolge bestätigen und weiter';
}
$('prev').onclick = () => { state.at = Math.max(0, state.at - 1); save(); render(); scrollTo(0, 0); };
$('next').onclick = () => {
  state.done[q().id] = true; order();
  if (state.at < data.questions.length - 1) state.at += 1;
  save(); render(); scrollTo(0, 0);
};
$('rater').value = state.rater || '';
$('rater').oninput = e => { state.rater = e.target.value; save(); };
function result() {
  const rankings = {};
  data.questions.forEach(item => { if (state.done[item.id]) rankings[item.id] = state.order[item.id].map(i => item.cards[i].nrs); });
  return {format: 'styrke-rangfolge-1', bogen: data.id, bewerter: (state.rater || '').trim(), zeit: new Date().toISOString(), rangfolgen: rankings};
}
$('export').onclick = () => {
  const out = result(), done = Object.keys(out.rangfolgen).length;
  if (!out.bewerter) { $('rater').focus(); $('count').textContent = 'Bitte zuerst deinen Namen eintragen'; return; }
  if (done < data.questions.length && !confirm(done + ' von ' + data.questions.length + ' Fragen sind bestätigt. Trotzdem speichern?')) return;
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'bewertung_' + out.bewerter.replace(/[^\wäöüÄÖÜß-]+/g, '_') + '.json'; document.body.appendChild(a); a.click(); a.remove();
};
window.styrkeResult = result;
render();
</script></body></html>
"""


def build_page(sheet_id: str, questions: list[dict[str, Any]]) -> str:
    """``questions``: ``{"id", "frage", "referenz", "cards": [{"nrs": [...], "text": ...}]}``."""
    payload = json.dumps({"id": sheet_id, "questions": questions}, ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__DATA__", payload)
