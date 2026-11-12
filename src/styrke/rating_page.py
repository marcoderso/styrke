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
<meta name="color-scheme" content="light only">
<title>Antworten bewerten</title>
<style>
:root{color-scheme:light only;--bg:#f6f5f1;--card:#fff;--ink:#1f1e1d;--ink2:#5f5e58;--line:#dcdbd3;--accent:#1f5fbf;--ok:#1a7f4b;--soft:#ecebe5}
*{box-sizing:border-box}html{background:var(--bg)}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header strong{margin-right:auto}main{max-width:860px;margin:0 auto;padding:16px 16px 80px}
input[type=text]{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);width:190px}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button:disabled{opacity:.45;cursor:default}
.dots{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 16px}.dots button{width:34px;height:34px;padding:0;border-radius:50%;font-size:13px}
.dots .done{background:var(--ok);border-color:var(--ok);color:#fff}.dots .here{outline:3px solid var(--accent);outline-offset:2px}
h1{font-size:21px;margin:8px 0 6px;white-space:pre-wrap}.muted{color:var(--ink2);font-size:14px}
.ref{background:var(--soft);border-radius:10px;padding:10px 14px;margin:10px 0 16px;white-space:pre-wrap}
.ref b{display:block;font-size:13px;color:var(--ink2);font-weight:600;margin-bottom:2px}
.edge{font-size:13px;color:var(--ink2);text-align:center;margin:6px 0}
ol{list-style:none;margin:0;padding:0}
li.card{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:8px 0;display:flex;gap:10px;padding:10px 12px;
  cursor:grab;user-select:none;-webkit-user-select:none;position:relative}
li.card:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
li.card.dragging{cursor:grabbing;z-index:4;border-color:var(--accent);box-shadow:0 10px 28px rgba(0,0,0,.22)}
body.sorting{cursor:grabbing}
.grip{flex:none;width:28px;display:flex;align-items:center;justify-content:center;color:var(--ink2);font-size:20px;touch-action:none}
.rank{flex:none;width:34px;height:34px;border-radius:50%;background:var(--soft);display:flex;align-items:center;justify-content:center;font-weight:700}
.body{flex:1;min-width:0}.text{white-space:pre-wrap;overflow-wrap:anywhere;max-height:15em;overflow:hidden}
.text.open{max-height:none}.more{display:block;margin-top:4px;font-size:14px;color:var(--accent);background:none;border:none;padding:0}
nav{display:flex;gap:10px;justify-content:space-between;margin-top:18px;flex-wrap:wrap}
.note{background:var(--soft);border-radius:10px;padding:12px 14px;margin:14px 0}
</style></head><body>
<header><strong>Antworten bewerten</strong>
<label class="muted">Dein Name <input type="text" id="rater" placeholder="Vorname genügt"></label>
<span class="muted" id="count"></span><button class="primary" id="export">Ergebnis speichern</button></header>
<main>
<div class="note" id="intro">Zu jeder Frage siehst du mehrere Antworten. Bring sie in eine Reihenfolge: die beste nach oben, die schlechteste nach unten. Greif dazu eine Karte und zieh sie an ihre Stelle; am Touchscreen am Griff links. Maßstab ist die Referenzantwort; der Wortlaut darf abweichen. Fremde Schriftzeichen oder Wörter in anderen Sprachen mitten im Text sind ein Mangel. Wenn du fertig bist, klicke oben auf „Ergebnis speichern“ und gib die Datei zurück.</div>
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
const list = $('cards');
const q = () => data.questions[state.at];
const order = () => { const k = q().id; if (!state.order[k]) state.order[k] = q().cards.map((_, i) => i); return state.order[k]; };

// Reihenfolge aus der Seite übernehmen. Die Karten bleiben stehen, nur die Nummern ändern sich: nichts springt.
function commit() {
  const now = [...list.children].map(li => parseInt(li.dataset.card, 10));
  if (now.join() !== order().join()) { state.order[q().id] = now; delete state.done[q().id]; save(); }
  [...list.children].forEach((li, pos) => { li.querySelector('.rank').textContent = pos + 1; });
  meta();
}
function meta() {
  const question = q();
  $('where').textContent = 'Frage ' + (state.at + 1) + ' von ' + data.questions.length + (state.done[question.id] ? ' · bestätigt' : '');
  const dots = $('dots'); dots.textContent = '';
  data.questions.forEach((item, i) => {
    const b = document.createElement('button'); b.textContent = i + 1; b.title = item.frage.slice(0, 80);
    if (state.done[item.id]) b.classList.add('done'); if (i === state.at) b.classList.add('here');
    b.onclick = () => go(i); dots.appendChild(b);
  });
  const done = data.questions.filter(item => state.done[item.id]).length;
  $('count').textContent = done + ' von ' + data.questions.length + ' bestätigt';
  $('prev').disabled = state.at === 0;
  $('next').textContent = state.at === data.questions.length - 1 ? 'Reihenfolge bestätigen' : 'Reihenfolge bestätigen und weiter';
}
function show() {  // eine Frage neu aufbauen; nur beim Wechsel der Frage
  const question = q();
  $('question').textContent = question.frage; $('reference').textContent = question.referenz || '–';
  list.textContent = '';
  order().forEach((cardIndex, pos) => {
    const li = document.createElement('li'); li.className = 'card'; li.dataset.card = cardIndex; li.tabIndex = 0;
    li.title = 'Ziehen zum Verschieben (Tastatur: Alt + Pfeil hoch oder runter)';
    const grip = document.createElement('div'); grip.className = 'grip'; grip.textContent = '⠿'; grip.setAttribute('aria-hidden', 'true');
    const rank = document.createElement('div'); rank.className = 'rank'; rank.textContent = pos + 1;
    const body = document.createElement('div'); body.className = 'body';
    const text = document.createElement('div'); text.className = 'text'; text.textContent = question.cards[cardIndex].text || '(keine Antwort)';
    body.appendChild(text); li.append(grip, rank, body); list.appendChild(li);
    if (text.scrollHeight > text.clientHeight + 4) {
      const more = document.createElement('button'); more.className = 'more'; more.textContent = 'ganz anzeigen';
      more.onclick = () => { text.classList.toggle('open'); more.textContent = text.classList.contains('open') ? 'einklappen' : 'ganz anzeigen'; };
      body.appendChild(more);
    }
  });
  meta();
}
function go(i) { state.at = Math.max(0, Math.min(data.questions.length - 1, i)); save(); show(); scrollTo(0, 0); }

// Ziehen: Die gegriffene Karte folgt dem Zeiger, die anderen rücken auf. Am Bildschirmrand rollt die Seite mit.
let drag = null;
function place() {
  if (!drag || !drag.active) return;
  const li = drag.li; li.style.transform = '';
  let top = li.getBoundingClientRect().top;
  const center = drag.y - drag.grab + li.offsetHeight / 2;
  for (;;) {  // Nachbarn tauschen, sobald die Mitte der Karte über deren Mitte hinaus ist
    const prev = li.previousElementSibling, next = li.nextElementSibling;
    if (prev && center < prev.getBoundingClientRect().top + prev.offsetHeight / 2) list.insertBefore(li, prev);
    else if (next && center > next.getBoundingClientRect().top + next.offsetHeight / 2) list.insertBefore(next, li);
    else break;
    top = li.getBoundingClientRect().top;
  }
  li.style.transform = 'translateY(' + (drag.y - drag.grab - top) + 'px)';
}
function autoscroll() {
  if (!drag || !drag.active) return;
  const edge = 90, speed = 14;
  if (drag.y < edge) scrollBy(0, -speed * (1 - drag.y / edge)); else if (drag.y > innerHeight - edge) scrollBy(0, speed * (1 - (innerHeight - drag.y) / edge));
  place(); drag.frame = requestAnimationFrame(autoscroll);
}
list.addEventListener('pointerdown', e => {
  const li = e.target.closest('li.card');
  if (!li || e.target.closest('button') || (e.pointerType === 'mouse' && e.button !== 0)) return;
  if (e.pointerType !== 'mouse' && !e.target.closest('.grip')) return;  // am Touchscreen nur am Griff, sonst ließe sich nicht mehr rollen
  drag = {li, id: e.pointerId, y: e.clientY, startY: e.clientY, grab: e.clientY - li.getBoundingClientRect().top, active: false, frame: 0};
});
addEventListener('pointermove', e => {
  if (!drag || e.pointerId !== drag.id) return;
  drag.y = e.clientY;
  if (!drag.active) {
    if (Math.abs(e.clientY - drag.startY) < 6) return;
    drag.active = true; drag.li.classList.add('dragging'); document.body.classList.add('sorting');
    try { drag.li.setPointerCapture(drag.id); } catch (err) {}
    drag.frame = requestAnimationFrame(autoscroll);
  }
  e.preventDefault(); place();
}, {passive: false});
function drop() {
  if (!drag) return;
  const d = drag; drag = null; cancelAnimationFrame(d.frame);
  if (!d.active) return;
  d.li.style.transform = ''; d.li.classList.remove('dragging'); document.body.classList.remove('sorting'); commit();
}
addEventListener('pointerup', drop); addEventListener('pointercancel', drop);
list.addEventListener('keydown', e => {
  const li = e.target.closest('li.card'); if (!li || !e.altKey || (e.key !== 'ArrowUp' && e.key !== 'ArrowDown')) return;
  e.preventDefault();
  if (e.key === 'ArrowUp' && li.previousElementSibling) list.insertBefore(li, li.previousElementSibling);
  if (e.key === 'ArrowDown' && li.nextElementSibling) list.insertBefore(li.nextElementSibling, li);
  li.focus(); commit();
});

$('prev').onclick = () => go(state.at - 1);
$('next').onclick = () => { state.done[q().id] = true; order(); save(); if (state.at < data.questions.length - 1) go(state.at + 1); else meta(); };
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
show();
</script></body></html>
"""


def build_page(sheet_id: str, questions: list[dict[str, Any]]) -> str:
    """``questions``: ``{"id", "frage", "referenz", "cards": [{"nrs": [...], "text": ...}]}``."""
    payload = json.dumps({"id": sheet_id, "questions": questions}, ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__DATA__", payload)
