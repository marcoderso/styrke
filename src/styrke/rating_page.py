"""Die Bewertungsseite: eine einzelne HTML-Datei, in der man die Antworten je Frage nebeneinander beurteilt.

Die Seite braucht kein Netz und keinen Server. Sie stellt Markdown dar, merkt
sich den Stand im Browser und gibt das Ergebnis als JSON-Datei aus, die
``styrke bewertung import`` einliest. Von welchem Modell eine Antwort stammt,
steht nicht in der Seite; das weiß nur die Schlüsseldatei.
"""

from __future__ import annotations

import json
from typing import Any

PAGE = r"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light only">
<title>Antworten bewerten</title>
<style>
:root{color-scheme:light only;--bg:#f6f5f1;--card:#fff;--ink:#1f1e1d;--ink2:#5f5e58;--line:#dcdbd3;--accent:#1f5fbf;--soft:#ecebe5;
  --good:#1a7f4b;--mid:#9a6700;--bad:#b3261e}
*{box-sizing:border-box}html{background:var(--bg)}body{margin:0;display:flex;flex-direction:column;height:100vh;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header.top{flex:none;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header.top strong{margin-right:auto}main{flex:1;min-height:0;display:flex;flex-direction:column;width:100%;max-width:1500px;margin:0 auto;padding:12px 16px 14px}
input[type=text]{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);width:190px}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button:disabled{opacity:.45;cursor:default}
.dots{flex:none;display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 10px}.dots button{width:34px;height:34px;padding:0;border-radius:50%;font-size:13px}
.dots .done{background:var(--good);border-color:var(--good);color:#fff}.dots .here{outline:3px solid var(--accent);outline-offset:2px}
h1{font-size:20px;margin:6px 0;white-space:pre-wrap}.muted{color:var(--ink2);font-size:14px}
.ref{background:var(--soft);border-radius:10px;padding:10px 14px;margin:8px 0 12px;white-space:pre-wrap;max-height:20vh;overflow-y:auto;flex:none}
.ref b{display:block;font-size:13px;color:var(--ink2);font-weight:600;margin-bottom:2px}
.note{background:var(--soft);border-radius:10px;padding:10px 14px;margin:0 0 12px;font-size:15px}.note ul{margin:6px 0 0;padding-left:20px}
.cols{flex:1;min-height:0;display:grid;grid-auto-flow:column;grid-auto-columns:minmax(300px,1fr);grid-template-rows:minmax(0,1fr);gap:12px;overflow-x:auto;padding-bottom:8px;scroll-snap-type:x proximity}
.col{scroll-snap-align:start;display:flex;flex-direction:column;background:var(--card);border:1px solid var(--line);border-radius:12px;
  min-height:0;overflow:hidden}
.col.judged{border-color:var(--ink2)}
.col h2{font-size:14px;margin:0;padding:8px 12px;border-bottom:1px solid var(--line);color:var(--ink2);font-weight:600}
.answer{flex:1;overflow-y:auto;padding:4px 14px 12px;overflow-wrap:anywhere}
.answer p{margin:8px 0}.answer h3,.answer h4,.answer h5{margin:12px 0 4px;line-height:1.3}.answer h3{font-size:18px}.answer h4{font-size:16px}.answer h5{font-size:15px}
.answer ul,.answer ol{margin:6px 0;padding-left:22px}.answer li{margin:2px 0}
.answer code{background:var(--soft);border-radius:4px;padding:1px 4px;font-size:14px}
.answer pre{background:var(--soft);border-radius:8px;padding:8px 10px;overflow-x:auto}.answer pre code{background:none;padding:0}
.answer blockquote{margin:8px 0;padding:2px 12px;border-left:3px solid var(--line);color:var(--ink2)}
.answer table{border-collapse:collapse;margin:8px 0;font-size:14px;display:block;overflow-x:auto}
.answer th,.answer td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}.answer hr{border:none;border-top:1px solid var(--line);margin:12px 0}
.verdict{display:flex;gap:6px;padding:10px;border-top:1px solid var(--line);background:var(--bg)}
.verdict button{flex:1;padding:9px 4px;font-weight:600}
.verdict button[aria-pressed=true]{color:#fff}.verdict .gut[aria-pressed=true]{background:var(--good);border-color:var(--good)}
.verdict .teils[aria-pressed=true]{background:var(--mid);border-color:var(--mid)}.verdict .schlecht[aria-pressed=true]{background:var(--bad);border-color:var(--bad)}
nav{flex:none;display:flex;gap:10px;justify-content:space-between;margin-top:10px;flex-wrap:wrap}
/* Kleine Fenster: die Seite rollt als Ganzes, jede Spalte hat eine feste Höhe. */
@media (max-width:700px),(max-height:620px){body{display:block;height:auto}main{display:block}.cols{grid-auto-columns:88vw;grid-template-rows:none}.col{height:72vh}}
</style></head><body>
<header class="top"><strong>Antworten bewerten</strong>
<label class="muted">Dein Name <input type="text" id="rater" placeholder="Vorname genügt"></label>
<span class="muted" id="count"></span><button class="primary" id="export">Ergebnis speichern</button></header>
<main>
<details class="note" id="intro" open><summary><b>So geht es</b></summary>
Zu jeder Frage stehen mehrere Antworten nebeneinander. Gib unter jeder Antwort ein Urteil ab. Maßstab ist die Referenzantwort; der Wortlaut darf abweichen.
<ul><li><b>✓ gut:</b> entspricht dem Maßstab.</li>
<li><b>~ teils:</b> im Kern richtig, aber mit Mängeln, etwa unnötigen oder falschen Zusätzen.</li>
<li><b>✗ schlecht:</b> falsch, erfunden oder unbrauchbar. Auch fremde Schriftzeichen mitten im Text sind ein Mangel.</li></ul>
Wenn du fertig bist, klicke oben auf „Ergebnis speichern“ und gib die Datei zurück. Du kannst jederzeit unterbrechen; der Stand bleibt in diesem Browser erhalten.</details>
<div class="dots" id="dots"></div>
<div class="muted" id="where"></div><h1 id="question"></h1>
<div class="ref"><b>Referenzantwort (Maßstab)</b><span id="reference"></span></div>
<div class="cols" id="cols"></div>
<nav><button id="prev">Zurück</button><button class="primary" id="next">Weiter</button></nav>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const KEY = 'styrke-urteile-' + data.id;
let state = {rater: '', verdicts: {}, at: 0, introSeen: false};
try { const saved = JSON.parse(localStorage.getItem(KEY)); if (saved && saved.verdicts) state = saved; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
const $ = id => document.getElementById(id);
const q = () => data.questions[state.at];
const given = item => Object.keys(state.verdicts[item.id] || {}).length;
const complete = item => given(item) === item.cards.length;
const LABELS = [['gut', '✓ gut'], ['teils', '~ teils'], ['schlecht', '✗ schlecht']];

// -- Markdown: erst alles entschärfen, dann die üblichen Auszeichnungen in HTML wandeln --
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function inline(s) {
  return esc(s).replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>').replace(/__([^_]+?)__/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\s](?:[^*]*?[^*\s])?)\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\([^)\s]+\)/g, '$1');
}
const isList = l => /^\s*(?:[-*+]|\d+[.)])\s+/.test(l);
const isBlock = l => /^\s*$/.test(l) || /^```/.test(l) || /^#{1,6}\s/.test(l) || isList(l) || /^>/.test(l) || /^\s*([-*_])(?:\s*\1){2,}\s*$/.test(l);
const cells = l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
function md(src) {
  const lines = (src || '').replace(/\r/g, '').split('\n'), out = []; let i = 0;
  while (i < lines.length) {
    const line = lines[i]; let m;
    if (/^```/.test(line)) { const code = []; i++; while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]); i++; out.push('<pre><code>' + esc(code.join('\n')) + '</code></pre>'); continue; }
    if (/^\s*$/.test(line)) { i++; continue; }
    if ((m = /^(#{1,6})\s+(.*?)\s*#*$/.exec(line))) { const h = Math.min(5, m[1].length + 2); out.push('<h' + h + '>' + inline(m[2]) + '</h' + h + '>'); i++; continue; }
    if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) { out.push('<hr>'); i++; continue; }
    if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(lines[i + 1])) {
      let html = '<table><thead><tr>' + cells(line).map(c => '<th>' + inline(c) + '</th>').join('') + '</tr></thead><tbody>'; i += 2;
      while (i < lines.length && lines[i].includes('|') && !/^\s*$/.test(lines[i])) html += '<tr>' + cells(lines[i++]).map(c => '<td>' + inline(c) + '</td>').join('') + '</tr>';
      out.push(html + '</tbody></table>'); continue;
    }
    if (isList(line)) {
      const ordered = /^\s*\d+[.)]\s/.test(line), items = [];
      while (i < lines.length && (isList(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        if (isList(lines[i])) items.push(lines[i].replace(/^\s*(?:[-*+]|\d+[.)])\s+/, '')); else items[items.length - 1] += ' ' + lines[i].trim();
        i++;
      }
      out.push((ordered ? '<ol>' : '<ul>') + items.map(t => '<li>' + inline(t) + '</li>').join('') + (ordered ? '</ol>' : '</ul>')); continue;
    }
    if (/^>/.test(line)) { const quote = []; while (i < lines.length && /^>/.test(lines[i])) quote.push(lines[i++].replace(/^>\s?/, '')); out.push('<blockquote>' + quote.map(inline).join('<br>') + '</blockquote>'); continue; }
    const para = [line]; i++;
    while (i < lines.length && !isBlock(lines[i])) para.push(lines[i++]);
    out.push('<p>' + para.map(inline).join('<br>') + '</p>');
  }
  return out.join('');
}

function meta() {
  const question = q();
  $('where').textContent = 'Frage ' + (state.at + 1) + ' von ' + data.questions.length + ' · ' + given(question) + ' von ' + question.cards.length + ' Antworten beurteilt';
  const dots = $('dots'); dots.textContent = '';
  data.questions.forEach((item, i) => {
    const b = document.createElement('button'); b.textContent = i + 1; b.title = item.frage.slice(0, 80);
    if (complete(item)) b.classList.add('done'); if (i === state.at) b.classList.add('here');
    b.onclick = () => go(i); dots.appendChild(b);
  });
  $('count').textContent = data.questions.filter(complete).length + ' von ' + data.questions.length + ' Fragen fertig';
  $('prev').disabled = state.at === 0; $('next').disabled = state.at === data.questions.length - 1;
}
function show() {
  const question = q(), cols = $('cols');
  $('question').textContent = question.frage; $('reference').textContent = question.referenz || '–';
  cols.textContent = ''; cols.scrollLeft = 0;
  question.cards.forEach((card, index) => {
    const col = document.createElement('section'); col.className = 'col';
    const head = document.createElement('h2'); head.textContent = 'Antwort ' + String.fromCharCode(65 + index);
    const answer = document.createElement('div'); answer.className = 'answer'; answer.tabIndex = 0;
    answer.innerHTML = md(card.text) || '<p>(keine Antwort)</p>';
    const bar = document.createElement('div'); bar.className = 'verdict';
    const mark = () => {
      const now = (state.verdicts[question.id] || {})[index];
      col.classList.toggle('judged', !!now);
      [...bar.children].forEach(b => b.setAttribute('aria-pressed', b.dataset.v === now ? 'true' : 'false'));
    };
    LABELS.forEach(([value, label]) => {
      const b = document.createElement('button'); b.className = value; b.dataset.v = value; b.textContent = label;
      b.onclick = () => {
        const mine = state.verdicts[question.id] || (state.verdicts[question.id] = {});
        if (mine[index] === value) delete mine[index]; else mine[index] = value;
        save(); mark(); meta();
      };
      bar.appendChild(b);
    });
    col.append(head, answer, bar); cols.appendChild(col); mark();
  });
  meta();
}
function go(i) { state.at = Math.max(0, Math.min(data.questions.length - 1, i)); save(); show(); scrollTo(0, 0); }
$('prev').onclick = () => go(state.at - 1); $('next').onclick = () => go(state.at + 1);
$('rater').value = state.rater || '';
$('rater').oninput = e => { state.rater = e.target.value; save(); };
$('intro').open = !state.introSeen;
$('intro').addEventListener('toggle', () => { if (!$('intro').open) { state.introSeen = true; save(); } });
function result() {
  const verdicts = {};
  data.questions.forEach(item => {
    const mine = state.verdicts[item.id] || {}, out = {};
    Object.keys(mine).forEach(index => item.cards[index].nrs.forEach(nr => { out[nr] = mine[index]; }));  // wortgleiche Antworten teilen das Urteil
    if (Object.keys(out).length) verdicts[item.id] = out;
  });
  return {format: 'styrke-urteile-1', bogen: data.id, bewerter: (state.rater || '').trim(), zeit: new Date().toISOString(), urteile: verdicts};
}
$('export').onclick = () => {
  const out = result(), done = data.questions.filter(complete).length;
  if (!out.bewerter) { $('rater').focus(); $('count').textContent = 'Bitte zuerst deinen Namen eintragen'; return; }
  if (done < data.questions.length && !confirm(done + ' von ' + data.questions.length + ' Fragen sind vollständig beurteilt. Trotzdem speichern?')) return;
  const blob = new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'bewertung_' + out.bewerter.replace(/[^\wäöüÄÖÜß-]+/g, '_') + '.json'; document.body.appendChild(a); a.click(); a.remove();
};
window.styrkeResult = result; window.styrkeMarkdown = md;
show();
</script></body></html>
"""


def build_page(sheet_id: str, questions: list[dict[str, Any]]) -> str:
    """``questions``: ``{"id", "frage", "referenz", "cards": [{"nrs": [...], "text": ...}]}``."""
    payload = json.dumps({"id": sheet_id, "questions": questions}, ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__DATA__", payload)
