// Page of the chess GUI (ADR-0029). chess.js keeps the game and its legal moves, chessboard.js
// draws the board; after every move the position is sent to the local server, which answers with
// the network's probabilities, the oracle's verdict and the saliency of each square (ADR-0035),
// drawn over the board: red where a square supports the network's answer, blue where it goes
// against it.
'use strict';

const game = new Chess();
let playable = true; // false for a pasted FEN that chess.js refuses: shown and analysed, not played
let request = 0;     // only the answer to the latest request is shown
let lastResult = null; // the latest analysis, to redraw the map after a flip or a resize

const $ = (id) => document.getElementById(id);

const board = Chessboard('board', {
  draggable: true,
  position: 'start',
  pieceTheme: 'vendor/img/{piece}.png',
  onDragStart: (source, piece) =>
    playable && !game.game_over() && piece[0] === game.turn(),
  onDrop: (source, target) => {
    // Always promote to a queen: enough to reach any check or mate position worth analysing.
    if (game.move({ from: source, to: target, promotion: 'q' }) === null)
      return 'snapback';
    update();
  },
  onSnapEnd: () => board.position(game.fen()),
});

function showMoves() {
  const list = $('moves');
  list.replaceChildren();
  const moves = game.history();
  for (let i = 0; i < moves.length; i += 2) {
    const item = document.createElement('li');
    item.textContent = moves[i] + (moves[i + 1] ? '  ' + moves[i + 1] : '');
    list.appendChild(item);
  }
}

function showBars(probabilities, prediction) {
  const bars = $('bars');
  bars.replaceChildren();
  for (const { label, p } of probabilities) {
    const row = document.createElement('div');
    row.className = 'bar' + (label === prediction ? ' top' : '');
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = label;
    const track = document.createElement('span');
    track.className = 'track';
    const fill = document.createElement('span');
    fill.className = 'fill';
    fill.style.display = 'block';
    fill.style.width = (100 * p).toFixed(1) + '%';
    track.appendChild(fill);
    const value = document.createElement('span');
    value.className = 'value';
    value.textContent = (100 * p).toFixed(1) + ' %';
    row.append(name, track, value);
    bars.appendChild(row);
  }
}

const HELP = {
  off: '',
  occlusion: 'Each piece is removed in turn: the deeper the red, the more the network\'s answer ' +
    'loses probability without it. Empty squares stay blank.',
  gradient: 'How much each square\'s inputs (its piece and whether each side attacks it) push ' +
    'the probability of the answer up (red) or down (blue).',
};

// Square index of the server, 0 = a8 to 63 = h1, to the chessboard.js name ("a8"…"h1").
const squareName = (index) => 'abcdefgh'[index % 8] + (8 - Math.floor(index / 8));

function showSaliency() {
  const mode = $('saliency').value;
  $('saliency-help').textContent = HELP[mode];
  for (const element of document.querySelectorAll('#board [data-square]'))
    element.style.boxShadow = '';
  if (mode === 'off' || !lastResult)
    return;
  const values = lastResult[mode];
  const peak = Math.max(...values.map(Math.abs));
  if (!(peak > 0))
    return;
  values.forEach((value, index) => {
    const element = document.querySelector(`#board [data-square="${squareName(index)}"]`);
    if (!element || value === 0)
      return;
    const alpha = (0.75 * Math.abs(value) / peak).toFixed(3);
    const colour = value > 0 ? `rgba(214, 39, 40, ${alpha})` : `rgba(31, 119, 180, ${alpha})`;
    element.style.boxShadow = `inset 0 0 0 100px ${colour}`;
  });
}

async function analyse(fen) {
  const id = ++request;
  $('fen').value = fen;
  try {
    const response = await fetch('/api/analyze?fen=' + encodeURIComponent(fen));
    const result = await response.json();
    if (id !== request)
      return;
    if (!response.ok)
      throw new Error(result.error || response.statusText);
    $('prediction').textContent = result.prediction;
    $('oracle').textContent = result.oracle;
    showBars(result.probabilities, result.prediction);
    lastResult = result;
    showSaliency();
    const agreement = $('agreement');
    if (result.oracle === 'Illegal') {
      agreement.textContent = 'Illegal position: the side not to move is in check.';
      agreement.className = 'bad';
    } else if (result.oracle === result.prediction) {
      agreement.textContent = 'The network agrees with the rules.';
      agreement.className = 'ok';
    } else {
      agreement.textContent = 'The network is wrong on this position.';
      agreement.className = 'bad';
    }
    $('status').textContent = playable ? '' :
      'chess.js cannot play from this position; it is analysed only.';
  } catch (error) {
    if (id === request)
      $('status').textContent = 'Error: ' + error.message;
  }
}

function update() {
  showMoves();
  analyse(game.fen());
}

$('new').addEventListener('click', () => {
  game.reset();
  playable = true;
  board.start();
  update();
});

$('undo').addEventListener('click', () => {
  if (playable && game.undo() !== null) {
    board.position(game.fen());
    update();
  }
});

$('flip').addEventListener('click', () => {
  board.flip();
  showSaliency();
});

$('saliency').addEventListener('change', showSaliency);

$('fen-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const fen = $('fen').value.trim();
  playable = game.load(fen);
  board.position(fen.split(' ')[0]);
  showMoves();
  analyse(fen);
});

fetch('/api/info')
  .then((response) => response.json())
  .then((info) => {
    $('network').textContent = `Network ${info.network}, encoding ${info.encoding}, ` +
      `${info.labels.length} classes.`;
  })
  .catch(() => { $('network').textContent = 'Server not reachable.'; });

window.addEventListener('resize', () => {
  board.resize();
  showSaliency();
});

// "?fen=…" opens the page on that position (a link to share a position).
const startFen = new URLSearchParams(window.location.search).get('fen');
if (startFen) {
  $('fen').value = startFen;
  $('fen-form').requestSubmit();
} else {
  update();
}
