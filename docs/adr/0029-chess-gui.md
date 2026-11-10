# ADR-0029: Chess GUI: a page with chess.js and chessboard.js, served by a local C++ server

- **Status**: Accepted (2026-10-29)
- **Date**: 2026-10-28
- **Deciders**: Ridwane777, approved by Meric-BG

## Context
The analyzer reads a file of FEN lines and prints one label per line: right for the evaluation harness,
hard to show. At a demonstration, playing moves on a board and watching the network's answer change
after each one shows what the network does, and where it fails, far better. A playable board needs
every rule of the game (castling, en passant, promotion), and our oracle does not generate castling
(ADR-0006). This first version analyses positions; playing against an AI (#27) comes later and
should reuse it.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| **Page with chess.js (rules) and chessboard.js (board), local C++ server (chosen)** | Complete, tested rules in a few kilobytes; drag-and-drop board; the server reuses the library, the encodings and the oracle unchanged | Three JavaScript dependencies and piece images to credit; a server to start |
| Page with our own rules (oracle completed with castling, exposed by the server) | Everything is ours | Castling and a move API to write first: most of #27's move generation, before any interface |
| Library compiled to WebAssembly (Emscripten), page without a server | Opens as a static page, could be hosted | One more toolchain; the network file must be shipped to the browser; two builds to keep in step |
| Terminal interface (ncurses) | No browser, no JavaScript | Poor board drawing, no drag-and-drop: not the better experience asked for |

## Decision
`bonus/gui/`: `my_torch_gui NETWORK [PORT]` loads the network once and listens on 127.0.0.1 only.
It serves the page of `bonus/gui/web/` and answers `GET /api/analyze?fen=` with the network's
probability for each label and the oracle's verdict. The page keeps the game with chess.js, draws
it with chessboard.js, and shows after each move the network's answer, its probabilities, the
oracle's answer and whether they agree. The JavaScript files and piece images are copied into the
repository, so the demonstration works without network access.

## Consequences
- ✅ A move-by-move demonstration of the network against the rules, mistakes highlighted.
- ✅ Nothing changes in the library or the analyzer; the server uses their public headers.
- ✅ The analysis is tested apart from the sockets (`bonus/tests/test_gui.cpp`): URL decoding, JSON
  escaping, the start position, a mate, invalid FENs.
- ❌ Third-party code in the repository (180 KB): chess.js (BSD-2-Clause), chessboard.js and
  jQuery (MIT), piece images (CC BY-SA 3.0), listed in `docs/legal/licenses.md`.
- ❌ Two sources of rules: chess.js decides which moves are legal on the page, our oracle gives the
  verdict shown. A pasted position that chess.js refuses is analysed but cannot be played from.
- ❌ A minimal HTTP server: one request at a time, GET only, no TLS. Enough on 127.0.0.1 for one
  user; not a server to expose on a network.
- ❌ Promotion always to a queen on the page.
- Follow-up: playing against an AI (#27) adds a route that returns a move; the page stays.
