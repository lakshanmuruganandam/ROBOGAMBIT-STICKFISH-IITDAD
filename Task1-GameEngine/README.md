# Task 1 — Game Engine

A standalone chess engine for the **RoboGambit 6×6 chess variant**, built for ELO-rated tournament play.

## Variant Rules

| Rule | Detail |
|------|--------|
| Board | 6×6 (files A–F, ranks 1–6) |
| Pieces | Pawn, Knight, Bishop, Queen, King (no Rooks) |
| Pawn movement | 1 square forward only (no double push) |
| Promotion | Only to previously captured pieces |
| No castling | No en passant |
| Starting position | Fischer Random (randomized back rank) |

## Piece Encoding

| ID | Piece | ID | Piece |
|----|-------|----|-------|
| 0 | Empty | — | — |
| 1 | White Pawn | 6 | Black Pawn |
| 2 | White Knight | 7 | Black Knight |
| 3 | White Bishop | 8 | Black Bishop |
| 4 | White Queen | 9 | Black Queen |
| 5 | White King | 10 | Black King |

## Engine Architecture

`game.py` is a single-file, self-contained engine (Python 3.8+, NumPy only).

- **Board representation** — flat 36-element NumPy array with Zobrist hashing
- **Move generation** — full legal move gen with pin/check detection
- **Search** — Negamax with alpha-beta pruning, featuring:
  - Iterative deepening with aspiration windows
  - Principal Variation Search (PVS)
  - Null-move pruning
  - Late Move Reductions (LMR)
  - Check extensions
  - Quiescence search with SEE (Static Exchange Evaluation)
  - Transposition table (4M entries)
  - Killer moves and history heuristic for move ordering
- **Evaluation** — material + piece-square tables with tapered eval (midgame/endgame interpolation)
- **Time management** — adaptive allocation over a 900s total clock with safety buffer

## API

### `get_move(board, turn) -> str`

Tournament entry point.

- **board**: 6×6 NumPy `int` array
- **turn**: `1` for white, `2` for black
- **Returns**: move string, e.g. `"4:D1->D3"`

### `get_best_move(board, playing_white) -> Optional[str]`

Internal entry point with boolean side flag. Returns `None` if no legal move exists.

## Usage

```bash
# Dependencies
pip install numpy

# Smoke test
python game.py
```

## Move Format

Moves are returned as `"piece_id:FILE_RANK->FILE_RANK"`, e.g. `"2:B1->C3"` (knight from B1 to C3). Promotions append the promoted piece id: `"1:C5->C6=4"` (pawn promotes to queen).
