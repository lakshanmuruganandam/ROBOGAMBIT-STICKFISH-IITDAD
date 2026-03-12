"""
Standalone — numpy + stdlib only.
"""

import time
import random
import numpy as np
from math import log
from typing import Optional

# ===================================================================
# Constants
# ===================================================================

EMPTY = 0
WP, WN, WB, WQ, WK = 1, 2, 3, 4, 5
BP, BN, BB, BQ, BK = 6, 7, 8, 9, 10

N = 36
MATE = 100000
MAX_PLY = 64
TOTAL_CLOCK = 900.0
SAFETY_BUFFER = 5.0
MIN_MOVE_TIME = 0.5
MAX_MOVE_TIME = 30.0
TT_SIZE = 1 << 22
TT_MASK = TT_SIZE - 1

CONTEMPT = 50  # higher contempt = fight harder to avoid draws
INF = MATE + 1

COL_TO_FILE = ('A', 'B', 'C', 'D', 'E', 'F')

MAT = [0, 120, 310, 330, 950, 20000, 120, 310, 330, 950, 20000]
SMAT = [0, 120, 310, 330, 950, 20000, -120, -310, -330, -950, -20000]

SIDE = [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
SIGN = [0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
PHASE_W = [0, 0, 1, 1, 4, 0, 0, 1, 1, 4, 0]

INIT_CNT = [0, 6, 2, 2, 1, 1, 6, 2, 2, 1, 1]
W_PROMO = (WQ, WN, WB)
B_PROMO = (BQ, BN, BB)

# ===================================================================
# Precomputed Geometry
# ===================================================================

_KN_OFF = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
_KG_OFF = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

KN = []
KG = []
for _s in range(N):
    _r, _c = _s // 6, _s % 6
    KN.append(tuple((_r + dr) * 6 + (_c + dc) for dr, dc in _KN_OFF
                     if 0 <= _r + dr < 6 and 0 <= _c + dc < 6))
    KG.append(tuple((_r + dr) * 6 + (_c + dc) for dr, dc in _KG_OFF
                     if 0 <= _r + dr < 6 and 0 <= _c + dc < 6))
KN = tuple(KN)
KG = tuple(KG)

_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1))
_RAYS_RAW = [[None]*8 for _ in range(N)]
for _s in range(N):
    _r, _c = _s // 6, _s % 6
    for _d, (_dr, _dc) in enumerate(_DIRS):
        ray = []
        _nr, _nc = _r + _dr, _c + _dc
        while 0 <= _nr < 6 and 0 <= _nc < 6:
            ray.append(_nr * 6 + _nc)
            _nr += _dr
            _nc += _dc
        _RAYS_RAW[_s][_d] = tuple(ray)

DR = [tuple(_RAYS_RAW[s][d] for d in (0,1,2,3)) for s in range(N)]
SR = [tuple(_RAYS_RAW[s][d] for d in (4,5,6,7)) for s in range(N)]
AR = [tuple(_RAYS_RAW[s][d] for d in range(8)) for s in range(N)]
del _RAYS_RAW

MIR = tuple((5 - (i // 6)) * 6 + (i % 6) for i in range(N))

# Precompute king zone (king square + adjacent) as sets for fast 'in' check
KING_ZONE = []
for _s in range(N):
    zone = set()
    zone.add(_s)
    for adj in KG[_s]:
        zone.add(adj)
    KING_ZONE.append(frozenset(zone))
KING_ZONE = tuple(KING_ZONE)

# ===================================================================
# Piece-Square Tables
# ===================================================================

_PAWN_T = (
      0,   0,   0,   0,   0,   0,
      5,   8,  12,  12,   8,   5,
     14,  16,  32,  32,  16,  14,
     22,  28,  42,  42,  28,  22,
     50,  55,  65,  65,  55,  50,
     90,  90,  90,  90,  90,  90,
)

_KNIGHT_T = (
    -28, -15,  -5,  -5, -15, -28,
    -15,   2,  20,  20,   2, -15,
     -5,  24,  42,  42,  24,  -5,
     -5,  24,  42,  42,  24,  -5,
    -15,  12,  24,  24,  12, -15,
    -28, -15,  -5,  -5, -15, -28,
)

_BISHOP_T = (
    -12,  -6,  -8,  -6,  -6, -12,
     -6,  14,   8,  10,  14,  -6,
     -6,   8,  24,  24,   8,  -6,
     -6,   8,  24,  24,   8,  -6,
     -6,  14,  10,  10,  14,  -6,
    -12,  -6,  -8,  -6,  -6, -12,
)

_QUEEN_T = (
    -12,  -5,  -5,  -5,  -5, -12,
     -5,   5,   8,   8,   5,  -5,
     -5,   8,  14,  14,   8,  -5,
     -5,   8,  14,  14,   8,  -5,
     -5,   5,   8,   8,   5,  -5,
    -12,  -5,  -5,  -5,  -5, -12,
)

_KING_MG = (
     32,  42,  10,   0,  10,  32,
     18,  18,  -8, -12,  -8,  18,
    -15, -22, -30, -30, -22, -15,
    -28, -38, -45, -45, -38, -28,
    -40, -50, -60, -60, -50, -40,
    -55, -65, -75, -75, -65, -55,
)

_KING_EG = (
    -25, -12,  -5,  -5, -12, -25,
    -12,   5,  18,  18,   5, -12,
     -5,  18,  30,  30,  18,  -5,
     -5,  18,  30,  30,  18,  -5,
    -12,   5,  18,  18,   5, -12,
    -25, -12,  -5,  -5, -12, -25,
)

PST_MG = [None] * 11
PST_EG = [None] * 11

_WMG = {WP: _PAWN_T, WN: _KNIGHT_T, WB: _BISHOP_T, WQ: _QUEEN_T, WK: _KING_MG}
_WEG = {WP: _PAWN_T, WN: _KNIGHT_T, WB: _BISHOP_T, WQ: _QUEEN_T, WK: _KING_EG}

for _p in range(1, 6):
    PST_MG[_p] = _WMG[_p]
    PST_EG[_p] = _WEG[_p]
for _p in range(6, 11):
    PST_MG[_p] = tuple(_WMG[_p - 5][MIR[i]] for i in range(N))
    PST_EG[_p] = tuple(_WEG[_p - 5][MIR[i]] for i in range(N))

SVAL_MG = [None] * 11
SVAL_EG = [None] * 11

for _p in range(1, 11):
    mg = [0] * N
    eg = [0] * N
    for _sq in range(N):
        mg[_sq] = SIGN[_p] * (MAT[_p] + PST_MG[_p][_sq])
        eg[_sq] = SIGN[_p] * (MAT[_p] + PST_EG[_p][_sq])
    SVAL_MG[_p] = tuple(mg)
    SVAL_EG[_p] = tuple(eg)

PASSED_W = (0, 12, 28, 55, 85, 120)
PASSED_B = (120, 85, 55, 28, 12, 0)

# Connected pawn bonus by rank
CONNECTED_W = (0, 5, 7, 12, 20, 30)
CONNECTED_B = (30, 20, 12, 7, 5, 0)

# ===================================================================
# Zobrist Hashing
# ===================================================================

_rng = random.Random(0xDEAD_BEEF_CAFE)
ZTBL = [[_rng.getrandbits(64) for _ in range(N)] for _ in range(11)]
ZSIDE = _rng.getrandbits(64)

# ===================================================================
# LMR Table
# ===================================================================

LMR = [[0] * MAX_PLY for _ in range(MAX_PLY)]
for _d in range(1, MAX_PLY):
    for _m in range(1, MAX_PLY):
        LMR[_d][_m] = max(1, int(0.75 + log(_d) * log(_m + 1) * 0.6))

LMP_LIMITS = (0, 8, 14, 22, 32)

# ===================================================================
# Transposition Table with aging
# ===================================================================

_TT_KEY = [0] * TT_SIZE
_TT_DAT = [None] * TT_SIZE
_TT_AGE = [0] * TT_SIZE

EXACT, LOWER, UPPER = 0, 1, 2

_tt_generation = 0


def _tt_store(h, d, s, f, mv):
    i = h & TT_MASK
    old = _TT_DAT[i]
    if old is None or _TT_KEY[i] == h or old[0] <= d or _TT_AGE[i] < _tt_generation:
        _TT_KEY[i] = h
        _TT_DAT[i] = (d, s, f, mv)
        _TT_AGE[i] = _tt_generation


def _tt_probe(h, d, a, b):
    i = h & TT_MASK
    if _TT_KEY[i] != h or _TT_DAT[i] is None:
        return False, 0, None
    td, ts, tf, tm = _TT_DAT[i]
    if td >= d:
        if tf == EXACT:
            return True, ts, tm
        if tf == LOWER and ts >= b:
            return True, ts, tm
        if tf == UPPER and ts <= a:
            return True, ts, tm
    return False, 0, tm


# ===================================================================
# Asymmetric Play
# ===================================================================

_side_to_move = True


# ===================================================================
# Board Class
# ===================================================================

class Board:
    __slots__ = ('sq', 'zh', 'cnt', 'ksq', 'smg', 'seg', 'ph')

    def __init__(self, arr):
        self.sq = [0] * N
        self.zh = 0
        self.cnt = [0] * 11
        self.ksq = [-1, -1]
        self.smg = 0
        self.seg = 0
        self.ph = 0
        for r in range(6):
            for c in range(6):
                p = int(arr[r][c])
                i = r * 6 + c
                self.sq[i] = p
                if p:
                    self.zh ^= ZTBL[p][i]
                    self.cnt[p] += 1
                    self.smg += SVAL_MG[p][i]
                    self.seg += SVAL_EG[p][i]
                    self.ph += PHASE_W[p]
                    if p == WK:
                        self.ksq[0] = i
                    elif p == BK:
                        self.ksq[1] = i

    def taper(self):
        ph16 = min(self.ph, 16)
        return (self.smg * ph16 + self.seg * (16 - ph16)) >> 4

    def make(self, fr, to, pc, cap, pro):
        sq = self.sq
        self.zh ^= ZTBL[pc][fr]
        self.smg -= SVAL_MG[pc][fr]
        self.seg -= SVAL_EG[pc][fr]
        if cap:
            self.cnt[cap] -= 1
            self.zh ^= ZTBL[cap][to]
            self.smg -= SVAL_MG[cap][to]
            self.seg -= SVAL_EG[cap][to]
            self.ph -= PHASE_W[cap]
        pl = pro or pc
        sq[fr] = 0
        sq[to] = pl
        self.zh ^= ZTBL[pl][to]
        self.smg += SVAL_MG[pl][to]
        self.seg += SVAL_EG[pl][to]
        if pro:
            self.cnt[pc] -= 1
            self.cnt[pl] += 1
            self.ph += PHASE_W[pl] - PHASE_W[pc]
        if pc == WK:
            self.ksq[0] = to
        elif pc == BK:
            self.ksq[1] = to
        self.zh ^= ZSIDE

    def unmake(self, fr, to, pc, cap, pro):
        sq = self.sq
        pl = pro or pc
        self.zh ^= ZTBL[pl][to]
        self.smg -= SVAL_MG[pl][to]
        self.seg -= SVAL_EG[pl][to]
        sq[to] = 0
        if pro:
            self.cnt[pl] -= 1
            self.cnt[pc] += 1
            self.ph -= PHASE_W[pl] - PHASE_W[pc]
        sq[fr] = pc
        self.zh ^= ZTBL[pc][fr]
        self.smg += SVAL_MG[pc][fr]
        self.seg += SVAL_EG[pc][fr]
        if cap:
            sq[to] = cap
            self.cnt[cap] += 1
            self.zh ^= ZTBL[cap][to]
            self.smg += SVAL_MG[cap][to]
            self.seg += SVAL_EG[cap][to]
            self.ph += PHASE_W[cap]
        if pc == WK:
            self.ksq[0] = fr
        elif pc == BK:
            self.ksq[1] = fr
        self.zh ^= ZSIDE

    def attacked(self, si, by_w):
        sq = self.sq
        r, c = si // 6, si % 6
        if by_w:
            pr = r - 1
            if pr >= 0:
                if c > 0 and sq[pr * 6 + c - 1] == WP:
                    return True
                if c < 5 and sq[pr * 6 + c + 1] == WP:
                    return True
            kn, bi, qu, ki = WN, WB, WQ, WK
        else:
            pr = r + 1
            if pr < 6:
                if c > 0 and sq[pr * 6 + c - 1] == BP:
                    return True
                if c < 5 and sq[pr * 6 + c + 1] == BP:
                    return True
            kn, bi, qu, ki = BN, BB, BQ, BK
        for ni in KN[si]:
            if sq[ni] == kn:
                return True
        for ni in KG[si]:
            if sq[ni] == ki:
                return True
        for ray in DR[si]:
            for ts in ray:
                p = sq[ts]
                if p:
                    if p == bi or p == qu:
                        return True
                    break
        for ray in SR[si]:
            for ts in ray:
                p = sq[ts]
                if p:
                    if p == qu:
                        return True
                    break
        return False

    def in_check(self, w):
        ki = self.ksq[0 if w else 1]
        if ki < 0:
            return True
        return self.attacked(ki, not w)

    def promos(self, w):
        cnt = self.cnt
        o = W_PROMO if w else B_PROMO
        return tuple(pt for pt in o if cnt[pt] < INIT_CNT[pt])

    def gen(self, w):
        sq = self.sq
        mvs = []
        s = 1 if w else 2
        op = 3 - s
        for fr in range(N):
            p = sq[fr]
            if not p or SIDE[p] != s:
                continue
            r, c = fr // 6, fr % 6
            if p == WP:
                nr = r + 1
                if nr < 6:
                    to = nr * 6 + c
                    if not sq[to]:
                        if nr == 5:
                            for pp in self.promos(True):
                                mvs.append((fr, to, p, 0, pp))
                        else:
                            mvs.append((fr, to, p, 0, 0))
                    for dc in (-1, 1):
                        nc = c + dc
                        if 0 <= nc < 6:
                            to = nr * 6 + nc
                            cap = sq[to]
                            if cap and SIDE[cap] == 2:
                                if nr == 5:
                                    for pp in self.promos(True):
                                        mvs.append((fr, to, p, cap, pp))
                                else:
                                    mvs.append((fr, to, p, cap, 0))
            elif p == BP:
                nr = r - 1
                if nr >= 0:
                    to = nr * 6 + c
                    if not sq[to]:
                        if nr == 0:
                            for pp in self.promos(False):
                                mvs.append((fr, to, p, 0, pp))
                        else:
                            mvs.append((fr, to, p, 0, 0))
                    for dc in (-1, 1):
                        nc = c + dc
                        if 0 <= nc < 6:
                            to = nr * 6 + nc
                            cap = sq[to]
                            if cap and SIDE[cap] == 1:
                                if nr == 0:
                                    for pp in self.promos(False):
                                        mvs.append((fr, to, p, cap, pp))
                                else:
                                    mvs.append((fr, to, p, cap, 0))
            elif p == WN or p == BN:
                for to in KN[fr]:
                    cap = sq[to]
                    if not cap or SIDE[cap] == op:
                        mvs.append((fr, to, p, cap, 0))
            elif p == WK or p == BK:
                for to in KG[fr]:
                    cap = sq[to]
                    if not cap or SIDE[cap] == op:
                        mvs.append((fr, to, p, cap, 0))
            elif p == WB or p == BB:
                for ray in DR[fr]:
                    for to in ray:
                        cap = sq[to]
                        if not cap:
                            mvs.append((fr, to, p, 0, 0))
                        else:
                            if SIDE[cap] == op:
                                mvs.append((fr, to, p, cap, 0))
                            break
            elif p == WQ or p == BQ:
                for ray in AR[fr]:
                    for to in ray:
                        cap = sq[to]
                        if not cap:
                            mvs.append((fr, to, p, 0, 0))
                        else:
                            if SIDE[cap] == op:
                                mvs.append((fr, to, p, cap, 0))
                            break
        return mvs

    def gen_caps(self, w):
        sq = self.sq
        mvs = []
        s = 1 if w else 2
        op = 3 - s
        for fr in range(N):
            p = sq[fr]
            if not p or SIDE[p] != s:
                continue
            r, c = fr // 6, fr % 6
            if p == WP:
                nr = r + 1
                if nr < 6:
                    if nr == 5:
                        to = nr * 6 + c
                        if not sq[to]:
                            for pp in self.promos(True):
                                mvs.append((fr, to, p, 0, pp))
                    for dc in (-1, 1):
                        nc = c + dc
                        if 0 <= nc < 6:
                            to = nr * 6 + nc
                            cap = sq[to]
                            if cap and SIDE[cap] == 2:
                                if nr == 5:
                                    for pp in self.promos(True):
                                        mvs.append((fr, to, p, cap, pp))
                                else:
                                    mvs.append((fr, to, p, cap, 0))
            elif p == BP:
                nr = r - 1
                if nr >= 0:
                    if nr == 0:
                        to = nr * 6 + c
                        if not sq[to]:
                            for pp in self.promos(False):
                                mvs.append((fr, to, p, 0, pp))
                    for dc in (-1, 1):
                        nc = c + dc
                        if 0 <= nc < 6:
                            to = nr * 6 + nc
                            cap = sq[to]
                            if cap and SIDE[cap] == 1:
                                if nr == 0:
                                    for pp in self.promos(False):
                                        mvs.append((fr, to, p, cap, pp))
                                else:
                                    mvs.append((fr, to, p, cap, 0))
            elif p == WN or p == BN:
                for to in KN[fr]:
                    cap = sq[to]
                    if cap and SIDE[cap] == op:
                        mvs.append((fr, to, p, cap, 0))
            elif p == WK or p == BK:
                for to in KG[fr]:
                    cap = sq[to]
                    if cap and SIDE[cap] == op:
                        mvs.append((fr, to, p, cap, 0))
            elif p == WB or p == BB:
                for ray in DR[fr]:
                    for to in ray:
                        cap = sq[to]
                        if not cap:
                            continue
                        if SIDE[cap] == op:
                            mvs.append((fr, to, p, cap, 0))
                        break
            elif p == WQ or p == BQ:
                for ray in AR[fr]:
                    for to in ray:
                        cap = sq[to]
                        if not cap:
                            continue
                        if SIDE[cap] == op:
                            mvs.append((fr, to, p, cap, 0))
                        break
        return mvs

    def legal(self, w):
        out = []
        for mv in self.gen(w):
            self.make(*mv)
            if not self.in_check(w):
                out.append(mv)
            self.unmake(*mv)
        return out

    def legal_caps(self, w):
        out = []
        for mv in self.gen_caps(w):
            self.make(*mv)
            if not self.in_check(w):
                out.append(mv)
            self.unmake(*mv)
        return out

    # --- SINGLE-PASS evaluation (mobility + threats + king zone in ONE loop) ---

    def evaluate(self):
        sq = self.sq
        cnt = self.cnt
        ph = self.ph
        is_eg = ph <= 4

        score = self.taper()

        wki = self.ksq[0]
        bki = self.ksq[1]
        wpf = [0] * 6
        bpf = [0] * 6

        wm = bm = 0

        # Collect pawn files + connected pawns
        for i in range(N):
            p = sq[i]
            if p == WP:
                c = i % 6
                wpf[c] += 1
                r = i // 6
                if r > 0:
                    if c > 0 and sq[(r-1)*6+c-1] == WP:
                        score += CONNECTED_W[r]
                    elif c < 5 and sq[(r-1)*6+c+1] == WP:
                        score += CONNECTED_W[r]
            elif p == BP:
                c = i % 6
                bpf[c] += 1
                r = i // 6
                if r < 5:
                    if c > 0 and sq[(r+1)*6+c-1] == BP:
                        score -= CONNECTED_B[r]
                    elif c < 5 and sq[(r+1)*6+c+1] == BP:
                        score -= CONNECTED_B[r]

        # Mobility (same as v1 — fast)
        wm = bm = 0
        for i in range(N):
            p = sq[i]
            if p == WN:
                for to in KN[i]:
                    if not sq[to] or SIDE[sq[to]] == 2:
                        wm += 1
            elif p == BN:
                for to in KN[i]:
                    if not sq[to] or SIDE[sq[to]] == 1:
                        bm += 1
            elif p == WB:
                for ray in DR[i]:
                    for to in ray:
                        t = sq[to]
                        if not t:
                            wm += 1
                        else:
                            if SIDE[t] == 2:
                                wm += 1
                            break
            elif p == BB:
                for ray in DR[i]:
                    for to in ray:
                        t = sq[to]
                        if not t:
                            bm += 1
                        else:
                            if SIDE[t] == 1:
                                bm += 1
                            break
            elif p == WQ:
                for ray in AR[i]:
                    for to in ray:
                        t = sq[to]
                        if not t:
                            wm += 1
                        else:
                            if SIDE[t] == 2:
                                wm += 1
                            break
            elif p == BQ:
                for ray in AR[i]:
                    for to in ray:
                        t = sq[to]
                        if not t:
                            bm += 1
                        else:
                            if SIDE[t] == 1:
                                bm += 1
                            break
        score += 4 * (wm - bm)

        # Bishop pair
        if cnt[WB] >= 2:
            score += 55
        if cnt[BB] >= 2:
            score -= 55

        # Pawn structure: doubled + isolated
        for c in range(6):
            wp = wpf[c]
            bp = bpf[c]
            if wp > 1:
                score -= 18 * (wp - 1)
            if bp > 1:
                score += 18 * (bp - 1)
            lw = wpf[c - 1] if c > 0 else 0
            rw = wpf[c + 1] if c < 5 else 0
            if wp > 0 and not lw and not rw:
                score -= 18
            lb = bpf[c - 1] if c > 0 else 0
            rb = bpf[c + 1] if c < 5 else 0
            if bp > 0 and not lb and not rb:
                score += 18

        # Passed pawns + Unstoppable
        w_has_promos = bool(self.promos(True))
        b_has_promos = bool(self.promos(False))

        for i in range(N):
            p = sq[i]
            if p == WP:
                r, c = i // 6, i % 6
                ok = True
                for rr in range(r + 1, 6):
                    for dc in (-1, 0, 1):
                        cc = c + dc
                        if 0 <= cc < 6 and sq[rr * 6 + cc] == BP:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    passed_bonus = PASSED_W[r]
                    pawn_dist = 5 - r
                    if pawn_dist <= 3:
                        unstoppable = True
                        for ei in range(N):
                            ep = sq[ei]
                            if ep and SIDE[ep] == 2 and ep != BK:
                                min_d = 99
                                for pr in range(r + 1, 6):
                                    md = abs(ei // 6 - pr) + abs(ei % 6 - c)
                                    if md < min_d:
                                        min_d = md
                                if min_d <= pawn_dist:
                                    unstoppable = False
                                    break
                        if unstoppable:
                            passed_bonus = 600
                    score += passed_bonus
            elif p == BP:
                r, c = i // 6, i % 6
                ok = True
                for rr in range(0, r):
                    for dc in (-1, 0, 1):
                        cc = c + dc
                        if 0 <= cc < 6 and sq[rr * 6 + cc] == WP:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    passed_bonus = PASSED_B[r]
                    pawn_dist = r
                    if pawn_dist <= 3:
                        unstoppable = True
                        for ei in range(N):
                            ep = sq[ei]
                            if ep and SIDE[ep] == 1 and ep != WK:
                                min_d = 99
                                for pr in range(0, r):
                                    md = abs(ei // 6 - pr) + abs(ei % 6 - c)
                                    if md < min_d:
                                        min_d = md
                                if min_d <= pawn_dist:
                                    unstoppable = False
                                    break
                        if unstoppable:
                            passed_bonus = 600
                    score -= passed_bonus

        # Promotion threat
        for c in range(6):
            if sq[4 * 6 + c] == WP and sq[5 * 6 + c] == EMPTY:
                if w_has_promos:
                    score += 150
            if sq[1 * 6 + c] == BP and sq[0 * 6 + c] == EMPTY:
                if b_has_promos:
                    score -= 150

        # Center control
        for ci in (14, 15, 20, 21):
            p = sq[ci]
            if SIDE[p] == 1:
                score += 10
            elif SIDE[p] == 2:
                score -= 10

        # King safety (midgame only)
        if not is_eg:
            w_shield_val = 20
            b_shield_val = 20
            if not _side_to_move:
                b_shield_val = 30

            if wki >= 0:
                wkr, wkc = wki // 6, wki % 6
                sh = 0
                for dc in (-1, 0, 1):
                    nc = wkc + dc
                    if 0 <= nc < 6 and wkr + 1 < 6 and sq[(wkr + 1) * 6 + nc] == WP:
                        sh += 1
                score += sh * w_shield_val
                for ni in KG[wki]:
                    t = sq[ni]
                    if not t or SIDE[t] == 2:
                        score -= 6
            if bki >= 0:
                bkr, bkc = bki // 6, bki % 6
                sh = 0
                for dc in (-1, 0, 1):
                    nc = bkc + dc
                    if 0 <= nc < 6 and bkr - 1 >= 0 and sq[(bkr - 1) * 6 + nc] == BP:
                        sh += 1
                score -= sh * b_shield_val
                for ni in KG[bki]:
                    t = sq[ni]
                    if not t or SIDE[t] == 1:
                        score += 6

        # King tropism
        w_tropism_mult = 1.5 if _side_to_move else 1.0
        b_tropism_mult = 1.0 if _side_to_move else 1.5

        if bki >= 0:
            bkr, bkc = bki // 6, bki % 6
            for i in range(N):
                p = sq[i]
                if p == WN:
                    d = abs(i // 6 - bkr) + abs(i % 6 - bkc)
                    if d <= 3:
                        score += int((4 - d) * 8 * w_tropism_mult)
                elif p == WQ:
                    d = abs(i // 6 - bkr) + abs(i % 6 - bkc)
                    if d <= 4:
                        score += int((5 - d) * 5 * w_tropism_mult)
        if wki >= 0:
            wkr, wkc = wki // 6, wki % 6
            for i in range(N):
                p = sq[i]
                if p == BN:
                    d = abs(i // 6 - wkr) + abs(i % 6 - wkc)
                    if d <= 3:
                        score -= int((4 - d) * 8 * b_tropism_mult)
                elif p == BQ:
                    d = abs(i // 6 - wkr) + abs(i % 6 - wkc)
                    if d <= 4:
                        score -= int((5 - d) * 5 * b_tropism_mult)

        # Asymmetric pawn advancement
        if _side_to_move:
            for i in range(N):
                if sq[i] == WP and i // 6 >= 4:
                    score += 15

        # Hanging piece penalty (undefended pieces attacked by enemy)
        for i in range(N):
            p = sq[i]
            if p in (WN, WB, WQ):
                if not self.attacked(i, True):
                    if self.attacked(i, False):
                        score -= MAT[p] // 4
                    else:
                        score -= 8
            elif p in (BN, BB, BQ):
                if not self.attacked(i, False):
                    if self.attacked(i, True):
                        score += MAT[p] // 4
                    else:
                        score += 8

        # Endgame king cornering (stronger gradient for big advantages)
        if is_eg and wki >= 0 and bki >= 0:
            dist = abs(wki // 6 - bki // 6) + abs(wki % 6 - bki % 6)
            if score > 80:
                if score >= 500:
                    bkr, bkc = bki // 6, bki % 6
                    edge_dist = min(bkr, 5 - bkr, bkc, 5 - bkc)
                    score += (8 - dist) * 15
                    score += (3 - edge_dist) * 25
                    if bkr in (0, 5) and bkc in (0, 5):
                        score += 30
                else:
                    bkr, bkc = bki // 6, bki % 6
                    edge_dist = min(bkr, 5 - bkr, bkc, 5 - bkc)
                    score += (8 - dist) * 8
                    score += (3 - edge_dist) * 8
            elif score < -80:
                if score <= -500:
                    wkr, wkc = wki // 6, wki % 6
                    edge_dist = min(wkr, 5 - wkr, wkc, 5 - wkc)
                    score -= (8 - dist) * 15
                    score -= (3 - edge_dist) * 25
                    if wkr in (0, 5) and wkc in (0, 5):
                        score -= 30
                else:
                    wkr, wkc = wki // 6, wki % 6
                    edge_dist = min(wkr, 5 - wkr, wkc, 5 - wkc)
                    score -= (8 - dist) * 8
                    score -= (3 - edge_dist) * 8

        # Knight outposts
        for i in range(N):
            p = sq[i]
            r, c = i // 6, i % 6
            if p == WN and r >= 3:
                for dc in (-1, 1):
                    nc = c + dc
                    if 0 <= nc < 6 and r - 1 >= 0 and sq[(r - 1) * 6 + nc] == WP:
                        score += 18
                        break
            elif p == BN and r <= 2:
                for dc in (-1, 1):
                    nc = c + dc
                    if 0 <= nc < 6 and r + 1 < 6 and sq[(r + 1) * 6 + nc] == BP:
                        score -= 18
                        break

        # Check bonus
        if wki >= 0 and self.attacked(wki, False):
            score -= 45
        if bki >= 0 and self.attacked(bki, True):
            score += 45

        return score

    def evaluate_fast(self):
        cnt = self.cnt
        score = self.taper()
        if cnt[WB] >= 2:
            score += 55
        if cnt[BB] >= 2:
            score -= 55
        return score


# ===================================================================
# SEE
# ===================================================================

def _lva(sq_arr, sq_idx, side):
    r, c = sq_idx // 6, sq_idx % 6
    if side == 1:
        for dc in (-1, 1):
            nc = c + dc
            if 0 <= nc < 6:
                fr = (r - 1) * 6 + nc
                if fr >= 0 and sq_arr[fr] == WP:
                    return fr, WP
        kn, bi, qu, ki = WN, WB, WQ, WK
    else:
        for dc in (-1, 1):
            nc = c + dc
            if 0 <= nc < 6:
                fr = (r + 1) * 6 + nc
                if fr < N and sq_arr[fr] == BP:
                    return fr, BP
        kn, bi, qu, ki = BN, BB, BQ, BK
    for ni in KN[sq_idx]:
        if sq_arr[ni] == kn:
            return ni, kn
    for ray in DR[sq_idx]:
        for ts in ray:
            p = sq_arr[ts]
            if p:
                if p == bi or p == qu:
                    return ts, p
                break
    for ray in SR[sq_idx]:
        for ts in ray:
            p = sq_arr[ts]
            if p:
                if p == qu:
                    return ts, p
                break
    for ni in KG[sq_idx]:
        if sq_arr[ni] == ki:
            return ni, ki
    return -1, 0


def _see(bd, fr, to):
    target = bd.sq[to]
    if not target:
        return 0
    piece = bd.sq[fr]
    sq = list(bd.sq)
    vc = [MAT[target]]
    sq[to] = piece
    sq[fr] = EMPTY
    side = 3 - SIDE[piece]
    for _ in range(14):
        f2, p2 = _lva(sq, to, side)
        if f2 < 0:
            break
        vc.append(MAT[sq[to]])
        sq[to] = p2
        sq[f2] = EMPTY
        side = 3 - side
    if len(vc) == 1:
        return vc[0]
    result = max(0, vc[-1])
    for i in range(len(vc) - 2, 0, -1):
        result = max(0, vc[i] - result)
    return vc[0] - result


def _min_attacker(sq_arr, sq_idx, by_w):
    r, c = sq_idx // 6, sq_idx % 6
    best = INF
    if by_w:
        pr = r - 1
        if pr >= 0:
            if c > 0 and sq_arr[pr * 6 + c - 1] == WP:
                return MAT[WP]
            if c < 5 and sq_arr[pr * 6 + c + 1] == WP:
                return MAT[WP]
        kn, bi, qu, ki = WN, WB, WQ, WK
    else:
        pr = r + 1
        if pr < 6:
            if c > 0 and sq_arr[pr * 6 + c - 1] == BP:
                return MAT[BP]
            if c < 5 and sq_arr[pr * 6 + c + 1] == BP:
                return MAT[BP]
        kn, bi, qu, ki = BN, BB, BQ, BK
    for ni in KN[sq_idx]:
        if sq_arr[ni] == kn:
            best = MAT[kn]
            break
    if best <= MAT[kn]:
        return best
    for ray in DR[sq_idx]:
        for ts in ray:
            p = sq_arr[ts]
            if p:
                if p == bi and MAT[bi] < best:
                    best = MAT[bi]
                elif p == qu and MAT[qu] < best:
                    best = MAT[qu]
                break
    for ray in SR[sq_idx]:
        for ts in ray:
            p = sq_arr[ts]
            if p:
                if p == qu and MAT[qu] < best:
                    best = MAT[qu]
                break
    for ni in KG[sq_idx]:
        if sq_arr[ni] == ki and MAT[ki] < best:
            best = MAT[ki]
    return best if best < INF else 0


# ===================================================================
# Engine State
# ===================================================================

_killers = [[None, None, None] for _ in range(MAX_PLY)]
_history = {}
_cap_hist = {}
_cmoves = [[None] * N for _ in range(11)]
_cmhist = {}
_nodes = 0
_t0 = 0.0
_tlim = 15.0
_stop = False
_pos_history = set()
_game_hist = {}
_game_last_pieces = 0
_clock_remaining = TOTAL_CLOCK
_move_number = 0
_score_history = []


def _allocate_time(bd):
    global _clock_remaining
    usable = _clock_remaining - SAFETY_BUFFER
    if usable <= 0:
        return MIN_MOVE_TIME

    phase = bd.ph
    if phase >= 6:
        est_remaining_moves = 35
    elif phase >= 3:
        est_remaining_moves = 25
    else:
        est_remaining_moves = 15

    base = usable / max(est_remaining_moves, 1)

    if phase >= 3 and _move_number >= 4:
        base *= 1.3
    if _move_number < 4:
        base *= 0.6

    # Score instability
    if len(_score_history) >= 2:
        delta = abs(_score_history[-1] - _score_history[-2])
        if delta > 60:
            base *= 1.3
        elif delta > 30:
            base *= 1.1

    return max(MIN_MOVE_TIME, min(base, MAX_MOVE_TIME, usable * 0.4))


def _record_position(board_arr, playing_white):
    h = 0
    for r in range(6):
        for c in range(6):
            p = int(board_arr[r][c])
            if p:
                h ^= ZTBL[p][r * 6 + c]
    if not playing_white:
        h ^= ZSIDE
    _game_hist[h] = _game_hist.get(h, 0) + 1
    return h


def _reset(tl):
    global _nodes, _t0, _tlim, _stop, _tt_generation
    _nodes = 0
    _t0 = time.time()
    _tlim = tl
    _stop = False
    _tt_generation += 1
    for i in range(MAX_PLY):
        _killers[i][0] = _killers[i][1] = _killers[i][2] = None
    for k in list(_history):
        _history[k] >>= 2
        if _history[k] == 0:
            del _history[k]


def _tick():
    global _nodes, _stop
    _nodes += 1
    if _nodes & 4095 == 0 and time.time() - _t0 >= _tlim:
        _stop = True


# ===================================================================
# Move Ordering
# ===================================================================

def _mscore(mv, tt_mv, ply, prev_mv, bd):
    fr, to, pc, cap, pro = mv
    if tt_mv and mv == tt_mv:
        return 10_000_000
    if cap:
        see_val = _see(bd, fr, to)
        if see_val >= 0:
            return 8_000_000 + see_val
        else:
            return 900_000 + see_val
    if pro:
        return 7_000_000 + MAT[pro]
    kl = _killers[ply] if ply < MAX_PLY else [None, None]
    if mv == kl[0]:
        return 6_000_000
    if mv == kl[1]:
        return 5_500_000
    if prev_mv:
        cm = _cmoves[prev_mv[2]][prev_mv[1]]
        if cm and mv == cm:
            return 5_000_000
    hist = _history.get((pc, to), 0)
    if MAT[pc] >= MAT[WN]:
        opp_side = 3 - SIDE[pc]
        by_w = (opp_side == 1)
        sq = bd.sq
        old_fr = sq[fr]; old_to = sq[to]
        sq[fr] = EMPTY; sq[to] = pc
        min_att = _min_attacker(sq, to, by_w)
        unsafe = False; loss = 0
        if 0 < min_att < MAT[pc]:
            unsafe = True; loss = MAT[pc] - min_att
        elif min_att > 0:
            our_def = _min_attacker(sq, to, not by_w)
            if our_def == 0:
                unsafe = True; loss = MAT[pc]
        sq[fr] = old_fr; sq[to] = old_to
        if unsafe:
            return -2_000_000 - loss + hist // 100
    return hist


def _order(mvs, tt_mv, ply, prev_mv, bd):
    return sorted(mvs, key=lambda m: -_mscore(m, tt_mv, ply, prev_mv, bd))


# ===================================================================
# Quiescence Search
# ===================================================================

def _qs(bd, a, b, w, ply):
    _tick()
    if _stop:
        return 0

    in_check = bd.in_check(w)

    ev = bd.evaluate_fast()
    if not w:
        ev = -ev

    if not in_check:
        if ev >= b:
            return b
        if ply > 12:
            return ev
        if ev > a:
            a = ev
    else:
        if ply > 12:
            return ev

    if in_check:
        # In check: search ALL legal moves (evasions), not just captures
        moves = bd.legal(w)
        if not moves:
            return -(MATE - ply)  # checkmate!
        scored = []
        for mv in moves:
            fr, to, pc, cap, pro = mv
            se = _see(bd, fr, to) if cap else 0
            scored.append((mv, se + (MAT[pro] if pro else 0) + (MAT[cap] if cap else 0)))
        scored.sort(key=lambda x: -x[1])
    else:
        scored = []
        for mv in bd.legal_caps(w):
            fr, to, pc, cap, pro = mv
            se = _see(bd, fr, to) if cap else 0
            if se >= -50 or pro:
                scored.append((mv, se + (MAT[pro] if pro else 0)))
        if not scored:
            return a
        scored.sort(key=lambda x: -x[1])

    for mv, _ in scored:
        fr, to, pc, cap, pro = mv
        if not in_check and cap and not pro:
            if ev + MAT[cap] + 150 < a:
                continue
        bd.make(*mv)
        s = -_qs(bd, -b, -a, not w, ply + 1)
        bd.unmake(*mv)
        if _stop:
            return 0
        if s >= b:
            return b
        if s > a:
            a = s
    return a


# ===================================================================
# Negamax PVS — speed-optimized (no SE/probcut, max NPS)
# ===================================================================

def _negamax(bd, d, a, b, w, ply, prev_mv=None, check_ext=0, prev_eval=-MATE):
    global _stop

    _tick()
    if _stop:
        return 0

    h = bd.zh
    oa = a

    # Dynamic contempt — penalize draws harder when winning
    if ply > 0 and h in _pos_history:
        ev = bd.evaluate_fast()
        if not w:
            ev = -ev
        dyn = min(200, CONTEMPT + max(0, abs(ev) - 80))
        if ev > 80:
            return -dyn  # we're winning, draw is bad
        elif ev < -80:
            return dyn   # we're losing, draw is good
        else:
            return -CONTEMPT

    if ply > 0:
        gc = _game_hist.get(h, 0)
        if gc >= 2:
            return -CONTEMPT
        if gc == 1:
            ev = bd.evaluate_fast()
            if not w:
                ev = -ev
            dyn = min(200, CONTEMPT + max(0, abs(ev) - 80))
            if ev > 80:
                return -dyn
            elif ev < -80:
                return dyn
            else:
                return -CONTEMPT

    # TT probe (inlined for speed)
    tt_mv = None
    i_tt = h & TT_MASK
    if _TT_KEY[i_tt] == h and _TT_DAT[i_tt] is not None:
        td, ts, tf, tt_mv = _TT_DAT[i_tt]
        if td >= d and ply > 0:
            if tf == EXACT:
                return ts
            if tf == LOWER and ts >= b:
                return ts
            if tf == UPPER and ts <= a:
                return ts

    ic = bd.in_check(w)

    if ic and check_ext < 3:
        d += 1
        check_ext += 1

    if d <= 0:
        return _qs(bd, a, b, w, ply)

    a = max(a, -(MATE - ply))
    b = min(b, MATE - ply - 1)
    if a >= b:
        return a

    static_eval = None
    if not ic:
        se = bd.taper()
        if w:
            se += 10
        else:
            se -= 10
        if not w:
            se = -se
        static_eval = se

    # Reverse futility (extended to d<=4)
    if not ic and ply > 0 and d <= 4 and abs(a) < MATE - 200:
        rfp_margin = 75 * d
        if static_eval - rfp_margin >= b:
            return static_eval - rfp_margin

    # Razoring
    if not ic and ply > 0 and d <= 2 and abs(a) < MATE - 200:
        if static_eval + 280 <= a:
            rs = _qs(bd, a, b, w, ply)
            if rs <= a:
                return rs

    improving = static_eval is not None and static_eval > prev_eval

    # Null-move (aggressive adaptive R)
    if not ic and ply > 0 and d >= 3 and bd.ph > 4:
        r = 3 + d // 5  # more aggressive than d//6
        if static_eval is not None and static_eval > b:
            r += min((static_eval - b) // 180, 3)
        if not improving:
            r += 1
        r = min(r, d - 1)
        bd.zh ^= ZSIDE
        _pos_history.add(h)
        ns = -_negamax(bd, d - 1 - r, -b, -b + 1, not w, ply + 1, None, check_ext, -MATE)
        _pos_history.discard(h)
        bd.zh ^= ZSIDE
        if _stop:
            return 0
        if ns >= b:
            return b

    mvs = bd.legal(w)
    if not mvs:
        return -(MATE - ply) if ic else 0

    # Internal iterative reduction
    if tt_mv is None and d >= 4 and not ic:
        d -= 1

    # Futility (extended to d<=4)
    futile = False
    fut_margins = (0, 110, 220, 330, 440)
    if not ic and d <= 4 and abs(a) < MATE - 200:
        if static_eval + fut_margins[d] <= a:
            futile = True

    lmp_limit = LMP_LIMITS[d] if d <= 4 else 9999
    if not improving:
        lmp_limit = max(lmp_limit // 2, 3)  # tighter LMP when not improving

    mvs = _order(mvs, tt_mv, ply, prev_mv, bd)
    best = -(MATE + 1)
    bmv = mvs[0]
    cnt = 0

    for mv in mvs:
        if _stop:
            return 0

        fr, to, pc, cap, pro = mv
        tac = cap or pro

        if futile and cnt > 0 and not tac:
            continue
        if not ic and d <= 4 and cnt >= lmp_limit and not tac:
            continue

        # SEE pruning for quiet moves — skip moves to attacked squares
        if not ic and d <= 6 and cnt > 0 and not tac and MAT[pc] >= MAT[WN]:
            opp = 3 - SIDE[pc]
            by_w = (opp == 1)
            ma = _min_attacker(bd.sq, to, by_w)
            if 0 < ma < MAT[pc] and _history.get((pc, to), 0) < 500:
                continue

        bd.make(*mv)
        _pos_history.add(h)

        gc = (not tac) and bd.in_check(not w)

        if cnt >= 3 and d >= 3 and not tac and not ic and not gc:
            di = min(d, MAX_PLY - 1)
            mi = min(cnt, MAX_PLY - 1)
            red = LMR[di][mi]
            h_score = _history.get((pc, to), 0)
            if h_score > 300:
                red = max(1, red - 1)
            if h_score < -100:
                red += 1
            if not improving:
                red += 1
            red = min(red, d - 2)
            red = max(1, red)
            s = -_negamax(bd, d - 1 - red, -a - 1, -a, not w, ply + 1, mv, check_ext,
                          static_eval if static_eval is not None else -MATE)
            if not _stop and s > a:
                s = -_negamax(bd, d - 1, -b, -a, not w, ply + 1, mv, check_ext,
                              static_eval if static_eval is not None else -MATE)
        elif cnt >= 1:
            s = -_negamax(bd, d - 1, -a - 1, -a, not w, ply + 1, mv, check_ext,
                          static_eval if static_eval is not None else -MATE)
            if not _stop and a < s < b:
                s = -_negamax(bd, d - 1, -b, -a, not w, ply + 1, mv, check_ext,
                              static_eval if static_eval is not None else -MATE)
        else:
            s = -_negamax(bd, d - 1, -b, -a, not w, ply + 1, mv, check_ext,
                          static_eval if static_eval is not None else -MATE)

        _pos_history.discard(h)
        bd.unmake(*mv)

        if _stop:
            break
        cnt += 1

        if s > best:
            best = s
            bmv = mv
        if s > a:
            a = s
        if a >= b:
            if not tac and ply < MAX_PLY:
                kl = _killers[ply]
                if kl[0] != mv:
                    kl[2] = kl[1]
                    kl[1] = kl[0]
                    kl[0] = mv
                k = (pc, to)
                _history[k] = min(_history.get(k, 0) + d * d, 100000)
                if prev_mv:
                    _cmoves[prev_mv[2]][prev_mv[1]] = mv
            break
        else:
            if not tac and s <= oa:
                k = (pc, to)
                _history[k] = max(_history.get(k, 0) - d * d, -100000)

    if not _stop:
        fl = EXACT if oa < best < b else (LOWER if best >= b else UPPER)
        _tt_store(h, d, best, fl, bmv)
    return best


# ===================================================================
# Root Search
# ===================================================================

def _root(bd, d, a, b, w):
    h = bd.zh
    oa = a
    _, _, tt_mv = _tt_probe(h, 0, a, b)
    mvs = bd.legal(w)
    if not mvs:
        return 0, None
    mvs = _order(mvs, tt_mv, 0, None, bd)
    best = -(MATE + 1)
    bmv = mvs[0]

    for i, mv in enumerate(mvs):
        if _stop:
            break
        bd.make(*mv)
        _pos_history.add(h)

        # Penalize moves that lead to positions we've seen in this game
        rep_penalty = 0
        post_hash = bd.zh
        gc = _game_hist.get(post_hash, 0)
        if gc >= 1:
            rep_penalty = CONTEMPT + 30 * gc  # increasingly avoid repeated positions

        if i == 0:
            s = -_negamax(bd, d - 1, -b, -a, not w, 1, mv, 0, -MATE)
        else:
            s = -_negamax(bd, d - 1, -a - 1, -a, not w, 1, mv, 0, -MATE)
            if not _stop and a < s < b:
                s = -_negamax(bd, d - 1, -b, -a, not w, 1, mv, 0, -MATE)

        s -= rep_penalty  # apply penalty AFTER search

        _pos_history.discard(h)
        bd.unmake(*mv)

        if _stop:
            break
        if s > best:
            best = s
            bmv = mv
        if s > a:
            a = s
        if a >= b:
            break

    if not _stop:
        fl = EXACT if oa < best < b else (LOWER if best >= b else UPPER)
        _tt_store(h, d, best, fl, bmv)
    return best, bmv


# ===================================================================
# Iterative Deepening
# ===================================================================

def _go(bd, w, move_time):
    global _pos_history
    hard = min(move_time, _clock_remaining - SAFETY_BUFFER)
    hard = max(hard, MIN_MOVE_TIME)
    _reset(hard)

    if not w:
        bd.zh ^= ZSIDE

    _pos_history = set()
    _pos_history.add(bd.zh)

    mvs = bd.legal(w)
    if not mvs:
        if not w:
            bd.zh ^= ZSIDE
        return None, 0
    if len(mvs) == 1:
        if not w:
            bd.zh ^= ZSIDE
        return mvs[0], 0

    best_mv = mvs[0]
    best_sc = 0
    soft_limit = hard * 0.55
    stable_count = 0
    prev_best_mv = None
    best_move_changes = 0

    for d in range(1, MAX_PLY):
        if _stop:
            break

        if d >= 5 and abs(best_sc) < MATE - 100:
            sc = None
            mv = None
            for asp in (40, 200, 600):
                sc, mv = _root(bd, d, best_sc - asp, best_sc + asp, w)
                if not _stop and best_sc - asp < sc < best_sc + asp:
                    break
            else:
                if _stop:
                    break
                sc, mv = _root(bd, d, -(MATE + 1), MATE + 1, w)
        else:
            sc, mv = _root(bd, d, -(MATE + 1), MATE + 1, w)

        if not _stop and mv:
            if mv == prev_best_mv:
                stable_count += 1
            else:
                stable_count = 0
                best_move_changes += 1
            prev_best_mv = mv
            best_mv = mv
            best_sc = sc

        if abs(best_sc) >= MATE - 100:
            break

        effective_limit = soft_limit
        if stable_count >= 4:
            effective_limit *= 0.75
        elif stable_count >= 2:
            effective_limit *= 0.9
        if abs(best_sc) > 500:
            effective_limit *= 0.7
        if best_move_changes >= 3 and d >= 6:
            effective_limit *= 1.25

        if time.time() - _t0 >= effective_limit:
            break

    if not w:
        bd.zh ^= ZSIDE
    return best_mv, best_sc


# ===================================================================
# Format Move
# ===================================================================

def _fmt(mv):
    fr, to, pc, cap, pro = mv
    sr, sc = fr // 6, fr % 6
    dr, dc = to // 6, to % 6
    base = f"{pc}:{COL_TO_FILE[sc]}{sr + 1}->{COL_TO_FILE[dc]}{dr + 1}"
    return f"{base}={pro}" if pro else base


# ===================================================================
# Entry Point
# ===================================================================

def get_best_move(board: np.ndarray, playing_white: bool = True) -> Optional[str]:
    global _side_to_move, _clock_remaining, _move_number, _game_last_pieces, _score_history
    _side_to_move = playing_white

    bd = Board(np.array(board, dtype=int))
    piece_count = sum(1 for sq in bd.sq if sq != EMPTY)
    if piece_count > _game_last_pieces + 2:
        _clock_remaining = TOTAL_CLOCK
        _move_number = 0
        _game_hist.clear()
        _score_history = []
    _game_last_pieces = piece_count

    _record_position(board, playing_white)
    move_time = _allocate_time(bd)

    t_start = time.time()
    mv, score = _go(bd, playing_white, move_time)
    elapsed = time.time() - t_start

    _clock_remaining -= elapsed
    _move_number += 1
    _score_history.append(score)

    return _fmt(mv) if mv else None


def get_move(board: np.ndarray, turn: int) -> str:
    playing_white = (turn == 1)
    result = get_best_move(board, playing_white)
    return result if result else "0:A1->A1"


# ===================================================================
# Smoke Test
# ===================================================================

if __name__ == "__main__":
    initial = np.array([
        [2, 3, 4, 5, 3, 2],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [6, 6, 6, 6, 6, 6],
        [7, 8, 9, 10, 8, 7],
    ], dtype=int)

    print("Prototype Clock v2 — Enhanced RoboGambit Bot")
    print("=" * 50)

    t0 = time.time()
    mw = get_best_move(initial, True)
    print(f"White: {mw}  ({time.time() - t0:.2f}s, {_nodes} nodes)")

    t0 = time.time()
    mb = get_best_move(initial, False)
    print(f"Black: {mb}  ({time.time() - t0:.2f}s, {_nodes} nodes)")
