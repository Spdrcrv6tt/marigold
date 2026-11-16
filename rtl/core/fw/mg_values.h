// Marigold value head: raw slot values from the CTC tokens, a line-for-line port of sw/intent/values.py raw_values
// (with items and _number_run), integer only. Level words become codes (-1 low, -2 medium, -3 high, -4 max, -5 min);
// the island maps the codes through the slot table and clamps to [min, max] (docs/decisions.md, Nov 15).
// Python's integers are unbounded: every quantity here is >= 0 and only grows, so values are held in i64 and clamped
// at VSAT = 2^40 (any value that large ends at the slot's max after the island's clamp, as Python's would); the
// value written to VAL is clamped to the int32 range. No 64-bit division (no libgcc on the chip): udivmod64.
#ifndef MG_VALUES_H
#define MG_VALUES_H
#include "hal.h"

enum { TK_BLANK = 0, TK_ZERO = 1, TK_TWENTY = 21, TK_HUNDRED = 29, TK_THOUSAND = 30, TK_A = 31, TK_HALF = 32,
       TK_QUARTER = 33, TK_SECOND = 34, TK_MINUTE = 35, TK_HOUR = 36, TK_DEGREE = 37, TK_PERCENT = 38, TK_LEVEL = 39,
       TK_LOW = 40, TK_MEDIUM = 41, TK_HIGH = 42, TK_MAX = 43, TK_MIN = 44 };
enum { IT_DUR = 1, IT_DEG, IT_PCT, IT_LVL, IT_WORD, IT_NUM, IT_CLK };   // items(): ('dur', s), ('deg', n), ...
#define SQ_N 'n'                  // seq entries: a merged number run, or a word (value = token id)
#define SQ_W 'w'
#define VSAT ((i64)1 << 40)
#define MG_VALUES_SCRATCH(nt) (20u * (nt) + 32u)   // scratch bytes for nt tokens (8-byte aligned)

static i64 vsat(i64 x) { return x > VSAT ? VSAT : x; }
static u64 udivmod64(u64 n, u32 d, u32 *rem) {
    u64 q = 0, r = 0;
    for (int b = 63; b >= 0; b--) {
        r = (r << 1) | ((n >> b) & 1);
        if (r >= d) { r -= d; q |= (u64)1 << b; }
    }
    if (rem) *rem = (u32)r;
    return q;
}
static int tk_numw(int id) { return id >= 1 && id <= 30; }                 // NUMV, hundred, thousand
static i64 tk_numv(int id) { return id <= 20 ? id - 1 : 20 + 10 * (id - 21); }
static int tk_unit(i64 id) { return id == TK_SECOND || id == TK_MINUTE || id == TK_HOUR; }
static i64 unit_s(i64 id) { return id == TK_SECOND ? 1 : id == TK_MINUTE ? 60 : 3600; }
static int tk_frac(i64 id) { return id == TK_HALF || id == TK_QUARTER; }

// _number_run: a run of number words -> integer; *clock = a clock reading ("three fifty")
static i64 number_run(const u8 *ws, u32 n, int *clock) {
    i64 total = 0, group = 0;
    int prev = 0, clk = 0;                                   // prev: 0 '', 'H', 'K', 'o', 't', 'T'
    for (u32 k = 0; k < n; k++) {
        int w = ws[k];
        if (w == TK_HUNDRED) { group = vsat((group > 1 ? group : 1) * 100); prev = 'H'; continue; }
        if (w == TK_THOUSAND) { total = vsat(total + vsat((group > 1 ? group : 1) * 1000)); group = 0; prev = 'K'; continue; }
        i64 v = tk_numv(w);
        int kind = v < 10 ? 'o' : v < 20 ? 't' : 'T';
        if (prev == 0) group = v;
        else if (prev == 'H' || prev == 'K') group = vsat(group + v);
        else if (kind == 'o') {
            u32 r; udivmod64((u64)group, 10, &r);
            group = (prev == 'T' && r == 0) ? vsat(group + v) : vsat(group * 10 + v);
        } else { group = vsat(group * 100 + v); clk = 1; }
        prev = kind;
    }
    *clock = clk && total == 0;
    return vsat(total + group);
}

// items: tokens -> item kinds it_k[] and values it_v[]; returns the item count. sq_*: scratch for nt entries.
static u32 items(const u8 *tok, u32 nt, u8 *sq_k, u8 *sq_c, i64 *sq_v, u8 *it_k, i64 *it_v) {
    u32 ns = 0, i = 0, no = 0;
    while (i < nt) {
        if (tk_numw(tok[i])) {
            u32 j = i;
            while (j < nt && tk_numw(tok[j])) j++;
            int clk; i64 v = number_run(tok + i, j - i, &clk);
            sq_k[ns] = SQ_N; sq_c[ns] = (u8)clk; sq_v[ns] = v; ns++; i = j;
        } else { sq_k[ns] = SQ_W; sq_c[ns] = 0; sq_v[ns] = tok[i]; ns++; i++; }
    }
    i64 dur = 0, last_unit = 0;
    int have_dur = 0;
    no = 1;                                                  // slot 0 is kept for ('dur', dur)
    i = 0;
#define SQK(j) ((j) < ns ? sq_k[j] : 0)
#define SQV(j) ((j) < ns ? sq_v[j] : -1)
#define PUT(kk, vv) do { it_k[no] = (u8)(kk); it_v[no] = (vv); no++; } while (0)
    while (i < ns) {
        int k = sq_k[i]; i64 v = sq_v[i];
        int nk = SQK(i + 1); i64 nv = SQV(i + 1);
        int n2k = SQK(i + 2); i64 n2v = SQV(i + 2);
        if (k == SQ_W && v == TK_A && nk == SQ_W && tk_frac(nv)) {          // "a half" == "half"
            k = SQ_W; v = nv; i += 1; nk = n2k; nv = n2v;
            n2k = SQK(i + 2); n2v = SQV(i + 2);
        }
        if ((k == SQ_N || (k == SQ_W && v == TK_A)) && nk == SQ_W && tk_unit(nv)) {     // n unit, a unit
            i64 u = unit_s(nv);
            dur = vsat(dur + vsat((k == SQ_N ? v : 1) * u)); last_unit = u; have_dur = 1; i += 2; continue;
        }
        if (k == SQ_N && nk == SQ_W && nv == TK_A && n2k == SQ_W && tk_frac(n2v)) {      // "two a half minute"
            nk = n2k; nv = n2v; i += 1;
        }
        if ((k == SQ_N || (k == SQ_W && v == TK_A)) && nk == SQ_W && tk_frac(nv)) {      // n + fraction + unit
            i64 m = nv == TK_HALF ? 2 : 4, n = k == SQ_N ? v : 1;
            u32 j = i + 2;
            if (j < ns && sq_k[j] == SQ_W && sq_v[j] == TK_A) j++;
            if (j < ns && sq_k[j] == SQ_W && tk_unit(sq_v[j])) {
                i64 u = unit_s(sq_v[j]);
                dur = vsat(dur + vsat(n * u) + (i64)((i32)u / (i32)m)); last_unit = u; have_dur = 1; i = j + 1; continue;
            }
        }
        if (k == SQ_W && tk_frac(v)) {
            i64 m = v == TK_HALF ? 2 : 4;
            u32 j = i + 1;
            if (j < ns && sq_k[j] == SQ_W && sq_v[j] == TK_A) j++;
            if (j < ns && sq_k[j] == SQ_W && tk_unit(sq_v[j])) {                          // "half a minute"
                i64 u = unit_s(sq_v[j]);
                dur = vsat(dur + (i64)((i32)u / (i32)m)); last_unit = u; have_dur = 1; i = j + 1; continue;
            }
            if (last_unit) { dur = vsat(dur + (i64)((i32)last_unit / (i32)m)); i += 1; continue; }   // "a minute and a half"
            PUT(IT_WORD, v == TK_HALF ? TK_MEDIUM : TK_QUARTER); i += 1; continue;
        }
        if (k == SQ_N && nk == SQ_W && nv == TK_DEGREE) { PUT(IT_DEG, v); i += 2; continue; }
        if (k == SQ_N && nk == SQ_W && nv == TK_PERCENT) { PUT(IT_PCT, v); i += 2; continue; }
        if (k == SQ_W && v == TK_LEVEL && nk == SQ_N) { PUT(IT_LVL, nv); i += 2; continue; }
        if (k == SQ_W && v == TK_LEVEL && nk == SQ_W && nv == TK_LEVEL) { i += 1; continue; }
        if (k == SQ_N && (last_unit == 60 || last_unit == 3600) &&
            (i == 0 || (sq_k[i - 1] == SQ_W && tk_unit(sq_v[i - 1])))) {                 // "one minute thirty"
            dur = vsat(dur + vsat(v * (i64)((i32)last_unit / 60))); last_unit = (i64)((i32)last_unit / 60); i += 1; continue;
        }
        if (k == SQ_N) { PUT(sq_c[i] ? IT_CLK : IT_NUM, v); i += 1; continue; }
        if (v == TK_LOW || v == TK_MEDIUM || v == TK_HIGH || v == TK_MAX || v == TK_MIN) PUT(IT_WORD, v);
        i += 1;
    }
#undef SQK
#undef SQV
#undef PUT
    if (have_dur) { it_k[0] = IT_DUR; it_v[0] = dur; return no; }
    for (u32 j = 1; j < no; j++) { it_k[j - 1] = it_k[j]; it_v[j - 1] = it_v[j]; }
    return no - 1;
}

static int first_it(const u8 *it_k, u32 n, int k1, int k2) {
    for (u32 j = 0; j < n; j++) if (it_k[j] == k1 || it_k[j] == k2) return (int)j;
    return -1;
}

// raw_values: one raw value per slot (types[s]: 0 number, 1 duration in s, 2 level; others as number), each slot on
// its own (the island picks the action's slots). scratch: MG_VALUES_SCRATCH(nt) bytes, 8-byte aligned.
static void mg_raw_values(const u8 *tok, u32 nt, const u8 *types, u32 nslot, i32 *raw, u8 *scratch) {
    i64 *sq_v = (i64 *)scratch, *it_v = sq_v + nt + 1;
    u8 *sq_k = (u8 *)(it_v + nt + 1), *sq_c = sq_k + nt + 1, *it_k = sq_c + nt + 1;
    u32 n = items(tok, nt, sq_k, sq_c, sq_v, it_k, it_v);
    for (u32 s = 0; s < nslot; s++) {
        int t = types[s], x, have = 0;
        i64 v = 0;
        if (t == 1) {
            x = first_it(it_k, n, IT_DUR, IT_DUR);
            if (x < 0) x = first_it(it_k, n, IT_CLK, IT_CLK);
            if (x < 0) x = first_it(it_k, n, IT_NUM, IT_NUM);
            if (x >= 0) {
                have = 1; v = it_v[x];
                if (it_k[x] == IT_CLK) { u32 r; u64 q = udivmod64((u64)v, 100, &r); v = (i64)q * 60 + r; }
            }
        } else if (t == 2) {
            x = first_it(it_k, n, IT_LVL, IT_LVL);
            if (x >= 0) { have = 1; v = it_v[x]; }
            else {
                x = first_it(it_k, n, IT_WORD, IT_WORD);
                if (x >= 0) {
                    i64 w = it_v[x];
                    if (w == TK_LOW) { have = 1; v = -1; } else if (w == TK_MEDIUM) { have = 1; v = -2; }
                    else if (w == TK_HIGH) { have = 1; v = -3; } else if (w == TK_MAX) { have = 1; v = -4; }
                    else if (w == TK_MIN) { have = 1; v = -5; }                            // "quarter": no code
                }
                if (!have) {
                    x = first_it(it_k, n, IT_NUM, IT_PCT);
                    if (x >= 0) { have = 1; v = it_v[x]; }
                }
            }
        } else {
            x = first_it(it_k, n, IT_DEG, IT_PCT);
            if (x < 0) x = first_it(it_k, n, IT_NUM, IT_CLK);
            if (x < 0) x = first_it(it_k, n, IT_LVL, IT_LVL);
            if (x >= 0) { have = 1; v = it_v[x]; }
        }
        if (!have) v = 0;
        raw[s] = v > 2147483647 ? 2147483647 : (i32)v;
    }
}
#endif
