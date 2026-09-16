# Turkish Learning Assistant

A rule-system approach to Turkish. Nothing here is a word list to memorize —
every surface form is derivable as `root + rule(s) applied → form`.

## Repository layout

| File | Purpose |
| --- | --- |
| `CLAUDE.md` | Tutor operating instructions (session structure, verification protocol) |
| `progress.md` | Running state: harmony mastered, suffix slots covered, root inventory, exceptions |
| `README.md` | Rule reference (this file) |

## Notation

Archiphonemes — capital letters stand for a slot whose realization the rules fix:

| Symbol | Realizations | Selected by |
| --- | --- | --- |
| `A` | a, e | 2-way vowel harmony |
| `I` | ı, i, u, ü | 4-way vowel harmony |
| `D` | d, t | voicing assimilation |
| `C` | c, ç | voicing assimilation |
| `K` | k, ğ | intervocalic lenition |
| `(y)`, `(n)`, `(s)` | present or null | buffer consonant, inserted only to break V-V hiatus |

So the locative is one suffix, `-DA`, with four surface forms — not four things to learn.

## Rule 1 — Vowel harmony

Look at the **last vowel of the stem** (including a preceding suffix's vowel —
harmony is iterative, it re-applies at every slot boundary).

**2-way (low, `A`):**

| Last stem vowel | `A` → |
| --- | --- |
| a, ı, o, u | **a** |
| e, i, ö, ü | **e** |

**4-way (high, `I`):**

| Last stem vowel | `I` → |
| --- | --- |
| a, ı | **ı** |
| e, i | **i** |
| o, u | **u** |
| ö, ü | **ü** |

Driving features: `A` tracks backness only; `I` tracks backness **and** rounding.

## Rule 2 — Consonant assimilation (suffix-initial)

If the stem ends in a voiceless consonant — **p, ç, t, k, s, ş, h, f**
(mnemonic set *fıstıkçı şahap*) — a suffix-initial `D` → t and `C` → ç.
Otherwise `D` → d, `C` → c.

## Rule 3 — Stem-final lenition (suffix-initial vowel)

A vowel-initial suffix softens a stem-final voiceless stop:

`k → ğ`, `p → b`, `t → d`, `ç → c`   (kitap + `-I` → kitab**ı**)

Conditions: applies to most polysyllabic stems. Monosyllabic stems mostly
**block** it (top → topu, at → atı, saç → saçı) — that block is the regular
case, not an exception. Genuine exceptions are logged in `progress.md`.

Also: `nk → ng` (renk → rengi).

## Rule 4 — Buffer consonants

Turkish does not allow two vowels to collide across a morpheme boundary.

- `-y-` between stem vowel and suffix vowel: araba + `-A` → araba**y**a
- `-n-` after a 3rd-person possessive or a pronominal: ev-i + `-DA` → evi**n**de
- `-s-` in the 3sg possessive after a vowel: araba + `-I` → araba**s**ı

## Slot template (noun)

```
ROOT - DERIV - PLURAL - POSSESSIVE - CASE
                -lAr     -(I)m etc.   -DA etc.
```

Order is fixed. `ev-ler-im-de` = house-PL-1SG.POSS-LOC = "in my houses".

## Verification protocol

A rule-generated form is a hypothesis until checked against real usage.
Search the quoted form; if it is well-formed but unattested, it is flagged
unidiomatic, not wrong. If it is wrong, the diagnosis names **which rule**
misapplied — harmony, assimilation, lenition, or a lexical exception.
