"""Pair-type helpers (shared by the VM analysis scripts)."""
import re
from difflib import SequenceMatcher
import france_lookalike_filter as F

NUM = re.compile(r'^0*(\d+)[a-z]*$')


def num(addr):
    t = re.sub(r'[^a-z0-9 ]', ' ', F.fold(addr)).split()
    x = next((x for x in t if x[:1].isdigit()), '')
    m = NUM.match(x)
    return m.group(1) if m else x


def close(a, b):
    return SequenceMatcher(None, a, b).ratio() >= 0.6 or a[:3] == b[:3] or F.is_abbrev(a, b)


def ptype(sn, sa, rn, ra):
    ta, tb = set(F.toks(sn)), set(F.toks(rn))
    wa, wb = ta - tb, tb - ta
    if not wa and not wb:
        nm = 'exact'
    elif not (ta & tb):
        nm = 'no shared word'
    elif all(any(close(a, b) for b in wb) for a in wa) and all(any(close(a, b) for a in wa) for b in wb):
        nm = 'typo only'
    elif not wa:
        nm = 'record adds word(s)'
    elif not wb:
        nm = 'record drops word(s)'
    elif len(wa) == 1 and len(wb) == 1:
        nm = 'one word swapped'
    else:
        nm = 'several words differ'
    if ra == '':
        ad = 'no address'
    else:
        na, nb = num(sa), num(ra)
        street = bool(F.addr_key(sa)[1] & F.addr_key(ra)[1])
        ad = ('no number' if not (na and nb) else 'same number' if na == nb else 'diff number') + (' +street' if street else ' -street')
    return nm, ad


