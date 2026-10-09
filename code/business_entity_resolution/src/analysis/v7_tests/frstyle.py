"""French-style rewrite used by E7/E14 (general French conventions only), extracted verbatim for v6.1."""
import re

import numpy as np

LEGAL = [(r"\b(llc|l\.l\.c\.)\b", ["SARL", "SAS", "EURL"]), (r"\b(inc|inc\.|incorporated)\b", ["SA", "SAS", "SASU"]),
         (r"\b(corp|corp\.|corporation)\b", ["SA", "SAS"]), (r"\b(ltd|ltd\.|limited)\b", ["SARL", "SA"]),
         (r"\b(co|co\.|company)\b", ["Cie", "Compagnie"])]
WORDS = {"services": "Services", "group": "Groupe", "associates": "Associés", "center": "Centre", "centre": "Centre",
         "school": "École", "sons": "Fils", "friends": "Amis", "health": "Santé", "development": "Développement",
         "management": "Gestion", "consulting": "Conseil", "solutions": "Solutions", "holdings": "Participations",
         "partners": "Partenaires", "international": "International", "medical": "Médical", "dental": "Dentaire"}
STREET = {"street": "Rue", "st": "Rue", "avenue": "Avenue", "ave": "Avenue", "boulevard": "Boulevard", "blvd": "Bd",
          "road": "Chemin", "rd": "Chemin", "drive": "Allée", "dr": "Allée", "lane": "Impasse", "ln": "Impasse",
          "place": "Place", "pl": "Place", "court": "Cour", "ct": "Cour", "way": "Voie"}
ACC = str.maketrans({"e": "é", "a": "à", "c": "ç", "o": "ô"})


def fr_name(n, seed):
    g = np.random.default_rng(seed)
    for pat, reps in LEGAL:
        n = re.sub(pat, lambda m: reps[g.integers(len(reps))], n, flags=re.I)
    toks = [WORDS.get(t.lower(), t) for t in n.split()]
    if g.random() < 0.15 and toks:  # occasional accent on one word (French text is accented)
        i = int(g.integers(len(toks))); toks[i] = toks[i][:1] + toks[i][1:].translate(ACC) if len(toks[i]) > 3 else toks[i]
    return " ".join(toks)


def fr_addr(a, seed):
    if not a.strip():
        return a
    g = np.random.default_rng(seed)
    parts = [x.strip() for x in a.split(",")]
    first = parts[0].split()
    if len(first) >= 3 and first[0].isdigit() and first[-1].lower().strip(".") in STREET:  # "12 Main Street" -> "12 Rue Main"
        num, typ, name = first[0], STREET[first[-1].lower().strip(".")], " ".join(first[1:-1])
        if g.random() < 0.08:
            num += g.choice([" bis", " ter"])
        parts[0] = f"{num} {typ} {name}"
    if len(parts) >= 3 and g.random() < 0.5:  # city/region order varies in French data
        parts[1], parts[2] = parts[2], parts[1]
    return ", ".join(parts)
