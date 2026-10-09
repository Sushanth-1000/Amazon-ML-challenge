"""Text normalization for business names and addresses (US / India / France / any Latin-script country).

Every record becomes a handful of clean string fields that blocking and features consume:
  name     - core business name: transliterated, lowercased, junk/legal suffixes/honorifics removed
  name_ns  - same without spaces (robust to word splits, website-style names)
  addr     - normalized address tokens (street-type, state and city-alias forms unified)
  nums     - space-joined digit groups found in the address (house / plot / PIN numbers)
  nonlatin - original name contained non-Latin script (Devanagari, Kannada, ...)
  name_sk  - consonant skeleton of the name: makes "शिवम सिस्टम्स" (anyascii "shivm sistms")
             comparable with "Shivam Systems" (both -> "sts"-like skeletons)
  suffix   - legal-form class of the name ("llc", "pvt ltd", "sarl", ...), for agree/conflict features
  tail     - normalized last two comma-separated address parts (usually city / district / state)
  pcode    - postcode: 6-digit PIN or 5-digit ZIP / code postal, if present

v3 changes: anyascii (ISC) replaces unidecode (GPL); phone numbers and e-mails are stripped; legal words
written in Indian scripts are removed; Indian-script state names map to state codes; common Indian city
aliases are unified; digit-for-letter OCR swaps are repaired; legal form kept as its own field.
"""
import re

from anyascii import anyascii

# legal-form token -> class (the class is kept in `suffix`, the token is removed from `name`)
LEGAL_CLASS = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llc": "llc", "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "plc": "plc", "llp": "llp",
    "lp": "lp", "pllc": "pllc", "pc": "pc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa",
    "eurl": "eurl", "sci": "sci", "snc": "snc",
    # Indian-script spellings after anyascii ("प्राइवेट लिमिटेड", "பிரைவேட் லிமிடெட்", "എൽഎൽപി", ...)
    "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "praivrr": "pvt", "praivett": "pvt",
    "limitet": "ltd", "limirrd": "ltd", "limitedd": "ltd", "limited": "ltd", "limittedd": "ltd",
    "elelpi": "llp", "elelsi": "llc",
}
LEGAL = set(LEGAL_CLASS)
HONORIFIC = set("the dr mr mrs ms sri shri smt".split())
TRADE_NAME = re.compile(r"\b(?:dba|d b a|formerly|fka|f k a|aka|a k a|t a|trading as)\b:?")
DOMAIN = re.compile(r"^(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|co|fr|us|biz|info)$")
EMAIL = re.compile(r"\S+@\S+\.\S+")
PHONE = re.compile(r"\+?\d[\d\s\-()]{8,}\d")
OCR = str.maketrans({"0": "o", "3": "e", "5": "s", "8": "b"})
OCR_TOKEN = re.compile(r"^(?=(?:.*[a-z]){2})(?=.*[0358])[a-z0358]+$")  # letters + a few look-alike digits
WORD_FIX = {"lndia": "india", "indla": "india"}

# Address tokens mapped to one canonical short form. Street types for US, India and France.
ADDR_ABBR = {
    "road": "rd", "street": "st", "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd",
    "bld": "blvd", "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir", "place": "pl",
    "highway": "hwy", "parkway": "pkwy", "terrace": "ter", "trail": "trl", "square": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt", "saint": "st",
    "rue": "r", "chemin": "ch", "allee": "all", "impasse": "imp", "route": "rte", "quai": "q",
    "near": "nr", "opposite": "opp", "sector": "sec", "building": "bldg", "cross": "crs",
    "main": "mn", "nagar": "ngr", "colony": "col", "first": "1", "second": "2", "third": "3",
    # Indian city aliases (old / new / local names) -> one form
    "bombay": "mumbai", "calcutta": "kolkata", "koklata": "kolkata", "madras": "chennai",
    "bengaluru": "bangalore", "gurugram": "gurgaon", "trivandrum": "thiruvananthapuram",
    "poona": "pune", "baroda": "vadodara", "mysuru": "mysore", "cochin": "kochi", "calicut": "kozhikode",
    "allahabad": "prayagraj", "benares": "varanasi", "banaras": "varanasi", "keralam": "kerala",
}
# Tokens that carry no identity (unit / house-number prefixes that sources add or drop at random).
ADDR_DROP = set("no h hno door unit apt apartment ste suite fl floor po box cedex".split())

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "keralam": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "tamilnadu": "tn", "telangana": "ts", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "west bengal": "wb", "paschim bangal": "wb", "paschimbanga": "wb", "paschim banga": "wb",
    "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk", "chandigarh": "ch",
    "puducherry": "py", "pondicherry": "py",
}
# Longest names first so "west virginia" wins over "virginia".
_STATE_RE = {
    c: re.compile(r"\b(" + "|".join(sorted(map(re.escape, d), key=len, reverse=True)) + r")\b")
    for c, d in (("us", US_STATES), ("india", IN_STATES))
}
_NONLATIN = re.compile(r"[^\x00-ɏ]")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")
_ORDINAL = re.compile(r"^0*(\d+)(?:st|nd|rd|th)?$")  # "0018" -> "18", "131th" -> "131"
_PCODE = re.compile(r"\b(\d{6}|\d{5})\b")

# Skeleton-level legal words (English + transliterated Indian-script "प्राइवेट लिमिटेड", "प्रा. लि.", "एलएलपी").
SKEL_LEGAL = {"prvt", "prvr", "pvt", "pr", "prbt", "lmtd", "lmt", "lmrd", "ltd", "lm", "l", "llp", "lp", "lc", "ll",
              "lt", "ln", "lnrd", "lnr", "kpn", "knpn", "krprtn", "nkrprtd",
              "ink", "krp", "kmpn", "k"}


def skeleton(s):
    s = anyascii(s).lower().split("|")[0].replace("ph", "f").replace("&", " ")
    s = re.sub(r"([bcdgjklpst])h", r"\1", s)  # aspirates bh/dh/th/kh -> b/d/t/k
    s = re.sub(r"c(?=[eiy])", "s", s)  # soft c: "finance" -> "finanse" (Hindi writes it with s)
    s = s.replace("q", "k").replace("x", "ks").replace("c", "k").replace("w", "v").replace("z", "j")
    s = re.sub(r"[aeiouy]", "", s)  # vowels vary most across transliterations
    s = re.sub(r"[^a-z ]", " ", s).replace("m", "n")  # anusvara is written m or n ("investmemt", "bmgal")
    s = re.sub(r"(.)\1+", r"\1", s)
    s = re.sub(r"n(?=[bdgjkpt])", "", s)  # nasal before a stop is spelled inconsistently
    s = re.sub(r"(.)\1+", r"\1", s)
    toks = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in s.split()]
    return " ".join(t for t in toks if t not in SKEL_LEGAL)


# state names in skeleton space, so "महाराष्ट्र" (anyascii "mharastr") finds "maharashtra" -> "mh"
_IN_STATE_SK = {skeleton(k): v for k, v in IN_STATES.items() if len(skeleton(k).replace(" ", "")) >= 3}


def _clean_name_tokens(s):
    """Tokens of a lowercased ASCII name with OCR swaps repaired; returns (core tokens, legal classes)."""
    toks, legal = [], set()
    for t in _NONALNUM.sub(" ", s).split():
        if OCR_TOKEN.match(t):
            t = t.translate(OCR)
        t = WORD_FIX.get(t, t)
        if t in LEGAL:
            legal.add(LEGAL_CLASS[t])
        elif t not in HONORIFIC and t != "and":
            toks.append(t)
    if "pra" in toks and "li" in toks:  # "प्रा. लि." -> "pra li" (abbreviated Private Limited)
        toks = [t for t in toks if t not in ("pra", "li")]
        legal |= {"pvt", "ltd"}
    return toks, legal


def name_fields(s):
    """(core name, legal class string)"""
    s = anyascii(s).lower()
    s = s.split("|")[0].strip() or s  # "X LLC | www.x.com" -> "X LLC"
    s = PHONE.sub(" ", EMAIL.sub(" ", s)).strip() or s
    m = DOMAIN.match(s.replace(" ", ""))
    if m:  # "lifeinvestments.com" -> "lifeinvestments"
        return m.group(1).replace("-", ""), ""
    parts = TRADE_NAME.split(s)
    if len(parts) > 1 and parts[-1].strip():  # "Avikor DBA: Life Investments" -> "life investments"
        s = parts[-1]
    s = s.replace("&", " and ").replace(".", "")  # "l.l.c." -> "llc", "pvt." -> "pvt"
    toks, legal = _clean_name_tokens(s)
    return " ".join(toks), " ".join(sorted(legal))


def norm_name(s):
    return name_fields(s)[0]


def _native_states(toks):
    """Map 1-2 token Indian-script state names (skeleton match) to state codes."""
    out, i = [], 0
    while i < len(toks):
        for n in (2, 1):
            if i + n <= len(toks):
                code = _IN_STATE_SK.get(skeleton(" ".join(toks[i:i + n])))
                if code:
                    out.append(code)
                    i += n
                    break
        else:
            out.append(toks[i])
            i += 1
    return out


def norm_addr(s, country, native=None):
    native = bool(_NONLATIN.search(s)) if native is None else native
    s = PHONE.sub(" ", anyascii(s).lower())
    c = country.lower()
    rx = _STATE_RE.get(c)
    if rx is not None:
        d = US_STATES if c == "us" else IN_STATES
        s = rx.sub(lambda m: d[m.group(1)], s)
    toks = _NONALNUM.sub(" ", s).split()
    if native and c == "india":
        toks = _native_states(toks)
    toks = [ADDR_ABBR.get(t, t) for t in toks]
    toks = [m.group(1) if (m := _ORDINAL.match(t)) else t for t in toks]
    return " ".join(t for t in toks if t not in ADDR_DROP)


def tail_of(addr, country):
    parts = [p for p in str(addr).split(",") if p.strip()]
    return norm_addr(",".join(parts[-2:]), country) if parts else ""


def normalize_record(name, addr, country):
    n, suffix = name_fields(name)
    a = norm_addr(addr, country)
    pc = _PCODE.findall(PHONE.sub(" ", anyascii(addr)))
    return (n, n.replace(" ", ""), a, " ".join(_DIGITS.findall(a)), bool(_NONLATIN.search(name)), skeleton(name),
            suffix, tail_of(addr, country), pc[-1] if pc else "")


NORM_COLS = ["name", "name_ns", "addr", "nums", "nonlatin", "name_sk", "suffix", "tail", "pcode"]


if __name__ == "__main__":
    assert norm_name("*** Blue Hypnósis") == "blue hypnosis"
    assert norm_name("Espinoza & Davis Industrials L.L.C.") == "espinoza davis industrials"
    assert name_fields("Espinoza & Davis Industrials L.L.C.")[1] == "llc"
    assert norm_name("Avikor DBA: Life Investments") == "life investments"
    assert norm_name("lifeinvestments.com") == "lifeinvestments"
    assert norm_name("Mr Jay Finance Ltd") == "jay finance"
    assert norm_name("@estrada - 8200465095") == "estrada"
    assert norm_name("Sharma 5ervices 8rothers lndia") == "sharma services brothers india"
    assert name_fields("ग्लोबल इन्वेस्टमेंट प्रा. लि.") == ("globl investmemt", "ltd pvt")
    assert name_fields("स्मार्ट बिल्डर्स प्राइवेट लिमिटेड") == ("smart bildrs", "ltd pvt")
    assert norm_addr("2 Muldowney Cir, # Apartment B, Poughkeepsie, New York", "US") == "2 muldowney cir b poughkeepsie ny"
    assert norm_addr("B-20, Regency Plaza, Kalyan, Thane, Maharashtra", "India") == "b 20 regency plaza kalyan thane mh"
    assert norm_addr("PLOT NO. ##74, M.P. NAGAR, BHOPAL, मध्य प्रदेश", "India") == "plot 74 m p ngr bhopal mp"
    assert norm_addr("DOOR NO 21, NASHIK, महाराष्ट्र", "India") == "21 nashik mh"
    assert norm_addr("11B Dr Rajendra Road, Calcutta, West Bengal", "India") == "11b dr rajendra rd kolkata wb"
    assert norm_addr("20 BD Jules Verne, Nantes", "France") == "20 blvd jules verne nantes"
    assert norm_addr("5 r de l'Église, 75001 Paris Cedex", "France") == "5 r de l eglise 75001 paris"
    assert normalize_record("लाइफ इन्वेस्टमेंट्स", "Office No 212, Thane", "India")[3:5] == ("212", True)
    assert normalize_record("X", "12 Main St, Austin, TX 78701", "US")[7:] == ("austin tx 78701", "78701")
    assert norm_name("Eye Center L.L.C. | www.eyecenter.com") == "eye center"
    assert norm_addr("001823 P St, 13 131th St", "US") == "1823 p st 13 131 st"
    assert skeleton("स्मार्ट बिल्डर्स प्राइवेट लिमिटेड") == skeleton("Smart Builders Private Limited") == "snrt bldr"
    assert skeleton("Apex Finance LLP") == skeleton("एपेक्स फाइनेंस एलएलपी") == "pks fns"
    print("normalize ok")
