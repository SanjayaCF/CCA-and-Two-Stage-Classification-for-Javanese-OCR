"""
Javanese Unicode composition and Latin transliteration.
Adapted to match the actual Stage 2 model class labels.
"""

# Base character → Javanese Unicode codepoint
BASE_UNICODE = {
    "ha": "ꦲ", "na": "ꦤ", "ca": "ꦕ", "ra": "ꦫ",
    "ka": "ꦏ", "da": "ꦢ", "ta": "ꦠ",
    "sa": "ꦱ", "wa": "ꦮ", "la": "ꦭ",
    "pa": "ꦥ", "dha": "ꦝ", "ja": "ꦗ", "ya": "ꦪ",
    "nya": "ꦚ", "ma": "ꦩ", "ga": "ꦒ", "ba": "ꦧ",
    "nga": "ꦔ",
    # Pasangan forms classified in the base position
    "_ha": "꧀ꦲ", "_pa": "꧀ꦥ", "_sa": "꧀ꦱ",
    # Punctuation / lone marks that can be anchors
    "pada_lingsa": "꧈", "pada_lungsi": "꧉",
    "taling": "ꦺ", "tarung": "ꦴ", "wignyan": "ꦃ",
}

BASE_LATIN = {
    "ha": "ha", "na": "na", "ca": "ca", "ra": "ra", "ka": "ka",
    "da": "da", "ta": "ta", "sa": "sa", "wa": "wa", "la": "la",
    "pa": "pa", "dha": "dha", "ja": "ja", "ya": "ya", "nya": "nya",
    "ma": "ma", "ga": "ga", "ba": "ba", "nga": "nga",
    "_ha": "ha", "_pa": "pa", "_sa": "sa",
    "pada_lingsa": ",", "pada_lungsi": ".",
    "taling": "e", "tarung": "ā", "wignyan": "h",
}

# Sandhangan (diacritical marks) → Unicode
SANDH_ABOVE = {
    "wulu":       "ꦶ",   # i
    "wulu_cecak": "ꦷ",   # î
    "wulu_layar": "ꦶꦂ",  # i + final -r
    "pepet":      "ꦼ",   # ê
    "pepet_cecak":"ꦼꦁ",  # ê + final -ng
    "pepet_layar":"ꦼꦂ",  # ê + final -r
    "cecak":      "ꦁ",   # final -ng
    "layar":      "ꦂ",   # final -r
}
SANDH_BELOW = {
    "suku":    "ꦸ",    # u
    "pengkal": "ꦾ",    # consonant sign ya (conjunct)
    "_ba":  "꧀ꦧ", "_ca":  "꧀ꦕ", "_da":  "꧀ꦢ",
    "_dha": "꧀ꦝ", "_ja":  "꧀ꦗ", "_ka":  "꧀ꦏ",
    "_la":  "꧀ꦭ", "_na":  "꧀ꦤ", "_nga": "꧀ꦔ",
    "_ta":  "꧀ꦠ", "_wa":  "꧀ꦮ", "_ya":  "꧀ꦪ",
}
SANDH_BESIDE = {
    "pangkon": "꧀",             # virama (dead consonant)
}


def compose(base, above=None, below=None, beside=None, wrapped=None):
    """Return (unicode_str, latin_str) or (None, None) if base is unrecognized."""
    uni = BASE_UNICODE.get(base)
    if uni is None:
        return None, None

    # Punctuation / isolated marks — no sandhangan
    if base in ("pada_lingsa", "pada_lungsi", "taling", "tarung", "wignyan"):
        return uni, BASE_LATIN[base]

    if above  in SANDH_ABOVE:  uni += SANDH_ABOVE[above]
    if below  in SANDH_BELOW:  uni += SANDH_BELOW[below]
    if beside in SANDH_BESIDE: uni += SANDH_BESIDE[beside]

    # Latin: strip inherent vowel "a" to get bare consonant
    # e.g. "ka" → "k",  "nga" → "ng",  "_ha" → "h"
    raw = base.lstrip("_")
    consonant = raw[:-1] if raw.endswith("a") else raw

    if beside == "pangkon":          vowel = ""
    elif above == "wulu":            vowel = "i"
    elif above == "wulu_cecak":      vowel = "î"
    elif above == "wulu_layar":      vowel = "i"
    elif above == "pepet":           vowel = "ê"
    elif above == "pepet_cecak":     vowel = "ê"
    elif above == "pepet_layar":     vowel = "ê"
    elif below == "suku":            vowel = "u"
    else:                            vowel = "a"

    suffix = ""
    if above == "cecak":             suffix = "ng"
    elif above == "pepet_cecak":     suffix = "ng"
    elif above == "layar":           suffix = "r"
    elif above == "wulu_layar":      suffix = "r"
    elif above == "pepet_layar":     suffix = "r"

    # Pasangan conjunct (below position)
    if below == "pengkal":
        return uni, consonant + vowel + "y" + suffix
    if below and below.startswith("_"):
        pas_raw  = below[1:]
        pas_cons = pas_raw[:-1] if pas_raw.endswith("a") else pas_raw
        return uni, consonant + vowel + pas_cons + suffix

    return uni, consonant + vowel + suffix


def _kill_vowel(latin_str):
    """Strip the inherent vowel from a Latin syllable before a pasangan conjunct."""
    if latin_str.endswith("ang"):
        return latin_str[:-3] + "ng"
    if latin_str.endswith("ar"):
        return latin_str[:-2] + "r"
    for v in ("a", "i", "u", "ê", "î", "ā", "o", "e"):
        if latin_str.endswith(v):
            return latin_str[:-len(v)]
    return latin_str


def _apply_taling_vowel(latin_str, new_vowel):
    """Replace the inherent 'a' vowel in a Latin syllable with new_vowel.
    Handles common suffixes: 'ng' (cecak) and 'r' (layar)."""
    if latin_str.endswith("ang"):
        return latin_str[:-3] + new_vowel + "ng"
    if latin_str.endswith("ar"):
        return latin_str[:-2] + new_vowel + "r"
    if latin_str.endswith("a"):
        return latin_str[:-1] + new_vowel
    return latin_str  # already has a non-'a' vowel; leave unchanged


def compose_line(compositions):
    """
    Apply cross-character Javanese composition rules to a sequence of
    compose_crop() result dicts.

    Rules:
      - taling before next aksara  → Unicode reorder (aksara + ꦺ); Latin vowel 'e'
      - taling + aksara + tarung   → aksara + ꦺ + ꦴ; Latin vowel 'o'
      - standalone tarung          → append ꦴ to previous; Latin 'a' → 'ā'
      - standalone wignyan         → append ꦃ to previous; append 'h' to previous Latin

    Returns (unicode_str, latin_str) for the whole line.
    """
    if not compositions:
        return "", ""

    tokens = []
    for c in compositions:
        if c is None:
            continue
        tokens.append([c.get("unicode") or "", c.get("latin") or ""])

    result = []
    i = 0
    while i < len(tokens):
        u, l = tokens[i]

        if u == "ꦺ":  # taling — look ahead
            if i + 1 < len(tokens):
                next_u, next_l = tokens[i + 1]
                if i + 2 < len(tokens) and tokens[i + 2][0] == "ꦴ":
                    # taling + aksara + tarung → 'o'
                    result.append([next_u + "ꦺ" + "ꦴ", _apply_taling_vowel(next_l, "o")])
                    i += 3
                elif (i + 2 < len(tokens)
                      and tokens[i + 2][0].startswith("꧀")
                      and len(tokens[i + 2][0]) > 1):
                    # taling + aksara + pasangan_as_base → taling belongs to the pasangan
                    # e.g. taling + pa + _pa → dead-pa + pe  (not pe + dead-pa)
                    pas_u, pas_l = tokens[i + 2]
                    result.append([next_u, _kill_vowel(next_l)])
                    result.append([pas_u + "ꦺ", _apply_taling_vowel(pas_l, "e")])
                    i += 3
                else:
                    # taling + aksara → 'e'
                    result.append([next_u + "ꦺ", _apply_taling_vowel(next_l, "e")])
                    i += 2
            else:
                result.append([u, l])
                i += 1

        elif u == "ꦴ":  # standalone tarung → attach to previous
            if result:
                result[-1][0] += "ꦴ"
                result[-1][1] = _apply_taling_vowel(result[-1][1], "ā")
            else:
                result.append([u, l])
            i += 1

        elif u == "ꦃ":  # standalone wignyan → attach to previous
            if result:
                result[-1][0] += "ꦃ"
                result[-1][1] += "h"
            else:
                result.append([u, l])
            i += 1

        elif u.startswith("꧀") and len(u) > 1:  # pasangan as base (_ha/_pa/_sa)
            # Kill the trailing vowel of the previous syllable in Latin
            if result:
                result[-1][1] = _kill_vowel(result[-1][1])
            result.append([u, l])
            i += 1

        else:
            result.append([u, l])
            i += 1

    return (
        "".join(t[0] for t in result),
        "-".join(t[1] for t in result),
        [t[0] for t in result],   # per-token unicode list for span rendering
    )


def compose_crop(s2_result):
    """
    Given a stage2.predict_crop() result dict, return a composition dict.
    Keys: unicode, latin, labels{base, above, below, beside, wrapped}
    """
    if not s2_result:
        return None

    def lbl(pos):
        pred = s2_result.get(pos)
        return pred["label"] if pred else None

    base    = lbl("base")
    above   = lbl("above")
    below   = lbl("below")
    beside  = lbl("beside")
    wrapped = lbl("wrapped")

    uni, latin = compose(base, above, below, beside, wrapped)
    return {
        "unicode": uni,
        "latin":   latin,
        "labels": {
            "base": base, "above": above, "below": below,
            "beside": beside, "wrapped": wrapped,
        },
    }
