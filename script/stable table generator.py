"""The stability screen behind Table S3.

Each metal of `data/M-N-C data set.xlsx` is screened on three grounds, and the
two indicators of the screen are reported beside them:

  stability   the descriptors of the data set: E_b/eV, E_f/eV and the two
              dissolution potentials U_diss_acid/V and U_diss_base/V;
  price       the USGS Mineral Commodity Summaries 2026, read through
              `pdftotext`: the 2025 annual average in USD per kg, of the form
              and grade the chapter quotes;
  scale-up    the same Summaries: the world production of 2025 and the world
              reserves, in tonnes;
  supply risk mendeleev: the share of world production held by the top
              producer, and the political stability of that producer;
  toxicity    PubChem PUG-View, heading "GHS Classification", which aggregates
              the ECHA C&L Inventory; the health hazard classes of the elemental
              substance are counted.

The USGS chapters are cached under `data/usgs-mcs` and the mendeleev/PubChem
answers in `supply risk and toxicity cache.json`, so that a re-run is offline.

Writes `stable table.csv` beside this script and `data/stable M-N4-C report.txt`.

Usage

    python "stable table generator.py"                 uses the caches
    python "stable table generator.py" --refresh-usgs  re-reads the chapters
    python "stable table generator.py" --probe         lists the price rows
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import warnings
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

import pandas as pd
import requests
from mendeleev import element

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

DATASET = os.path.join(ROOT, "data/M-N-C data set.xlsx")
STABLE_CSV = os.path.join(SCRIPT_DIR, "stable table.csv")
REPORT = os.path.join(ROOT, "data/stable M-N4-C report.txt")

# the USGS chapters are cached here
CACHE_DIR = os.path.join(ROOT, "data/usgs-mcs")
EDITION = 2026
BASE = "https://pubs.usgs.gov/periodicals/mcs%d" % EDITION
os.makedirs(CACHE_DIR, exist_ok=True)

# the mendeleev and PubChem answers are cached here
CACHE_PATH = os.path.join(SCRIPT_DIR, "supply risk and toxicity cache.json")

# a metal above this price is treated as expensive
THRESHOLD = 1000.0

REFRESH_USGS = "--refresh-usgs" in sys.argv


# metal -> the MCS chapter that quotes it.  Several metals share a chapter.
CHAPTER = {
    "Li": "lithium", "Na": "salt", "K": "potash", "Rb": "rubidium",
    "Cs": "cesium", "Be": "beryllium", "Mg": "magnesium-metal", "Ca": "lime",
    "Sr": "strontium", "Ba": "barite", "Sc": "scandium", "Ti": "titanium",
    "V": "vanadium", "Cr": "chromium", "Mn": "manganese", "Fe": "iron-ore",
    "Co": "cobalt", "Ni": "nickel", "Cu": "copper", "Zn": "zinc",
    "Y": "yttrium", "Zr": "zirconium-hafnium", "Nb": "niobium",
    "Mo": "molybdenum", "Ru": "platinum-group", "Rh": "platinum-group",
    "Pd": "platinum-group", "Ag": "silver", "Cd": "cadmium",
    "Hf": "zirconium-hafnium", "Ta": "tantalum", "W": "tungsten",
    "Re": "rhenium", "Os": "platinum-group", "Ir": "platinum-group",
    "Pt": "platinum-group", "Au": "gold", "Hg": "mercury", "Al": "aluminum",
    "Ga": "gallium", "Ge": "germanium", "In": "indium", "Sn": "tin",
    "Sb": "antimony", "Tl": "thallium", "Pb": "lead", "Bi": "bismuth",
}
for _m in ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
           "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]:
    CHAPTER[_m] = "rare-earths"

# On a chapter that quotes several commodities at once, the row to read.  Where
# a metal has no entry the chapter quotes a single price and that row is taken;
# a metal whose pattern does not match is reported as having no price.
ROW_HINT = {
    # the rare earths are quoted as oxides, and only seven of the fourteen; the
    # other seven match nothing, so that they are not given a neighbour's price
    "La": r"lanthanum", "Ce": r"cerium", "Pr": r"praseodymium",
    "Nd": r"neodymium oxide", "Sm": r"samarium", "Eu": r"europium",
    "Gd": r"gadolinium",
    "Tb": r"terbium oxide", "Dy": r"dysprosium oxide", "Ho": r"holmium oxide",
    "Er": r"erbium oxide", "Tm": r"thulium oxide", "Yb": r"ytterbium oxide",
    "Lu": r"lutetium oxide",
    # the platinum group, quoted metal by metal
    "Ru": r"^\s*ruthenium", "Rh": r"^\s*rhodium", "Pd": r"^\s*palladium",
    "Os": r"^\s*osmium", "Ir": r"^\s*iridium", "Pt": r"^\s*platinum",
    # chapters with more than one row, of which one is the metal
    "Mg": r"u\.s\. spot",              # the European row is a different market
    "Ti": r"dollars per kilogram",     # sponge, not the metric ton row
    "Zr": r"zirconium, sponge",
    "Hf": r"hafnium",
    "Cr": r"chromium metal",           # not the chromite ore or the ferroalloy
    "Sc": r"scandium metal",           # not the oxide or the master alloy
    "Y": r"yttrium metal",
    "Ni": r"dollars per metric ton",   # the pound row quotes the same price
}

# The form and grade the price refers to.  The USGS quotes a commodity, not an
# element: the alkali and alkaline earth metals as their compounds, iron as
# ore, molybdenum as its oxide, and the rare earths as oxides, so the price is
# a price of that commodity.
USGS_FORM = {
    "Li": "battery-grade lithium carbonate",
    "Na": "salt, vacuum and open pan",
    "K": "potash, all products",
    "Be": "beryllium-copper master alloy",
    "Mg": "magnesium metal, U.S. spot Western",
    "Ca": "quicklime",
    "Sr": "celestite, imports",
    "Ba": "ground barite, ex-works",
    "Sc": "scandium metal, ingot",
    "Ti": "titanium sponge",
    "V": "vanadium pentoxide",
    "Cr": "chromium metal, gross weight",
    "Mn": "manganese content, CIF",
    "Fe": "iron ore",
    "Co": "cobalt cathode, U.S. spot",
    "Ni": "nickel metal, LME cash",
    "Cu": "copper cathode, COMEX plus premium",
    "Zn": "zinc metal, North American",
    "Y": "yttrium metal",
    "Zr": "zirconium sponge, ex-works China",
    "Nb": "ferroniobium",
    "Mo": "molybdic oxide (MoO3), 57% molybdenum",
    "Ru": "ruthenium metal",
    "Rh": "rhodium metal",
    "Pd": "palladium metal",
    "Ag": "silver bullion",
    "Cd": "cadmium metal",
    "Hf": "hafnium, unwrought",
    "Ta": "tantalite, tantalum oxide content",
    "W": "tungsten concentrate, Rotterdam",
    "Re": "rhenium metal",
    "Ir": "iridium metal",
    "Pt": "platinum metal",
    "Au": "gold bullion",
    "Hg": "mercury, imports",
    "Al": "aluminum ingot, U.S. market spot",
    "Ga": "gallium, imports",
    "Ge": "germanium metal",
    "In": "indium, U.S. warehouse free on board",
    "Sn": "tin metal, New York dealer",
    "Sb": "antimony metal",
    "Tl": "thallium metal",
    "Pb": "lead metal, North American",
    "Bi": "bismuth metal",
    "La": "lanthanum oxide",
    "Ce": "cerium oxide",
    "Pr": "praseodymium oxide",
    "Nd": "neodymium oxide",
    "Sm": "samarium oxide",
    "Eu": "europium oxide",
    "Gd": "gadolinium oxide",
}


# ---------------------------------------------------------------------------
# the price, from the layout text of the chapters
# ---------------------------------------------------------------------------
PRICE_HEADER = re.compile(r"^\s*Price\b")
PRICE_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# the price table ends at the first of these
TABLE_END = re.compile(
    r"^\s*(Employment|Net import|Recycling|Import Sources|Tariff|Depletion"
    r"|Prepared by|Government Stockpile|Events, Trends|World|Substitutes"
    r"|Salient|Consumer|Stocks|LME U\.S\.)", re.I)

UNITS = [
    ("dollars per kilogram", 1.0),
    ("dollars per troy ounce", 1000.0 / 31.1034768),
    # a metric ton unit is 1% of a tonne, so a price per unit is a price per
    # 10 kg of contained metal
    ("dollars per dry metric ton unit", 1.0 / 10.0),
    ("dollars per metric ton unit", 1.0 / 10.0),
    ("dollars per metric ton", 1.0 / 1000.0),
    ("dollars per ton", 1.0 / 1000.0),
    ("cents per pound", 0.01 / 0.45359237),
    ("dollars per pound", 1.0 / 0.45359237),
]


def ChapterText(chapter):
    """Download a chapter once and return its layout text."""
    path = os.path.join(CACHE_DIR, "mcs%d-%s.pdf" % (EDITION, chapter))
    if REFRESH_USGS or not os.path.exists(path):
        url = "%s/mcs%d-%s.pdf" % (BASE, EDITION, chapter)
        # the USGS server rejects requests without a User-Agent
        request = urllib.request.Request(
            url, headers={"User-Agent": "materials-stability-screen/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, \
                open(path, "wb") as f:
            f.write(response.read())
        time.sleep(0.3)
    return subprocess.run(["pdftotext", "-q", "-layout", path, "-"],
                          capture_output=True, text=True).stdout


def PriceUnit(fragment):
    """The unit a price is quoted in, from a flattened fragment of the page.

    Three layout traps are handled: the unit is often not on the label line, a
    line break can fall inside it, and the annual values can sit between its
    two halves.  The numbers are struck out before the unit is looked for."""
    flat = re.sub(PRICE_NUMBER, " ", fragment)
    flat = re.sub(r"\s+", " ", flat).lower()
    for unit, factor in UNITS:
        if unit in flat:
            return unit, factor
    return None, None


def NumbersOn(line):
    return [float(n.replace(",", "")) for n in PRICE_NUMBER.findall(line)]


def PriceRows(text):
    """Every row of every price table of a page.

    A row is `label ... five annual values`, on the label line itself or on the
    lines that follow it; both layouts occur."""
    lines = text.split("\n")
    rows = []
    for i, line in enumerate(lines):
        if not PRICE_HEADER.match(line):
            continue
        unit, factor = PriceUnit(" ".join(lines[i:i + 3]))
        # the header line may carry the values itself, with the unit below it
        if len(NumbersOn(line)) >= 3:
            if unit is None and i + 1 < len(lines):
                unit, factor = PriceUnit(line + " " + lines[i + 1])
            rows.append({"row": line.strip(), "label": line.strip(),
                         "unit": unit, "factor": factor,
                         "series": NumbersOn(line)[-5:]})
            continue
        # otherwise read the following rows until the table ends
        j = i + 1
        misses = 0
        while j < len(lines):
            line_j = lines[j]
            numbers = NumbersOn(line_j)
            if len(numbers) >= 3:
                label = PRICE_NUMBER.split(line_j)[0].strip(" ,:")
                # a row may name its own unit, which overrides the header; the
                # next line is consulted only when this one names none, because
                # it may be another row quoting a different unit
                row_unit, row_factor = PriceUnit(line_j)
                if row_unit is None and j + 1 < len(lines):
                    row_unit, row_factor = PriceUnit(line_j + " " + lines[j + 1])
                # stock rows carry a tonnage, not a price
                if not re.match(r"\s*stocks", label, re.I):
                    rows.append({"row": line_j.strip()[:110],
                                 "label": label or lines[i].strip(),
                                 "unit": row_unit or unit,
                                 "factor": row_factor or factor,
                                 "series": numbers[-5:]})
                misses = 0
            else:
                if TABLE_END.match(line_j):
                    break
                # the price table is indented, so an unindented line that is
                # not the header itself starts the next section
                if misses and line_j[:2].strip() and line_j.strip():
                    break
                misses += 1
                if misses > 6:
                    break
            j += 1
    return rows


def SelectPriceRow(metal, rows):
    """The price row that belongs to a metal."""
    hint = ROW_HINT.get(metal)
    if hint is None:
        return rows[0] if rows else None
    for r in rows:
        if re.search(hint, r["label"], re.I):
            return r
    return None                 # the chapter quotes other commodities only


def FormOnly(label):
    """The quoted row with the figures and their footnote markers removed.

    The USGS row label is itself the record of the form and grade, so it is
    kept verbatim apart from the columns of figures; rewriting it into a
    shorter commodity name would be an interpretation, not the record."""
    text = PRICE_NUMBER.sub(" ", str(label))
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;:")


def SeriesYears(count):
    """The years covered by a series of `count` annual values.

    The edition holds the annual averages of the five years that end one year
    before it, the newest of which is an estimate for that year."""
    last = EDITION - 1
    return "%d-%d" % (last - count + 1, last)


def ProbePriceRows():
    """Every price row of every chapter, for re-deriving ROW_HINT."""
    for chapter in sorted(set(CHAPTER.values())):
        rows = PriceRows(ChapterText(chapter))
        print("\n=== %s (%d rows) ===" % (chapter, len(rows)))
        for r in rows:
            print("   %-46s unit=%-26s %s"
                  % (r["label"][:46], r["unit"], r["series"]))
    print("\nPRICE_PROBE_DONE")


def ReadUSGSPrice(elements):
    """The price of each metal: the 2025 annual average, in USD per kg.

    The price is converted to USD per kg year by year, so that the spread of
    the series is a spread of comparable numbers."""
    collected = []
    chapters = {}
    for metal in elements:
        chapter = CHAPTER[metal]
        if chapter not in chapters:
            chapters[chapter] = PriceRows(ChapterText(chapter))
        row = SelectPriceRow(metal, chapters[chapter])
        record = {"element": metal, "MCS chapter": chapter,
                  "form and purity": "", "USGS form": USGS_FORM.get(metal, ""),
                  "price year": "", "unit": "", "price / USD kg-1": None,
                  "price series years": "", "price minimum / USD kg-1": None,
                  "price maximum / USD kg-1": None,
                  "price span, maximum / minimum": None}
        if row is not None and row["factor"] is not None \
                and len(row["series"]) >= 3:
            series = [value * row["factor"] for value in row["series"]]
            record.update({
                "form and purity": FormOnly(row["label"]),
                "price year": EDITION - 1,
                "unit": row["unit"],
                "price / USD kg-1": series[-1],
                "price series years": SeriesYears(len(series)),
                "price minimum / USD kg-1": min(series),
                "price maximum / USD kg-1": max(series),
                "price span, maximum / minimum":
                    max(series) / min(series) if min(series) > 0 else None,
            })
            print("  %-3s %-18s %-44s %10.2f USD/kg"
                  % (metal, chapter, row["label"][:44], series[-1]))
        collected.append(record)
    return pd.DataFrame(collected)


# ---------------------------------------------------------------------------
# the production and the reserves, from the word boxes of the chapters
# ---------------------------------------------------------------------------
# The word boxes are used rather than the text because the layout carries three
# traps: a chapter runs over two pages whose coordinates restart; a footnote is
# a superscript beside the value it qualifies; and the unit is not in the table
# but in the chapter subtitle, which a short chapter wraps its running text
# through.
PROD_NUMBER = re.compile(r">?\d[\d,]*(?:\.\d+)?")
RESERVES = re.compile(r"Reserves:?\d*", re.I)
SUBTITLE = re.compile(r"Data in (.+?)\s*,?\s*unless otherwise specified", re.I)
# a chapter whose subtitle carries no such clause ends at the closing parenthesis
SUBTITLE_FALLBACK = re.compile(r"Data in (.+?)\)", re.I)
UNIT_PHRASE = [("thousand metric tons", 1e3), ("thousand tons", 1e3),
               ("million metric tons", 1e6), ("million tons", 1e6),
               ("kilograms", 1e-3), ("metric tons", 1.0)]
MIN_HEIGHT = 11.0           # a value is set larger than a superscript
HEADER_REACH = 240.0        # how far above the world total the heading is looked for
RESERVES_TOLERANCE = 40.0   # how near a reserves value has to sit to its heading


def DocumentWords(chapter):
    """Every word of a chapter, as (page, xMin, yMin, xMax, yMax, text)."""
    path = os.path.join(CACHE_DIR, "mcs%d-%s.pdf" % (EDITION, chapter))
    out = subprocess.run(["pdftotext", "-bbox", path, "-"],
                         capture_output=True, text=True).stdout
    root = ET.fromstring(out[out.index("<html"):])
    words = []
    for page, node in enumerate(root.iter("{http://www.w3.org/1999/xhtml}page")):
        for w in node.iter("{http://www.w3.org/1999/xhtml}word"):
            words.append((page, float(w.get("xMin")), float(w.get("yMin")),
                          float(w.get("xMax")), float(w.get("yMax")),
                          w.text or ""))
    return words


def RowsOf(words, tolerance=0.5):
    """The words grouped into the lines of a page, page by page, top to bottom.

    The page is part of the key of a line, because the second page of a chapter
    restarts its own coordinates."""
    rows = []
    for word in sorted(words, key=lambda w: (w[0], w[2], w[1])):
        if rows and rows[-1][0][0] == word[0] \
                and abs(rows[-1][0][2] - word[2]) < tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])
    return [sorted(row, key=lambda w: w[1]) for row in rows]


def WordHeight(word):
    return word[4] - word[2]


def ValuesOf(row):
    """The values of a row, a superscript of the layout being a small word."""
    return [w for w in row if WordHeight(w) > MIN_HEIGHT
            and PROD_NUMBER.fullmatch(w[5])]


def WorldTotal(rows):
    """The index of the row of the world total of the table."""
    for i, row in enumerate(rows):
        text = [w[5] for w in row]
        if "World" in text and "total" in text:
            return i
    return None


def ChapterSubtitle(words):
    """The statement of the unit of a chapter, from its subtitle.

    It is the one statement that covers the whole chapter, and it is read in
    document order because the running text of a short chapter is set through
    it."""
    flat = " ".join(w[5] for w in words if w[0] == 0 and w[2] < 100)
    match = SUBTITLE.search(flat) or SUBTITLE_FALLBACK.search(flat)
    if match is None:
        return ""
    # a footnote mark stands as a number of its own outside any parenthesis
    kept, depth = [], 0
    for token in match.group(1).lower().split():
        depth += token.count("(") - token.count(")")
        if token.isdigit() and depth == 0:
            continue
        kept.append(token)
    return " ".join(kept).strip().rstrip(",").strip()


def ProductionUnit(text, default="metric tons"):
    for phrase, _ in UNIT_PHRASE:
        if phrase in text:
            return phrase
    return default


def FactorOf(phrase):
    for name, factor in UNIT_PHRASE:
        if name == phrase:
            return factor
    return 1.0


def ReservesHeading(rows, wt):
    """The heading of the reserves column, or None where the table has none.

    The heading that belongs to the table is the lowest one above the world
    total on the same page."""
    page, y = rows[wt][0][0], rows[wt][0][2]
    heading = None
    for row in rows:
        if row[0][0] != page or not y - HEADER_REACH < row[0][2] < y - 2:
            continue
        for w in row:
            if WordHeight(w) > MIN_HEIGHT and RESERVES.fullmatch(w[5]):
                heading = w
    return heading


def ReservesOf(rows, wt, values):
    """The value of the world total that sits in the reserves column.

    A value belongs to the heading when its right edge is that of the heading,
    the values being right aligned.  A table whose reserves are not reported
    leaves the world total with production values only."""
    heading = ReservesHeading(rows, wt)
    if heading is None or not values:
        return None
    nearest = min(values, key=lambda w: abs(w[3] - heading[3]))
    return nearest if abs(nearest[3] - heading[3]) <= RESERVES_TOLERANCE \
        else None


def ReservesUnit(rows, reserves, default):
    """The unit of the reserves, which a table may state on its own.

    The reserves of iron ore are given in million metric tons while its
    production is given in thousand metric tons, and the table says so in a
    parenthetical set over the reserves columns."""
    text = " ".join(w[5] for row in rows
                    if row[0][0] == reserves[0]
                    and 0 < reserves[2] - row[0][2] < HEADER_REACH
                    for w in row if abs(w[3] - reserves[3]) < 80).lower()
    for phrase, factor in UNIT_PHRASE:
        if factor != 1.0 and "(%s)" % phrase in text:
            return phrase
    return default


def ExtractProduction(chapter):
    """The world production of 2024 and 2025, and the world reserves, in tons.

    The values of the world total row are its largest words; the one that sits
    in the reserves column is the reserves, and the leftmost of the rest are
    the production of the two years the table reports."""
    words = DocumentWords(chapter)
    rows = RowsOf(words)
    wt = WorldTotal(rows)
    if wt is None:
        return None
    values = sorted(ValuesOf(rows[wt]), key=lambda w: w[1])
    reserves = ReservesOf(rows, wt, values)
    production = [w for w in values if w is not reserves]
    unit = ProductionUnit(ChapterSubtitle(words))
    reserves_unit = unit if reserves is None else ReservesUnit(
        rows, reserves, unit)
    production_factor = FactorOf(unit)
    reserves_factor = FactorOf(reserves_unit)
    return {"production 2024 / t": None if len(production) < 1
            else ToNumber(production[0][5]) * production_factor,
            "production 2025 / t": None if len(production) < 2
            else ToNumber(production[1][5]) * production_factor,
            "reserves / t": None if reserves is None
            else ToNumber(reserves[5]) * reserves_factor,
            "unit": unit, "form": ChapterSubtitle(words)}


def ToNumber(text):
    return float(text.lstrip(">").replace(",", ""))


def ReadUSGSProduction(elements):
    """The world production of 2024 and 2025, and the world reserves, in tons."""
    chapters = {c: ExtractProduction(c)
                for c in sorted(set(CHAPTER.values()))}
    collected = []
    for metal in elements:
        chapter = CHAPTER[metal]
        read = chapters[chapter]
        if read is None:
            collected.append({"element": metal, "MCS chapter": chapter,
                              "world production 2025 / t": None,
                              "world reserves / t": None, "form": ""})
            continue
        collected.append({
            "element": metal, "MCS chapter": chapter,
            "world production 2025 / t": read["production 2025 / t"],
            "world reserves / t": read["reserves / t"],
            "form": read["form"]})
        print("  %-3s %-18s 2025=%-12s reserves=%-12s [%s]"
              % (metal, chapter,
                 "n/a" if read["production 2025 / t"] is None
                 else "%.4g" % read["production 2025 / t"],
                 "n/a" if read["reserves / t"] is None
                 else "%.4g" % read["reserves / t"], read["unit"]))
    return pd.DataFrame(collected)


# ---------------------------------------------------------------------------
# supply risk and toxicity
# ---------------------------------------------------------------------------
# GHS hazard statement -> hazard class.  Only the classes that matter for a
# catalyst screen are kept; the physical hazards are collected as a count but
# are not used as a toxicity measure.
HAZARD_CLASSES = {
    "carcinogenicity":            ["H350", "H351"],
    "germ cell mutagenicity":     ["H340", "H341"],
    "reproductive toxicity":      ["H360", "H361", "H362"],
    "organ toxicity (STOT)":      ["H370", "H371", "H372", "H373", "H335"],
    "acute toxicity":             ["H300", "H301", "H302", "H310",
                                   "H311", "H312", "H330", "H331", "H332"],
    "aquatic toxicity":           ["H400", "H410", "H411", "H412", "H413"],
    "skin/respiratory sensitisation": ["H317", "H334"],
    "corrosion/irritation":       ["H314", "H315", "H318", "H319"],
}

# the classes that make a metal a health hazard rather than only an
# environmental or physical one
HEALTH_CLASSES = ["carcinogenicity", "germ cell mutagenicity",
                  "reproductive toxicity", "organ toxicity (STOT)"]

SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "materials-stability-screen/1.0 (research use)"})

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, encoding="utf-8") as f:
        cache = json.load(f)
else:
    cache = {}


def cached(key, fetch):
    # nothing worth caching is written, so that a re-run retries the lookup.
    if key not in cache:
        value = fetch()
        empty = (not value) or (isinstance(value, tuple) and not value[0])
        if not empty:
            cache[key] = value
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=1, ensure_ascii=False)
        time.sleep(0.34)          # PubChem allows 5 requests per second
        return value
    return cache[key]


def pubchem_cid(M):
    """mendeleev element -> PubChem CID of the elemental substance.

    The CAS registry number is tried first, because it identifies the element
    unambiguously; the element name is the fallback, because PubChem does not
    expose every element through its CAS xref endpoint."""
    if M.cas:
        r = SESSION.get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
                        "/xref/RegistryID/%s/cids/JSON" % M.cas, timeout=25)
        if r.ok and "IdentifierList" in r.json():
            return r.json()["IdentifierList"]["CID"][0]
        time.sleep(0.34)
    r = SESSION.get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
                    "/name/%s/cids/JSON" % M.name.lower(), timeout=25)
    if r.ok and "IdentifierList" in r.json():
        return r.json()["IdentifierList"]["CID"][0]
    return None


def ghs_statements(cid):
    """Hazard statements of the elemental substance, plus the signal word.

    The statement text carries the code and the signal word, and the union over
    all notifiers is taken: a statement present in any harmonised notification
    counts as a classification of the substance."""
    if not cid:
        return {}, None
    r = SESSION.get("https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data"
                    "/compound/%d/JSON?heading=GHS+Classification" % cid,
                    timeout=30)
    if not r.ok:
        return {}, None
    text = r.text
    signals = {}
    # `H350: May cause cancer [Danger Carcinogenicity]` and the ECHA variant
    # `H330 (92.2%): Fatal if inhaled [Danger Acute toxicity, inhalation]`
    for code, tail in re.findall(r"(H\d{3}[A-Za-z]{0,2})\s*(?:\([\d.]+%\))?"
                                 r"\s*:[^\[\]]*\[(Danger|Warning)", text):
        # a statement seen as Danger anywhere is recorded as Danger
        if signals.get(code) != "Danger":
            signals[code] = tail
    if not signals:                       # fall back on a plain code sweep
        for code in set(re.findall(r"H\d{3}[A-Za-z]{0,2}", text)):
            signals[code] = "Warning" if "Warning" in text else None
    return signals, ("Danger" if "Danger" in signals.values()
                     else ("Warning" if signals else None))


def SupplyRiskToxicity(elements):
    """The supply-risk and GHS hazard indicators of each metal."""
    rows = []
    for symbol in elements:
        M = element(symbol)

        def get(attr):
            try:
                return getattr(M, attr)
            except Exception:
                return None

        cid = cached("cid:%s" % symbol, lambda M=M: pubchem_cid(M))
        signals, signal = cached("ghs:%s" % symbol, lambda: ghs_statements(cid))

        # hazard class -> was any statement of that class issued, and as what
        classes = {}
        for klass, codes in HAZARD_CLASSES.items():
            hit = {c: signals[c] for c in signals if c in codes}
            if hit:
                classes[klass] = ("Danger" if "Danger" in hit.values()
                                  else "Warning")

        health = [k for k in HEALTH_CLASSES if k in classes]
        rows.append({
            "element": symbol,
            "production concentration / %": get("production_concentration"),
            "political stability of top producer":
                get("political_stability_of_top_producer"),
            "PubChem CID": cid,
            "GHS signal": signal,
            # the headline indicator: how many of the four health hazard classes
            # the substance carries a harmonised classification for.  a binary
            # flag would not discriminate, because most metals are Danger.
            "toxicity severity": len(health),
            "health hazard": "yes" if health else "no",
            "hazard classes": "; ".join(classes),
        })
        print("  %-3s CID=%-9s signal=%-8s classes=%d"
              % (symbol, cid, signal, len(classes)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the screen
# ---------------------------------------------------------------------------
def Stability(eb, ef, acid, base):
    """The stability of a site, from the three descriptors of the data set."""
    if eb > 0 or ef > 0:
        return "E_b && E_f unstable"
    if acid >= 0 and base > 0:
        return "Both stable"
    if acid >= 0 and base <= 0:
        return "Acid stable"
    if acid <= 0 and base >= 0:
        return "Base stable"
    return "U_diss unstable"


def main():
    data = pd.read_excel(DATASET)
    metals = data.iloc[:, 0].astype(str).to_list()
    print("Metals in the data set: %d" % len(metals))

    print("\nReading the USGS prices...")
    price = ReadUSGSPrice(metals)
    print("\nReading the USGS production...")
    production = ReadUSGSProduction(metals).drop(columns=["MCS chapter"])
    print("\nReading the supply risk and the toxicity...")
    supply = SupplyRiskToxicity(metals)

    table = pd.DataFrame({
        "element": metals,
        "atomic number": data["atomic number"].to_numpy(),
        "E_b/eV": data["E_b/eV"].to_numpy(),
        "E_f/eV": data["E_f/eV"].to_numpy(),
        "U_diss_acid/V": data["U_diss_acid/V"].to_numpy(),
        "U_diss_base/V": data["U_diss_base/V"].to_numpy(),
    })
    table["stability"] = [
        Stability(r["E_b/eV"], r["E_f/eV"], r["U_diss_acid/V"],
                  r["U_diss_base/V"])
        for _, r in table.iterrows()
    ]
    table["stable"] = table["stability"].isin(
        ["Both stable", "Acid stable", "Base stable"])

    table = table.set_index("element").join(
        [price.set_index("element"), production.set_index("element"),
         supply.set_index("element")], how="left").reset_index()
    table["expensive"] = table["price / USD kg-1"] >= THRESHOLD

    table.to_csv(STABLE_CSV, index=False)

    lines = []
    lines.append("Stable M-N4-C of the USGS Mineral Commodity Summaries %d"
                 % EDITION)
    lines.append("=" * 78)
    lines.append("")
    lines.append("Sources")
    lines.append("  price and volume  U.S. Geological Survey, Mineral Commodity")
    lines.append("                    Summaries %d, %s" % (EDITION, BASE))
    lines.append("  supply risk       mendeleev, EU critical-raw-materials pillars")
    lines.append("  toxicity          PubChem PUG-View, GHS classification")
    lines.append("  stability         the descriptors of the data set")
    lines.append("")
    lines.append("Stable M-N4-C are:")
    for _, r in table[table["stable"]].sort_values("E_f/eV").iterrows():
        note = "$" if r["expensive"] else " "
        lines.append("  %-3s %-14s %s  %s"
                     % (r["element"], r["stability"], note,
                        "" if pd.isna(r["price / USD kg-1"])
                        else "%.0f USD/kg" % r["price / USD kg-1"]))
    lines.append("")
    lines.append("Stable material number: %d" % int(table["stable"].sum()))
    lines.append('Note: "$" is an expensive element, price >= %.0f USD/kg.'
                 % THRESHOLD)

    report = "\n".join(lines)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n" + report)
    print("\nSTABLE_TABLE_DONE")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        ProbePriceRows()
        sys.exit(0)
    main()
