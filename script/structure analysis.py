"""The structure of the M-N4-C site, and the bonding of the lanthanides.

The script holds three parts, run in this order when it is called with no
argument:

  1. the geometry of the 61 sites, read from the relaxed CONTCAR: the average
     M-N distance, the average N-M-N angle and the out-of-plane displacement of
     the metal, their correlation with the three stability measures, and the two
     figures drawn from them;
  2. the lanthanide bonding descriptors, read from the DOSCAR, the Bader
     partition and the LOBSTER projection of the fourteen lanthanides: the d and
     f band centres and occupations, the M-N bond order and bond strength, and
     the Bader charge transfer, drawn as `Lanthanide bonding.png`;
  3. the LOBSTER inputs of that series, written into the VASP working
     directories by `--lobster-setup`.

Usage

  python "structure analysis.py"                  parts 1 and 2
  python "structure analysis.py" --redraw         redraw `Lanthanide
                                                  bonding.png` from the CSV
  python "structure analysis.py" --lobster-setup  write `lobsterin` and
                                                  `run_lobster.sh`

The VASP working directories of the lanthanides are the live scratch copies,
because LOBSTER reads the WAVECAR and CHGCAR of the run that produced them; the
curated copy under `data/vasp-file` is used as a fallback where the scratch copy
of one file is incomplete.

Everything is written next to this script and into
`../figures/structure analysis/`.
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms

from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from utils.vaspfile import (GetAngle, GetBandCenter, GetDistance, GetOutOfPlane)
from utils.doscar import SplitDOSCAR, SplitDoscar

ROOT = os.path.dirname(SCRIPT_DIR)
CURATED = os.path.join(ROOT, "data/vasp-file/M-N-C")
VASP = CURATED
WORK = os.path.join(os.path.dirname(ROOT), "VASP_files/work")
DATASET = os.path.join(ROOT, "data/M-N-C data set.xlsx")
FIG_DIR = os.path.join(ROOT, "figures/structure analysis")
os.makedirs(FIG_DIR, exist_ok=True)

# the site: the metal is atom 49 and its four N are 45..48, 1-based
M_IDX = 49
N_IDX = [45, 46, 47, 48]
TRANS_PAIRS = [(45, 48), (46, 47)]

LANTHANIDES = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
               "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]

# the two-tone palette of the figures: a light fill under a dark outline, one
# hue per family of sites
BLUE, BLUE_FILL = "#3a4b6e", "#9fbbd5"
RED, RED_FILL = "#ba3e45", "#d69d98"
GRID = dict(linestyle="--", alpha=0.6, color="grey")


# ---------------------------------------------------------------------------
# part 1. the geometry of the site
# ---------------------------------------------------------------------------
# the mean N-M-N angle that separates the square-planar from the square-pyramidal
# family.  the widest pyramidal angle is 167.8 and the narrowest planar one is
# 176.7, so any cut inside that empty window gives the same 43 / 18 split; 175 is
# used here.
ANGLE_THRESHOLD = 175

# ----------------------------------------------------------------------------
# the two-tone palette used throughout the rest of the figures of this study:
# a light fill with a dark outline, one family for the planar / majority class
# and one for the pyramidal / minority class
# ----------------------------------------------------------------------------


def Style(ax):
    """The grid of the other figures of this study."""
    ax.grid(True, **GRID)
    return ax


# the elements of each block of the periodic table, used to colour the geometry
# panels of the last figure


BLOCK_OF = {}
for _block, _members in {"s": ["Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba"],
                         "p": ["Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"],
                         "d": ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                               "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
                               "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"],
                         "f": LANTHANIDES}.items():
    for _member in _members:
        BLOCK_OF[_member] = _block


def GeometryAnalysis():
    """The geometry of the 61 sites, its correlation with stability,
    and the two figures drawn from it."""
    print("Reading the structural descriptors of 61 systems...")

    rows = []
    for folder in tqdm(sorted(os.listdir(os.path.join(VASP, "energy"))), desc="Descriptors"):
        if not folder.endswith("-N-C"):
            continue
        metal = folder[:-4]
        contcar = os.path.join(VASP, "energy", folder, "CONTCAR")
        if not os.path.isfile(contcar):
            continue

        # geometry of the site
        d_mn = np.mean([GetDistance(contcar, n, M_IDX) for n in N_IDX])
        trans = [GetAngle(contcar, a, M_IDX, b) for a, b in TRANS_PAIRS]
        all_angles = [GetAngle(contcar, N_IDX[i], M_IDX, N_IDX[j])
                      for i in range(4) for j in range(i + 1, 4)]

        rows.append({"metal": metal,
                     "d_M-N": d_mn,
                     "angle trans 1": trans[0],
                     "angle trans 2": trans[1],
                     "angle mean": float(np.mean(trans)),
                     "angle max": float(np.max(trans)),
                     "angle min": float(np.min(all_angles)),
                     "out of plane": GetOutOfPlane(contcar, idx_M=M_IDX, idx_N=N_IDX)})

    descriptors = pd.DataFrame(rows).set_index("metal")
    descriptors.to_csv(os.path.join(SCRIPT_DIR, "structure analysis descriptors.csv"))

    # the stability measures of the data set, merged on the element
    dataset = pd.read_excel(DATASET).set_index("Unnamed: 0")
    merged = descriptors.join(dataset[["E_b/eV", "E_f/eV", "U_diss_acid/V"]])

    report = open(os.path.join(SCRIPT_DIR, "structure analysis report.txt"),
                  "w+", encoding="utf-8")


    def Write(text=""):
        print(text)
        report.write(text + "\n")


    Write("=" * 78)
    Write("Structure analysis of the M-N4-C data set")
    Write("=" * 78)
    Write()
    Write("The geometric descriptors: the average M-N distance, the average N-M-N")
    Write("angle and the out-of-plane displacement of the metal, each read from the")
    Write("relaxed CONTCAR of the 61 systems.")

    # ----------------------------------------------------------------------------
    # 2. the structural thresholds and their correlation with stability
    # ----------------------------------------------------------------------------
    Write()
    Write("-" * 78)
    Write("1. Structural descriptors and the %d degree criterion" % ANGLE_THRESHOLD)
    Write("-" * 78)

    angles = merged["angle mean"].to_numpy(dtype=float)
    Write()
    Write("N-M-N angle of the 61 sites:")
    Write("  min = %.1f   max = %.1f   mean = %.1f"
          % (angles.min(), angles.max(), angles.mean()))
    # the criterion separates the two families when the distribution is empty around
    # the threshold
    below = (angles < ANGLE_THRESHOLD).sum()
    above = (angles >= ANGLE_THRESHOLD).sum()
    near = ((angles >= ANGLE_THRESHOLD - 2.5) & (angles <= ANGLE_THRESHOLD + 2.5)).sum()
    Write("  systems below %d degrees (square pyramidal family): %d"
          % (ANGLE_THRESHOLD, below))
    Write("  systems at or above %d degrees (square planar family): %d"
          % (ANGLE_THRESHOLD, above))
    Write("  systems within 2.5 degrees of the threshold: %d" % near)
    gap_lo = angles[angles < ANGLE_THRESHOLD].max() if below else np.nan
    gap_hi = angles[angles >= ANGLE_THRESHOLD].min() if above else np.nan
    Write("  the empty window around the threshold is %.1f - %.1f degrees, which is"
          % (gap_lo, gap_hi))
    Write("  what makes a single cut at %d degrees well defined.  any threshold"
          % ANGLE_THRESHOLD)
    Write("  inside that window returns the same %d / %d split." % (below, above))

    # how the two families differ in the geometry and in stability
    planar = merged[merged["angle mean"] >= ANGLE_THRESHOLD]
    pyramidal = merged[merged["angle mean"] < ANGLE_THRESHOLD]
    Write()
    Write("  %-14s %8s %12s %12s %12s %12s" % ("family", "n", "out of plane",
                                                "d_M-N", "E_b", "E_f"))
    for label, frame in [("square planar", planar), ("square pyramidal", pyramidal)]:
        Write("  %-14s %8d %12.3f %12.3f %12.3f %12.3f"
              % (label, len(frame), frame["out of plane"].mean(),
                 frame["d_M-N"].mean(), frame["E_b/eV"].mean(), frame["E_f/eV"].mean()))

    Write()
    Write("Correlation of the stability measures with the geometry")
    Write("(Pearson r, p value, and Spearman rho between the brackets):")
    corr_rows = []
    for stability in ["E_b/eV", "E_f/eV", "U_diss_acid/V"]:
        for geometry in ["out of plane", "d_M-N", "angle mean", "angle max"]:
            sub = merged[[geometry, stability]].dropna()
            r, p = stats.pearsonr(sub[geometry], sub[stability])
            rho, p_rho = stats.spearmanr(sub[geometry], sub[stability])
            corr_rows.append({"stability": stability, "geometry": geometry,
                              "n": len(sub), "Pearson r": r, "p": p,
                              "Spearman rho": rho, "p (spearman)": p_rho})
            Write("  %-14s vs %-12s r = %+6.3f (p = %-6.3f)  rho = %+6.3f (p = %-6.3f)"
                  % (stability, geometry, r, p, rho, p_rho))
    pd.DataFrame(corr_rows).to_csv(
        os.path.join(SCRIPT_DIR, "structure analysis correlations.csv"), index=False)

    # the correlation inside each block, which is the fairer test because the
    # lanthanides span a much narrower E_f range than the s and p blocks
    Write()
    Write("The same correlations computed inside each block of the periodic table:")
    blocks = {"s": ["Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba"],
              "p": ["Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"],
              "d": ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                    "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
                    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"],
              "f": LANTHANIDES}
    for block, members in blocks.items():
        frame = merged.loc[[m for m in members if m in merged.index]]
        for stability in ["E_b/eV", "E_f/eV"]:
            for geometry in ["out of plane", "angle mean"]:
                sub = frame[[geometry, stability]].dropna()
                if len(sub) < 4 or sub[geometry].std() == 0:
                    continue
                r, p = stats.pearsonr(sub[geometry], sub[stability])
                Write("  block %s  %-10s vs %-12s r = %+6.3f (p = %.3f, n = %d)"
                      % (block, stability, geometry, r, p, len(sub)))

    # ----------------------------------------------------------------------------
    # 3. figures
    # ----------------------------------------------------------------------------
    Write()
    Write("Writing the figures to %s" % FIG_DIR)

    # 3a. the angle distribution and the threshold criterion
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    bins = np.arange(angles.min() - 1, angles.max() + 1, 1.0)
    ax = axes[0]
    ax.axvspan(gap_lo, gap_hi, color="lightgrey", alpha=0.9, zorder=1)
    # the two families are histogrammed separately so that every bar takes the colour
    # of the family it belongs to
    ax.hist(angles[angles < ANGLE_THRESHOLD], bins=bins, color=RED_FILL,
            edgecolor=RED_EDGE, linewidth=1.5, zorder=3, label="Square pyramidal")
    ax.hist(angles[angles >= ANGLE_THRESHOLD], bins=bins, color=BLUE_FILL,
            edgecolor=BLUE_EDGE, linewidth=1.5, zorder=3, label="Square planar")
    ax.axvline(ANGLE_THRESHOLD, color="grey", linestyle="--", linewidth=2.5,
               label="%d$^{\\circ}$ threshold" % ANGLE_THRESHOLD, zorder=5)
    ax.set_xlabel("Average $\\theta_{N-M-N}$ (degree)", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14, loc="upper left")
    Style(ax)

    ax = axes[1]
    ax.axvspan(gap_lo, gap_hi, color="lightgrey", alpha=0.9, zorder=1)
    ax.scatter(planar["angle mean"], planar["out of plane"], s=50, color=BLUE_FILL,
               edgecolor=BLUE_EDGE, linewidths=1.5,
               label="Square planar  ($\\geq$ %d$^{\\circ}$)" % ANGLE_THRESHOLD, zorder=3)
    ax.scatter(pyramidal["angle mean"], pyramidal["out of plane"], s=50, color=RED_FILL,
               edgecolor=RED_EDGE, linewidths=1.5,
               label="Square pyramidal  ($\\leq$ %d$^{\\circ}$)" % ANGLE_THRESHOLD, zorder=3)
    ax.axvline(ANGLE_THRESHOLD, color="grey", linestyle="--", linewidth=2.5, zorder=2)
    ax.set_xlabel("Average $\\theta_{N-M-N}$ (degree)", fontsize=16)
    ax.set_ylabel("$d_{z}$ ($\\AA$)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14, loc="upper right")
    Style(ax)
    # the two panels label themselves, so the figure carries no overall title
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "angle threshold.png"))
    plt.close(fig)

    # 3b. stability against the structural descriptors
    fig, axes = plt.subplots(3, 3, figsize=(20, 18), dpi=200)
    geometries = [("out of plane", "$d_{z}$ ($\\AA$)"),
                  ("d_M-N", "$d_{M-N}$ ($\\AA$)"),
                  ("angle mean", "$\\theta_{N-M-N}$ (degree)")]
    stabilities = [("E_b/eV", "$E_{b}$ (eV)"),
                   ("E_f/eV", "$E_{f}$ (eV)"),
                   ("U_diss_acid/V", "$U_{diss}$ (V)")]
    for i, (stability, s_label) in enumerate(stabilities):
        for j, (geometry, g_label) in enumerate(geometries):
            ax = axes[i, j]
            sub = merged[[geometry, stability]].dropna()
            lan_sel = sub[[BLOCK_OF.get(m) == "f" for m in sub.index]]
            oth_sel = sub[[BLOCK_OF.get(m) != "f" for m in sub.index]]
            ax.scatter(oth_sel[geometry], oth_sel[stability], s=50, color=BLUE_FILL,
                       edgecolor=BLUE_EDGE, linewidths=1.5, zorder=3,
                       label="Other metals")
            ax.scatter(lan_sel[geometry], lan_sel[stability], s=50, color=RED_FILL,
                       edgecolor=RED_EDGE, linewidths=1.5, zorder=4,
                       label="Lanthanides")
            r, p = stats.pearsonr(sub[geometry], sub[stability])
            # least-squares line, drawn only to guide the eye
            slope, intercept = np.polyfit(sub[geometry], sub[stability], 1)
            xs = np.linspace(sub[geometry].min(), sub[geometry].max(), 50)
            ax.plot(xs, slope * xs + intercept, color="#85120f", linewidth=2, zorder=2)
            ax.set_xlabel(g_label, fontsize=17)
            ax.set_ylabel(s_label, fontsize=17)
            ax.set_title("r = %+.3f   (p = %.3f)" % (r, p), fontsize=17)
            ax.tick_params(labelsize=13)
            if i == 0 and j == 0:
                ax.legend(fontsize=15, loc="upper right")
            Style(ax)
    fig.suptitle("Stability against the structural descriptors of the M-N$_4$-C site",
                 fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG_DIR, "stability vs geometry.png"))
    plt.close(fig)

    report.close()
    print()
    print("Report: %s" % os.path.join(SCRIPT_DIR, "structure analysis report.txt"))
    print("Descriptors: %s" % os.path.join(SCRIPT_DIR, "structure analysis descriptors.csv"))
    print("Correlations: %s" % os.path.join(SCRIPT_DIR, "structure analysis correlations.csv"))


# ---------------------------------------------------------------------------
# part 2. the lanthanide bonding descriptors
# ---------------------------------------------------------------------------
# the lanthanide series, the f-shielding argument: whether a half-filled or
# filled f shell weakens the M-N bond, and whether Eu (f7) and Yb (f14) sit at
# an extremum of the series on the descriptors that measure it.
# ---------------------------------------------------------------------------
# the window of the band center, taken from `dataset generator.py` so that the
# d band center of the lanthanides is the same quantity as the descriptor the
# machine-learning models were trained on
E_WINDOW = (-10.0, 5.0)

# the window of the occupation integrals.  The upper bound is the Fermi level,
# and the lower bound is not decoration: DFT+U can leave a spurious almost flat
# band far below the valence states, and an integral that is open at the bottom
# counts it.  Ho has such a band at -116 eV, purely f, the same energy at every
# k point of the mesh, which carries 5.2 f electrons and pushes its f count to
# 15.4, past the 14 the shell can hold.  The deepest physical state of the
# series is the 5s/5p semicore of Lu at -60.7 eV, so -70 eV lies below every
# real state of every element and above every spurious one.
OCC_WINDOW = (-70.0, 0.0)

# the two data-integrity tests, both of them against the series itself.  The f
# shell of a lanthanide holds at most 14 electrons, but the LORBIT=11 projection
# spills a few tenths of an electron past the capacity of a filled shell (Lu
# reaches 14.11), so the bound is set at half an electron over it, which still
# separates that spillover from the 15.5 of a projection that has gone wrong.
# The lowest Kohn-Sham eigenvalue falls smoothly across the series, so an
# element whose lowest band sits many median-absolute-deviations below its
# neighbours was not computed from the same Hamiltonian as the rest.
F_OCC_MAX = 14
F_OCC_TOL = 0.5
OUTLIER_MAD = 5.0


# the drawing of the figure
#
# The whole of it: the palette and the fonts, the collision search that places
# every number, the four panels, and the pixel check on the search.  It is kept
# here rather than in a file of its own so that the descriptors measured above
# and the figure drawn from them cannot drift apart.
# ---------------------------------------------------------------------------
# the Z order of the series, promethium in its place
ORDER = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
         "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
# the only member of the series with no stable isotope, and the only one the
# data set cannot carry; it keeps its slot on the axis, in red, but no bar and
# no marker, the curve closing over it
RADIOACTIVE = "Pm"

# the 4f configuration the f panel prints above each bar: the ion in the cage,
# not the free atom.  The series is trivalent, so the count runs f0, f1, ...,
# f14 in step with Z, except for the divalent Eu and Yb, which stop at the
# half-filled f7 and the filled f14.  The number printed under the
# configuration is the integrated occupation of `lanthanide bonding.csv`.
#
# Sm and Tm carry a pair of configurations rather than a single count, because
# the ion sits between two: Sm between f5 and the half-filled f6, Tm between
# f12 and the filled f13, and the occupations the panel plots (5.66 for Sm,
# 12.51 for Tm) fall between the two, so the single count the rest of the
# series carries would be a claim the number under it contradicts.  Both
# configurations are written out in full, f5-f6 and f12-f13, with the f
# repeated before each count and a hyphen -- not the minus a bare `-` renders
# as in math mode -- between them, because a range and a subtraction are not
# the same mark: f5-6, one f over a range, reads as a subtraction, and it is
# the second f that says there are two configurations here.
F_SHELL = {"La": "0", "Ce": "1", "Pr": "2", "Nd": "3", "Pm": "4",
           "Sm": "5", "Eu": "7", "Gd": "7", "Tb": "8",
           "Dy": "9", "Ho": "10", "Er": "11", "Tm": "12",
           "Yb": "14", "Lu": "14"}
# the two ions of the series whose configuration is not a single count, and the
# label each of them prints in place of the one derived above.  Both
# configurations are written out in full and both counts carry the same raised
# position and the same size the rest of the series prints its single count at:
# the range line between them is a hyphen, not the minus a bare `-` renders as
# in math mode, and the second f is what says there are two configurations
# here rather than one count spanning a range
F_SHELL_PAIR = {"Sm": r"\mathrm{f}^{5}\text{-}\mathrm{f}^{6}",
                "Tm": r"\mathrm{f}^{12}\text{-}\mathrm{f}^{13}"}

# the palette of the figure this replaces: blue for the f shell, the bond order,
# the d band center and the charge transfer, red for the bond strength
HIGHLIGHT = "#e2e2e2"       # the band that picks out Eu and Yb
# the band sits behind the markers, so it is kept faint enough to read as a
# backdrop rather than as a value of its own
BAND_ALPHA = 0.55

# the two curves of the bond order panel, as the legend names them.  They are
# named for the descriptor and not the element, because the point of the pair is
# that the two descriptors agree with each other across the series
LEGEND_LABELS = ("ICOBI", "$-$ICOHP")

# the fonts of the previous figure
TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 13
VALUE_SIZE = 10
FCONF_SIZE = 13             # the 4f configuration, the label the panel is read for
# the rise a requested place climbs, and the step it climbs in, in display
# pixels.  The first candidate of `firsts` holds its width and is tried at every
# height from its own to `RISE` above its point: the column it stands in may be
# the column of the tallest bar in the panel, so the range is set to clear that
# bar rather than to the panel's ordinary ladder.
RISE = 150.0
STEP = 2.0
# where the two paired configurations are asked to sit, in display pixels
# relative to the centre of their own bar: the offset and the rise above the
# point, as (dx, dy).  Both stand beside a taller neighbour -- Sm beside Eu, Tm
# beside Yb -- so both are asked for the place their own crowding leaves them,
# a little to the left of the bar.
F_PAIR_PLACE = {"Sm": (-12.0, 22.0), "Tm": (-20.0, 22.0)}
# the shallow arm of the bimodal E_b ladder.  They carry the grey band of the
# old figure everywhere; in the panels where they are a marker or a bar rather
# than a band they are drawn in the red of the palette as well, and their
# numbers are written in the same red, so that they are picked out by colour and
# not by the band alone
RED_METALS = ("Eu", "Yb")

# the least a label stands clear of the point it names, in pixels.  It is two
# things at once and both are wanted: the slack added to a label's box when it
# is tested against a curve, a marker or another label, and the distance the
# closest candidate keeps from that obstacle.  A label whose box only clears the
# ink it sits beside by a pixel reads as touching it, however exactly the two
# were placed, so the clearance is set large enough to be seen.
PADDING = 6.0

# the clearance the f panel keeps instead: the value the figure was tuned with,
# unchanged.  Its slot is the narrowest thing in the figure -- fifteen labels
# across one panel -- and enlarging the clearance costs it the width it needs to
# stand at its own x, which is what the panel's labels were fixed for.  The bar
# also separates a label from it on its own, being a solid block whose top edge
# is drawn a few pixels below the number that names it.
F_PADDING = 1.6

# the margin tight_layout leaves between the panels and around them, in units
# where the default is 1.08.  Slightly above that: the panels are read as four
# separate figures, and the wider margin is also the blank the panel letters are
# added in by hand.
LAYOUT_PAD = 2.0

# the size of the figure in inches.  It is tied to the layout margin above: the
# margin is paid for out of the panels, and a panel can only be squeezed so far
# before the fifteen labels of the f shell no longer fit their slots one to a
# bar.  The figure is grown a little as the margin is widened, so that the extra
# space between the panels is space the figure has, not space taken from them.
FIG_SIZE = (13.6, 9.4)

# the ladder the bar panel's labels climb.  It starts further out than the
# ladder the curves use because a bar is the one obstacle whose ink reaches
# past the point the label is anchored to: the label is put on the bar top, and
# the drawn outline of the bar rises half a line width above it, so the nearest
# candidate is never clear and a ladder that begins at nought sends every label
# out to whichever step it clears first.  Beginning past the outline instead
# puts them all on the same step, in one row.
BAR_STEPS = (3.0, 6.0, 10.0, 16.0, 23.0, 31.0, 40.0, 50.0, 62.0)

# the band of the bond order panel each of its two series is drawn in, as the
# fraction of the axis its smallest and largest value occupy.  The two axes are
# scaled independently, and that freedom is used to give each curve a lane: the
# ICOBI curve low, the -ICOHP curve high, so that the numbers under the one and
# the numbers over the other have a column to themselves instead of sharing the
# middle of the panel, which is where the two curves used to cross and their
# labels with them.  The lanes are kept as close as the labels allow -- the
# point of drawing the two on one panel is to compare them, which a wide gap
# defeats -- and the values here were arrived at by drawing the panel and
# counting: all twenty-eight numbers keep the x of the point they name, none
# overlaps another, and neither lane is so narrow that a curve flattening out
# can no longer be followed.
ICOBI_BAND = (0.16, 0.48)
ICOHP_BAND = (0.52, 0.84)

pos = np.arange(len(ORDER))


# ---------------------------------------------------------------------------
# label placement
# ---------------------------------------------------------------------------
def SegmentHitsRect(p0, p1, rect):
    """Whether the segment p0-p1 meets the rectangle, by Liang-Barsky clipping.

    A segment that misses is rejected here, so a label is never counted as clear
    of a curve that only came near it in the bounding box of the whole path.
    """
    x0, y0, x1, y1 = rect
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - x0), (dx, x1 - p0[0]),
                 (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if p == 0.0:
            if q < 0.0:
                return False
        else:
            r = q / p
            if p < 0.0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return t0 <= t1


def RectOverlap(a, b):
    """The area the two rectangles share, zero when they are apart."""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def CollectCurves(axes):
    """Every drawn stroke of the panels, in display coordinates: the curve
    segments split at their gaps, the marker squares, and the bars.

    Everything is inflated by half its own line width, because a stroke is drawn
    centred on its path and therefore reaches half a line width past the
    geometry that describes it.  Without that, a label the search believed to be
    clear still lands on the drawn ink.
    """
    segments, blocks = [], []
    for ax in axes:
        renderer = ax.figure.canvas.get_renderer()
        to_pixels = ax.figure.dpi / 72.0
        for line in ax.get_lines():
            # the reach of the stroke around its path
            ink = max(line.get_linewidth(), line.get_markeredgewidth()) \
                * to_pixels / 2.0
            xy = np.asarray(line.get_xydata(), dtype=float)
            if len(xy) < 2:
                if len(xy) == 1 and np.isfinite(xy[0]).all():
                    mid = ax.transData.transform(xy[0])
                    half = line.get_markersize() * to_pixels / 2.0 + ink
                    blocks.append((mid[0] - half, mid[1] - half,
                                   mid[0] + half, mid[1] + half))
                continue
            # a NaN breaks the path, so the stroke is a set of runs
            good = np.isfinite(xy).all(axis=1)
            run = []
            for point, ok in zip(xy, good):
                if ok:
                    run.append(point)
                else:
                    if len(run) > 1:
                        segments.append((ax.transData.transform(np.array(run)),
                                         ink))
                    run = []
            if len(run) > 1:
                segments.append((ax.transData.transform(np.array(run)), ink))
        for patch in ax.patches:
            # the Eu/Yb shading is a background band, not a stroke: a label is
            # meant to sit on it, and treating it as an obstacle would forbid
            # every position over two whole columns of the panel
            if mcolors.to_hex(patch.get_facecolor()) == mcolors.to_hex(HIGHLIGHT):
                continue
            box = tuple(patch.get_window_extent(renderer).extents)
            ink = patch.get_linewidth() * to_pixels / 2.0
            blocks.append((box[0] - ink, box[1] - ink,
                           box[2] + ink, box[3] + ink))
    return segments, blocks


def Angles(primary, fallback=None):
    """The directions a label may take, most wanted first.

    `primary` is the direction the panel asks for -- the two panels that share a
    pair of axes want their two series on opposite sides, so that curves lying
    nearly on top of one another still read as two.  `fallback` is tried only if
    nothing in `primary` is clear, so the preference never costs a collision.
    """
    order = list(primary) + [a for a in (fallback or ()) if a not in primary]
    return order


def Candidates(width, height, angles, padding, extra_steps):
    """The offsets to try for one label, in pixels, most wanted first.

    The first offset of a direction is the smallest that gets the label clear of
    its own data point: the support of the label box along that direction plus
    the padding.  The rest step further out.  Computing it from the label's own
    size is what the fixed list of offsets got wrong -- a tenth of a point of
    text is 14 pixels at this resolution, so an offset chosen for a small label
    leaves a large one sitting on its marker.
    """
    half_w, half_h = width / 2.0, height / 2.0
    out = []
    for extra in extra_steps:
        for angle in angles:
            theta = np.radians(angle)
            cos, sin = np.cos(theta), np.sin(theta)
            reach = (half_w * abs(cos) + half_h * abs(sin) + padding + extra)
            out.append((reach * cos, reach * sin))
    return out


def Place(ax, points, texts, colour, fontsize, segments, blocks, placed,
          angles, padding=None, extra_steps=(0.0, 3.0, 6.0, 10.0, 14.0, 19.0,
                                            25.0, 32.0, 40.0, 50.0, 62.0,
                                            76.0), fallback=(),
          fallback_steps=None, shifts=None, colours=None, firsts=None):
    """Lay out one series of labels.

    Each label is tried at every candidate offset until one clears the curves
    (`segments`), the markers and bars (`blocks`) and the labels already placed
    (`placed`).  The search is in display coordinates, where an offset is the
    same number of pixels whatever the axis scale.  When nothing is clear the
    offset with the least overlap is kept, so a label is never dropped.

    `angles` are the directions the panel wants and `fallback` the rest.  Every
    candidate of `angles` -- all of them -- is tried before any candidate of
    `fallback`, so a panel that asks for its numbers straight above or below
    their points gets them there wherever there is room, however far up or down
    that is, and only sends a number off to the side when its own column has
    none.  Merging the two lists into one and letting the nearest candidate win
    would not do this: a diagonal a few pixels away would beat a vertical
    position a few more, which is the arrangement being avoided.

    `extra_steps` is the ladder the wanted directions climb and
    `fallback_steps` the shorter one the rest are held to -- a label put off to
    the side is a last resort, and should not wander far from its point while it
    is at it.  `padding` is the clearance a label keeps from whatever it stands
    beside; left unset it is the panel-wide `PADDING`.

    `shifts` is a per-label (dx, dy) in display pixels added to the point a
    label is anchored on, before any candidate offset is tried, so that it moves
    the whole ladder of candidates rather than one position on it.  It is what
    pulls a label off the centre of its own bar and holds it off: the search
    alone puts a label wherever it first clears the ink, which for the two bars
    whose configuration is written as a pair is a place chosen by the crowding,
    not by the reader.  Left unset every label is anchored on its point.

    `colours` is the per-label ink, one entry per point, for the panels that
    pick two elements out in red: their numbers are written in the same red as
    the bar or the marker they name, so that the number is read as belonging to
    that series.  Left unset every label takes `colour`.

    `firsts` is one offset per label, in display pixels, tried before the whole
    ladder of `angles`.  `shifts` moves the anchor and with it every candidate,
    which is not the same thing: the label still lands wherever the ladder first
    clears the ink, so a requested offset is only honoured when the search
    happens to choose the direction it was requested in.  A label whose own
    column is blocked -- the pair of a bar that stands beside a taller bar --
    never gets that direction, and its offset comes out of the collision rather
    than out of the request.  `firsts` asks for the place itself: the offset's
    width is held and only the rise is left to the search, so the label stands
    where it was asked to as long as any height above its point is clear, and
    the ladder takes over only when no height is.
    """
    if padding is None:
        padding = PADDING
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer).extents
    kept, artists = [], []
    dropped = []
    for index, (point, label) in enumerate(zip(points, texts)):
        if point is None or label is None:
            continue
        anchor = ax.transData.transform(point)
        if shifts is not None:
            anchor = (anchor[0] + shifts[index][0], anchor[1] + shifts[index][1])

        # the size of the label, measured once, so that every candidate is a box
        # of the same size slid to a new place rather than a new text artist
        probe = ax.text(anchor[0], anchor[1], label, ha="center", va="center",
                        fontsize=fontsize, color=colour, zorder=6)
        probe.set_transform(mtransforms.IdentityTransform())
        width, height = probe.get_window_extent(renderer).size
        probe.remove()

        wanted = Candidates(width, height, angles, padding, extra_steps)
        if fallback:
            # the fallback directions come after every wanted one, at every
            # step, and are held to a shorter ladder of their own
            wider = list(angles) + [a for a in fallback
                                    if a not in set(angles)]
            wanted += Candidates(width, height, wider, padding,
                                 extra_steps if fallback_steps is None
                                 else fallback_steps)
        if firsts is not None and firsts[index] is not None:
            # the requested place, ahead of the whole ladder: the requested
            # offset held fixed while the rise climbs, in small steps, to well
            # above anything a label of this panel has to clear.  Only the
            # height is left to the search, so that the label stands where it
            # was asked to stand as long as any height above the point is
            # clear, instead of the offset coming out of the collision.  The
            # rise goes further than the panel's own ladder because the place
            # requested is not always one the ladder would have offered: a
            # label asked for a column it does not sit in the middle of has to
            # rise past whatever stands in that column, and the steps it needs
            # are not the steps of the centred case.
            fx, fy = firsts[index]
            wanted = ([(fx, fy + s) for s in np.arange(0.0, RISE, STEP)]
                      + wanted)
        best, best_cost = None, None
        for dx, dy in wanted:
            box = (anchor[0] + dx - width / 2.0, anchor[1] + dy - height / 2.0,
                   anchor[0] + dx + width / 2.0, anchor[1] + dy + height / 2.0)
            # a label that would leave the panel is not a candidate at all
            if (box[0] < axes_box[0] or box[2] > axes_box[2]
                    or box[1] < axes_box[1] or box[3] > axes_box[3]):
                continue
            padded = (box[0] - padding, box[1] - padding,
                      box[2] + padding, box[3] + padding)
            cost = 0.0
            for path, ink in segments:
                probe = (padded[0] - ink, padded[1] - ink,
                         padded[2] + ink, padded[3] + ink)
                for i in range(len(path) - 1):
                    if SegmentHitsRect(path[i], path[i + 1], probe):
                        cost += 1e6
                        break
            for block in blocks:
                cost += RectOverlap(padded, block)
            for other in placed + kept:
                cost += RectOverlap(padded, other)
            if cost == 0.0:
                best, best_cost = box, cost
                break
            if best_cost is None or cost < best_cost:
                best, best_cost = box, cost
        if best is None:
            # a measured point with nowhere to put its label: a fault, not a
            # tidy omission, so it is said out loud rather than left off
            dropped.append(label)
            continue
        artist = ax.text((best[0] + best[2]) / 2.0, (best[1] + best[3]) / 2.0,
                         label, ha="center", va="center", fontsize=fontsize,
                         color=colour if colours is None else colours[index],
                         zorder=6)
        artist.set_transform(mtransforms.IdentityTransform())
        # the data point this label belongs to, kept for the audits, which need
        # to say which element a stray label came from
        artist.anchor_data = point
        kept.append(best)
        placed.append(best)
        artists.append(artist)
    if dropped:
        sys.stderr.write("warning: %d label(s) placed nowhere: %s\n"
                         % (len(dropped), ", ".join(dropped)))
    return kept, artists


def Audit(fig, labels):
    """Rasterise the figure twice, once with the value labels alone and once
    with the curves and bars alone, and count the pixels they share.

    The placement above decides overlap from geometry.  This decides it from the
    drawing, so a curve that was drawn thicker than its path, or a label whose
    ink reaches past the box it was measured in, is caught rather than assumed
    away.

    Only the value labels count as text, and only the curves and bars count as
    strokes.  The panel letters, the titles, the ticks, the axis labels and the
    Eu/Yb shading belong to neither: the first four are not what the search
    places, and the shading is a background band a label is meant to sit on.
    """
    axes = list(fig.axes)
    labels = list(labels)
    bands = [patch for ax in axes for patch in ax.patches
             if mcolors.to_hex(patch.get_facecolor()) == mcolors.to_hex(HIGHLIGHT)]
    # every glyph of the figure, so that the stroke mask can be emptied of all
    # lettering including the letters drawn on the axes
    glyphs = [t for t in fig.findobj(mtext.Text)
              if not any(t is label for label in labels)]
    # every artist the masks below touch, with the visibility it had before they
    # touched it.  The patches have to be recorded as carefully as the lines.
    # The twin axis of panel (a) is an axes in its own right whose patch is
    # invisible, and an axes drawn after another paints over it: a patch left
    # visible rather than put back the way it was covers the whole of the ICOBI
    # series behind the -ICOHP one, which costs the panel its curve, its
    # markers and both its sets of numbers
    strokes = [(ax, [(artist, artist.get_visible()) for artist in ax.get_lines()],
                [(artist, artist.get_visible()) for artist in ax.patches])
               for ax in axes]
    frames = [(ax.patch, ax.patch.get_visible()) for ax in axes]
    drawn = [(t, t.get_visible()) for t in fig.findobj(mtext.Text)]
    axes_on = [(ax, ax.axison) for ax in axes]

    def Raster(show_text):
        for ax in axes:
            ax.set_axis_off()
            ax.patch.set_visible(False)
        for ax, lines, patches in strokes:
            for line, _ in lines:
                line.set_visible(not show_text)
            for patch, _ in patches:
                patch.set_visible(not show_text and patch not in bands)
        for glyph in glyphs:
            glyph.set_visible(False)
        for label in labels:
            label.set_visible(show_text)
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
        return buffer.sum(axis=2) < 720      # ink is anything not near-white

    text_mask = Raster(True)
    stroke_mask = Raster(False)

    # put the artists back exactly as they were: the masks above are a
    # measurement, not a state the figure is left in
    for ax, was_on in axes_on:
        ax.set_axis_on() if was_on else ax.set_axis_off()
    for patch, was_visible in frames:
        patch.set_visible(was_visible)
    for artist, was_visible in drawn:
        artist.set_visible(was_visible)
    for ax, lines, patches in strokes:
        for artist, was_visible in lines + patches:
            artist.set_visible(was_visible)
    fig.canvas.draw()

    # both masks are eroded by one pixel before they are intersected, so that
    # anti-aliased edges -- where a curve's halo and a glyph's halo touch
    # without either being drawn over the other -- are not counted as a
    # collision.  What is counted is a stroke and a glyph sharing ink.
    def Erode(mask):
        return (mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
                & mask[1:-1, :-2] & mask[1:-1, 2:])

    return int(np.logical_and(Erode(text_mask), Erode(stroke_mask)).sum())


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
def Build(frame):
    """The revised four-panel figure.  `frame` is the descriptor table of
    `lanthanide bonding.csv`, indexed by element."""
    def Series(column):
        """One descriptor over `ORDER`, promethium and unusable rows as gaps."""
        if column not in frame.columns:
            return np.full(len(ORDER), np.nan)
        series = frame[column].reindex(ORDER).astype(float)
        if "available" in frame.columns:
            usable = frame["available"].reindex(ORDER).eq(True).fillna(False)
            series = series.where(usable.to_numpy(), np.nan)
        return series.to_numpy()

    suspect = (frame["suspect"].reindex(ORDER).eq(True).fillna(False).to_numpy()
               if "suspect" in frame.columns else np.zeros(len(ORDER), bool))

    def Closed(values):
        """The same series with the promethium slot taken by the mean of its two
        neighbours, for the line only, so that the curve runs unbroken.  No
        marker and no label is drawn there: the segments either side of it are
        joined, and the value itself is not reported."""
        closed = values.copy()
        i = ORDER.index(RADIOACTIVE)
        if 0 < i < len(closed) - 1:
            left, right = closed[i - 1], closed[i + 1]
            if np.isfinite(left) and np.isfinite(right):
                closed[i] = 0.5 * (left + right)
        return closed

    def Bands(ax):
        """The Eu/Yb band of the old figure.  The radioactive slot is left
        unshaded: only its tick label is red."""
        for metal in ("Eu", "Yb"):
            i = ORDER.index(metal)
            ax.axvspan(i - 0.5, i + 0.5, color=HIGHLIGHT, alpha=BAND_ALPHA, lw=0,
                       zorder=0)

    def Style(ax, ylabel):
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", alpha=0.5, color="grey")
        ax.set_xlim(-0.5, len(ORDER) - 0.5)
        ax.set_xticks(pos)
        ax.set_xticklabels(ORDER, fontsize=TICK_SIZE)
        for label, metal in zip(ax.get_xticklabels(), ORDER):
            if metal == RADIOACTIVE:
                label.set_color(RED)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_SIZE)

    def Limits(ax, arrays, frac=0.16, zero=False, band=None):
        """Pin the axis range.

        `band` is the fraction of the axis the data is to occupy, as a pair
        `(bottom, top)`.  Given it, the range is set so that the smallest value
        lands at `bottom` and the largest at `top`, which is how the two series
        of the bond order panel are held apart: the ICOBI curve is drawn low and
        the -ICOHP curve high, so that a number put under the one has an empty
        column under the other.  Without it the data is centred with a fraction
        `frac` of its own span as padding on each side.
        """
        values = np.concatenate([a[np.isfinite(a)] for a in arrays])
        lo, hi = float(values.min()), float(values.max())
        if band is not None:
            low, high = band
            span = ((hi - lo) or 1.0) / (high - low)
            ax.set_ylim(lo - low * span, lo + (1.0 - low) * span)
            return
        pad = (hi - lo) * frac or 0.1
        ax.set_ylim(0.0 if zero else lo - pad, hi + pad)

    def Gradient(ax, x0, y0, x1, y1, c0, c1, lw=1.9, zorder=2, steps=24):
        """A straight stroke whose colour runs from `c0` at one end to `c1` at
        the other, drawn as `steps` short segments of interpolated colour.

        Matplotlib has no gradient line, and a LineCollection would take the
        curve out of `ax.lines`, which is where the label search looks for the
        ink it must clear.  Drawing it as ordinary segments keeps the curve
        where everything else expects to find it, at the cost of a few dozen
        artists on the two segments per red element that use this.
        """
        start, end = np.array(mcolors.to_rgb(c0)), np.array(mcolors.to_rgb(c1))
        t = np.linspace(0.0, 1.0, steps + 1)
        xs = x0 + (x1 - x0) * t
        ys = y0 + (y1 - y0) * t
        for i in range(steps):
            mid = 0.5 * (t[i] + t[i + 1])
            ax.plot(xs[i:i + 2], ys[i:i + 2], "-",
                    color=start + (end - start) * mid, lw=lw,
                    solid_capstyle="butt", zorder=zorder)

    def Curve(ax, values, colour, fill, marker="o", red=None):
        """A marker at every measured element, joined by a line that runs
        through promethium without stopping at it.  The marker is hollow in the
        theme's light tint and outlined in the same deep tone as its curve, so
        that a point reads as a point rather than as a lump on the line.

        `red` names the elements whose marker is taken out of the curve's own
        colour and drawn in the palette's red instead, and the segments either
        side of them are shaded from the one colour to the other, so that the
        stroke arrives at a red point as red and leaves it as red rather than
        changing colour underneath the marker.  The default of `None` draws the
        whole curve in its own colour -- the bond strength panel, whose markers
        and line are red throughout.

        A proxy handle is returned for the legend: the stroke is a run of short
        segments when it carries a gradient, and no one of them stands for the
        curve as a whole.
        """
        red = () if red is None else tuple(red)
        closed = Closed(values)
        for i in range(len(pos) - 1):
            y0, y1 = closed[i], closed[i + 1]
            if not (np.isfinite(y0) and np.isfinite(y1)):
                continue
            c0 = RED if ORDER[i] in red else colour
            c1 = RED if ORDER[i + 1] in red else colour
            if c0 == c1:
                ax.plot(pos[i:i + 2], [y0, y1], "-", color=c0, lw=1.9,
                        zorder=2)
            else:
                Gradient(ax, pos[i], y0, pos[i + 1], y1, c0, c1)
        for x, value, flag, metal in zip(pos, values, suspect, ORDER):
            if not np.isfinite(value):
                continue
            edge, face = (RED, RED_FILL) if metal in red else (colour, fill)
            ax.plot([x], [value], marker="X" if flag else marker, ms=7.5,
                    mfc=face, mec=edge, mew=1.5, color=edge, zorder=3)
        return mlines.Line2D([], [], color=colour, lw=1.9)

    def Points(values, fmt):
        """The (point, label) pairs a panel has to place."""
        return [(None if not np.isfinite(v) else (float(x), float(v)),
                 None if not np.isfinite(v) else fmt % v)
                for x, v in zip(pos, values)]

    def Fitted(ax, texts, fontsize, colour):
        """The size at which the widest of `texts` still fits the slot of one
        column.

        The f panel names every bar and its widest name is wider than the slot
        its bar stands in -- 79 pixels of text in a 70 pixel slot at the size
        the other panels use.  Two neighbouring names can then never stand at
        the same height, and the search, which is not allowed to drop one, is
        left staggering them up and down the panel: that is what made the panel
        unreadable.  Sizing the names to the slot instead of the slot to the
        names keeps them in one row, which is the only arrangement that reads
        across fifteen bars.
        """
        renderer = ax.figure.canvas.get_renderer()
        slot = ax.get_window_extent(renderer).width / len(ORDER)
        widest = 0.0
        for text in texts:
            probe = ax.text(0, 0, text, fontsize=fontsize, color=colour)
            probe.set_transform(mtransforms.IdentityTransform())
            widest = max(widest, probe.get_window_extent(renderer).width)
            probe.remove()
        if widest <= slot - 4.0:
            return fontsize
        return fontsize * (slot - 4.0) / widest

    # the height of the figure, brought down from the old six-panel layout: four
    # panels need less room than six, and the vertical gap the letters are added
    # in by hand is the only reason it is not shorter still
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE, dpi=200)

    # --- the two bond order panels, sharing one panel and a pair of y axes.
    # Each axis is scaled to its own series: the bond energy of the series spans
    # 2.4 to 4.0 eV, nearly four times the span of ICOBI, so a shared range would
    # flatten the ICOBI curve into a straight line.  Held apart this way the two
    # curves still track one another, which is the point of putting them
    # together, but the ICOBI trend is legible on its own.  It stands in the top
    # left corner, where the figure is read from, because the M-N bond is what
    # the argument is about.
    ax = axes[0, 0]
    Bands(ax)
    icobi, icohp = Series("ICOBI M-N"), Series("-ICOHP M-N")
    Style(ax, "M-N bond order (ICOBI)")
    Limits(ax, [icobi], band=ICOBI_BAND)
    icobi_line = Curve(ax, icobi, BLUE, BLUE_FILL)
    ax.set_title("M-N bond order and strength", fontsize=TITLE_SIZE)

    ax2 = ax.twinx()
    ax2.grid(False)
    icohp_line = Curve(ax2, icohp, RED, RED_FILL, marker="s")
    ax2.set_ylabel("M-N bond strength ($-$ICOHP, eV)", fontsize=LABEL_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_SIZE)
    Limits(ax2, [icohp], band=ICOHP_BAND)

    # the legend of the two curves, in the bottom left corner of the panel.  It
    # stands where neither curve goes: the ICOBI lane is drawn above this corner
    # and the -ICOHP lane above that again, so the space below the blue curve is
    # the one part of the panel no curve and no number uses.  The handles are the
    # curves themselves, so the swatches cannot come loose from what they label.
    bond_legend = ax.legend((icobi_line, icohp_line), LEGEND_LABELS,
                            loc="lower left", fontsize=TICK_SIZE,
                            framealpha=0.92, edgecolor="grey",
                            borderpad=0.5, labelspacing=0.4, handlelength=2.0)

    # --- the Bader charge transfer, in the top right corner, beside the bond
    # panel: it is the descriptor that says how much charge the metal gave up,
    # which is the quantity the rest of the panel row acts on
    ax = axes[0, 1]
    Bands(ax)
    transfer = Series("charge transfer/e")
    Style(ax, "Bader charge transfer (e)")
    Limits(ax, [transfer])
    Curve(ax, transfer, BLUE, BLUE_FILL, red=RED_METALS)
    ax.set_title("Bader charge transfer", fontsize=TITLE_SIZE)

    # --- the f shell, a bar chart, in the bottom left corner.  It is drawn in
    # the blue of the palette even though the older figure had it red, because
    # red is the bond strength here and the f shell is a descriptor like the
    # other three.  Each bar is named twice, the 4f configuration of the ion
    # above the integrated occupation it is derived from.  It is the one panel
    # whose labels are allowed to leave the vertical, because fifteen names have
    # to fit fifteen slots
    ax_f = axes[1, 0]
    Bands(ax_f)
    occupation = Series("f occupation")
    finite = occupation[np.isfinite(occupation)]
    top = float(finite.max()) if len(finite) else 1.0
    ax_f.bar(pos, np.nan_to_num(occupation, nan=0.0), width=0.62,
             color=[RED_FILL if m in RED_METALS else BLUE_FILL for m in ORDER],
             edgecolor=[RED if m in RED_METALS else BLUE for m in ORDER],
             linewidth=1.5, zorder=3)
    Style(ax_f, "f shell occupation (electrons)")
    ax_f.set_ylim(0.0, top * 1.40)
    ax_f.set_title("f shell occupation", fontsize=TITLE_SIZE)

    # --- the d band center, in the bottom right corner
    ax = axes[1, 1]
    Bands(ax)
    center = Series("d band center/eV")
    Style(ax, "d band center (eV)")
    Limits(ax, [center])
    Curve(ax, center, BLUE, BLUE_FILL, red=RED_METALS)
    ax.set_title("d band center", fontsize=TITLE_SIZE)

    # the panel letters are not drawn: they are left to be added by hand

    # the layout has to be final before a label is placed, because the search
    # works in display coordinates and a later tight_layout would move every
    # point under the labels already placed on it.
    #
    # The gap between the panels, and between them and the edge of the figure,
    # is `LAYOUT_PAD`, in units where the default of 1.08 is the packed one.  It
    # is deliberately above the default: the two columns are read against each
    # other and the value labels stand in the space above and below the curves,
    # so the panels are given room to be read separately, and the wider margins
    # are also where the panel letters are added by hand
    fig.tight_layout(pad=LAYOUT_PAD)
    fig.canvas.draw()

    # --- the labels, one series at a time, each clear of what is already there
    placed, labels = [], []

    # the directions a label may take around its point.  The vertical pair is
    # tried first and on its own: a number standing at the same x as the point
    # it names is read as that point's value without any other cue, so the
    # panels that have room for it keep their numbers straight above or straight
    # below their markers.  Only a label with no clear vertical position -- one
    # whose column is already taken by a curve or by a neighbouring number --
    # falls back on the diagonals and the shallower angles
    VERTICAL = (90, 270)
    AROUND = (45, 135, 225, 315, 60, 120, 30, 150, 0, 180,
              15, 75, 105, 165, 195, 255, 285, 345)
    # the ladder the vertical directions climb, longer than the default because
    # a panel is tall and a number may have to stand well clear of its point
    # before it is out of its own curve; and the shorter ladder the directions
    # off the vertical are held to, so that a number pushed to the side stays
    # near the point it names
    UPRIGHT = (0.0, 3.0, 6.0, 10.0, 14.0, 19.0, 25.0, 32.0, 40.0, 50.0, 62.0,
               76.0, 92.0, 110.0, 130.0)
    SIDEWAYS = (0.0, 3.0, 6.0, 10.0, 14.0, 19.0, 25.0, 32.0, 40.0, 50.0, 62.0)

    ax_bond = axes[0, 0]
    twin = [a for a in fig.axes if a is not ax_bond and a.bbox.bounds ==
            ax_bond.bbox.bounds]
    segments, blocks = CollectCurves([ax_bond] + twin)
    # the legend of that panel is an obstacle like any other: a number placed on
    # it would be read as part of it, so the search is told to keep off its box
    legend_box = tuple(bond_legend.get_window_extent(
        fig.canvas.get_renderer()).extents)
    blocks = blocks + [legend_box]
    _, added = Place(ax_bond, [p for p, _ in Points(icobi, "%.2f")],
                     [t for _, t in Points(icobi, "%.2f")], BLUE, VALUE_SIZE,
                     segments, blocks, placed,
                     Angles((270, 90)), extra_steps=UPRIGHT,
                     fallback=AROUND, fallback_steps=SIDEWAYS)
    labels += added
    # the -ICOHP series is drawn on the twin axis, so its labels have to be
    # placed through that axis: a value of 3.4 eV transformed by the ICOBI axis
    # of the left side lands nowhere near the curve it belongs to
    _, added = Place(ax2, [p for p, _ in Points(icohp, "%.2f")],
                     [t for _, t in Points(icohp, "%.2f")], RED, VALUE_SIZE,
                     segments, blocks, placed,
                     Angles((90, 270)), extra_steps=UPRIGHT,
                     fallback=AROUND, fallback_steps=SIDEWAYS)
    labels += added

    for panel, values, colour in [(axes[0, 1], transfer, BLUE),
                                 (axes[1, 1], center, BLUE)]:
        segments, blocks = CollectCurves([panel])
        pairs = Points(values, "%.2f")
        # the number of a red element is written in red, like the marker it
        # names, and the ones between two red markers are left blue
        inks = [RED if m in RED_METALS else colour for m in ORDER]
        _, added = Place(panel, [p for p, _ in pairs], [t for _, t in pairs],
                         colour, VALUE_SIZE, segments, blocks, placed,
                         Angles(VERTICAL), extra_steps=UPRIGHT,
                         fallback=AROUND, fallback_steps=SIDEWAYS,
                         colours=inks)
        labels += added

    # --- the f panel: the value sits on the bar, the configuration above it,
    # the second placed after the first so that it is put clear of it.  The
    # value is sized to the slot, because fifteen names do not fit a slot at
    # the size the four-value panels use, and the ladder starts clear of the
    # bar outline: a label put exactly on the bar top still shares its ink with
    # the drawn edge, so the first candidate is never clear and every label
    # drifts out to whatever step it happens to clear first
    segments, blocks = CollectCurves([ax_f])
    pairs = Points(occupation, "%.2f")
    fsize = Fitted(ax_f, [t for _, t in pairs], VALUE_SIZE, BLUE)
    # the occupation of a red bar is written in red, like the bar itself
    _, added = Place(ax_f, [p for p, _ in pairs], [t for _, t in pairs], BLUE,
                     fsize, segments, blocks, placed,
                     Angles((90, 45, 135), (270, 225, 315, 0, 180, 60, 120)),
                     padding=F_PADDING, extra_steps=BAR_STEPS,
                     colours=[RED if m in RED_METALS else BLUE for m in ORDER])
    labels += added
    conf_pairs = [(p, None if p is None
                   else (r"$%s$" % F_SHELL_PAIR[m]
                         if m in F_SHELL_PAIR
                         else r"$\mathrm{f}^{%s}$" % F_SHELL[m]))
                  for (p, _), m in zip(pairs, ORDER)]
    conf_firsts = [F_PAIR_PLACE.get(m) for m in ORDER]
    # the configuration of a red ion is written in red as well, like the bar it
    # names and like the occupation printed under it
    conf_inks = [RED if m in RED_METALS else BLUE for m in ORDER]
    _, added = Place(ax_f, [p for p, _ in conf_pairs],
                     [t for _, t in conf_pairs], BLUE,
                     FCONF_SIZE, segments, blocks, placed,
                     Angles((90, 45, 135), (270, 225, 315, 0, 180, 60, 120)),
                     padding=F_PADDING, extra_steps=BAR_STEPS,
                     firsts=conf_firsts, colours=conf_inks)
    labels += added
    return fig, labels


def Draw(frame, out_path):
    """Build the figure, check the labels against the pixels, and save it."""
    fig, labels = Build(frame)
    shared = Audit(fig, labels)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path, shared


# reading the VASP output
# ---------------------------------------------------------------------------
def EnsureSplit(dos_dir):
    """Split the DOSCAR of a directory into DOS0..DOSn, once.

    `GetBandCenter` and the occupation integrals read the split form, which is
    the convention the rest of the study uses; the files are small next to the
    WAVECAR of the same directory.
    """
    if not os.path.exists(os.path.join(dos_dir, "DOS1")):
        SplitDOSCAR(dos_dir)
    return dos_dir


def BandCenter(dos_dir, orbital):
    """(center, width) of one orbital of the metal, in eV relative to E_F."""
    try:
        center, width = GetBandCenter(dos_dir=dos_dir, idx=M_IDX,
                                      orbital=orbital, e_range=list(E_WINDOW))
    except Exception:
        return np.nan, np.nan
    return float(center), float(width)


def Occupation(dos_dir, orbital):
    """Electrons in one orbital of the metal, up to the Fermi level.

    The occupied integral of the spin-summed PDOS, which is the f count the
    argument is stated in: 7 for Eu and 14 for Yb.  The integral is taken over
    `OCC_WINDOW`, so a spurious band left far below the valence states by the
    +U treatment is not counted; without the lower bound Ho integrates to 15.4
    f electrons instead of 10.2.
    """
    doscar = SplitDoscar(dos_dir=dos_dir, ispin=2)
    energy = doscar.energy - doscar.efermi
    pdos = doscar.pdos_sum([M_IDX - 1], l=orbital)
    mask = (energy >= OCC_WINDOW[0]) & (energy <= OCC_WINDOW[1])
    if mask.sum() < 2:
        return np.nan
    return float(np.trapz(pdos[mask], energy[mask]))


def ReadPotcarZval(potcar_path):
    """Valence electron count of the metal, the last ZVAL of the POTCAR.

    The POTCAR is ordered C, N, metal, so the metal is the last entry; reading
    the last ZVAL rather than the first keeps that true whatever the order of
    the species inside one element block.
    """
    values = []
    if not os.path.isfile(potcar_path):
        return np.nan
    with open(potcar_path, errors="ignore") as fh:
        for line in fh:
            if "ZVAL" in line:
                match = re.search(r"ZVAL\s*=\s*([0-9.]+)", line)
                if match:
                    values.append(float(match.group(1)))
    return values[-1] if values else np.nan


def ReadBaderCharge(acf_path, ion):
    """Bader electron count of one ion, from an ACF.dat."""
    if not os.path.isfile(acf_path):
        return np.nan
    with open(acf_path, errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    if int(parts[0]) == ion:
                        return float(parts[4])
                except ValueError:
                    continue
    return np.nan


def LobsterLabel(label):
    """The atom index behind a LOBSTER label, or None for an orbital row.

    LOBSTER 6 writes labels rather than bare indices: `N45` is atom 45, while
    the orbital-resolved rows that `orbitalWise` adds are written `N45_2s`,
    `Eu49_4f_z^3` and so on.  Only the atom-wise rows, without an orbital
    suffix, carry the total of a pair, so the suffixed ones are recognised and
    skipped: their sum would be the same interaction counted many times over.
    """
    if "_" in label:
        return None
    digits = ""
    for char in reversed(label):
        if char.isdigit():
            digits = char + digits
        else:
            break
    return int(digits) if digits else None


def ReadLobsterList(path):
    """The atom-wise bond rows of an ICOHPLIST / ICOOPLIST / ICOBILIST.

    Returns a list of (atom1, atom2, cell, length, value), the value spin
    summed.  A row of LOBSTER 6 is

        index  atom1  atom2  length  t_x t_y t_z  value_up  value_down

    where the atoms are labels (`N45`, `Eu49`), the three integers between the
    length and the values are the cell of the second atom, and the two value
    columns are the spin channels.  The orbital-resolved rows interleave with
    the atom-wise ones and are skipped, the atom-wise row being their sum.

    The two-block layout of LOBSTER 3.1.1, where the spin-down values are a
    second block of the same bonds instead of a second column, is still folded:
    a bond and cell seen twice is that signature, and the pair is then summed.
    In 6.0 every bond appears once and nothing folds.
    """
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 8:
                continue
            atom1, atom2 = LobsterLabel(parts[1]), LobsterLabel(parts[2])
            if atom1 is None or atom2 is None:
                continue
            try:
                length = float(parts[3])
                cell = tuple(int(float(t)) for t in parts[4:7])
            except ValueError:
                continue
            values = []
            for token in parts[7:]:
                try:
                    values.append(float(token))
                except ValueError:
                    pass
            if values:
                rows.append((atom1, atom2, cell, length, sum(values)))

    keyed = {}
    for row in rows:
        keyed.setdefault((row[0], row[1], row[2]), []).append(row)
    if rows and all(len(group) == 2 for group in keyed.values()):
        rows = [(a, b, cell, length, up + down)
                for (a, b, cell), [(_, _, _, length, up), (_, _, _, _, down)]
                in keyed.items()]
    return rows


def LobsterBondValues(directory):
    """The M-N bond order of one system, from whatever LOBSTER wrote.

    Returns a dict with the mean ICOBI (bond order), the mean ICOHP (bond
    strength, whose sign is flipped so that a larger number is a stronger
    bond), and the per-bond values behind them.  Empty when LOBSTER has not
    been run in the directory.
    """
    out = {"ICOBI": np.nan, "ICOHP": np.nan, "n_bonds": 0, "source": []}
    for key, name in [("ICOBI", "ICOBILIST.lobster"),
                      ("ICOHP", "ICOHPLIST.lobster")]:
        path = os.path.join(directory, name)
        rows = ReadLobsterList(path)
        if not rows:
            continue
        values = []
        for atom1, atom2, _cell, _length, value in rows:
            pair = {atom1, atom2}
            if M_IDX in pair and pair & set(N_IDX):
                values.append(value)
        if not values:
            continue
        # ICOHP of a bonding state is negative, so the sign is flipped to make
        # the quantity a bond strength that grows upwards like the others
        out[key] = float(-np.mean(values)) if key == "ICOHP" else float(np.mean(values))
        out["n_bonds"] = len(values)
        out["source"].append(name)
    return out


def LobsterMetalBasis(directory, metal):
    """The shells LOBSTER projected the metal onto, as read from `lobsterout`.

    LOBSTER lists the basis it used as `<element> (basisSet) <shells>`, with the
    functions written orbital by orbital (`5d_xy`, `4f_z^3`, ...), so the shells
    are the part before the first underscore.  This is the quantity that says
    whether two elements of the series can be compared at all: with a 5d
    function the metal d states can form part of the bond order, without it they
    cannot, and the M-N bond is carried by 5d.
    """
    path = os.path.join(directory, "lobsterout")
    if not os.path.isfile(path):
        return set()
    shells = set()
    with open(path, errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == metal and "pbevaspfit" in parts[1]:
                for token in parts[2:]:
                    shells.add(token.split("_")[0].rstrip(","))
    return shells


# ---------------------------------------------------------------------------
# the sanity check on the VASP output itself
# ---------------------------------------------------------------------------
def ReadEigenvalRange(path):
    """(lowest, highest) Kohn-Sham eigenvalue of an EIGENVAL, in eV.

    Every band at every k point is listed as `index energy occupation`, behind
    a header and one k-point line per k point, so the two counts in the header
    are enough to walk the file.  The lowest eigenvalue is the quantity that
    exposes a run whose states do not belong to the same Hamiltonian as those
    of its neighbours: in this series it falls smoothly from La to Lu, and a
    single element far below that trend cannot be a chemical result.
    """
    if not os.path.isfile(path):
        return np.nan, np.nan
    tokens = [line.split() for line in open(path, errors="ignore")
              if line.strip()]
    if len(tokens) < 7:
        return np.nan, np.nan
    try:
        nkpts, nbands = int(tokens[5][1]), int(tokens[5][2])
    except (IndexError, ValueError):
        return np.nan, np.nan

    i, energies = 6, []
    for _ in range(nkpts):
        i += 1                                  # the k-point line
        for _ in range(nbands):
            if i >= len(tokens):
                break
            try:
                energies.append(float(tokens[i][1]))
            except (IndexError, ValueError):
                pass
            i += 1
    if not energies:
        return np.nan, np.nan
    return float(np.min(energies)), float(np.max(energies))


def ReadDoscarWindow(path):
    """(E_max, E_min) of the energy grid a DOSCAR was projected on, in eV.

    A window far wider than the one of the neighbouring elements means the
    projection carried states the others do not have, which is the second sign
    of a run that does not belong to the series.
    """
    if not os.path.isfile(path):
        return np.nan, np.nan
    with open(path, errors="ignore") as fh:
        line = ""
        for _ in range(6):
            line = fh.readline()
    parts = line.split()
    try:
        return float(parts[0]), float(parts[1])
    except (IndexError, ValueError):
        return np.nan, np.nan


def DoscarIsComplete(path):
    """Whether a DOSCAR holds its whole projection, header to last row.

    A DOSCAR is a header, one total-DOS block, then one block of NEDOS rows per
    atom, so its line count is fixed by the two numbers in its own header.  The
    count is what tells a finished projection from one that is still being
    written by a job running in the same directory, or from the total-only
    DOSCAR that a run without LORBIT leaves behind; reading either would either
    crash or silently take a partial file for a result.
    """
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        return False
    try:
        with open(path, errors="ignore") as fh:
            lines = fh.readlines()
        natom = int(lines[0].split()[0])
        nedos = int(lines[5].split()[2])
    except (IndexError, ValueError):
        return False
    # the header, the total-DOS block, then one header line plus NEDOS rows for
    # each of the NIONS atoms, with the two counts read from the header itself
    if len(lines) != 6 + nedos + natom * (nedos + 1):
        return False
    try:
        [float(v) for v in lines[-1].split()]
    except ValueError:
        return False
    return True


def LanthanideBonding(redraw=False):
    """The four descriptors of the fourteen lanthanides, the CSV, the
    report and the figure."""
    if redraw:
        frame = pd.read_csv(os.path.join(
            SCRIPT_DIR, "lanthanide bonding.csv")).set_index("metal")
        print("rows read from the CSV: %d" % len(frame))
        print("available             : %d"
              % int(frame["available"].sum()))
        written, shared = Draw(frame, os.path.join(
            FIG_DIR, "Lanthanide bonding.png"))
        print("labels sharing pixels with a curve: %d" % shared)
        print("written to %s" % written)
        return
    print("Reading the lanthanide bonding descriptors...")

    def PickSource(folder, metal, kind, name, valid):
        """(directory, label) to read one artifact of one system from.

        The working directory is preferred, because it is where LOBSTER ran and it
        holds the state being analysed.  When the copy there is missing, or is there
        but not whole, the curated copy of the data set is used instead, and the
        choice is reported: the two are the same run for every element that did not
        have to be repeated, so a fallback changes which files are read, not which
        system is described.  A file a running job has only half written is the case
        that matters, which is why the test is `valid` and not a size.
        """
        live = os.path.join(folder, name)
        if os.path.isfile(live) and valid(live):
            return folder, "working copy"
        ref = os.path.join(CURATED, kind, "%s-N-C" % metal)
        if os.path.isfile(os.path.join(ref, name)) and valid(os.path.join(ref, name)):
            return ref, "data set"
        return folder, "absent"


    NonEmpty = lambda path: os.path.getsize(path) > 0

    rows = []
    for metal in LANTHANIDES:
        folder = os.path.join(WORK, "%s-N-C" % metal)

        # the electronic structure and the geometry are picked separately: a job
        # that aborted may have left one of them usable and the other not
        dos_dir, dos_src = PickSource(folder, metal, "dos", "DOSCAR",
                                      DoscarIsComplete)
        geom_dir, geom_src = PickSource(folder, metal, "energy", "CONTCAR", NonEmpty)

        row = {"metal": metal, "available": True, "note": "", "source": ""}
        if dos_src == "absent" or geom_src == "absent":
            row.update({"available": False,
                        "note": "VASP output absent, truncated, or still being "
                                "written by a job running in that directory"})
            rows.append(row)
            print("  %-3s incomplete, skipped" % metal)
            continue

        if dos_src != "working copy" or geom_src != "working copy":
            row["source"] = "DOS from the %s, geometry from the %s" % (dos_src,
                                                                       geom_src)
            if dos_src != "working copy":
                row["note"] = "no complete DOSCAR in the working directory"

        # --- electronic structure, from the DOSCAR -----------------------------
        EnsureSplit(dos_dir)
        for orbital, tag in [("d", "d"), ("f", "f")]:
            center, width = BandCenter(dos_dir, orbital)
            row["%s band center/eV" % tag] = center
            row["%s band width/eV" % tag] = width
            row["%s occupation" % tag] = Occupation(dos_dir, orbital)

        # --- geometry, from CONTCAR -------------------------------------------
        # a CONTCAR is only written at the end of a run, so a partial one means a
        # job is still going in that directory; it is reported rather than read
        contcar = os.path.join(geom_dir, "CONTCAR")
        try:
            row["d_M-N/angstrom"] = float(np.mean(
                [GetDistance(contcar, n, M_IDX) for n in N_IDX]))
            row["d_M-N max/angstrom"] = float(np.max(
                [GetDistance(contcar, n, M_IDX) for n in N_IDX]))
            row["angle mean/degree"] = float(np.mean(
                [GetAngle(contcar, a, M_IDX, b) for a, b in TRANS_PAIRS]))
            row["out of plane/angstrom"] = GetOutOfPlane(contcar, idx_M=M_IDX,
                                                         idx_N=N_IDX)
        except Exception as exc:
            row.update({"available": False,
                        "note": "CONTCAR could not be read: %s" % exc})
            rows.append(row)
            print("  %-3s CONTCAR unreadable, skipped" % metal)
            continue

        # --- Bader charge, from the partition of CHGCAR ------------------------
        zval = ReadPotcarZval(os.path.join(geom_dir, "POTCAR"))
        acf = os.path.join(folder, "ACF.dat")
        if not os.path.isfile(acf):
            acf = os.path.join(CURATED, "energy", "%s-N-C" % metal, "ACF.dat")
        q_bader = ReadBaderCharge(acf, M_IDX)
        row["ZVAL"] = zval
        row["q_Bader/e"] = q_bader
        row["charge transfer/e"] = (zval - q_bader
                                    if np.isfinite(zval) and np.isfinite(q_bader)
                                    else np.nan)
        if not np.isfinite(q_bader):
            row["note"] = "no ACF.dat, run `run bader lanthanides.sh`"

        # --- bond order, from LOBSTER ------------------------------------------
        lobster = LobsterBondValues(folder)
        row["ICOBI M-N"] = lobster["ICOBI"]
        row["-ICOHP M-N"] = lobster["ICOHP"]
        row["Lobster bonds"] = lobster["n_bonds"]
        if not lobster["source"]:
            row["note"] = (row["note"] + "; " if row["note"] else "") + \
                "no LOBSTER output, run `run_lobster.sh`"

        # the basis LOBSTER actually projected on, and the check that matters more
        # than the number it produced: if the metal has no 5d function in its basis,
        # the bond order is not comparable with the rest of the series, whatever its
        # value is, so it is withheld and said so rather than plotted.  LOBSTER on
        # its own gives a 5d function only to the elements whose POTCAR reference
        # configuration occupies 5d, which in this series is La, Ce, Gd and Lu.
        shells = LobsterMetalBasis(folder, metal)
        row["Lobster basis"] = " ".join(sorted(shells))
        if np.isfinite(row["ICOBI M-N"]) and not any(s.startswith("5d")
                                                     for s in shells):
            row["ICOBI M-N"] = np.nan
            row["-ICOHP M-N"] = np.nan
            row["note"] = (row["note"] + "; " if row["note"] else "") + \
                ("the LOBSTER basis of the metal has no 5d function (%s), so its "
                 "bond order is not comparable with the rest; state the basis in "
                 "lobsterin and run it again"
                 % (row["Lobster basis"] or "none recorded"))

        # --- the sanity check on the run itself --------------------------------
        # the two numbers that tell a run of this series from one that is not:
        # where its lowest band sits, and how wide a window the projection needed
        eig = os.path.join(folder, "EIGENVAL")
        if not os.path.isfile(eig):
            eig = os.path.join(dos_dir, "EIGENVAL")
        row["lowest band/eV"], row["highest band/eV"] = ReadEigenvalRange(eig)
        emax, emin = ReadDoscarWindow(os.path.join(dos_dir, "DOSCAR"))
        row["DOSCAR E_max/eV"] = emax
        row["DOSCAR E_min/eV"] = emin
        row["DOSCAR span/eV"] = (emax - emin
                                 if np.isfinite(emax) and np.isfinite(emin)
                                 else np.nan)

        rows.append(row)
        print("  %-3s d center %7.3f eV   d_M-N %6.3f A   q transfer %6.3f e"
              % (metal, row["d band center/eV"], row["d_M-N/angstrom"],
                 row["charge transfer/e"]))

    frame = pd.DataFrame(rows).set_index("metal")


    def FlagSuspects(frame):
        """Which rows cannot be trusted, and why.

        Two tests, both of them against the series rather than against a constant
        that would have to be justified:

          * the f shell of a lanthanide holds at most 14 electrons, so a larger
            occupied integral is an artefact of the projection and not a result;
          * the lowest Kohn-Sham eigenvalue falls smoothly across the series, so an
            element sitting many median-absolute-deviations below its neighbours
            was not computed from the same Hamiltonian.  The bound is loose on
            purpose, because the lanthanide contraction does lower the value
            gradually from La to Lu and that part of the trend is real.

        Returns (suspect, reason), both indexed like the frame.
        """
        suspect = pd.Series(False, index=frame.index)
        reason = pd.Series("", index=frame.index)
        ok = frame[frame["available"]]

        # the f shell of a lanthanide holds at most 14 electrons
        over = ok["f occupation"] > F_OCC_MAX + F_OCC_TOL
        for metal in over[over].index:
            reason[metal] = ("f occupation %.2f, past the %d the shell holds"
                             % (ok.loc[metal, "f occupation"], F_OCC_MAX))
        suspect[over[over].index] = True

        # the lowest band, against the median of the series
        lowest = ok["lowest band/eV"].dropna()
        if len(lowest) >= 5:
            median = float(lowest.median())
            mad = float((lowest - median).abs().median())
            if mad > 0:
                far = lowest < median - OUTLIER_MAD * mad
                for metal in far[far].index:
                    suspect[metal] = True
                    reason[metal] = (reason[metal] + "; " if reason[metal] else "") + \
                        ("lowest band %.1f eV, %.0f eV below the series"
                         % (lowest[metal], median - lowest[metal]))
        return suspect, reason


    suspect, suspect_reason = FlagSuspects(frame)
    frame["suspect"] = suspect
    frame["suspect reason"] = suspect_reason

    CSV = os.path.join(SCRIPT_DIR, "lanthanide bonding.csv")
    frame.to_csv(CSV)

    # ---------------------------------------------------------------------------
    # the report
    # ---------------------------------------------------------------------------
    report = open(os.path.join(SCRIPT_DIR, "lanthanide bonding report.txt"),
                  "w+", encoding="utf-8")


    def Write(text=""):
        print(text)
        report.write(text + "\n")


    Write("=" * 78)
    Write("Lanthanide bonding descriptors")
    Write("=" * 78)
    Write()
    Write("The f shell of Eu (f7) and Yb (f14) is half filled and filled, and the")
    Write("argument of the manuscript is that this shell screens the 5d orbitals and")
    Write("weakens the M-N bond.  Four quantities test that: the d band center, the")
    Write("M-N bond length, the Bader charge transfer of the metal, and the M-N bond")
    Write("order of the LOBSTER projection, reported both as ICOBI (bond order) and")
    Write("as -ICOHP (covalent bond strength).")
    Write()
    Write("A weaker M-N bond shows up as a smaller bond order, a shorter M-N")
    Write("distance that relaxes back out, a smaller charge transfer, and a d band")
    Write("center that moves towards the Fermi level; the test is whether Eu and Yb")
    Write("sit at an extremum of the series on those four axes.")

    available = frame[frame["available"]]
    # the statistics and the figure are run on the rows that pass the integrity
    # check, so that one anomalous run cannot move the comparison
    included = available[~available["suspect"]]

    Write()
    Write("-" * 78)
    Write("The series")
    Write("-" * 78)
    Write()
    Write("  %-2s %-4s %9s %9s %9s %9s %9s %9s"
          % ("", "M", "d center", "f center", "d_M-N", "q trans", "ICOBI", "-ICOHP"))
    for metal, row in available.iterrows():
        Write("  %-2s %-4s %9.3f %9.3f %9.4f %9.3f %9s %9s"
              % ("!" if row["suspect"] else "", metal,
                 row["d band center/eV"], row["f band center/eV"],
                 row["d_M-N/angstrom"], row["charge transfer/e"],
                 ("%.4f" % row["ICOBI M-N"]) if np.isfinite(row["ICOBI M-N"]) else "pending",
                 ("%.4f" % row["-ICOHP M-N"]) if np.isfinite(row["-ICOHP M-N"]) else "pending"))
    Write()
    Write("  a leading ! marks a row that failed the integrity check described")
    Write("  below; it is shown here for completeness but left out of every")
    Write("  statistic.")
    Write()
    Write("  d center and f center are the first moments of the metal d and f PDOS")
    Write("  over %.0f to %.0f eV around the Fermi level, the window of the rest of"
          % E_WINDOW)
    Write("  the study.  d_M-N is the mean of the four metal-nitrogen distances, q")
    Write("  trans the Bader charge the metal has given up, ZVAL - q_Bader.")

    # the f count, which is what the argument is stated in
    Write()
    Write("  %-4s %12s %12s %12s %12s" % ("M", "d occ", "f occ", "ZVAL", "q_Bader"))
    for metal, row in available.iterrows():
        Write("  %-4s %12.3f %12.3f %12.1f %12.3f"
              % (metal, row["d occupation"], row["f occupation"], row["ZVAL"],
                 row["q_Bader/e"]))
    Write()
    Write("  the f occupation is the integral of the f PDOS of the metal from %.0f eV"
          % OCC_WINDOW[0])
    Write("  up to the Fermi level, and is the direct measure of the filling the")
    Write("  argument turns on: it rises through the series and is 7 at Eu and 14 at")
    Write("  Yb.  The integral is bounded at the bottom on purpose, because a +U run")
    Write("  can leave a spurious band far below the valence states that an open")
    Write("  integral would count.")

    # --- data integrity --------------------------------------------------------
    Write()
    Write("-" * 78)
    Write("Data integrity")
    Write("-" * 78)
    Write()
    Write("  Two tests, both of them against the series itself rather than against a")
    Write("  constant that would have to be justified.  A row that fails either test")
    Write("  is marked ! in the tables and left out of every statistic.")
    Write()
    Write("  %-4s %10s %13s %12s %10s" % ("M", "f occ", "lowest band", "DOS span", "flag"))
    for metal, row in available.iterrows():
        Write("  %-4s %10.3f %13.1f %12.1f %10s"
              % (metal, row["f occupation"], row["lowest band/eV"],
                 row["DOSCAR span/eV"], "suspect" if row["suspect"] else "ok"))

    lowest_all = available["lowest band/eV"].dropna()
    if len(lowest_all) >= 5:
        Write()
        Write("  the lowest band across the series: median %.1f eV, median absolute"
              % lowest_all.median())
        Write("  deviation %.2f eV; the flag is set %g deviations below the median."
              % ((lowest_all - lowest_all.median()).abs().median(), OUTLIER_MAD))

    flagged = available[available["suspect"]]
    Write()
    if len(flagged):
        Write("  rows left out of the statistics:")
        for metal, row in flagged.iterrows():
            Write("    %-4s %s" % (metal, row["suspect reason"]))
        Write()
        Write("  a flag is not a small error to be interpreted around: a projection")
        Write("  whose lowest band sits 100 eV below its neighbours, or whose f shell")
        Write("  integrates past the 14 electrons it can hold, is describing a")
        Write("  different Hamiltonian.  The run is one to repeat, and the chain that")
        Write("  repeats it is the one the rest of the data set used: an SCF under the")
        Write("  same +U settings, then the DOS from the state that SCF leaves behind.")
    else:
        Write("  every available row passed both tests.")

    # --- the double-double effect, tested element by element -------------------
    # the rows that failed the integrity check are left out of this comparison, so
    # that one anomalous run cannot move it
    special = [m for m in ["Eu", "Yb"] if m in included.index]
    rest = [m for m in included.index if m not in special]

    Write()
    Write("-" * 78)
    Write("Eu and Yb against the rest of the series")
    Write("-" * 78)
    Write()
    columns = ["d band center/eV", "d_M-N/angstrom", "charge transfer/e",
               "ICOBI M-N", "-ICOHP M-N", "d occupation", "f occupation"]
    Write("  %-34s %13s %13s %10s" % ("descriptor",
                                       "Eu, Yb (n=%d)" % len(special),
                                       "rest (n=%d)" % len(rest), "change"))
    for column in columns:
        a = included.loc[special, column].astype(float)
        b = included.loc[rest, column].astype(float)
        if a.notna().sum() == 0 or b.notna().sum() == 0:
            Write("  %-34s %10s %10s %10s" % (column, "pending", "pending", "-"))
            continue
        Write("  %-34s %10.4f %10.4f %+10.4f"
              % (column, a.mean(), b.mean(), a.mean() - b.mean()))
    Write()
    Write("  a negative change is the direction the argument predicts: the two")
    Write("  elements with the half-filled and the filled f shell are further along")
    Write("  than the rest of the series.")

    # --- what is still missing -------------------------------------------------
    Write()
    Write("-" * 78)
    Write("Outstanding runs")
    Write("-" * 78)
    Write()
    if not available["ICOBI M-N"].notna().any():
        Write("  LOBSTER has not been run in the working directory, so the M-N bond")
        Write("  order and the M-N bond strength are still pending.  The inputs are")
        Write("  in place:")
        Write("    python \"structure analysis.py\" --lobster-setup  wrote")
        Write("        `lobsterin` and `run_lobster.sh` beside every VASP run")
        Write("  put the `lobster` binary on PATH, run `run_lobster.sh` in each")
        Write("  %s-N-C directory," % "/".join(LANTHANIDES))
        Write("  then re-run this script, which fills the two columns with no other")
        Write("  change.")
    if not available["charge transfer/e"].notna().all():
        Write()
        Write("  the Bader charge is missing where ACF.dat is absent; run")
        Write("    bash \"run bader lanthanides.sh\"")
    if frame["available"].eq(False).any():
        Write()
        Write("  systems whose VASP job did not finish, and which cannot enter the")
        Write("  analysis until they are re-run:")
        for metal, row in frame[~frame["available"]].iterrows():
            Write("    %s-N-C  %s" % (metal, row["note"]))

    # ---------------------------------------------------------------------------
    # the figure
    # ---------------------------------------------------------------------------
    Write()
    Write("-" * 78)
    Write("The figure")
    Write("-" * 78)
    Write()
    Write("  four panels over the lanthanide series, in the reading order M-N bond")
    Write("  order and strength, Bader charge transfer, f shell, d band center.  The")
    Write("  M-N bond order and the M-N bond strength share a panel and a pair of")
    Write("  axes, every panel but the f shell is a curve rather than a bar chart, and")
    Write("  the f panel names each bar with the 4f configuration of the ion, the")
    Write("  series being trivalent apart from the divalent Eu and Yb.  Sm and Tm are")
    Write("  named by the pair of configurations they sit between, f5-f6 and f12-f13,")
    Write("  each of the four counts raised as the series raises its own, because the")
    Write("  occupation below the pair falls between the two.  Promethium")
    Write("  keeps its slot on the axis with a red tick label, no shading, and the")
    Write("  curves close over it.  Eu and Yb are picked out in grey, as in the")
    Write("  orbital-overlap figure, and in the f shell, the d band center and the")
    Write("  charge transfer -- the panels where they are a bar or a marker rather")
    Write("  than a band -- they are drawn in red as well, their numbers with them,")
    Write("  and the curve is shaded from blue to red across the segments that end on")
    Write("  one of them.  Every number is placed by")
    Write("  collision search, and")
    Write("  the search is checked against the drawn pixels before the figure is")
    Write("  written.")
    Write()

    FIGURE = os.path.join(FIG_DIR, "Lanthanide bonding.png")
    written, shared = Draw(available, FIGURE)
    plt.close("all")
    Write("  labels sharing pixels with a curve: %d" % shared)
    Write("  written to %s" % written)

    report.close()

    print()
    print("CSV: %s" % CSV)
    print("Report: %s" % os.path.join(SCRIPT_DIR, "lanthanide bonding report.txt"))


# ---------------------------------------------------------------------------
# part 3. the LOBSTER inputs of the lanthanide series
# ---------------------------------------------------------------------------
NITROGEN = "N"

# the build of LOBSTER that was installed for this analysis
DEFAULT_LOBSTER = ("/public/home/wangshen/LOBSTER/lobster-6.0.0/"
                   "lobster-6.0.0/lobster-6.0.0")

# the files LOBSTER reads.  POTCAR has to be the one of the WAVECAR run, and
# POSCAR the geometry that WAVECAR and CHGCAR describe, which is why the
# relaxation output CONTCAR is not used instead.
REQUIRED = ["POSCAR", "POTCAR", "WAVECAR", "CHGCAR"]

# the energy window of the bonding analysis, in eV relative to the Fermi level.
# The Kohn-Sham eigenvalues of this series reach down to -59 eV (Lu, whose
# 5s and 5p semicore states are in the valence of its PAW potential), so -70
# leaves every occupied state inside the window for every element, and +20 is
# above the top of the conduction bands.
COHP_START = -70.0
COHP_END = 20.0
COHP_STEPS = 1000

# the pair range of the COHP generator, in angstrom.  The M-N bonds of this
# series run from about 2.0 to 2.6 A, so a window that stops at 3.2 covers the
# four N of the site and nothing else; the C atoms of the ligand sit further out.
COHP_PAIR_MAX = 3.2

# LOBSTER 6.0 input.  The keywords are those of the user guide of that version;
# the comments give the reason each value has the value it has.  LOBSTER reads
# the file case-insensitively and takes !, # and // as comment markers.
LOBSTERIN = """! LOBSTER input for {metal}-N4-C, written by
! code/script/structure analysis.py.

! The minimal basis fitted by Maintz and co-workers to the GGA-PBE PAW
! wavefunctions of VASP.  It is available for every element up to lawrencium,
! so the lanthanides, whose f shell is in the valence of the PAW potential
! used here, are covered by the same set as C and N.
basisSet pbeVaspFit2015

! The basis functions, stated rather than recommended.  Left to itself LOBSTER
! builds the basis from the reference configuration of each POTCAR, and the
! lanthanide potentials differ in whether that configuration occupies 5d: La,
! Ce, Gd and Lu are given a 5d function and Pr, Nd, Eu, Tb, Dy, Er, Tm and Yb
! are not.  The M-N bond is carried by the metal 5d, so that difference alone
! moves the ICOBI of those eight elements by a factor of five against the other
! four, which is an artefact of the basis and not a chemical result.  Naming
! the functions here gives every element of the series the same basis.
basisFunctions C 2s 2p
basisFunctions N 2s 2p
basisFunctions {metal} 6s 5p 5s 5d 4f

! The energy window of the bonding analysis.  -70 eV is below the lowest
! Kohn-Sham eigenvalue of the series and +20 eV is above its highest, so every
! state that can carry M-N bonding is inside the window.
COHPStartEnergy {start}
COHPEndEnergy {end}
COHPSteps {steps}

! The pairs to analyse: the metal and nitrogen only, out to a distance that
! contains the four N of the site.  `orbitalWise` keeps the decomposition into
! the d-p and the f-p channel, which is what separates the covalency the f
! shell is expected to remove from the covalency it leaves behind.
cohpGenerator from 0.1 to {pair_max} type {metal} type {nitrogen} orbitalWise
"""

RUNNER = """#!/bin/bash
#SBATCH -J lobster-{metal}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition={partition}
#SBATCH -o lobster-%j.out

# written by code/script/structure analysis.py.  LOBSTER is called in the
# directory of the VASP run, because it reads POSCAR, POTCAR, WAVECAR and
# CHGCAR from the working directory and writes its output there as well.
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS={threads}

# the build installed for this analysis; override with LOBSTER=/path if needed
LOBSTER="${{LOBSTER:-{lobster}}}"

echo "running $LOBSTER in $(pwd) with $OMP_NUM_THREADS threads"
"$LOBSTER"
echo "lobster exit $?"
for f in ICOBILIST.lobster ICOHPLIST.lobster ICOOPLIST.lobster; do
    [ -e "$f" ] && echo "wrote $f ($(wc -l < "$f") lines)"
done
"""


def FileState(path):
    """(exists, size) of a file, so that a zero-byte file is caught as absent."""
    if not os.path.isfile(path):
        return False, 0
    return True, os.path.getsize(path)


def ReadVasprun(vasprun):
    """The scalar settings LOBSTER cares about, from a vasprun.xml.

    The INCAR that produced the WAVECAR is not necessarily the INCAR left in
    the directory, so the values are read from the machine-written record where
    it exists.
    """
    out = {}
    if not os.path.isfile(vasprun):
        return out
    wanted = ["NBANDS", "NELECT", "ISPIN", "ISYM"]
    with open(vasprun, errors="ignore") as fh:
        for line in fh:
            for name in wanted:
                if name in out:
                    continue
                key = 'name="%s"' % name
                if key in line:
                    text = line.split(">", 1)[-1].split("<")[0].strip()
                    try:
                        out[name] = int(float(text))
                    except ValueError:
                        pass
            if len(out) == len(wanted):
                break
    return out


def BuildOne(metal, dry_run, lobster, partition, threads):
    """Write lobsterin and run_lobster.sh for one system.  Returns a status."""
    folder = os.path.join(WORK, "%s-N-C" % metal)
    status = {"metal": metal, "folder": folder, "written": [],
              "missing": [], "empty": [], "vasprun": {}}

    if not os.path.isdir(folder):
        status["missing"].append("directory")
        return status

    for name in REQUIRED:
        exists, size = FileState(os.path.join(folder, name))
        if not exists:
            status["missing"].append(name)
        elif size == 0:
            status["empty"].append(name)

    status["vasprun"] = ReadVasprun(os.path.join(folder, "vasprun.xml"))

    # a system with no wavefunction or no charge density cannot be projected,
    # so nothing is written for it
    if status["missing"] or status["empty"]:
        return status

    lobsterin = os.path.join(folder, "lobsterin")
    runner = os.path.join(folder, "run_lobster.sh")

    if not dry_run:
        with open(lobsterin, "w") as fh:
            fh.write(LOBSTERIN.format(metal=metal, nitrogen=NITROGEN,
                                      start=COHP_START, end=COHP_END,
                                      steps=COHP_STEPS, pair_max=COHP_PAIR_MAX))
        with open(runner, "w") as fh:
            fh.write(RUNNER.format(metal=metal, partition=partition,
                                   threads=threads, lobster=lobster))
        os.chmod(runner, 0o755)

    status["written"] = [os.path.basename(p) for p in (lobsterin, runner)]
    return status


def LobsterSetup(argv):
    parser = argparse.ArgumentParser(
        description="Write the LOBSTER inputs of the lanthanide series.")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written, write nothing")
    parser.add_argument("--lobster", default=DEFAULT_LOBSTER,
                        help="path of the LOBSTER executable")
    parser.add_argument("--partition", default="hfacnormal01",
                        help="SLURM partition of the run_lobster.sh scripts")
    parser.add_argument("--threads", type=int, default=32,
                        help="OMP_NUM_THREADS of run_lobster.sh")
    args = parser.parse_args(argv[1:])

    print("LOBsTER preparation for the lanthanide M-N4-C series")
    print("VASP working directory: %s" % WORK)
    print("LOBsTER executable:     %s" % args.lobster)
    print("  present: %s" % os.path.isfile(args.lobster))
    print("COHP window: %.0f to %.0f eV in %d steps, pairs 0.1 to %.1f A, "
          "metal-nitrogen" % (COHP_START, COHP_END, COHP_STEPS, COHP_PAIR_MAX))
    print()

    ready, blocked = [], []
    for metal in LANTHANIDES:
        status = BuildOne(metal, args.dry_run, args.lobster, args.partition,
                          args.threads)
        if status["written"]:
            ready.append(status)
            print("  %-3s written: %s" % (metal, ", ".join(status["written"])))
        else:
            blocked.append(status)
            problem = []
            if status["missing"]:
                problem.append("absent: %s" % ", ".join(status["missing"]))
            if status["empty"]:
                problem.append("zero bytes: %s" % ", ".join(status["empty"]))
            print("  %-3s SKIPPED  %s" % (metal, "; ".join(problem)))

    print()
    print("ready: %d" % len(ready))
    print("skipped: %d" % len(blocked))

    # the settings LOBSTER reads from the run that wrote the WAVECAR
    print()
    print("the VASP settings LOBSTER needs, as recorded in vasprun.xml:")
    print("  %-4s %8s %8s %6s %6s %8s" % ("M", "NBANDS", "NELECT", "ISPIN",
                                          "ISYM", "n_occ"))
    for status in ready:
        v = status["vasprun"]
        nelect = v.get("NELECT")
        n_occ = int((nelect + 1) // 2) if nelect else None
        print("  %-4s %8s %8s %6s %6s %8s"
              % (status["metal"], v.get("NBANDS", "?"), nelect,
                 v.get("ISPIN", "?"), v.get("ISYM", "?"), n_occ))
    print()
    print("  NBANDS has to sit well above the occupied count for the projection")
    print("  to keep the empty states of the metal; at 256 against about 110")
    print("  occupied bands these runs have roughly twice the room needed.")

    if blocked:
        print()
        print("systems that cannot enter the analysis yet:")
        for status in blocked:
            print("  %s-N-C" % status["metal"])

    print()
    if args.dry_run:
        print("dry run: nothing was written")
    else:
        print("next: sbatch the run_lobster.sh of each system, then")
        print("      python \"structure analysis.py\"")
    return 0


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------
def main(argv):
    """Run the parts the arguments select."""
    if "--lobster-setup" in argv:
        return LobsterSetup([a for a in argv if a != "--lobster-setup"])
    if "--redraw" in argv:
        LanthanideBonding(redraw=True)
        return 0
    GeometryAnalysis()
    LanthanideBonding()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
