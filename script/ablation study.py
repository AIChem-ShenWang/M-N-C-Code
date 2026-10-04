"""Feature ablation study behind Table 1 of the manuscript.

For every stability target the table compares several kinds of feature set,
grouped by the `Kind` column:

  Kind           Row
  ------------   ------------------------------------------------------------
  baseline       All features, and the optimum found by `machine learning.py`
  leave-one-out  the optimum with one descriptor removed, one row each
  destruction    the optimum with the probed descriptor permuted, or replaced by
                 Gaussian noise of the same mean and spread
  ceiling        the probed descriptor on its own, plus any extra descriptor
                 the target names in `EXTRA_CEILING`

CM1 is probed wherever the greedy search kept it; for a target that dropped it,
the most important descriptor of that target's own selected set takes its place,
so every target gets the same kind of control.

For every candidate set all six estimators are re-trained under leave-one-out
cross-validation and the best of them is reported.  The destruction rows are the
exception: they run the single estimator that won the `Selected features` row,
and report the mean and spread over `N_SCRAMBLE` scrambles.

Writes `ablation study.csv`, `ablation study detail.csv` and
`ablation study report.txt` next to this script.

Usage: python "ablation study.py" [E_b|E_f|U_diss ...]
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score as R2

# the package sits one level above this script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ml import (MODEL_LIST, RMSE, MAE, feature_symbol_dict,
                      symbol_feature_dict, SymbolsToColumns, loocv)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. data
data = pd.read_excel(os.path.join(SCRIPT_DIR, "../data/M-N-C data set.xlsx"))
feat = data.iloc[:, 4:-7]                    # the complete feature pool

# frozen before any derived column is added, so that the `All features` row keeps
# describing the real pool of the data set
ALL_COLUMNS = list(feat.columns)

# the descriptor the destruction battery probes
PROBE_SYMBOL = "CM1"

# descriptors that get a single-descriptor ceiling row on top of the probed one.
# The probed descriptor always gets its own row; U_diss names GN here as well,
# so that its table reports both the descriptor its destruction rows break
# (CM1) and the other member of its retained set a reader is likely to ask
# about.  The rows are listed in this order, and a descriptor named twice is
# reported once.
EXTRA_CEILING = {"U_diss": ["GN"]}

# the permutation and noise controls are averaged over this many scrambles
N_SCRAMBLE = 20

TARGETS = {"E_b": "E_b/eV",
           "E_f": "E_f/eV",
           "U_diss": "U_diss_acid/V"}

# the label every artefact of a target is named after: the reports, the figures
# and the plotting data of `machine learning.py` all use the subscript form
MODES = {"E_b": "$E_{b}$",
         "E_f": "$E_{f}$",
         "U_diss": "$U_{diss}$"}

# the order the rows are written in
KIND_ORDER = ["baseline", "leave-one-out", "destruction", "ceiling"]
KIND_RANK = {kind: i for i, kind in enumerate(KIND_ORDER)}

# the targets to run: from the command line, or every target by default
argv = [a for a in sys.argv[1:] if a in TARGETS]
targets = argv if argv else list(TARGETS)


def ReadSelectedFeatures(target):
    """The optimum feature set of the best model, as written by `machine
    learning.py` in `<target> optimal features.csv`."""
    # the report files are named after the plain target key, not after the
    # LaTeX form used in figure labels
    path = os.path.join(SCRIPT_DIR, "%s optimal features.csv" % target)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Run `machine learning.py %s` first: %s is missing" % (target, path))

    table = pd.read_csv(path)
    best_row = table.loc[table["MAE"].idxmin()]
    symbols = [s for s in str(best_row["features"]).split(",") if s.strip()]
    return SymbolsToColumns(symbols), best_row["model"]


def ProbeFeature(target, selected):
    """The descriptor the destruction battery probes.

    CM1 wherever the greedy search kept it.  For a target that dropped CM1 the
    most important descriptor of its own selected set takes its place, ranked by
    the SHAP table `machine learning.py` wrote, so that every target gets the
    same kind of control."""
    if PROBE_SYMBOL in selected:
        return PROBE_SYMBOL

    shap_path = os.path.join(SCRIPT_DIR, "%s shap importance.csv" % target)
    if os.path.isfile(shap_path):
        table = pd.read_csv(shap_path)
        table = table.sort_values(table.columns[1], ascending=False)
        for symbol in table["feature"]:
            column = symbol_feature_dict.get(str(symbol).strip())
            if column in selected:
                return column
    return selected[0]


def ScoreSet(mode, columns, y, desc):
    """All six estimators on one candidate set; the best by MAE is reported."""
    X = np.array(feat[columns])
    scores = {}
    for name in tqdm(MODEL_LIST, desc=desc, leave=False):
        y_true, y_pred, _ = loocv(mode, name, X, y)
        scores[name] = {"R2": R2(y_true, y_pred),
                        "MAE": MAE(y_true, y_pred),
                        "RMSE": RMSE(y_true, y_pred)}
    best = min(MODEL_LIST, key=lambda n: scores[n]["MAE"])
    return best, scores[best], scores


def ScrambledX(columns, probe, how, rng):
    """The candidate matrix with the probed column broken.

    `permute` keeps every value but detaches it from its system, so the marginal
    distribution is untouched and only the pairing with the label is destroyed.
    `noise` replaces it by Gaussian draws of the same mean and spread."""
    X = np.array(feat[columns], dtype=float)
    j = columns.index(probe)
    if how == "permute":
        X[:, j] = rng.permutation(X[:, j])
    else:
        X[:, j] = rng.normal(X[:, j].mean(), X[:, j].std(), X.shape[0])
    return X


def ScoreScrambled(mode, name, columns, probe, how, y, rng, desc):
    """One fixed estimator over `N_SCRAMBLE` scrambles of the probed column."""
    acc = {"R2": [], "MAE": [], "RMSE": []}
    for _ in tqdm(range(N_SCRAMBLE), desc=desc, leave=False):
        X = ScrambledX(columns, probe, how, rng)
        y_true, y_pred, _ = loocv(mode, name, X, y)
        acc["R2"].append(R2(y_true, y_pred))
        acc["MAE"].append(MAE(y_true, y_pred))
        acc["RMSE"].append(RMSE(y_true, y_pred))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


report = open(os.path.join(SCRIPT_DIR, "ablation study report.txt"),
              "w+", encoding="utf-8")


def Report(text):
    """Write one line and push it out at once.

    The run takes tens of minutes, and an unflushed buffer would leave the
    report empty on disk for the whole time."""
    report.write(text + "\n")
    report.flush()


def Delta(text, score, reference):
    """One aligned delta line against the `Selected features` row."""
    return ("%-44s              dR2=%+.3f  dMAE=%+.3f  relative to the selected set"
            % (text, score["R2"] - reference["R2"], score["MAE"] - reference["MAE"]))


detail_rows = []
table_rows = []

for target in targets:
    mode = MODES[target]
    y = data.loc[:][TARGETS[target]].to_numpy(dtype=float)
    rng = np.random.RandomState(42)

    selected, selected_model = ReadSelectedFeatures(target)
    probe = ProbeFeature(target, selected)
    probe_symbol = feature_symbol_dict[probe]
    # the descriptors scored on their own: the probed one first, then any extra
    # this target names, without repeating a descriptor
    ceiling_symbols = list(dict.fromkeys([probe_symbol] + EXTRA_CEILING.get(target, [])))

    print("=" * 70)
    print("%s: selected features of %s are %s"
          % (mode, selected_model,
             ", ".join(feature_symbol_dict[c] for c in selected)))
    print("%s: the battery probes %s" % (mode, probe_symbol))

    Report("=" * 70)
    Report("%s prediction" % mode)
    Report("Selected features of the best model (%s): %s"
           % (selected_model,
              ", ".join(feature_symbol_dict[c] for c in selected)))
    Report("The destruction rows probe %s." % probe_symbol)
    if len(ceiling_symbols) > 1:
        # only worth a line where a target reports more than one ceiling row
        Report("The ceiling rows are %s."
               % ", ".join("%s alone" % symbol for symbol in ceiling_symbols))
    Report("")

    # every candidate feature set of this target, in the order it is reported
    candidates = [("All %d features" % len(ALL_COLUMNS), ALL_COLUMNS,
                   "baseline", "the whole pool of %d descriptors" % len(ALL_COLUMNS))]
    candidates.append(("Selected features", list(selected), "baseline",
                       "optimum of %s" % selected_model))
    for column in selected:
        candidates.append(("Selected features - %s" % feature_symbol_dict[column],
                           [c for c in selected if c != column],
                           "leave-one-out", "one descriptor dropped"))

    for symbol in ceiling_symbols:
        candidates.append(("%s alone" % symbol, [symbol_feature_dict[symbol]],
                           "ceiling", "single-descriptor ceiling"))

    target_rows = []
    target_detail = []
    selected_score = None
    selected_best = None

    for label, columns, kind, note in candidates:
        best_name, best_score, scores = ScoreSet(
            mode, columns, y, "%s | %s" % (mode, label))

        if label == "Selected features":
            # reference for every row below
            selected_score, selected_best = best_score, best_name

        target_rows.append({"Group": "%s prediction" % mode,
                            "Kind": kind,
                            "Feature": label,
                            "n_features": len(columns),
                            "Best model": best_name,
                            "R2": best_score["R2"],
                            "MAE": best_score["MAE"],
                            "RMSE": best_score["RMSE"],
                            "R2_std": "", "MAE_std": "", "RMSE_std": "",
                            "n_scrambles": "",
                            "Note": note})
        for name in MODEL_LIST:
            target_detail.append({"Group": "%s prediction" % mode,
                                  "Kind": kind,
                                  "Feature": label,
                                  "model": name,
                                  "R2": scores[name]["R2"],
                                  "MAE": scores[name]["MAE"],
                                  "RMSE": scores[name]["RMSE"]})

        Report("%-44s n=%2d  best=%-5s  R2=%.3f  MAE=%.3f  RMSE=%.3f"
               % (label, len(columns), best_name,
                  best_score["R2"], best_score["MAE"], best_score["RMSE"]))
        if selected_score is not None and kind != "baseline":
            Report(Delta("", best_score, selected_score))

    # the destruction controls: the estimator that won `Selected features`, over
    # many scrambles of the probed column
    for how, label in [("permute", "Selected features, %s permuted" % probe_symbol),
                       ("noise", "Selected features, %s -> noise" % probe_symbol)]:
        stats = ScoreScrambled(mode, selected_best, list(selected), probe, how, y,
                               rng, "%s | %s" % (mode, label))
        target_rows.append({"Group": "%s prediction" % mode,
                            "Kind": "destruction",
                            "Feature": label,
                            "n_features": len(selected),
                            "Best model": "%s (fixed)" % selected_best,
                            "R2": stats["R2"][0], "MAE": stats["MAE"][0],
                            "RMSE": stats["RMSE"][0],
                            "R2_std": stats["R2"][1], "MAE_std": stats["MAE"][1],
                            "RMSE_std": stats["RMSE"][1],
                            "n_scrambles": N_SCRAMBLE,
                            "Note": "%s of %s over %d scrambles"
                                    % (how, probe_symbol, N_SCRAMBLE)})

        Report("%-44s n=%2d  best=%-5s  R2=%.3f+-%.3f  MAE=%.3f+-%.3f  RMSE=%.3f+-%.3f"
               % (label, len(selected), selected_best,
                  stats["R2"][0], stats["R2"][1],
                  stats["MAE"][0], stats["MAE"][1],
                  stats["RMSE"][0], stats["RMSE"][1]))
        Report(Delta("", {"R2": stats["R2"][0], "MAE": stats["MAE"][0]},
                     selected_score))

    Report("")

    # a stable sort keeps the order inside every kind
    target_rows.sort(key=lambda r: KIND_RANK[r["Kind"]])
    target_detail.sort(key=lambda r: KIND_RANK[r["Kind"]])
    table_rows.extend(target_rows)
    detail_rows.extend(target_detail)

# 2. output
table = pd.DataFrame(table_rows)
table.to_csv(os.path.join(SCRIPT_DIR, "ablation study.csv"), index=False)
pd.DataFrame(detail_rows).to_csv(
    os.path.join(SCRIPT_DIR, "ablation study detail.csv"), index=False)
report.close()

print("=" * 70)
print(table.to_string(index=False))
print("\nWritten to %s" % SCRIPT_DIR)
