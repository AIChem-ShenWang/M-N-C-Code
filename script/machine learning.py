import os
import re
import sys
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as R2, r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.inspection import permutation_importance

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

import shap
from TorchSisso import SissoModel

# the package sits one level above this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

# the palette, the target labels and the figure-drawing functions are shared
# with `ablation study.py`, so they live in one module
from utils.ml import CUSTOM_CMAP, PlotElimination, PlotParity, PlotShap
from utils import ml


# --redraw redraws the figures from the cached tables, without repeating the
# search, which is the expensive part of the study
if "--redraw" in sys.argv:
    FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures", "ML")
    TARGETS = ["E_b", "E_f", "U_diss"]
    # the number of descriptors the elimination starts from, which turns the
    # retained count of `optimal features.csv` into the number eliminated
    N_POOL = 17

    def ReadCurves(tag):
        """The LOOCV curves of every model, and the optimum of each of them."""
        table = pd.read_csv(os.path.join(
            SCRIPT_DIR, "%s elimination curves.csv" % tag))
        mae_curves, r2_curves, k_opt = {}, {}, {}
        for name in table["model"].unique():
            sub = table[table.model == name].sort_values("n_eliminated")
            mae_curves[name] = sub["MAE"].to_numpy()
            r2_curves[name] = sub["R2"].to_numpy()
            k_opt[name] = int(
                sub["n_eliminated"].to_numpy()[mae_curves[name].argmin()])
        return mae_curves, r2_curves, k_opt

    def ReadOptimal(tag):
        """The retained feature set of every model, as the figure symbols."""
        table = pd.read_csv(os.path.join(
            SCRIPT_DIR, "%s optimal features.csv" % tag))
        sets = {}
        for _, row in table.iterrows():
            sets[row["model"]] = [s.strip() for s in str(row["features"]).split(",")]
        return sets

    def RedrawParity(tag, mode, y, metals, sets, data):
        """One parity plot per model, recomputed at the cached feature set.

        `metals` are the element symbols of the rows of `y`, in order; they are
        passed in because the data set is read without an index, so its index is
        the row number and would label the points 0, 1, ..."""
        # the labelled points are those missed by more than the constant
        # predictor, which is the baseline the figures were labelled with
        baseline_rmse = ml.RMSE(
            y.to_numpy(dtype=float),
            np.full(len(y), y.to_numpy(dtype=float).mean()))
        written = []
        for name in ml.MODEL_LIST:
            if name not in sets:
                print("   no optimum cached for %s, skipped" % name)
                continue
            columns = ml.SymbolsToColumns(sets[name])
            X = np.array(data[columns], dtype=float)
            y_true, y_pred, _ = ml.loocv(mode, name, X, y.to_numpy(dtype=float))
            path = os.path.join(FIG_DIR, "ML Performance of %s for %s.png"
                                % (name, tag))
            ml.PlotParity(name, mode, y_true, y_pred, metals, path,
                          label_threshold=baseline_rmse)
            written.append("   %-5s n=%-2d  R2=%.2f  MAE=%.2f  RMSE=%.2f"
                           % (name, len(columns), ml.R2(y_true, y_pred),
                              ml.MAE(y_true, y_pred), ml.RMSE(y_true, y_pred)))
        return written

    def RedrawShap(tag, mode):
        """The SHAP bars, read straight from the cached importance table."""
        table = pd.read_csv(os.path.join(
            SCRIPT_DIR, "%s shap importance.csv" % tag))
        table = table.sort_values("mean absolute SHAP")
        path = os.path.join(FIG_DIR, "SHAP_%s.png" % tag)
        ml.PlotShap(table["feature"].to_list(),
                    table["mean absolute SHAP"].to_numpy(), mode, path)
        return ("   %d descriptors, top = %s"
                % (len(table), table.iloc[-1]["feature"]))

    # the data set: the targets and the metal symbols of the first column, which
    # is unnamed in the spreadsheet and so comes back as `Unnamed: 0`
    data = pd.read_excel(os.path.join(SCRIPT_DIR, "../data/M-N-C data set.xlsx"))
    metals = data.iloc[:, 0].to_list()
    target_series = {"E_b": data["E_b/eV"].rename("E_b"),
                     "E_f": data["E_f/eV"].rename("E_f"),
                     "U_diss": data["U_diss_acid/V"].rename("U_diss")}

    requested = [a for a in sys.argv[1:] if a != "--redraw"] or TARGETS
    for tag in requested:
        assert tag in TARGETS, "the target should be one of %s" % ", ".join(TARGETS)

    for tag in requested:
        mode = ml.NormaliseMode(tag)
        print("== %s" % tag)
        mae_curves, r2_curves, k_opt = ReadCurves(tag)
        sets = ReadOptimal(tag)

        # the retained count of `optimal features.csv` and the optimum of the
        # curves are two views of one result, checked before either is drawn
        for name, symbols in sets.items():
            from_curves = k_opt[name]
            from_table = N_POOL - len(symbols)
            if from_curves != from_table:
                raise SystemExit(
                    "%s: %s has %d eliminated features in the curves but %d "
                    "(from %d retained of %d) in the optimal table"
                    % (tag, name, from_curves, from_table, len(symbols), N_POOL))

        for line in RedrawParity(tag, mode, target_series[tag], metals, sets, data):
            print(line)

        path = os.path.join(FIG_DIR, "Feature Elimination for %s.png" % tag)
        ml.PlotElimination(mae_curves, r2_curves, k_opt, mode, path)
        best = min(mae_curves, key=lambda n: mae_curves[n][k_opt[n]])
        print("   elimination grid, best model = %s at %d eliminated features"
              % (best, k_opt[best]))
        print(RedrawShap(tag, mode))

    print("figures written to %s" % os.path.abspath(FIG_DIR))
    sys.exit(0)


# args: mode & model selection
# usage: python "machine learning.py" [E_b|E_f|U_diss]
mode = sys.argv[1] if len(sys.argv) > 1 else "E_f"
assert mode in ["E_b", "E_f", "U_diss"], print("The mode should be E_b or E_f or U_diss.")

# `mode` carries LaTeX markup ($E_{b}$) because it is dropped straight into
# figure titles and axis labels.  File names must not contain those $ { }
# characters, so every path written to disk uses this plain tag instead.
mode_tag = mode.replace("$", "").replace("{", "").replace("}", "")
model_list = ["LR", "RF", "XGBR", "SVR", "kNN", "GP"]
# model_list = ["SVR"]

# number of shuffles used by permutation importance, the model-agnostic
# measure of how much a single feature contributes to the LOOCV score
n_permutation = 30

# 0. Preparation
# function
def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))

# the navy-white-crimson scale of the descriptor plots and the heatmaps comes
# from `utils/ml.py`; every other colour of the ML figures is used through the
# Plot* functions of that module rather than inline here
custom_cmap = CUSTOM_CMAP

# import data
data = pd.read_excel("../data/M-N-C data set.xlsx")
feat = data.iloc[:, 4:-7]
# the complete feature pool, kept aside because the backward elimination below
# starts from every feature rather than from the per-target subset
feat_pool = feat.copy()
E_b = data.loc[:]["E_b/eV"]
E_f = data.loc[:]["E_f/eV"]
U_diss_acid = data.loc[:]["U_diss_acid/V"]
U_diss_base = data.loc[:]["U_diss_base/V"]
# U_diss = []
# for i in range(len(U_diss_acid)):
#     U_diss.append(max(U_diss_base[i], U_diss_acid[i]))
# U_diss = pd.Series(np.array(U_diss))
U_diss = U_diss_acid

metals = data.iloc[:, 0].to_list()

if not os.path.exists("../figures/ML"):
    os.makedirs("../figures/ML")

# mode
if mode == "E_b":
    mode = "$E_{b}$"
    y = np.array(E_b)
    feat = feat[[
                 "CM2",
                 "band center/eV",
                 "heat_of_formation",
                "average distance of M-N bond/A",
                "covalent_radius_cordero/pm"
                 ]]

elif mode == "E_f":
    mode = "$E_{f}$"
    y = np.array(E_f)
    feat = feat[[
         "CM1",
         "covalent_radius_cordero/pm",
    ]]


elif mode == "U_diss":
    mode = "$U_{diss}$"
    y = np.array(U_diss)

    feat = feat[[
    "CM1",
    "CM2",
    "band width/eV",
    "atomic number",
    "covalent_radius_cordero/pm",
    "vdw_radius",
    "group number",
    ]]

# feature symbol
feature_symbol_dict = {"average distance of M-N bond/A": "$d_{M-N}$",
                      "average angle of M-N-C/degree": "$θ_{N-M-N}$",
                      "out-of-plane displacement of M/A": "$d_{z}$",
                      "CM1":"CM1",
                      "CM2":"CM2",
                      "band center/eV": "$ε_{e}$",
                      "band width/eV": "$ε_{w}$",
                      # atomic number of the central metal, i.e. its nuclear
                      # charge Z, the feature the Coulomb matrix is built from
                      "atomic number": "$Z$",
                      "atomic wight/g mol-1":"AW",
                      "atomic_radius/pm": "$R_{atom}$",
                      "covalent_radius_cordero/pm": "$R_{co}$",
                      "heat_of_formation": "$H_{f}$",
                      "molar_heat_capacity": "$C_{p}$",
                      "vdw_radius": "$R_{vdw}$",
                      "zeff": "$Z_{eff}$",
                      "group number": "GN",
                      "common valence": "CV",}



# 1. feature selection
X = np.array(feat)

# the pool on which the backward elimination is run: every feature of the dataset
X_pool = np.array(feat_pool)
feat_name_pool = [feature_symbol_dict[str(i)] for i in feat_pool.columns]

# the directory the reports and the plotting data are written to: this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def PlotCorrelationHeatmap(frame, path):
    """Pearson heatmap of `frame`, labelled with the symbols of the features."""
    metrics = frame.copy()
    metrics.columns = [feature_symbol_dict[str(c)] for c in metrics.columns]
    plt.figure(dpi=500, figsize=(10, 8))
    ax = sns.heatmap(metrics.corr(), annot=True, fmt=".2f", cmap=custom_cmap,
                     vmin=-1, vmax=1, annot_kws={'size': 7})
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    plt.title('Feature Correlation Heatmap', fontsize=18)
    plt.savefig(path)
    plt.close()


# heatmap of every feature of the data set
PlotCorrelationHeatmap(data.iloc[:, 4:-7], "../figures/ML/Heatmap for Features.png")


# 2. Train & Testing of ML models
# Feature importance is measured with permutation importance, which is the
# model-agnostic option of scikit-learn: one feature at a time is shuffled and
# the resulting degradation of the score is recorded, so the same measure can be
# applied to every estimator used here (LR / RF / XGBR / SVR / kNN / GP).
# scikit-learn's own RFE and RFECV cannot be used for this, because their
# importance_getter callable only receives the fitted estimator and never the
# data that permutation importance needs.  The backward elimination loop is
# therefore written explicitly.
def build_model(name):
    """Return a fresh estimator for `name`, with the hyper-parameters tuned for
    the current target.  The models are re-created for every fit so that no
    state leaks between leave-one-out splits."""
    kernel = (ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 1e2),
                                           nu=1.5) + WhiteKernel(noise_level=0.1,
                                                                 noise_level_bounds=(0.1, 10)))
    if mode == "$E_{b}$":
        if name == "LR":
            return Lasso(alpha=0.05, max_iter=10000, random_state=42)
        if name == "RF":
            return RandomForestRegressor(n_estimators=50, random_state=0, max_depth=10, max_features=0.5, min_samples_leaf=2)
        if name == "XGBR":
            return XGBRegressor(n_estimators=100, random_state=0, gamma=0.3)
        if name == "SVR":
            return SVR(kernel=kernel, C=5e5)
        if name == "kNN":
            return KNeighborsRegressor(n_neighbors=2)
        if name == "GP":
            return GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5, normalize_y=True)

    elif mode == "$E_{f}$":
        if name == "LR":
            return Lasso(alpha=0.08, max_iter=10000, random_state=42)
        if name == "RF":
            return RandomForestRegressor(n_estimators=50, random_state=0, max_depth=20, max_features=0.6)
        if name == "XGBR":
            return XGBRegressor(n_estimators=100, random_state=42, reg_alpha=0.5, gamma=0.01)
        if name == "SVR":
            return SVR(kernel=kernel, C=5e5)
        if name == "kNN":
            return KNeighborsRegressor(n_neighbors=3)
        if name == "GP":
            return GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5, normalize_y=True)

    elif mode == "$U_{diss}$":
        if name == "LR":
            return Lasso(alpha=0.001, max_iter=10000, random_state=42)
        if name == "RF":
            return RandomForestRegressor(n_estimators=50, random_state=0, max_depth=10, max_features=0.5)
        if name == "XGBR":
            return XGBRegressor(n_estimators=100, random_state=42, reg_alpha=0.1, gamma=0.2)
        if name == "SVR":
            return SVR(kernel=kernel, C=5e5)
        if name == "kNN":
            return KNeighborsRegressor(n_neighbors=2)
        if name == "GP":
            return GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=10, normalize_y=True)

    raise ValueError("Unknown model: %s" % name)


# Leave One Out Validation
loo = LeaveOneOut()


def loocv(name, X_sub):
    """Leave-one-out predictions for one feature subset.

    Returns the ground truth, the prediction, and the per-split training
    metrics (R2, MAE, RMSE) collected along the diagonal."""
    y_true = np.empty(len(y))
    y_pred = np.empty(len(y))
    train_metrics = np.empty((3, len(y)))

    for k, (train_index, test_index) in enumerate(loo.split(X_sub)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_sub[train_index])
        X_test = scaler.transform(X_sub[test_index])
        y_train, y_test = y[train_index], y[test_index]

        model = build_model(name)
        model.fit(X_train, y_train)

        prediction_train = model.predict(X_train)
        train_metrics[0, k] = R2(y_train, prediction_train)
        train_metrics[1, k] = MAE(y_train, prediction_train)
        train_metrics[2, k] = RMSE(y_train, prediction_train)

        y_true[k] = y_test[0]
        y_pred[k] = model.predict(X_test)[0]

    return y_true, y_pred, train_metrics


# backward elimination: start from every feature in the pool and, at each step,
# remove the feature whose removal leaves the best subset.  The subset left
# after a removal is scored directly by leave-one-out cross-validation, so the
# criterion of the search and the objective that is reported are the same thing.
# An earlier version ranked the features by in-sample permutation importance
# instead, which is decoupled from the LOOCV score and can therefore pick the
# wrong feature to drop.
print("Start Feature Elimination and LOOCV...")

elimination_mae = {}     # name -> LOOCV MAE after every elimination
elimination_r2 = {}      # name -> LOOCV R² after every elimination
eliminated_names = {}    # name -> retained feature positions at every step
best = {}                # name -> the optimum of that model

for name in tqdm(model_list, desc="Models"):
    remaining = list(range(X_pool.shape[1]))   # positions inside the feature pool
    mae_curve, r2_curve, retained = [], [], []

    # every subset is scored once and then remembered, which is what keeps the
    # search affordable: the subset finally chosen at one step is reused as the
    # starting point of the next
    cache = {}

    def evaluate(positions):
        key = tuple(sorted(positions))
        if key not in cache:
            cache[key] = loocv(name, X_pool[:, list(positions)])
        return cache[key]

    while True:
        y_true, y_pred, train_metrics = evaluate(remaining)
        mae_curve.append(MAE(y_true, y_pred))
        r2_curve.append(r2_score(y_true, y_pred))
        retained.append(list(remaining))

        if len(remaining) == 1:
            break

        # score each candidate removal and drop the feature whose absence costs
        # the least accuracy
        best_removal, best_mae = None, np.inf
        for position in remaining:
            candidate = [p for p in remaining if p != position]
            candidate_true, candidate_pred, _ = evaluate(candidate)
            candidate_mae = MAE(candidate_true, candidate_pred)
            if candidate_mae < best_mae:
                best_mae, best_removal = candidate_mae, position
        remaining.remove(best_removal)

    elimination_mae[name] = mae_curve
    elimination_r2[name] = r2_curve
    eliminated_names[name] = retained

    # the optimum of this model: the fewest features at the lowest LOOCV MAE
    k_opt = int(np.argmin(mae_curve))
    y_true, y_pred, train_metrics = loocv(name, X_pool[:, retained[k_opt]])
    best[name] = {"n_features": len(retained[k_opt]),
                  "n_eliminated": k_opt,
                  "features": [feat_name_pool[j] for j in retained[k_opt]],
                  "columns": [feat_pool.columns[j] for j in retained[k_opt]],
                  "mae": MAE(y_true, y_pred),
                  "y_true": y_true,
                  "y_pred": y_pred,
                  "train_metrics": train_metrics}


# 3. Performance Record & Plotting
if not os.path.exists("../data/ML-results"):
    os.makedirs("../data/ML-results")

# the report is written next to this script
report = open(os.path.join(SCRIPT_DIR, "%s ML report.txt" % mode_tag),
              mode="w+", encoding="utf-8")

# data set baseline: the mean of the target used as a constant predictor, kept
# for comparison against every model in the figures, and used as the threshold
# above which a metal is named in the parity plots
y_avg = [np.array(y).mean()] * len(y)
r2 = r2_score(y, y_avg)
mae = MAE(y, y_avg)
baseline_rmse = RMSE(y, y_avg)
report.write("Dataset Baseline:\n")
report.write("R²: %.2f\nMAE = %.2f\nRMSE = %.2f\n" % (r2, mae, baseline_rmse))
report.write("\n")
print("Dataset Baseline:")
print("R²: %.2f\nMAE = %.2f\nRMSE = %.2f\n" % (r2, mae, baseline_rmse))

# best model of the elimination loop, ranked by its LOOCV MAE at the optimum
best_model = min(model_list, key=lambda n: best[n]["mae"])

# the data behind every figure is written next to this script, so that the plots
# can be re-drawn without repeating the leave-one-out search (item 0).  It is
# saved before any figure is drawn, because the search above is the expensive
# part and must survive a plotting failure.
curve_rows = []
for name in model_list:
    for k, (mae_k, r2_k) in enumerate(zip(elimination_mae[name], elimination_r2[name])):
        curve_rows.append({"model": name,
                           "n_eliminated": k,
                           "n_features": X_pool.shape[1] - k,
                           "R2": r2_k,
                           "MAE": mae_k})
pd.DataFrame(curve_rows).to_csv(
    os.path.join(SCRIPT_DIR, "%s elimination curves.csv" % mode_tag), index=False)

pd.DataFrame({"metal": metals,
              "ground truth": best[best_model]["y_true"],
              "prediction": best[best_model]["y_pred"]}).to_csv(
    os.path.join(SCRIPT_DIR, "%s best model predictions.csv" % mode_tag), index=False)

pd.DataFrame([{"model": name,
               "n_features": best[name]["n_features"],
               "features": ", ".join(best[name]["features"]),
               "R2": r2_score(best[name]["y_true"], best[name]["y_pred"]),
               "MAE": best[name]["mae"]} for name in model_list]).to_csv(
    os.path.join(SCRIPT_DIR, "%s optimal features.csv" % mode_tag), index=False)

# the heatmap below describes the optimum of the best model, so it is drawn only
# once that feature set is known (item 1).  The optimum is expressed in the
# columns of the whole feature pool, so the pool is what has to be sliced here.
PlotCorrelationHeatmap(feat_pool[best[best_model]["columns"]],
                       "../figures/ML/Heatmap for Features in %s.png" % mode_tag)

evaluation_metrics_name = ["R²", "MAE", "RMSE"]

for name in model_list:
    y_true = best[name]["y_true"]
    y_pred = best[name]["y_pred"]

    # Model Performance, taken at the optimal number of features of this model
    train_performance_metrics = best[name]["train_metrics"]
    train_performace_avg = train_performance_metrics.mean(axis=1)
    train_performace_std = train_performance_metrics.std(axis=1)

    train_performance = str()
    test_performance = str()
    for i in range(3):
        train_performance += "%s = %.2f ± %.2f\n" % (evaluation_metrics_name[i],
                                                       train_performace_avg[i],
                                                       train_performace_std[i])
        test_performance += "%s = %.2f\n" % (evaluation_metrics_name[i],
                                             [r2_score(y_true, y_pred),
                                              MAE(y_true, y_pred),
                                              RMSE(y_true, y_pred)][i])

    performance = "train set performance:\n" + train_performance + "test set performance:\n" + test_performance
    report.write("ML Model: %s\n" % name)
    report.write(performance)
    report.write("Optimal number of features = %d (out of %d)\n"
                 % (best[name]["n_features"], X_pool.shape[1]))
    report.write("Retained features: %s\n" % ", ".join(best[name]["features"]))
    report.write("\n")
    print("ML Model: %s" % name)
    print(performance)
    print("Optimal number of features = %d" % best[name]["n_features"])

    # the parity plot is drawn by the shared function, so that the full run and
    # the redraw cannot drift apart in palette or in labels.  The metals named
    # inside it are the ones missed by more than the constant predictor.
    PlotParity(name, mode, y_true, y_pred, metals,
               "../figures/ML/ML Performance of %s for %s.png" % (name, mode_tag),
               label_threshold=baseline_rmse)

    print("Optimal features of %s: %s" % (name, ", ".join(best[name]["features"])))


# the best model of the loop, reported as the headline performance of the target
best_performance = best[best_model]
report.write("Best model of the elimination loop:\n")
report.write("Model = %s\n" % best_model)
report.write("Optimal number of features = %d (out of %d)\n"
             % (best_performance["n_features"], X_pool.shape[1]))
report.write("Retained features: %s\n" % ", ".join(best_performance["features"]))
report.write("LOOCV performance:\n")
report.write("R² = %.2f\nMAE = %.2f\nRMSE = %.2f\n"
             % (r2_score(best_performance["y_true"], best_performance["y_pred"]),
                MAE(best_performance["y_true"], best_performance["y_pred"]),
                RMSE(best_performance["y_true"], best_performance["y_pred"])))
report.write("train set performance:\n")
report.write("R² = %.2f ± %.2f\nMAE = %.2f ± %.2f\nRMSE = %.2f ± %.2f\n"
             % (best_performance["train_metrics"][0].mean(),
                best_performance["train_metrics"][0].std(),
                best_performance["train_metrics"][1].mean(),
                best_performance["train_metrics"][1].std(),
                best_performance["train_metrics"][2].mean(),
                best_performance["train_metrics"][2].std()))
report.write("\n")
print("Best model of the elimination loop: %s, %d features"
      % (best_model, best_performance["n_features"]))


# R² and MAE against the number of eliminated features, one subplot per model.
# The panel grid is drawn by the shared function of `utils/ml.py`, which the
# redraw path calls as well, so the two cannot drift apart.
PlotElimination(elimination_mae, elimination_r2,
                {name: best[name]["n_eliminated"] for name in model_list},
                mode, "../figures/ML/Feature Elimination for %s.png" % mode_tag)


# 4.Interpretability
# SHAP, computed on the optimum feature set of the best model (item 2)
best_columns = best[best_model]["columns"]
X_shap = StandardScaler().fit_transform(np.array(feat_pool[best_columns]))
model = build_model(best_model).fit(X_shap, y)

plt.figure(dpi=500, figsize=(5, 10))
if best_model in ["RF", "XGBR"]:
    explainer = shap.Explainer(model)
elif best_model in ["kNN", "SVR", "GP", "LR"]:
    explainer = shap.KernelExplainer(model.predict, X_shap)

shap_values = explainer(X_shap)

# SISSO Part 1: the target and the descriptor set of the symbolic regression
if mode == "$E_{b}$":
    y = E_b.to_numpy()
elif mode == "$E_{f}$":
    y = E_f.to_numpy()
elif mode == "$U_{diss}$":
    y = U_diss.to_numpy()

# SISSO Part 2.  The symbolic regression is run on a deliberately small
# descriptor set, because the expansion of a full optimum exhausts the memory:
# the two settings below are the largest that still complete for each stability
# measure, and they are also the ones that give the best equation, so they are
# chosen per target rather than shared.
#   top_n       how many of the descriptors, ranked by permutation importance,
#               are passed on
#   n_expansion how many levels the operator expansion is allowed to reach
SISSO_SETTING = {"$E_{b}$":    {"top_n": 3, "n_expansion": 3},
                 "$E_{f}$":    {"top_n": 2, "n_expansion": 4},
                 "$U_{diss}$": {"top_n": 2, "n_expansion": 4}}

sisso_pool = best[best_model]["columns"]
X_pool_scaled = StandardScaler().fit_transform(np.array(feat_pool[sisso_pool]))
sisso_model = build_model(best_model).fit(X_pool_scaled, y)
sisso_importance = permutation_importance(
    sisso_model, X_pool_scaled, y, scoring="neg_mean_absolute_error",
    n_repeats=n_permutation, random_state=42, n_jobs=1).importances_mean

sisso_ranked = [sisso_pool[i] for i in np.argsort(sisso_importance)[::-1]]
top_n = SISSO_SETTING[mode]["top_n"]
n_expansion = SISSO_SETTING[mode]["n_expansion"]
sisso_col = sisso_ranked[:top_n]
X = np.array(feat_pool[sisso_col])
# the equation itself keeps the two terms used throughout the study
n_terms = 2

print("SISSO descriptors of the best model (%s): %s"
      % (best_model, ", ".join(feature_symbol_dict[c] for c in sisso_col)))

# feature importance: one call to the shared function, so that this figure and
# the one the redraw path writes agree
if hasattr(shap_values, 'values'):
    shap_matrix = shap_values.values
else:
    shap_matrix = shap_values
mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)
shap_feat_name = [feature_symbol_dict[str(c)] for c in best_columns]
sorted_features = [shap_feat_name[i] for i in sorted_idx]
sorted_values = mean_abs_shap[sorted_idx]

# data behind the SHAP figure
pd.DataFrame({"feature": shap_feat_name,
              "mean absolute SHAP": mean_abs_shap}).to_csv(
    os.path.join(SCRIPT_DIR, "%s shap importance.csv" % mode_tag), index=False)

PlotShap(sorted_features, sorted_values, mode,
         "../figures/ML/SHAP_%s.png" % mode_tag)


# the report of the elimination loop is complete here, so it is closed before
# SISSO: the symbolic regression is the slowest and most fragile step and must
# not be able to take the rest of the results down with it
report.close()

# SISSO Part2: symbolic regression on the minimal feature set
data = pd.DataFrame(np.column_stack((y, X)), columns=["Target"] + [feature_symbol_dict[i] for i in sisso_col])
operators = ["+", "-", "*", "/", "Ln", "sin", "tan", "cos",  "pow(-1)", "pow(-2)"]
sm = SissoModel(data, operators, n_expansion=n_expansion, n_term=n_terms, k=5)
rmse, equation, r2,_ = sm.fit()
with open(os.path.join(SCRIPT_DIR, "%s ML report.txt" % mode_tag),
          "a", encoding="utf-8") as sisso_report:
    sisso_report.write("\nSISSO Model:\n")
    sisso_report.write("Descriptors: %s\n"
                       % ", ".join(feature_symbol_dict[c] for c in sisso_col))
    sisso_report.write("top_n = %d, n_expansion = %d\n" % (top_n, n_expansion))
    sisso_report.write("R²=%.2f\nRMSE=%.2f\nEq:%s" % (r2, rmse, equation))
print("SISSO Model:\nR²=%.2f\nRMSE=%.2f\nEq:%s" %(r2, rmse, equation))
