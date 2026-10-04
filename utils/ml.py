"""Definitions shared by the machine-learning scripts.

The hyper-parameters below are the ones tuned for each stability target, and the
symbol dictionary is the label set used in every figure and report.  Keeping both
in a single place means that the scripts always compare the same models and print
the same names.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, Matern

# the estimators compared throughout the study
MODEL_LIST = ["LR", "RF", "XGBR", "SVR", "kNN", "GP"]

# display name of every feature of the data set, used in the figures and reports
feature_symbol_dict = {"average distance of M-N bond/A": "$d_{M-N}$",
                       "average angle of M-N-C/degree": "$θ_{N-M-N}$",
                       "out-of-plane displacement of M/A": "$d_{z}$",
                       "CM1": "CM1",
                       "CM2": "CM2",
                       "band center/eV": "$ε_{e}$",
                       "band width/eV": "$ε_{w}$",
                       "atomic number": "$Z$",
                       "atomic wight/g mol-1": "AW",
                       "atomic_radius/pm": "$R_{atom}$",
                       "covalent_radius_cordero/pm": "$R_{co}$",
                       "heat_of_formation": "$H_{f}$",
                       "molar_heat_capacity": "$C_{p}$",
                       "vdw_radius": "$R_{vdw}$",
                       "zeff": "$Z_{eff}$",
                       "group number": "GN",
                       "common valence": "CV"}

# display name -> column name of the data set
symbol_feature_dict = {v: k for k, v in feature_symbol_dict.items()}


def SymbolsToColumns(symbols):
    """Map a list of display names back to the columns of the data set."""
    out = []
    for s in symbols:
        s = s.strip()
        if s not in symbol_feature_dict:
            raise KeyError("Unknown feature symbol: %s" % s)
        out.append(symbol_feature_dict[s])
    return out


def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))


# the three targets, in the tag they are written on disk and the LaTeX label
# they carry in the figures.  The tag is what the file names use, because a
# file name cannot hold the $ { } of the label.
TARGET_LABEL = {"E_b": "$E_{b}$", "E_f": "$E_{f}$", "U_diss": "$U_{diss}$"}
LABEL_TARGET = {v: k for k, v in TARGET_LABEL.items()}


def NormaliseMode(mode):
    """Accept the short target name (E_b) and its plotting label ($E_{b}$)."""
    mode = str(mode)
    if mode in TARGET_LABEL:
        return TARGET_LABEL[mode]
    if mode.startswith("$") and mode.endswith("$") and len(mode) > 2:
        return mode
    # anything else is assumed to be a bare target name
    if mode in LABEL_TARGET.values():
        return mode
    raise ValueError("Unknown target: %s (expected one of %s)"
                     % (mode, ", ".join(TARGET_LABEL)))


def _kernel():
    """The kernel shared by SVR and GP: a smooth Matern plus a noise term."""
    return (ConstantKernel(1.0)
            * Matern(length_scale=1.0, length_scale_bounds=(0.1, 1e2), nu=1.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(0.1, 10)))


def build_model(mode, name):
    """Return a fresh estimator for `name` under the hyper-parameters of `mode`.

    A new object is returned on every call so that no state leaks between the
    leave-one-out splits."""
    mode = NormaliseMode(mode)
    kernel = _kernel()

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
LOO = LeaveOneOut()


def loocv(mode, name, X, y):
    """Leave-one-out predictions for one feature subset.

    Returns the ground truth, the prediction, and the per-split training
    metrics (R2, MAE, RMSE) collected along the diagonal."""
    y_true = np.empty(len(y))
    y_pred = np.empty(len(y))
    train_metrics = np.empty((3, len(y)))

    for k, (train_index, test_index) in enumerate(LOO.split(X)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_index])
        X_test = scaler.transform(X[test_index])
        y_train, y_test = y[train_index], y[test_index]

        model = build_model(mode, name)
        model.fit(X_train, y_train)

        prediction_train = model.predict(X_train)
        train_metrics[0, k] = R2(y_train, prediction_train)
        train_metrics[1, k] = MAE(y_train, prediction_train)
        train_metrics[2, k] = RMSE(y_train, prediction_train)

        y_true[k] = y_test[0]
        y_pred[k] = model.predict(X_test)[0]

    return y_true, y_pred, train_metrics


# ---------------------------------------------------------------------------
# the palette
# ---------------------------------------------------------------------------
# Every figure of the machine-learning part of the study is drawn from this one
# block.  The colours used to live in each script separately, which is how the
# elimination curves and the parity plots came to be drawn in two different reds
# and two different blues; they are defined once here now.
#
#   PARITY_*  the parity plots of `ML Performance of <model> for <target>.png`
#   ELIM_*    the two curve colours of the elimination figures
#   SHAP_*    the SHAP bars of each target (defined below, from the two above)
CMAP_COLORS = ["#003153", "#f5f5f5", "#85120f"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom", CMAP_COLORS, N=256)

PARITY_FILL, PARITY_EDGE = "#9fbbd5", "#3a4b6e"
PARITY_LINE, PARITY_BAND = "#b8474d", "#d69d98"
PARITY_OUTLIER = "#3a4b6e"

ELIM_RED, ELIM_BLUE = "#b8474d", "#3a4b6e"

# The SHAP bars carry the same two-tone treatment as the parity markers -- a
# light fill under a dark outline -- and keep the hue each target is known by in
# the other figures, so the SHAP panels sit in the same palette as the rest.
SHAP_FILL = {"$E_{b}$": PARITY_FILL, "$E_{f}$": PARITY_BAND,
             "$U_{diss}$": "#d9d9d9"}
SHAP_EDGE = {"$E_{b}$": PARITY_EDGE, "$E_{f}$": PARITY_LINE,
             "$U_{diss}$": "#7f7f7f"}

# the R2 of the labels inside the figures is set in math mode: matplotlib does
# not know the Unicode superscript two with every font, and the Roman R of the
# plain text version does not match the italic R2 of the paper
METRIC_LABELS = ["$R^{2}$", "MAE", "RMSE"]
METRIC_LABELS_PLAIN = ["R²", "MAE", "RMSE"]

# how far past the data the parity axes run, and the offsets of the text
PARITY_EXTEND = 0.5
PARITY_TEXT_SIZE = 12
PARITY_TITLE_SIZE = 15
PARITY_LABEL_SIZE = 16

# the unit of each target, for the axis labels and the MAE label
TARGET_UNIT = {"$E_{b}$": "eV", "$E_{f}$": "eV", "$U_{diss}$": "V"}

# where the metal labels and the metric box sit in a parity plot.  The offsets
# are in the units of the target, so each target carries its own: the span of
# U_diss across the 61 metals is about 4.5 V against 14 eV for E_f, and the
# same offset would collide in one figure and float in the other.
PARITY_OFFSET = {
    #                outlier label      metric box
    "$E_{b}$":    ((0.25, 0.25), (0.4, 0.5)),
    "$E_{f}$":    ((0.35, 0.35), (0.6, 0.7)),
    "$U_{diss}$": ((0.10, 0.11), (0.2, 0.28)),
}


def PlotParity(name, mode, y_true, y_pred, metals, path, label_threshold=None):
    """The parity plot of one model on one target.

    `label_threshold` is the error above which a metal is named inside the plot.
    It is a parameter because it is the baseline of the study rather than the
    RMSE of this model: the manuscript names the metals that the model misses by
    more than the constant predictor does on average, so passing the model's own
    RMSE here would name a different -- and much longer -- list.  Left as None,
    the model's own test RMSE is used.

    `R2 = ...` inside the box is set in math mode, so that it is italic like
    every other R2 of the paper."""
    mode = NormaliseMode(mode)
    unit = TARGET_UNIT[mode]
    if label_threshold is None:
        label_threshold = RMSE(y_true, y_pred)

    figure = plt.figure(dpi=300, figsize=(6, 5))
    minimum = min(min(y_true), min(y_pred))
    maximum = max(max(y_true), max(y_pred))
    plot_min = minimum - PARITY_EXTEND
    plot_max = maximum + PARITY_EXTEND
    plt.plot([plot_min, plot_max], [plot_min, plot_max],
             linestyle="--", color=PARITY_LINE, linewidth=2)

    test_rmse = RMSE(y_true, y_pred)

    # the band and its two edges mark the RMSE of the test set
    plt.fill_between([plot_min, plot_max],
                     [plot_min - test_rmse, plot_max - test_rmse],
                     [plot_min + test_rmse, plot_max + test_rmse],
                     color=PARITY_BAND, alpha=0.3)
    plt.plot([plot_min, plot_max], [plot_min + test_rmse, plot_max + test_rmse],
             linestyle=":", color=PARITY_LINE, linewidth=1.5)
    plt.plot([plot_min, plot_max], [plot_min - test_rmse, plot_max - test_rmse],
             linestyle=":", color=PARITY_LINE, linewidth=1.5)

    plt.title("Performance of %s in %s" % (name, mode),
              fontsize=PARITY_TITLE_SIZE)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([plot_min, plot_max])
    plt.ylim([plot_min, plot_max])

    plt.scatter(y_true, y_pred, color=PARITY_FILL, edgecolors=PARITY_EDGE,
                linewidth=1, s=60)

    # only the metals the model misses by more than the baseline are named
    (outlier_dx, outlier_dy), (box_dx, box_dy) = PARITY_OFFSET[mode]
    for i in range(len(y_true)):
        if abs(y_true[i] - y_pred[i]) > label_threshold:
            plt.text(y_true[i] - outlier_dx, y_pred[i] + outlier_dy,
                     "%s" % metals[i],
                     fontdict={"size": 11, "color": PARITY_OUTLIER})

    performance = "\n".join("%s = %.2f" % (METRIC_LABELS[i], value)
                            for i, value in enumerate(
                                [R2(y_true, y_pred), MAE(y_true, y_pred),
                                 test_rmse]))
    plt.text(plot_min + box_dx, plot_max - box_dy, performance,
             fontsize=PARITY_TEXT_SIZE, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.xlabel("Ground Truth (%s)" % unit, fontsize=PARITY_LABEL_SIZE)
    plt.ylabel("Predicted (%s)" % unit, fontsize=PARITY_LABEL_SIZE)
    plt.grid(True, alpha=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(figure)


def PlotElimination(mae_curves, r2_curves, k_opt, mode, path, n_columns=3):
    """The grid of elimination curves, one panel per model.

    The left axis carries the LOOCV R2 and the right axis the LOOCV MAE, so
    that the accuracy and the error of the same feature set can be read off
    together; the grey line marks the optimum of that model."""
    model_list = list(mae_curves)
    n_rows = int(np.ceil(len(model_list) / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(16, 7.5), dpi=500)
    axes = np.atleast_1d(axes).ravel()

    unit = TARGET_UNIT[NormaliseMode(mode)]

    for ax, name in zip(axes, model_list):
        mae_curve = mae_curves[name]
        r2_curve = r2_curves[name]
        x_axis = np.arange(len(mae_curve))

        left, = ax.plot(x_axis, r2_curve, color=ELIM_RED, linewidth=2,
                        marker="o", markersize=5,
                        markerfacecolor="#f5f5f5",
                        markeredgecolor=ELIM_RED, markeredgewidth=1.3,
                        label="$R^{2}$")
        ax.set_ylabel("$R^{2}$", fontsize=12)
        ax.set_xlabel("Number of eliminated features", fontsize=12)
        ax.set_title(name, fontsize=14)

        ax_mae = ax.twinx()
        right, = ax_mae.plot(x_axis, mae_curve, color=ELIM_BLUE, linewidth=2,
                             marker="s", markersize=5,
                             markerfacecolor="#f5f5f5",
                             markeredgecolor=ELIM_BLUE, markeredgewidth=1.3,
                             label="MAE")
        ax_mae.set_ylabel("MAE (%s)" % unit, fontsize=12)

        ax.axvline(k_opt[name], color="grey", linestyle="--", linewidth=1.6,
                   alpha=0.85, zorder=1)

        ax.tick_params(axis="both", labelsize=10)
        ax_mae.tick_params(axis="y", labelsize=10)
        # both y axes are pinned to two decimals, otherwise matplotlib picks the
        # precision per subplot and the panels end up labelled inconsistently
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax_mae.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, linestyle="--", alpha=0.5)

        # the legend is attached to the twin axis: matplotlib draws a twinx
        # axes on top of its parent, so a legend living on `ax` would always be
        # hidden behind the MAE curve of `ax_mae`.  putting it on `ax_mae` with
        # a high zorder and an opaque frame keeps it on top of both curves.
        leg = ax_mae.legend(handles=[left, right], loc="lower left",
                            fontsize=10, framealpha=0.75, facecolor="white",
                            edgecolor="grey")
        leg.set_zorder(20)

    for ax in axes[len(model_list):]:
        ax.axis("off")

    figure.suptitle("Feature Elimination Method for %s Prediction"
                    % NormaliseMode(mode), fontsize=17)
    figure.tight_layout(rect=(0, 0, 1, 0.96), w_pad=0.7, h_pad=1.2)
    figure.savefig(path)
    plt.close(figure)


def PlotShap(features, values, mode, path):
    """The mean absolute SHAP of the descriptors of one target.

    The target picks the hue -- the default blue for the binding energy, red for
    the formation energy and grey for the dissolution potential -- and the bars
    are drawn as a light fill inside a dark outline, the way every other marker
    in these figures is drawn."""
    mode = NormaliseMode(mode)
    figure = plt.figure(figsize=(6, 5), dpi=300)
    plt.barh(features, values, color=SHAP_FILL[mode],
             edgecolor=SHAP_EDGE[mode], linewidth=1.5)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.title("SHAP Explanation for %s" % mode, fontsize=18)
    plt.xlabel("Importance", fontsize=16)
    plt.ylabel("Features", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(figure)
