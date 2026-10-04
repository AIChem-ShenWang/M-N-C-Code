import os
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import numpy as np
import seaborn as sns
from mendeleev import element

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

# the display name of every column, so that the factor figures are driven by the
# very symbols the reports print rather than by a list copied into this file
from utils.ml import symbol_feature_dict

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 0. preparation
# function
def Read_ML_Result(file_path:str, combine:bool=True):
    f = open(file_path, "r", encoding="utf-8")
    text = f.read()

    # Get Baseline
    baseline_pattern = r"Dataset Baseline:\s*R²:\s*([\d.]+)\s*MAE = ([\d.]+)\s*RMSE = ([\d.]+)"
    baseline_match = re.search(baseline_pattern, text)
    if baseline_match:
        baseline = {
            "R²": float(baseline_match.group(1)),
            "MAE": float(baseline_match.group(2)),
            "RMSE": float(baseline_match.group(3))
        }

    # mode test set
    model_pattern = r"ML Model: (\w+).*?test set performance:\s*R² = (-?[\d.]+)\s*MAE = (-?[\d.]+)\s*RMSE = (-?[\d.]+)"
    model_matches = re.findall(model_pattern, text, re.DOTALL)

    models = {}
    if combine:
        models["baseline"] = baseline

    for match in model_matches:
        model_name = match[0]
        models[model_name] = {
            "R²": float(match[1]),
            "MAE": float(match[2]),
            "RMSE": float(match[3])
        }
    if combine:
        return models
    else:
        return baseline, models


def RetainedFeatures(target):
    """The feature set the elimination loop kept for one target.

    The report prints `Retained features:` once per model and then a final time
    under `Best model of the elimination loop:`; it is that last line that names
    the optimum now carried by the factor figures, so the last match wins."""
    text = open("%s ML report.txt" % target, "r", encoding="utf-8").read()
    matches = re.findall(r"Retained features:\s*(.+)", text)
    if not matches:
        raise RuntimeError("no retained features in the %s report" % target)
    symbols = [s.strip() for s in matches[-1].split(",")]
    return symbols, [symbol_feature_dict[s] for s in symbols]

# import data
data = pd.read_excel("../data/M-N-C data set.xlsx")
metals = data.iloc[:, 0].to_list()
GN = data.loc[:]["group number"]
AN =data.loc[:]["atomic number"]
E_b = data.loc[:]["E_b/eV"]
E_f = data.loc[:]["E_f/eV"]
U_diss_acid = data.loc[:]["U_diss_acid/V"]
U_diss_base = data.loc[:]["U_diss_base/V"]
U_diss = []
for i in range(len(U_diss_acid)):
    U_diss.append(max(U_diss_base[i], U_diss_acid[i]))
U_diss = pd.Series(np.array(U_diss))

if not os.path.exists("../figures/data analysis"):
    os.makedirs("../figures/data analysis")



# 1.plotting for data distribution
# 1.1 E_b
plt.figure(figsize=(13,3), dpi=500)

# reindex by group number and atomic number
sorted_df = pd.DataFrame({
    'GN': GN,
    'AN': AN,
    'E_b': E_b
})
sorted_df.index = metals
sorted_df = sorted_df.sort_values(by=['GN', 'AN'])
sorted_E_b = sorted_df['E_b'].values
sorted_GN = sorted_df['GN'].values
sorted_metals = sorted_df.index

x_pos = []
x_gap = []
GN_temp = 0

for i in range(len(sorted_GN)):
    if sorted_GN[i]-1 != GN_temp:
        GN_temp = sorted_GN[i]-1
        x_gap.append(int(i + GN_temp))
    x_pos.append(int(i + GN_temp) + 1)

# beautify
plt.grid(True, linestyle="--", color="grey", alpha=0.2)
plt.xticks([])
plt.yticks(fontsize=10)
plt.xlabel("Group Number", fontsize=12)
plt.ylabel("Binding Energy / eV", fontsize=12)
extend = 1.2
plt.xlim(x_pos[0] - extend, x_pos[-1] + extend)
ymin = min(E_b)- 1.5 * extend
plt.ylim(ymin, max(E_b) + extend)

# splitting of group number
alpha = np.linspace(0.05, 0.95, len(x_gap)+1)
for i in range(len(x_gap)):
    # division line
    # plt.axvline(x_gap[i], color="gray", alpha=0.6, linestyle="--")

    # region
    if i == 0:
        plt.fill_between([x_pos[0] - extend, x_gap[0]], ymin, max(E_b) + extend, color="#9fbbd5", alpha=alpha[i])
    if i == len(x_gap)-1:
        plt.fill_between([x_gap[-1], x_pos[-1] + extend], ymin, max(E_b) + extend, color="#9fbbd5", alpha=alpha[i+1])
    plt.fill_between([x_gap[i-1], x_gap[i]], ymin, max(E_b) + extend, color="#9fbbd5", alpha=alpha[i])

# plt.axhline(0, color="grey", linestyle="--")

# data points
plt.scatter(x_pos, sorted_E_b, s=40, color='#9fbbd5', edgecolor="#3a4b6e", linewidths=1.5)

# metal names
for i in range(len(sorted_metals)):
    M = sorted_metals[i]
    # manually adjust
    x_adjust = 0
    if M in  ["Mg", "Nd", "Mn"]:
        x_adjust = 0.2
    plt.text(x_pos[i] - x_adjust, sorted_E_b[i] + 0.28, "%s" % sorted_metals[i], ha="center", va="bottom", color="#3a4b6e", fontsize=8)

# group number label
unique_GNs = np.unique(sorted_GN)
y_bottom = plt.ylim()[0] + 0.2
for idx, gn in enumerate(unique_GNs):
    if idx == 0:
        left = x_pos[0] - extend
        right = x_gap[0]
    elif idx == len(unique_GNs) - 1:
        left = x_gap[-1]
        right = x_pos[-1] + extend
    else:
        left = x_gap[idx - 1]
        right = x_gap[idx]
    x_center = (left + right) / 2
    plt.text(x_center, y_bottom, str(gn),
             ha='center', va='bottom', fontsize=12, color='#3a4b6e')

plt.title("$E_{b}$ distribution", fontsize=18)
plt.savefig("../figures/data analysis/E_b distribution.png")


# 1.2 Ef & U_diss
# set background
sns.set_style("white")
sns.set_context("notebook", font_scale=1.2)

# scatter plot
g = sns.jointplot(x=E_f, y=U_diss, kind='scatter',
                  height=5,
                  color='#9fbbd5',
                  edgecolor='#3a4b6e',
                  alpha=0.9,
                  s=50,
                  linewidths=1,
                  marginal_kws={
                      'kde': True,
                      'color': '#9fbbd5',
                      'linewidth': 1.5,
                      'fill': True,
                  })

for i in range(len(metals)):
    g.ax_joint.text(x=E_f[i], y=U_diss[i] + 0.08, s=metals[i], ha='center', va='bottom', fontsize=10, color='#3a4b6e')

# beautify
g.fig.set_size_inches(7, 5)
g.fig.suptitle('$E_{f}$ & $U_{diss}$ Distribution', fontsize=18, y=1.05)
g.set_axis_labels('$E_{f}$ / eV', '$U_{diss}$ / V', fontsize=12)
g.ax_joint.grid(True, linestyle=':', alpha=0.6)
g.ax_marg_x.tick_params(labelsize=6)
g.ax_marg_y.tick_params(labelsize=6)

# the marginals are keyed to the background colours of the joint plot and to
# their own colour family: the top strip takes the blue region colour and the
# right strip the red one.  The bars are filled with exactly the colour of the
# shaded region beside them -- the same base colour at the same 0.25 of alpha --
# so the strip reads as a continuation of that region, while the outline and the
# kernel curve both use the base colour at full strength, which is what actually
# defines the strip's hue.
for ax_marg, base in ((g.ax_marg_x, '#9fbbd5'), (g.ax_marg_y, '#d69d98')):
    for patch in ax_marg.patches:
        patch.set_facecolor(to_rgba(base, 0.25))
        patch.set_edgecolor(base)
        patch.set_linewidth(1.2)
    for line in ax_marg.lines:
        line.set_color(base)
        line.set_linewidth(1.8)

# stable metals
# line
g.ax_joint.axvline(x=0, linestyle='--', color='gray', linewidth=1, alpha=0.7)
g.ax_joint.axhline(y=0, linestyle='--', color='gray', linewidth=1, alpha=0.7)
# region
xlim = g.ax_joint.get_xlim()
ylim = g.ax_joint.get_ylim()

# x-axis for U_diss
space_extend = 20
x_start = max(0, xlim[0])
x_end = xlim[1]
if x_start < x_end:
    x_positive = np.linspace(x_start, x_end+space_extend, 2)
    g.ax_joint.fill_between(x_positive, ylim[0]-space_extend, ylim[1]+space_extend,
                             color='#d69d98', alpha=0.25, zorder=0)

# y-axis for E_f
y_bottom = ylim[0]
y_top = min(0, ylim[1])
if y_bottom < y_top:
    x_range = np.linspace(xlim[0]-space_extend, xlim[1]+space_extend, 2)
    g.ax_joint.fill_between(x_range, y_bottom-space_extend, y_top,
                             color='#d69d98', alpha=0.25, zorder=0)

# space with stable metals
g.ax_joint.fill_between([xlim[0]-space_extend, 0], 0, ylim[1]+space_extend,
                             color='#9fbbd5', alpha=0.25, zorder=0)


# unstable metal should have red color
for i in range(len(metals)):
    if E_f[i] >= 0 or U_diss[i] <= 0:
        g.ax_joint.text(x=E_f[i], y=U_diss[i] + 0.08, s=metals[i], color='#ba3e45', ha='center', va='bottom', fontsize=10)
        g.ax_joint.scatter(x=E_f[i], y=U_diss[i],
                  color='#d69d98',
                  edgecolor='#ba3e45',
                  alpha=0.9,
                  s=50,
                  linewidths=1)

extend = 0.5
g.ax_joint.set_xlim(xlim[0]-extend, xlim[1]+extend)
g.ax_joint.set_ylim(ylim[0]-extend, ylim[1]+extend)

plt.savefig("../figures/data analysis/E_f & U_diss distribution.png", dpi=500,
            bbox_inches='tight')



# 2.Plot for ML results
# read the ML learning performance result.  These reports are written by
# `machine learning.py` next to itself, in this directory.
# the report file names carry no LaTeX markup, only the figures do
E_b_res = Read_ML_Result("E_b ML report.txt")
E_f_res = Read_ML_Result("E_f ML report.txt")
U_diss_res = Read_ML_Result("U_diss ML report.txt")

# 3 data sets
datasets = [E_b_res, E_f_res, U_diss_res]
dataset_names = ['$E_{b}$', '$E_{f}$', '$U_{diss}$']
models = list(E_b_res.keys())
n_datasets = len(datasets)
n_models = len(models)
n_groups = n_datasets * n_models

# bar parameters
bar_width = 0.5
spacing_group = 0.3
group_centers = np.arange(n_groups) * (3*bar_width + spacing_group)
offset_R2 = -bar_width
offset_MAE = 0
offset_RMSE = bar_width

# data tuple
R2_values = []
MAE_values = []
RMSE_values = []
labels = []


for ds_name, ds_data in zip(dataset_names, datasets):
    for model in models:
        R2_values.append(ds_data[model]['R²'])
        MAE_values.append(ds_data[model]['MAE'])
        RMSE_values.append(ds_data[model]['RMSE'])
        labels.append(f"{model}")

# plotting
fig, ax1 = plt.subplots(figsize=(18, 4))

# left y-axis
ax1.set_ylabel('$R^{2}$', fontsize=12)
ax1.set_ylim(0, 1.1)
ax1.tick_params(axis='y', labelsize=10)

# right y-axis
ax2 = ax1.twinx()
all_mae_rmse = MAE_values + RMSE_values
ymin = min(all_mae_rmse) * 0.9
ymax = max(all_mae_rmse) * 1.1
ax2.set_ylim(ymin, ymax)
ax2.set_ylabel('MAE / RMSE', fontsize=12)
ax2.tick_params(axis='y', labelsize=10)

# bars
# R2
bars_R2 = ax1.bar(group_centers + offset_R2, R2_values, bar_width,
                   label='$R^{2}$', color='lightgrey', edgecolor='grey', linewidth=1)
for bar in bars_R2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', fontsize=6, color='grey', weight='bold')
# MAE
bars_MAE = ax2.bar(group_centers + offset_MAE, MAE_values, bar_width,
                    label='MAE', color='#9fbbd5', edgecolor='#3a4b6e', linewidth=1)
for bar in bars_MAE:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom', fontsize=6, color='#3a4b6e', weight='bold')
# RMSE
bars_RMSE = ax2.bar(group_centers + offset_RMSE, RMSE_values, bar_width,
                     label='RMSE', color='#d69d98', edgecolor='#ba3e45', linewidth=1)
for bar in bars_RMSE:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom', fontsize=6, color='#ba3e45', weight='bold')


# x axis
ax1.set_xticks(group_centers)
ax1.set_xticklabels(labels, ha='center', fontsize=12)
left_margin = group_centers[0] - 2 * bar_width
right_margin = group_centers[-1] + 2 * bar_width
ax1.set_xlim(left_margin, right_margin)

# legand
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

# division line
n_models_per_dataset = len(models)
for i in range(n_datasets - 1):
    last_idx = (i + 1) * n_models_per_dataset - 1
    next_idx = (i + 1) * n_models_per_dataset
    sep_x = (group_centers[last_idx] + group_centers[next_idx]) / 2
    ax1.axvline(x=sep_x, color='gray', linestyle='--', linewidth=1, alpha=0.7)


# data set title
for i in range(n_datasets):
    start_idx = i * n_models_per_dataset
    end_idx = (i + 1) * n_models_per_dataset - 1
    mid_x = (group_centers[start_idx] + group_centers[end_idx]) / 2
    ax1.text(mid_x, 0.91, dataset_names[i] + " prediction",
             transform=ax1.get_xaxis_transform(),
             ha='center', va='bottom',
             fontsize=15,
             bbox=dict(facecolor='white',
                       boxstyle='round,pad=0.05',
                       edgecolor='white',
                       linestyle='--',
                       linewidth=0.7))

ax1.grid(axis='y', linestyle='--', alpha=0.5)


plt.title('Model Performance', fontsize=20, y=1.01)

plt.tight_layout()
plt.savefig("../figures/data analysis/ML Performance.png", dpi=500)



# 3. the stable M-N4-C screen and the price table are not here
# The screen, the mendeleev price table that the threshold uses, Table S3 and
# the report of which metal fails which criterion all live in
# `stable table generator.py`.  They were moved out of this script so that the
# screen has one home and cannot drift from the table; this script only draws
# the figures.  Run that script first if the screening outputs are wanted.

# 4.d band center distribution
e_center = data.loc[:]["band center/eV"]
period = []
for M in metals:
    M = element(M)
    period.append(M.period)

sorted_df = pd.DataFrame({
    'AN': AN,
    'bc': e_center,
    "period": period,
})
sorted_df.index = metals
sorted_df = sorted_df.sort_values(by=['AN'])
sorted_bc = sorted_df['bc'].values
sorted_AN = sorted_df['AN'].values
sorted_period = sorted_df['period'].values
sorted_metals = sorted_df.index

x_pos = []
x_gap = []
period_temp = 1

for i in range(len(sorted_period)):
    if sorted_period[i]-1 != period_temp:
        period_temp = sorted_period[i]-1
        x_gap.append(int(i + period_temp))
    x_pos.append(int(i + period_temp) + 1)

# x_pos = []
# for i in range(len(sorted_AN)):
#     x_pos.append(i)

# plotting
plt.figure(figsize=(13,4), dpi=500)
for i, (x, an) in enumerate(zip(x_pos, sorted_AN)):
    M = sorted_metals[i]
    bc = sorted_bc[i]

    if bc >= 0:
        plt.plot([x, x], [0, bc], color="#3a4b6e", zorder=1)
        plt.scatter(x, bc, s=50, color="#9fbbd5", edgecolors="#3a4b6e", zorder=2)
        # the label of a positive band centre sits above its marker, at a
        # smaller offset than the negative case so that the crowded top half of
        # the figure stays readable
        plt.text(x, bc + 0.25, M, color="#3a4b6e", ha='center', va='bottom', fontsize=9.0)
    else:
        plt.plot([x, x], [bc, 0], color="#ba3e45", zorder=1)
        plt.scatter(x, bc, s=50, color="#d69d98", edgecolors="#ba3e45", zorder=2)
        # the label keeps the same 0.5 eV gap as the positive case, only on the
        # other side of the marker; anchoring it by its top edge puts it below
        plt.text(x, bc - 0.5, M, color="#ba3e45", ha='center', va='top', fontsize=9.0)

# division line
for i in range(len(x_gap)):
    plt.axvline(x_gap[i], color="gray", alpha=0.5, linestyle="--")

# data range
extend = 3
plt.ylim([min(e_center) - extend, max(e_center) + extend])
plt.xlim([min(x_pos) - extend, max(x_pos) + extend])
plt.axhline(0, color="grey", linestyle="-", alpha=0.7)

# background
ax = plt.gca()
ax.axhspan(0, ax.get_ylim()[1], facecolor='#9fbbd5', alpha=0.15, zorder=0)
ax.axhspan(ax.get_ylim()[0], 0, facecolor='#d69d98', alpha=0.15, zorder=0)

# period number label
unique_periods = np.unique(sorted_period)
y_bottom = plt.ylim()[0] + 0.2
for idx, gn in enumerate(unique_periods):
    if idx == 0:
        left = x_pos[0] - extend
        right = x_gap[0]
    elif idx == len(unique_periods) - 1:
        left = x_gap[-1]
        right = x_pos[-1] + extend
    else:
        left = x_gap[idx - 1]
        right = x_gap[idx]
    x_center = (left + right) / 2
    plt.text(x_center, y_bottom, str(gn),
             ha='center', va='bottom', fontsize=13.0, color="#ba3e45")

# label & title
plt.grid(True, linestyle="--", color="grey", alpha=0.2)
plt.xticks([])
plt.yticks(fontsize=11.5)
plt.ylabel("Band Center / eV", fontsize=15.5)
plt.xlabel("Period Number", fontsize=15.5)
plt.title("Band Center Distribution", fontsize=19.5)
plt.tight_layout()
plt.savefig("../figures/data analysis/band center distribution.png")



# 6. plotting for the descriptors of E_f, plus the atomic number they all follow
# CM1 is the average diagonal element of the Coulomb matrix of the M-N4 site.
CM1 = data.loc[:]["CM1"]
R_CO = data.loc[:]["covalent_radius_cordero/pm"]
CM2 = data.loc[:]["CM2"]
AW = data.loc[:]["atomic wight/g mol-1"]
GN = data.loc[:]["group number"]
colors = ['#003153', "#f5f5f5", '#85120f']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)


def multi_gaussian(x, *p):
    """A sum of (len(p) - 1) / 3 Gaussians plus a constant offset."""
    out = np.full_like(x, p[-1], dtype=float)
    for i in range((len(p) - 1) // 3):
        A, mu, sigma = p[3 * i], p[3 * i + 1], p[3 * i + 2]
        out = out + A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return out


def DoubleGaussianFit(x, y):
    """Double-Gaussian fit of E_f against one descriptor, and its R2.

    Two components with the amplitudes left free in sign, which is what this
    figure has always used: E_f against CM1 is a valley, so a component that
    describes it carries a negative amplitude and forcing the amplitudes positive
    is not the same fit.

    The centres, though, are held within one span of the data.  Without that the
    amplitude and the centre trade off along a flat direction and the fit runs to
    coefficients that mean nothing: for CM1 it reports an amplitude of -2.5e36
    with its centre at -129170, on a descriptor that lives between 44 and 4077.
    Such a component is a numerical artefact rather than a feature, since a peak
    whose centre is further from the data than the data is wide cannot be a peak
    in it.  Confining the centres costs nothing on the fit -- CM1 moves from
    0.8749 to 0.8743 -- and it makes the coefficients printable, which is what the
    equation inside the panel needs.  On AW and CM2 the confined search actually
    scores better (0.889 against 0.858, 0.827 against 0.777), because the free fit
    was settling in a worse optimum.

    The two peaks are seeded at the terciles of the descriptor and their widths at
    a tenth of its span."""
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    bounds = ([
        -np.inf, x.min() - span, 1e-9,
        -np.inf, x.min() - span, 1e-9,
        -np.inf,
    ], [
        np.inf, x.max() + span, 10 * span,
        np.inf, x.max() + span, 10 * span,
        np.inf,
    ])
    p0 = [y.max() - y.min(), np.quantile(x, 0.33), 0.1 * span,
          y.max() - y.min(), np.quantile(x, 0.66), 0.1 * span,
          y.mean()]
    popt, _ = curve_fit(multi_gaussian, x, y, p0=p0, bounds=bounds, maxfev=200000)
    return popt, r2_score(y, multi_gaussian(x, *popt))


# one panel per retained descriptor of E_f, in the order the report lists them
E_F_SYMBOLS, E_F_COLUMNS = RetainedFeatures("E_f")
E_F_FACTORS = [(s, data.loc[:, c]) for s, c in zip(E_F_SYMBOLS, E_F_COLUMNS)]


def MathItalic(label):
    """Set a descriptor name in italic, for the axis and panel titles.

    A name is the LaTeX of this study already if it carries a `$`, and wrapping
    that in `\\mathit{}` would nest the math delimiters -- `\\mathit{$Z$}` is
    not parseable -- so only the plain names are wrapped."""
    return label if "$" in label else "$\\mathit{%s}$" % label


def UprightName(label):
    """A descriptor name with its initialisms set upright.

    The counterpart of `MathItalic` for a sheet whose plain names are acronyms
    rather than symbols: `CM1`, `AW` and `GN` are initialisms, and an italic
    acronym reads as the product of its letters.  A name that is the LaTeX of this
    study already carries a `$` and keeps the italic of a symbol, and a plain
    single letter is a symbol too, so both are left to `MathItalic`."""
    if "$" in label or (len(label) > 1 and label.isupper()):
        return label
    return MathItalic(label)


def LinearFit(x, y):
    """The least-squares line y = c x + b, returned as (slope, intercept, R2)."""
    slope, intercept = np.polyfit(np.asarray(x, dtype=float),
                                  np.asarray(y, dtype=float), 1)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return slope, intercept, r2_score(y, slope * x + intercept)


def FormatNumber(value, digits=3):
    """A coefficient, in scientific notation when it is very small or large.

    The format is built by concatenation rather than by nesting `%` operators: a
    format string that has to produce another format string cannot be written as
    one conversion, and `"%%.%df" % digits` silently yields `%.-7f`."""
    if value == 0:
        return "0"
    spec = "%." + str(digits) + "f"
    if 1e-2 <= abs(value) < 1e4:
        return spec % value
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return (spec % mantissa) + "\\times10^{%d}" % exponent


def MathBody(label):
    """The inside of a LaTeX name, so that it can be embedded in a longer
    equation: `$E_{b}$` becomes `E_{b}`, and the equation supplies its own
    math delimiters rather than nesting a second pair."""
    return label.strip("$")


def LinearEquation(slope, intercept, variable, target):
    """The fitted line as the LaTeX written inside the panel.

    Only the formula is built here: the R2 that goes with it is carried by the
    legend, so the string stays a single mathematical statement."""
    sign = "+" if intercept >= 0 else "-"
    return ("$%s = %s\\,%s %s %s$"
            % (MathBody(target), FormatNumber(slope), MathBody(variable),
               sign, FormatNumber(abs(intercept))))


def SignedDifference(variable, centre):
    """`(x - c)` with the sign hoisted, so a negative centre adds rather than
    subtracting a negative: `(CM1--3989)` is not readable, `(CM1+3989)` is."""
    sign = "-" if centre >= 0 else "+"
    return "(%s %s %s)" % (variable, sign, FormatNumber(abs(centre)))


def DoubleGaussianEquation(popt, variable, target):
    """The fitted two-peak Gaussian, with every parameter substituted.

    Both components are printed out in full, and the second carries its own
    leading sign, so the expression can be evaluated as printed.  Each component
    gets a line of its own and the offset a third: the whole expression is far
    too wide to read as one or two lines, and splitting it is what keeps it
    legible in the report the equation is written to."""
    amplitude_1, centre_1, width_1 = popt[0], popt[1], popt[2]
    amplitude_2, centre_2, width_2 = popt[3], popt[4], popt[5]
    offset = popt[6]
    v = MathBody(variable)

    def peak(centre, width):
        return "e^{-%s^{2}/(2\\cdot%s^{2})}" % (
            SignedDifference(v, centre), FormatNumber(width))

    sign_2 = "+" if amplitude_2 >= 0 else "-"
    sign_e = "+" if offset >= 0 else "-"
    return ("$%s = %s\\,%s$\n$%s\\,%s\\,%s$\n$%s\\,%s$"
            % (MathBody(target), FormatNumber(amplitude_1),
               peak(centre_1, width_1),
               sign_2, FormatNumber(abs(amplitude_2)), peak(centre_2, width_2),
               sign_e, FormatNumber(abs(offset))))


def SinusoidBracket(constant, slope, variable):
    """One `(a + b x)` bracket of the sinusoid, with a negative slope as a minus.

    The sign is hoisted out of the coefficient so that `a + -0.14 x` never
    appears: the expression is meant to be read straight off the panel."""
    sign = "+" if slope >= 0 else "-"
    return "(%s %s %s\\,%s)" % (FormatNumber(constant), sign,
                                FormatNumber(abs(slope)), variable)


def SinusoidEquation(x, wavelength, chirp, coef, variable, target):
    """The fitted chirp sinusoid, with every parameter substituted as a number.

    Nothing is left as a symbol, so the expression can be evaluated as printed:
    the sin and cos coefficients, the offset, the period and the origin the phase
    is measured from all appear explicitly.  The origin needs stating as much as
    the period does -- the phase runs from the smallest x in the sample rather
    than from zero, so a reader given only `x_min` cannot evaluate the sine.

    The phase is spelled out separately, because the phase is where the growing
    period lives: it is the quadratic term in `chirp` that lets the period widen
    with x, and `chirp = 0` is the rigid sinusoid."""
    # `coef` multiplies [x sin, sin, x cos, cos, 1], so the coefficient of x comes
    # before the constant one in each bracket
    coefficient_xsin, constant_sin, coefficient_xcos, constant_cos, e = coef
    a, b = constant_sin, coefficient_xsin
    c, d = constant_cos, coefficient_xcos
    span = float(x.max() - x.min())
    origin = float(x.min())
    v = MathBody(variable)

    # the growing-period term, with the sign pulled out of the fraction so that a
    # negative chirp reads as a leading minus rather than as a fraction whose
    # numerator begins with one
    growth = ""
    if abs(chirp) > 1e-3:
        sign = "-" if chirp < 0 else "+"
        growth = (" %s \\frac{%s\\,(%s-%s)^{2}}{2\\cdot%s\\cdot%s}"
                  % (sign, FormatNumber(abs(chirp)), v, FormatNumber(origin),
                     FormatNumber(span), FormatNumber(wavelength)))

    offset_sign = "+" if e >= 0 else "-"
    return "\n".join([
        "$%s = %s\\sin\\varphi + %s\\cos\\varphi %s %s$"
        % (MathBody(target), SinusoidBracket(a, b, v),
           SinusoidBracket(c, d, v), offset_sign, FormatNumber(abs(e))),
        "$\\varphi = 2\\pi\\left[\\frac{%s - %s}{%s}%s\\right]$"
        % (v, FormatNumber(origin), FormatNumber(wavelength), growth),
    ])

def FactorGrid(n_panels, panel_width=10.5, panel_height=6.5, ncols=0):
    """A figure of `n_panels` factor panels.

    The retained feature set is read from the reports, so its size changes with
    the data set and the layout cannot be hard-coded: a single row keeps every
    panel in one band, and the figure grows sideways instead of wrapping.  Pass
    `ncols` to wrap the panels into that many columns instead, which trades the
    single band for a sheet that stays closer to square -- the E_b sheet asks for
    three columns, so its six panels sit as 2 x 3.  The panels are filled left to
    right, top to bottom, in the order of the retained feature set."""
    ncols = ncols if ncols > 0 else n_panels
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_width * ncols, panel_height * nrows),
                             dpi=500, squeeze=False)
    return fig, axes.ravel()


fig, axes = FactorGrid(len(E_F_FACTORS))
for ax, (label, values) in zip(axes, E_F_FACTORS):
    x = np.asarray(values, dtype=float)
    sc = ax.scatter(x, E_f, c=E_f, cmap=custom_cmap, s=100,
                    edgecolor='k', linewidth=0.5, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('$E_{f}$ value', fontsize=16)

    popt, r2 = DoubleGaussianFit(x, E_f)
    x_fit = np.linspace(x.min(), x.max(), 500)
    ax.plot(x_fit, multi_gaussian(x_fit, *popt), 'grey', lw=3,
            label='$R^{2}$=%.2f' % r2, zorder=4)

    ax.set_xlabel(UprightName(label), fontsize=16)
    ax.set_ylabel('$E_{f}$', fontsize=16)
    ax.set_title('%s vs. $E_{f}$' % UprightName(label), fontsize=20)
    ax.legend(loc="lower left", fontsize=16, prop={'size': 16})
    ax.grid(True, linestyle='--', alpha=0.6, color="grey")
fig.tight_layout()
plt.savefig("../figures/ML/E_f factors.png")
plt.close(fig)


# 6b. the same view for U_diss, with every retained panel fitted by the same
# functional form: a sinusoid whose period is allowed to grow with x.  The
# oscillation is genuine rather than an artefact of the eye -- against Z a grid
# search takes R2 to 0.63 and leave-one-out keeps 0.51 of it, where a straight
# line manages 0.08.
#
# One form is used throughout rather than a rigid sinusoid for some panels and a
# chirp for others.  The reason is that the chirp is only worth having where the
# axis warrants it, and the fit itself decides that: against CM1, which runs from
# 44 to 4077, the period is driven out more than threefold over the range, from
# 722 to 2391 (chirp -0.698, R2 0.54), and even against Z, which spans only 3 to
# 80, it widens a little, from 14.3 to 15.3 (chirp -0.065, R2 0.64).  Only AW is
# left on a rigid sinusoid (chirp 0.000, period 43.2, R2 0.62).  Applying the
# same form everywhere keeps the three panels directly comparable, which fitting
# them with two different forms would not.
#
# The chirp on CM1 is also physically sensible rather than a curve-fitting
# convenience: CM1 is a d-band integral that accumulates as the d shell fills, so
# the spacing between successive features is expected to widen along a period
# instead of repeating rigidly.
#
# A grid is used instead of `curve_fit` because the residual is badly non-convex
# in the frequency: an unconstrained `curve_fit` settles near R2 = 0.09 for Z, a
# local optimum.  A double Gaussian is the wrong shape here too: U_diss is single
# peaked and reaches only 0.33.
#
# The target is U_diss_acid, the acid value at pH 0, which is the column the
# machine-learning scripts predict.  It is NOT the `U_diss` series above, which is
# max(acid, base) and belongs to the stability screening instead.
def chirp_phase(x, wavelength, chirp):
    r"""The phase of a sinusoid whose instantaneous period grows with x.

    The local period is P(u) = wavelength / (1 + chirp u/span) in the shifted
    coordinate u = x - min(x), so chirp < 0 makes P grow as x increases.  The
    phase is 2 pi times the integral of 1/P, which is
        2 pi (u/wavelength + chirp u^2 / (2 span wavelength)),
    and chirp = 0 gives the rigid sinusoid back."""
    span = float(x.max() - x.min())
    u = x - x.min()
    return 2 * np.pi * (u / wavelength + chirp * u ** 2 / (2 * span * wavelength))


def sinusoid_design(x, wavelength, chirp=0.0):
    """(a + b x) sin(phi) + (c + d x) cos(phi) + e, with phi the chirp phase."""
    phi = chirp_phase(np.asarray(x, dtype=float), wavelength, chirp)
    return np.column_stack([x * np.sin(phi), np.sin(phi),
                            x * np.cos(phi), np.cos(phi), np.ones_like(x)])


def BestChirp(x, y, n_wavelength=200, n_chirp=42, n_refine=3, window=4.0):
    """The (wavelength, chirp) that maximises R2, on a grid then refined.

    A coarse grid over both parameters is followed by `n_refine` local passes
    that shrink the step size by 12 each time.  The refinement window is the
    thing to get right: R2 is oscillatory in the wavelength, so neighbouring
    harmonics are near-degenerate and the search can settle one harmonic away
    from the best.  A window of only one or two coarse steps therefore locks the
    answer in wherever the coarse pass happened to land; opening it to
    `window` coarse steps makes this staged search reach the optimum of a
    600 x 96 dense grid (R2 = 0.5274 for CM1) at about a twentieth of its cost.

    The rigid sinusoid is part of the search: chirp = 0 is in the grid and the
    search keeps it whenever the extra parameter does not pay, which is what
    happens for Z and AW."""
    span = x.max() - x.min()
    best = (-9.0, None, None, None)

    def sweep(wavelengths, chirps):
        nonlocal best
        for wavelength in wavelengths:
            for chirp in chirps:
                A = sinusoid_design(x, wavelength, chirp)
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                score = r2_score(y, A @ coef)
                if score > best[0]:
                    best = (score, wavelength, coef, chirp)

    wavelengths = np.linspace(span / 40.0, span * 2.0, n_wavelength)
    chirps = np.concatenate([[0.0], -np.linspace(0.05, 0.95, n_chirp - 1)])
    sweep(wavelengths, chirps)

    step_w = wavelengths[1] - wavelengths[0]
    step_c = 0.9 / (n_chirp - 1)
    for _ in range(n_refine):
        centre_w, centre_c = best[1], best[3]
        sweep(np.linspace(max(centre_w - window * step_w, 1e-6),
                          centre_w + window * step_w, 61),
              np.linspace(max(centre_c - window * step_c, -0.99),
                          min(centre_c + window * step_c, 0.0), 61))
        step_w /= 12.0
        step_c /= 12.0
    return best


# every descriptor the elimination loop retained for U_diss, one panel apiece, with
# the chirp sinusoid the study uses for U_diss drawn on it.  The one exception is
# GN: the sinusoid reproduces the other three descriptors of the sheet (R2 = 0.54,
# 0.64 and 0.62) but not this one (R2 = 0.32), so that panel alone is left as a
# plain scatter.  The four panels are wrapped into two columns, so the sheet reads
# as 2 x 2, which puts GN in the bottom-right corner.
U_DISS_SYMBOLS, U_DISS_COLUMNS = RetainedFeatures("U_diss")
U_DISS_FACTORS = [(s, data.loc[:, c])
                  for s, c in zip(U_DISS_SYMBOLS, U_DISS_COLUMNS)]
U_DISS_NO_FIT = {"GN"}      # the descriptor the sinusoid does not reproduce

fig, axes = FactorGrid(len(U_DISS_FACTORS), ncols=2)
for ax, (label, values) in zip(axes, U_DISS_FACTORS):
    x = np.asarray(values, dtype=float)

    sc = ax.scatter(x, U_diss_acid, c=U_diss_acid, cmap=custom_cmap, s=100,
                    edgecolor='k', linewidth=0.5, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('$U_{diss}$ value', fontsize=16)

    if label not in U_DISS_NO_FIT:
        x_fit = np.linspace(x.min(), x.max(), 800)
        score, wavelength, coef, chirp = BestChirp(x, U_diss_acid)
        ax.plot(x_fit, sinusoid_design(x_fit, wavelength, chirp) @ coef, 'grey',
                lw=3, zorder=4, label='$R^{2}$=%.2f' % score)
        ax.legend(loc="upper left", fontsize=16, prop={'size': 16})

        # the fitted period is not put in the legend, only reported on the console
        print("   %-8s period %8.1f -> %8.1f  (chirp %+.3f)  R2 = %.3f"
              % (label, wavelength, wavelength / (1.0 + chirp), chirp, score))
    else:
        print("   %-8s drawn as a plain scatter, the sinusoid is not fitted"
              % label)

    ax.set_xlabel(UprightName(label), fontsize=16)
    ax.set_ylabel('$U_{diss}$ (V)', fontsize=16)
    ax.set_title('%s vs. $U_{diss}$' % UprightName(label), fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.6, color="grey")
fig.tight_layout()
plt.savefig("../figures/ML/U_diss factors.png")
plt.close(fig)


# 6c. the same view for E_b, as a plain scatter.  Every descriptor the elimination
# loop retained for E_b is plotted against the binding energy, one panel apiece.
# Most of them carry no closed form: the point of the figure is to show what each
# descriptor does on its own, before any functional form is imposed on it, and only
# $H_f$ has a form that is worth drawing -- a straight line, which is the relation
# the SISSO equation of E_b and the permutation importance both single out.
# The six panels are wrapped into three columns, so the sheet reads as 2 x 3.
# The type is set a little larger than on the other two factor sheets, whose
# panels are fewer and wider: the tick labels there are matplotlib's default and
# read small on a panel of this size.  The step is deliberately modest, so that
# the sheet still sits beside the others.
E_B_FS_TICK = 14            # axis tick labels
E_B_FS_LABEL = 20           # axis labels and the colour-bar label
E_B_FS_TITLE = 26           # panel title
E_B_FS_LEGEND = 20          # the fit annotation of the $H_f$ panel

E_B_SYMBOLS, E_B_COLUMNS = RetainedFeatures("E_b")
E_B_LINEAR = "$H_{f}$"      # the one descriptor the straight line is drawn for
E_B_FACTORS = [(s, data.loc[:, c], s == E_B_LINEAR)
               for s, c in zip(E_B_SYMBOLS, E_B_COLUMNS)]

fig, axes = FactorGrid(len(E_B_FACTORS), ncols=3)
for ax, (label, values, linear) in zip(axes, E_B_FACTORS):
    x = np.asarray(values, dtype=float)
    sc = ax.scatter(x, E_b, c=E_b, cmap=custom_cmap, s=100,
                    edgecolor='k', linewidth=0.5, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('$E_{b}$ value', fontsize=E_B_FS_LABEL)
    cbar.ax.tick_params(labelsize=E_B_FS_TICK)

    # only $H_f$ is fitted, and only with a straight line: y = c x + b.  The
    # legend carries the R2 alone, as on the other two factor sheets; the equation
    # itself is written out in `Best descriptor for each target report.txt`.
    if linear:
        slope, intercept, r2 = LinearFit(x, E_b)
        x_fit = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_fit, slope * x_fit + intercept, 'grey', lw=3, zorder=4,
                label='$R^{2}$=%.2f' % r2)
        ax.legend(loc='lower left', fontsize=E_B_FS_LEGEND,
                  prop={'size': E_B_FS_LEGEND})

    ax.set_xlabel(label, fontsize=E_B_FS_LABEL)
    ax.set_ylabel('$E_{b}$', fontsize=E_B_FS_LABEL)
    ax.set_title('%s vs. $E_{b}$' % label, fontsize=E_B_FS_TITLE)
    ax.tick_params(axis='both', labelsize=E_B_FS_TICK)
    ax.grid(True, linestyle='--', alpha=0.6, color="grey")
fig.tight_layout()
plt.savefig("../figures/ML/E_b factors.png")
plt.close(fig)


# 6d. one sheet that shows only the single most informative descriptor of each
# target, the panel that matters out of each of the three figures above.  Which
# descriptor that is is not written down here: for every target each retained
# descriptor is fitted with the form this study uses for that target -- a straight
# line for E_b, a double Gaussian for E_f, a chirp sinusoid for U_diss -- and the
# one that reproduces the target best on its own is the panel that is drawn.  The
# set of candidates therefore follows the elimination loop, and the choice cannot
# drift out of step with `E_b/E_f/U_diss factors.png` when a feature is swapped.
#
# Each panel states the R2 in a legend, so the fit quality is read at a glance at
# a fixed spot.  The fitted equation is no longer written inside the panel: at a
# size legible enough to be worth reading it competed with the markers for the
# same space, so it is reported instead in `Best descriptor for each target
# report.txt`, one entry per panel, where it can be given in full and at a
# precision the panel never had room for.
#
# The E_f panel uses the same double Gaussian as `E_f factors.png`, so the two
# figures cannot disagree about the shape of the valley.  The centre bound in
# `DoubleGaussianFit` is what keeps the broad component from running off to a
# numerically degenerate coefficient.
def BestFactor(target, target_values, kind):
    """The retained descriptor of one target that fits it best on its own.

    Returns (symbol, values, R2) of the winner, fitted with the form this study
    uses for that target."""
    symbols, columns = RetainedFeatures(target)
    y = np.asarray(target_values, dtype=float)
    best = None
    for s, c in zip(symbols, columns):
        x = np.asarray(data.loc[:, c], dtype=float)
        if kind == "linear":
            _, _, r2 = LinearFit(x, y)
        elif kind == "double gaussian":
            _, r2 = DoubleGaussianFit(x, y)
        else:
            r2 = BestChirp(x, y)[0]
        print("   %-8s vs %-10s R2 = %.3f" % (s, target, r2))
        if best is None or r2 > best[0]:
            best = (r2, s, c)
    return best[1], data.loc[:, best[2]], best[0]


print("  best single descriptor of each target:")
SUMMARY = []
for target_name, target_values, kind, target_label in [
        ("E_b", E_b, "linear", "$E_{b}$"),
        ("E_f", E_f, "double gaussian", "$E_{f}$"),
        ("U_diss", U_diss_acid, "sinusoid", "$U_{diss}$")]:
    symbol, values, r2 = BestFactor(target_name, target_values, kind)
    print("   -> %s keeps %s  (R2 = %.3f)" % (target_name, symbol, r2))
    SUMMARY.append((symbol, values, target_values, target_label, kind))

# the panel fonts.  With the equation gone each panel carries only its data and
# its legend, so the text can be set larger than in the surrounding figures
# without crowding the markers.
PANEL_LABEL_SIZE = 21
PANEL_TICK_SIZE = 19
PANEL_TITLE_SIZE = 26
PANEL_LEGEND_SIZE = 20
PANEL_CBAR_LABEL_SIZE = 21
PANEL_CBAR_TICK_SIZE = 18

fig, axes = plt.subplots(1, len(SUMMARY),
                         figsize=(10.5 * len(SUMMARY), 6.5), dpi=500)
summary_scores = []
equations = []
MARKER_AREA = 100          # the markers' area, in points squared
MARKER_EDGE_WIDTH = 0.5
for ax, (label, values, target, target_label, kind) in zip(axes, SUMMARY):
    x = np.asarray(values, dtype=float)
    y = np.asarray(target, dtype=float)

    sc = ax.scatter(x, y, c=y, cmap=custom_cmap, s=MARKER_AREA,
                    edgecolor='k', linewidth=MARKER_EDGE_WIDTH, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('%s value' % target_label, fontsize=PANEL_CBAR_LABEL_SIZE)

    x_fit = np.linspace(x.min(), x.max(), 800)
    if kind == "linear":
        slope, intercept, r2 = LinearFit(x, y)
        curve = slope * x_fit + intercept
        equation = LinearEquation(slope, intercept, label, target_label)
        fitted = [("slope", slope), ("intercept", intercept)]
    elif kind == "double gaussian":
        params, r2 = DoubleGaussianFit(x, y)
        curve = multi_gaussian(x_fit, *params)
        equation = DoubleGaussianEquation(params, label, target_label)
        fitted = [("amplitude 1", params[0]), ("centre 1", params[1]),
                  ("width 1", params[2]), ("amplitude 2", params[3]),
                  ("centre 2", params[4]), ("width 2", params[5]),
                  ("offset", params[6])]
    else:
        r2, wavelength, coef, chirp = BestChirp(x, y)
        curve = sinusoid_design(x_fit, wavelength, chirp) @ coef
        equation = SinusoidEquation(x, wavelength, chirp, coef, label,
                                    target_label)
        # `coef` multiplies [x sin, sin, x cos, cos, 1]
        fitted = [("x*sin coefficient", coef[0]), ("sin coefficient", coef[1]),
                  ("x*cos coefficient", coef[2]), ("cos coefficient", coef[3]),
                  ("offset", coef[4]), ("period", wavelength),
                  ("chirp", chirp)]
    # the grey curve doubles as the legend handle, so the legend answers "how
    # good is the fit" without the reader having to see the equation for it.  The
    # legend holds a fixed corner, which is what makes the panels comparable
    ax.plot(x_fit, curve, 'grey', lw=3, zorder=4,
            label='$R^{2} = %.3f$' % r2)
    ax.legend(loc="upper left", fontsize=PANEL_LEGEND_SIZE, framealpha=0.9,
              edgecolor="0.75")

    ax.set_xlabel(MathItalic(label), fontsize=PANEL_LABEL_SIZE)
    ax.set_ylabel(target_label, fontsize=PANEL_LABEL_SIZE)
    ax.set_title('%s vs. %s' % (MathItalic(label), target_label),
                 fontsize=PANEL_TITLE_SIZE)
    # the grid is pushed behind the markers, which is what keeps a dashed grid
    # from speckling the points
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.5, color="grey")
    ax.tick_params(labelsize=PANEL_TICK_SIZE)
    cbar.ax.tick_params(labelsize=PANEL_CBAR_TICK_SIZE)

    summary_scores.append((label, target_label, kind, r2))
    equations.append((label, target_label, kind, r2, equation, fitted))

fig.tight_layout()
plt.savefig("../figures/ML/Best descriptor for each target.png")
plt.close(fig)
print("Best descriptor of each target -> "
      "../figures/ML/Best descriptor for each target.png")

# the fitted equations, which the figure no longer carries.  They are written out
# in full, with the coefficients they were built from, so that the fit can be
# read and reproduced without the panel having to hold it
lines = []
lines.append("Fitted equations of `Best descriptor for each target.png`")
lines.append("=" * 78)
lines.append("")
lines.append("One entry per panel of the figure.  Each panel states only its R2, in the")
lines.append("legend; the equations that used to be written inside the panels are given")
lines.append("here, together with the coefficients they were built from, so that either")
lines.append("form can be checked against the other.")
for label, target_label, kind, r2, equation, fitted in equations:
    lines.append("")
    lines.append("-" * 78)
    lines.append("%s vs. %s     fit: %s     R2 = %.4f"
                 % (MathBody(label), target_label, kind, r2))
    lines.append("-" * 78)
    lines.append("")
    lines.append("  equation")
    for line in equation.split("\n"):
        lines.append("    %s" % line)
    lines.append("")
    lines.append("  fitted coefficients")
    for name, value in fitted:
        lines.append("    %-20s %+.6e" % (name, value))
lines.append("")
report = "\n".join(lines)
with open("Best descriptor for each target report.txt", "w",
          encoding="utf-8") as f:
    f.write(report)
print("Equations of the same figure -> "
      "Best descriptor for each target report.txt")
for d, t, k, v in summary_scores:
    print("   %-6s vs %-9s %-8s R2 = %.3f" % (d, t, k, v))



# 7. structure configuration distribution
angle = data.loc[:]["average angle of M-N-C/degree"]

sorted_df = pd.DataFrame({
    'R_CO': R_CO,
    "angle": angle,
})
sorted_df.index = metals
sorted_df = sorted_df.sort_values(by=['R_CO'])
sorted_R_CO = sorted_df['R_CO'].values
sorted_angle = sorted_df['angle'].values
sorted_metals = sorted_df.index

y_pos = list(range(len(sorted_metals)))
# the square-planar / square-pyramidal criterion, kept in step with
# ANGLE_THRESHOLD in structure analysis.py; any value inside the 167.8-176.7
# degree gap gives the identical 43/18 split
angle_baseline = 175

plt.figure(figsize=(6, 0.14 * len(sorted_metals)), dpi=500)

# the metal label is nudged down within its own row so that it clears the
# horizontal line and the marker it belongs to
label_dy = -0.11

for i, (y, an) in enumerate(zip(y_pos, sorted_angle)):
    metal = sorted_metals[i]

    if an >= angle_baseline:
        plt.text(an + 4, y + label_dy, metal, color="#3a4b6e", fontsize=10, ha="center", va="center")
        plt.plot([angle_baseline, an], [y, y], color="#3a4b6e", zorder=1)
        plt.scatter(an, y, s=50, color="#9fbbd5", edgecolors="#3a4b6e", zorder=2)

    else:
        plt.text(an - 4, y + label_dy, metal, color="#ba3e45", fontsize=10, ha="center", va="center")
        plt.plot([an, angle_baseline], [y, y], color="#ba3e45", zorder=1)
        plt.scatter(an, y, s=50, color="#d69d98", edgecolors="#ba3e45", zorder=2)

x_margin = 10
plt.xlim([min(sorted_angle) - x_margin, max(sorted_angle) + x_margin])
plt.ylim([min(y_pos) - 1.2, max(y_pos) + 1.2])

plt.axvline(angle_baseline, color="grey", linestyle="-", alpha=0.7)
ax = plt.gca()
ax.axvspan(angle_baseline, ax.get_xlim()[1], facecolor='#9fbbd5', alpha=0.15, zorder=0)
ax.axvspan(ax.get_xlim()[0], angle_baseline, facecolor='#d69d98', alpha=0.15, zorder=0)


plt.grid(True, linestyle="--", color="grey", alpha=0.2, axis='x')
plt.xlabel("$θ_{N-M-N}$ (angle)", fontsize=16)
plt.ylabel("$R_{CO}$ / pm", fontsize=16)
plt.yticks(fontsize=16)
plt.title("$θ_{N-M-N}$ Distribution", fontsize=20)
y_pos = list(range(len(sorted_metals)))
step = 10
selected_indices = y_pos[::step]
selected_R_CO = sorted_R_CO[::step]
plt.yticks(ticks=selected_indices, labels=selected_R_CO, fontsize=16)
plt.tight_layout()
plt.savefig("../figures/data analysis/θ_N-M-N distribution.png")