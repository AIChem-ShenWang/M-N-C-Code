import os
import re
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns
from mendeleev import element

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

plt.title("$E_{b}$ distribution", fontsize=16)
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
g.fig.suptitle('$E_{f}$ & $U_{diss}$ Distribution', fontsize=20, y=1.0001)
g.set_axis_labels('$E_{f}$ / eV', '$U_{diss}$ / V', fontsize=12)
g.ax_joint.grid(True, linestyle=':', alpha=0.6)
g.ax_marg_x.tick_params(labelsize=6)
g.ax_marg_y.tick_params(labelsize=6)

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

plt.savefig("../figures/data analysis/E_f & U_diss distribution.png", dpi=500)



# 2.Plot for ML results
# read the ML learning performance result
E_b_res = Read_ML_Result("../data/ML-results/$E_{b}$ ML report.txt")
E_f_res = Read_ML_Result("../data/ML-results/$E_{f}$ ML report.txt")
U_diss_res = Read_ML_Result("../data/ML-results/$U_{diss}$ ML report.txt")

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
ax1.set_ylabel('R²', fontsize=12)
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
                   label='R²', color='lightgrey', edgecolor='grey', linewidth=1)
for bar in bars_R2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', fontsize=8, color='grey', weight='bold')
# MAE
bars_MAE = ax2.bar(group_centers + offset_MAE, MAE_values, bar_width,
                    label='MAE', color='#9fbbd5', edgecolor='#3a4b6e', linewidth=1)
for bar in bars_MAE:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom', fontsize=8, color='#3a4b6e', weight='bold')
# RMSE
bars_RMSE = ax2.bar(group_centers + offset_RMSE, RMSE_values, bar_width,
                     label='RMSE', color='#d69d98', edgecolor='#ba3e45', linewidth=1)
for bar in bars_RMSE:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom', fontsize=8, color='#ba3e45', weight='bold')


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


#
# # 3.stable M-N4-C screening
# # the price of all metals
# try:
#     price_table = []
#     for i in tqdm(range(len(metals)), desc="generating price table"):
#         M = element(metals[i])
#         price_table.append([M.symbol, M.atomic_number, M.price_per_kg])
#     price_table = pd.DataFrame(price_table, columns=['element', 'atomicnumber', 'USD per kg'])
#     price_table.to_excel("../data/atom-table/metal price.xlsx", index=False)
# except:
#     print("mendeleev package version error, price_table.xlsx generated failed.")
#
# price_table = pd.read_excel("../data/atom-table/metal price.xlsx")
# price = price_table.set_index('element')['USD per kg'].to_dict()
#
# # Screening
# report = open("../data/stable M-N4-C report.txt", "w+", encoding="utf-8")
# report.write("Stable M-N4-C are:\n")
# print("Stable M-N4-C are:")
# stable_metals = []
# for i in range(len(metals)):
#     M = metals[i]
#     text = f"{M}-N4-C|"
#
#     # price
#     if price[M] >= 1000:
#         text += "$|"
#
#     # Eb & Ef
#     if E_b[i] > 0 or E_f[i] > 0:
#         text += "E_b && E_f Unstable|"
#
#     # Udiss
#
#     if U_diss_acid[i] >= 0 and U_diss_base[i] > 0:
#         text += "Both Stable"
#         stable_metals.append(M)
#
#     elif U_diss_acid[i] >= 0 and U_diss_base[i] <= 0:
#         text += "Acid Stable"
#         stable_metals.append(M)
#
#     elif U_diss_acid[i] <= 0 and U_diss_base[i] >= 0:
#         text += "Base Stable"
#         stable_metals.append(M)
#     else:
#         text += "U_diss Untable"
#
#     report.write(f"{text}\n")
#     print(f"{text}")
#
# report.write("Stable material number: %s\n" % len(stable_metals))
# report.write('Note: "$" is expensive element.')
# print("Stable material number: %s" % len(stable_metals))
# print('Note: * is unstable in air but stable in water. "$" is expensive element with price >= 1000 USD/KG')
# report.close()
#
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
        plt.text(x, bc + 0.5, M, color="#3a4b6e", ha='center', va='bottom', fontsize=10)
    else:
        plt.plot([x, x], [bc, 0], color="#ba3e45", zorder=1)
        plt.scatter(x, bc, s=50, color="#d69d98", edgecolors="#ba3e45", zorder=2)
        plt.text(x, bc - 3, M, color="#ba3e45", ha='center', va='bottom', fontsize=10)

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
             ha='center', va='bottom', fontsize=14, color="#ba3e45")

# label & title
plt.grid(True, linestyle="--", color="grey", alpha=0.2)
plt.xticks([])
plt.yticks(fontsize=12)
plt.ylabel("Band Center / eV", fontsize=16)
plt.xlabel("Period Number", fontsize=16)
plt.title("Band Center Distribution", fontsize=20)
plt.tight_layout()
plt.savefig("../figures/data analysis/band center distribution.png")



# 6. plotting for CM1, R_CO and Ef
CM1 = data.loc[:]["CM1"]
R_CO = data.loc[:]["covalent_radius_cordero/pm"]
colors = ['#003153', "#f5f5f5", '#85120f']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

plt.figure(figsize=(16, 6), dpi=500)
# R_CO
plt.subplot(1, 2, 1)
sc = plt.scatter(R_CO, E_f, c=E_f, cmap=custom_cmap, s=100, edgecolor='k', linewidth=0.5)
cbar = plt.colorbar(sc)
cbar.set_label('$E_{f}$ value', fontsize=16)
plt.xlabel('$R_{CO}$', fontsize=16)
plt.ylabel('$E_{f}$', fontsize=16)
plt.title('$R_{CO}$ vs. $E_{f}$', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6, color="grey")

# CM1
plt.subplot(1, 2, 2)
sc = plt.scatter(CM1, E_f, c=E_f, cmap=custom_cmap, s=100, edgecolor='k', linewidth=0.5)
cbar = plt.colorbar(sc)

# fitting of the data points
def two_gaussian(x_scaled, A1, mu1, sigma1, A2, mu2, sigma2, C):
    return (A1 * np.exp(-(x_scaled - mu1)**2 / (2 * sigma1**2)) +
            A2 * np.exp(-(x_scaled - mu2)**2 / (2 * sigma2**2)) + C)

# bonunds and initial guess
bounds = ([-np.inf, -np.inf, 0, -np.inf, -np.inf, 0, -np.inf],
          [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
p0 = [2.5, 2.0, 0.5, 2.5, 10.0, 1.0, -5.0]

x = CM1
y = E_f
x_scaled = x / 1065.0

# model fitting
popt, _ = curve_fit(two_gaussian, x_scaled, y, p0=p0, bounds=bounds, maxfev=10000)
y_pred = two_gaussian(x_scaled, *popt)
r2 = r2_score(y, y_pred)

x_fit = np.linspace(x.min(), x.max(), 500)
x_fit_scaled = x_fit / 1065.0
y_fit = two_gaussian(x_fit_scaled, *popt)
plt.plot(x_fit, y_fit, 'grey', lw=3, label='$R^{2}$=%.2f' % r2)

cbar.set_label('$E_{f}$ value', fontsize=16)
plt.xlabel('$CM1$', fontsize=16)
plt.ylabel('$E_{f}$', fontsize=16)
plt.legend(loc="lower left", fontsize=16, prop={'size': 16})
plt.title('$CM1$ vs. $E_{f}$', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6, color="grey")
plt.tight_layout()
plt.savefig("../figures/data analysis/E_f factors.png")



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
angle_baseline = 175

plt.figure(figsize=(6, 0.14 * len(sorted_metals)), dpi=500)

for i, (y, an) in enumerate(zip(y_pos, sorted_angle)):
    metal = sorted_metals[i]

    if an >= angle_baseline:
        plt.text(an + 4, y, metal, color="#3a4b6e", fontsize=10, ha="center", va="center")
        plt.plot([angle_baseline, an], [y, y], color="#3a4b6e", zorder=1)
        plt.scatter(an, y, s=50, color="#9fbbd5", edgecolors="#3a4b6e", zorder=2)

    else:
        plt.text(an - 4, y, metal, color="#ba3e45", fontsize=10, ha="center", va="center")
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