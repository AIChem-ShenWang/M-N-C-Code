import os
import re
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as R2, r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

from tqdm import tqdm
import shap
from TorchSisso import SissoModel

# args: mode & model selection
mode = "E_f"
assert mode in ["E_b", "E_f", "U_diss"], print("The mode should be E_b or E_f or U_diss.")
model_list = ["LR", "RF", "XGBR", "SVR", "kNN", "GP"]
# model_list = ["SVR"]

# 0. Preparation
# function
def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))

# color
colors = ['#003153', "#f5f5f5", '#85120f']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

# import data
data = pd.read_excel("../data/M-N-C data set.xlsx")
feat = data.iloc[:, 4:-7]
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
                      "CM1":"CM1",
                      "CM2":"CM2",
                      "band center/eV": "$ε_{e}$",
                      "band width/eV": "$ε_{w}$",
                      "atomic number": "AN",
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
feat_metrics = feat.copy()
feat_metrics.columns = feat_metrics.columns.astype(str)
feat_metrics.columns = [feature_symbol_dict[i] for i in feat_metrics.columns]
corr_matrix = feat_metrics.corr()

X = np.array(feat)
feat_name = feat_metrics.columns

# Pearson coefficient heatmap
plt.figure(dpi=500, figsize=(10, 8))
ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=custom_cmap, vmin=-1, vmax=1, annot_kws={'size': 7})
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
plt.title('Feature Correlation Heatmap', fontsize=18)
plt.savefig("../figures/ML/Heatmap for Features in %s.png" % mode)

# Pearson coefficient heatmap
feat_all_metrics = data.iloc[:, 4:-7].copy()
feat_all_metrics.columns = feat_all_metrics.columns.astype(str)
feat_all_metrics.columns = [feature_symbol_dict[i] for i in feat_all_metrics.columns]
corr_matrix_all = feat_all_metrics.corr()

X_all = np.array(feat_all_metrics)
feat_name_all = feat_all_metrics.columns

# Pearson coefficient heatmap
plt.figure(dpi=500, figsize=(10, 8))
ax = sns.heatmap(corr_matrix_all, annot=True, fmt=".2f", cmap=custom_cmap, vmin=-1, vmax=1, annot_kws={'size': 7})
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
plt.title('Feature Correlation Heatmap', fontsize=18)
plt.savefig("../figures/ML/Heatmap for Features.png")


# 2. Train & Testing of ML models
# Leave One Out Validation
loo = LeaveOneOut()

# initialization
models = {}
for model in model_list:
    models[model] = []
    # [[], [], [], [], []] : store the value of train_R2, train_MAE, train_RMSE, test_true, test_predict
    for i in range(5):
        models[model].append([])

# parameters for plotting. Record a group of data when testing
interpret_model = {}

print("Start Training...")
for train_index, test_index in tqdm(loo.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for m in models.keys():
        if mode == "$E_{b}$":
            if m == "LR":
                model = Lasso(alpha=0.05, max_iter=10000, random_state=42)
            if m == "RF":
                model = RandomForestRegressor(n_estimators=50, random_state=0, max_depth=10, max_features=0.5, min_samples_leaf=2)
            if m == "XGBR":
                model = XGBRegressor(n_estimators=100, random_state=0, gamma=0.3)
            if m == "SVR":
                kernel = (ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 1e2),
                                                       nu=1.5) + WhiteKernel(noise_level=0.1,
                                                                             noise_level_bounds=(0.1, 10)))

                model = SVR(kernel=kernel, C=5e5)
            if m == "kNN":
                model = KNeighborsRegressor(n_neighbors=2)
            if m == "GP":
                kernel = (ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 1e2),
                                                       nu=1.5) + WhiteKernel(noise_level=0.1,
                                                                             noise_level_bounds=(0.1, 10)))
                model = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5, normalize_y=True)

        elif mode == "$E_{f}$":
            if m == "LR":
                model = Lasso(alpha=0.08, max_iter=10000, random_state=42)
            if m == "RF":
                model = RandomForestRegressor(n_estimators=50, random_state=0, max_depth=20, max_features=0.6)
            if m == "XGBR":
                model = XGBRegressor(n_estimators=100, random_state=42, reg_alpha=0.5, gamma=0.01)
            if m == "SVR":
                kernel = (ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 1e2), nu=1.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(0.1, 10)))

                model = SVR(kernel=kernel, C=5e5)
            if m == "kNN":
                model = KNeighborsRegressor(n_neighbors=3)
            if m == "GP":
                kernel = (ConstantKernel(1.0) * Matern(length_scale=1.0, length_scale_bounds=(0.1, 1e2), nu=1.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(0.1, 10)))
                model = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5, normalize_y=True)

        elif mode == "$U_{diss}$":
            if m == "LR":
                model = Lasso(alpha=0.001, max_iter=10000, random_state=42)
            if m == "RF":
                model = RandomForestRegressor(n_estimators=50, random_state=0, max_depth=10, max_features=0.5)
            if m == "XGBR":
                model = XGBRegressor(n_estimators=100, random_state=42, reg_alpha=0.1, gamma=0.2)
            if m == "SVR":
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=1.0, length_scale_bounds=(0.1, 1e2), nu=1.5) +
                          WhiteKernel(noise_level=0.1, noise_level_bounds=(0.1, 10)))

                model = SVR(kernel=kernel, C=5e5)
            if m == "kNN":
                model = KNeighborsRegressor(n_neighbors=2)
            if m == "GP":
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=1.0, length_scale_bounds=(0.1, 1e2), nu=1.5) +
                          WhiteKernel(noise_level=0.1, noise_level_bounds=(0.1, 10)))
                model = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=10, normalize_y=True)

        # Training
        model.fit(X_train_scaled, y_train)
        models[m][0].append(R2(y_train, model.predict(X_train_scaled)))
        models[m][1].append(MAE(y_train, model.predict(X_train_scaled)))
        models[m][2].append(RMSE(y_train, model.predict(X_train_scaled)))

        # Testing
        y_pred = model.predict(X_test_scaled)
        models[m][3].append(y_test[0])
        models[m][4].append(y_pred[0])

        interpret_model[m] = model



# 3. Performance Record & Plotting
if not os.path.exists("../data/ML-results"):
    os.makedirs("../data/ML-results")

report = open("../data/ML-results/%s ML report.txt" % mode, mode="w+", encoding="utf-8")

# data set baseline
y_true = models[m][3]
y_avg = [np.array(y_true).mean()] * len(y_true)
r2 = r2_score(y_true, y_avg)
mae = MAE(y_true, y_avg)
rmse = RMSE(y_true, y_avg)
report.write("Dataset Baseline:\n")
report.write("R²: %.2f\nMAE = %.2f\nRMSE = %.2f\n" % (r2, mae, rmse))
report.write("\n")
print("Dataset Baseline:")
print("R²: %.2f\nMAE = %.2f\nRMSE = %.2f\n" % (r2, mae, rmse))

# best model
rmse_min = np.inf
best_model = None

for m in models.keys():
    plt.figure(dpi=300, figsize=(6, 5))

    y_true = models[m][3]
    y_pred = models[m][4]

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    extend = 0.5
    plot_min = min_val - extend
    plot_max = max_val + extend
    plt.plot([plot_min, plot_max], [plot_min, plot_max],
             linestyle="--", color="#b8474d", linewidth=2)

    # error line
    test_rmse = RMSE(y_true, y_pred)
    if test_rmse < rmse_min:
        rmse_min = test_rmse
        best_model = m

    plt.fill_between([plot_min, plot_max],
                     [plot_min - test_rmse, plot_max - test_rmse],  # 下边界
                     [plot_min + test_rmse, plot_max + test_rmse],
                     color="#d69d98", alpha=0.3)
    plt.plot([plot_min, plot_max], [plot_min + test_rmse, plot_max + test_rmse],
             linestyle=":", color="#b8474d", linewidth=1.5)
    plt.plot([plot_min, plot_max], [plot_min - test_rmse, plot_max - test_rmse],
             linestyle=":", color="#b8474d", linewidth=1.5)

    plt.title("Performance of %s in %s" % (m, mode), fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([plot_min, plot_max])
    plt.ylim([plot_min, plot_max])

    # scatter
    plt.scatter(y_true, y_pred, color="#9fbbd5", edgecolors='#3a4b6e', linewidth=1, s=60)

    # Model Performance
    evaluation_metrics_name = ["R²", "MAE", "RMSE", "R²", "MAE", "RMSE"]

    train_performance_metrics = []
    for i in range(3):
        train_performance_metrics.append(models[m][i])
    train_performance_metrics = np.array(train_performance_metrics)
    train_performace_avg = train_performance_metrics.mean(axis=1)
    train_performace_std = train_performance_metrics.std(axis=1)

    test_performance_metrics = [r2_score(y_true, y_pred), MAE(y_true, y_pred), RMSE(y_true, y_pred)]

    train_performance = str()
    test_performance = str()

    for i in range(len(evaluation_metrics_name)):
        if i < 3:
            train_performance += "%s = %.2f ± %.2f\n" % (evaluation_metrics_name[i],
                                                           train_performace_avg[i],
                                                           train_performace_std[i])
        else:
            test_performance += "%s = %.2f\n" % (evaluation_metrics_name[i], test_performance_metrics[i-3])

    performance = "train set performance:\n" + train_performance + "test set performance:\n" + test_performance
    report.write("ML Model: %s\n" % m)
    report.write(performance)
    report.write("\n")
    print("ML Model: %s" % m)
    print(performance)

    # metal name && performance plot && axis label
    for i in range(len(y_true)):
        if mode == "$E_{b}$":
            if abs(y_true[i] - y_pred[i]) > rmse:
                # metal
                plt.text(y_true[i] - 0.25, y_pred[i] + 0.25, "%s" % metals[i],
                         fontdict={"size": 11, "color": '#3a4b6e'})
            # performance
            plt.text(plot_min + 0.4, plot_max - 0.5, test_performance[:-1],
                     fontsize=14, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            # axis labels
            plt.xlabel("Ground Truth (eV)", fontsize=16)
            plt.ylabel("Predicted (eV)", fontsize=16)
        elif mode == "$E_{f}$":
            if abs(y_true[i] - y_pred[i]) > rmse:
                # metal
                plt.text(y_true[i] - 0.35, y_pred[i] + 0.35, "%s" % metals[i],
                         fontdict={"size": 11, "color": '#3a4b6e'})
            # performance
            plt.text(plot_min + 0.6, plot_max - 0.7, test_performance[:-1],
                     fontsize=14, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            # axis labels
            plt.xlabel("Ground Truth (eV)", fontsize=16)
            plt.ylabel("Predicted (eV)", fontsize=16)

        elif mode == "$U_{diss}$":
            if abs(y_true[i] - y_pred[i]) > rmse:
                # metal
                plt.text(y_true[i] - 0.1, y_pred[i] + 0.11, "%s" % metals[i], fontdict={"size": 11, "color": '#3a4b6e'})
            # performance
            plt.text(plot_min + 0.2, plot_max - 0.28, test_performance[:-1],
                     fontsize=14, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            # axis labels
            plt.xlabel("Ground Truth (V)", fontsize=16)
            plt.ylabel("Predicted (V)", fontsize=16)


    plt.grid(True, alpha=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig("../figures/ML/ML Performance of %s for %s.png" % (m, mode))



# 4.Interpretability
# SHAP
# best_model = model_list[0]
plt.figure(dpi=500, figsize=(5, 10))
model = interpret_model[best_model]

X = StandardScaler().fit_transform(X)
if best_model in ["RF", "XGBR"]:
    explainer = shap.Explainer(model)
elif best_model in ["kNN", "SVR", "GP", "LR"]:
    explainer = shap.KernelExplainer(model.predict, X)

shap_values = explainer(X)

# SISSO Part 1
if mode == "$E_{b}$":
    y = E_b.to_numpy()
    sisso_col = [
        "heat_of_formation",
        "covalent_radius_cordero/pm"
    ]
    X = np.array(feat[sisso_col])
    n_terms = 2
    n_expansion = 4
elif mode == "$E_{f}$":
    y = E_f.to_numpy()
    sisso_col = [
                "covalent_radius_cordero/pm",
                "CM1"
    ]
    X = np.array(feat[sisso_col])
    n_terms = 2
    n_expansion = 4
elif mode == "$U_{diss}$":
    y = U_diss.to_numpy()
    sisso_col = [
                 "vdw_radius",
                 "covalent_radius_cordero/pm",
                 # "group number"
    ]
    X = np.array(feat[sisso_col])
    n_terms = 2
    n_expansion = 4

# plotting for feature importtance
plt.figure(figsize=(6, 5), dpi=300)

if hasattr(shap_values, 'values'):
    shap_matrix = shap_values.values
else:
    shap_matrix = shap_values
mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)
sorted_features = [feat_name[i] for i in sorted_idx]
sorted_values = mean_abs_shap[sorted_idx]

# chosen feature
N = len(sisso_col)
plt.barh(sorted_features[:-N], sorted_values[:-N],
         color='#9fbbd5', edgecolor='#3a4b6e', linewidth=1.5)
plt.barh(sorted_features[-N:], sorted_values[-N:],
         color="#d69d98", edgecolor="#b8474d", linewidth=1.5)

plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.title("SHAP Explanation for %s" % mode, fontsize=18)
plt.xlabel("Importance", fontsize=16)
plt.ylabel("Features", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("../figures/ML/SHAP_%s.png" % mode, bbox_inches='tight')


# SISSO Part2
data = pd.DataFrame(np.column_stack((y, X)), columns=["Target"] + [feature_symbol_dict[i] for i in sisso_col])
operators = ["+", "-", "*", "/", "Ln", "sin", "tan", "cos",  "pow(-1)", "pow(-2)"]
sm = SissoModel(data, operators, n_expansion=n_expansion, n_term=n_terms, k=5)
rmse, equation, r2,_ = sm.fit()
report.write("SISSO Model:\n")
report.write("R²=%.2f\nRMSE=%.2f\nEq:%s" %(r2, rmse, equation))
report.close()
print("SISSO Model:\nR²=%.2f\nRMSE=%.2f\nEq:%s" %(r2, rmse, equation))

