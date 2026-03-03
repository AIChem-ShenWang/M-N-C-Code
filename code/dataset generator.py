import os
import re
import warnings
warnings.filterwarnings("ignore")
from distutils.command.build_scripts import first_line_re

import mendeleev
from tqdm import tqdm

import numpy as np
import pandas as pd

from mendeleev import element

from utils.vaspfile import *

# 1.List of Metals
# Radioactive metals, alkali metals are not included
# AM alkaline metal
AM = ["Li", "Na", "K", "Rb", "Cs"]
# AEM alkaline-earth metal
AEM = ["Be", "Mg", "Ca", "Sr", "Ba"]
# TM transition metal
TM = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
      "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
      "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"]
# MGM main-group metal
MGM = ["Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]
# LnM lanthanide metal
LnM = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]

Metals = AM + AEM + TM + MGM + LnM

# # 2. VASP input file generation
# potential_db_path="../data/5.4.4 VASP POTCAR/PBE/potpaw_PBE"
# # structure for M_{SA}
# M_dir = "../data/vasp-file/M"
# if not os.path.exists(M_dir):
#     os.makedirs(M_dir)
#
# for i in tqdm(range(len(Metals)), desc="Generating M VASP input files"):
#     M = element(Metals[i])
#     M_path = os.path.join(M_dir, M.symbol)
#     if not os.path.exists(M_path):
#         os.mkdir(M_path)
#
#     # INCAR
#     GetINCAR(template_path="../data/template/INCAR_ENERGY_ATOM",
#              output_path=os.path.join(M_path, "INCAR"),
#              mat_name=M.symbol)
#     f = open(os.path.join(M_path, "INCAR"), "r")
#     lines = f.readlines()
#     for i in range(len(lines)):
#         line = lines[i]
#         if "ISPIN" in line:
#             lines[i] = "ISPIN = %s\n" % GetISPIN(M)
#     with open(os.path.join(M_path, "INCAR"), "w+") as f_new:
#         f_new.writelines(lines)
#
#     # KPOINT
#     GetKPOINT(template_path="../data/template/KPOINT_SM",
#              output_path=os.path.join(M_path, "KPOINT"))
#     # POSCAR
#     GetPOSCAR(template_path="../data/template/POSCAR_ATOM",
#               output_path=os.path.join(M_path, "POSCAR"),
#               replace_pair={1:M.symbol})
#     # POTCAR
#     GetPOTCAR(poscar_path=os.path.join(M_path, "POSCAR"),
#               output_path=os.path.join(M_path, "POTCAR"),
#               potential_db_path=potential_db_path)
#
# # structure for MNC
# MNC_dir = "../data/vasp-file/M-N-C"
# if not os.path.exists(MNC_dir):
#     os.makedirs(MNC_dir)
# # opt
# MNC_dir_opt = os.path.join(MNC_dir, "opt")
# if not os.path.exists(MNC_dir_opt):
#     os.makedirs(MNC_dir_opt)
# # energy
# MNC_dir_e = os.path.join(MNC_dir, "energy")
# if not os.path.exists(MNC_dir_e):
#     os.makedirs(MNC_dir_e)
#
# # Generation VASP input files for optimization
# for i in tqdm(range(len(Metals)), desc="Generating M-N-C VASP input files for optimization"):
#     M = element(Metals[i])
#     MNC_path = os.path.join(MNC_dir_opt, "%s-N-C" % M.symbol)
#     if not os.path.exists(MNC_path):
#         os.mkdir(MNC_path)
#     # INCAR file
#     GetINCAR(template_path="../data/template/INCAR_OPT",
#              output_path=os.path.join(MNC_path, "INCAR"),
#              mat_name="%s-N-C" % M.symbol)
#     # KPOINT
#     GetKPOINT(template_path="../data/template/KPOINT_OPT",
#              output_path=os.path.join(MNC_path, "KPOINT"))
#     # POSCAR
#     GetPOSCAR(template_path="../data/template/POSCAR_MNC",
#               output_path=os.path.join(MNC_path, "POSCAR"),
#               replace_pair={49:M.symbol})
#     # POTCAR
#     GetPOTCAR(poscar_path=os.path.join(MNC_path, "POSCAR"),
#               output_path=os.path.join(MNC_path, "POTCAR"),
#               potential_db_path=potential_db_path)
#
# # 3. Calculations for the formation energy
# # unzip files obtained from DFT
# # energy for single atom
# if not os.path.exists('../data/vasp-file/M/energy_zip'):
#     raise FileNotFoundError("Please put the calculation result in ../data/vasp-file/M/energy_zip")
#
# for filename in os.listdir('../data/vasp-file/M/energy_zip'):
#     unzip_dir(os.path.join('../data/vasp-file/M/energy_zip', filename), '../data/vasp-file/M/%s' % filename[:-4])
#
# # energy and dos for M-N-C
# for task in ["opt", "energy", "dos"]:
#     if not os.path.exists('../data/vasp-file/M-N-C/%s_zip' % task):
#         raise FileNotFoundError("Please put the calculation result in ../data/vasp-file/M-N-C/%s_zip" % task)
#     if not os.path.exists('../data/vasp-file/M-N-C/%s' % task):
#         os.makedirs('../data/vasp-file/M-N-C/%s' % task)
#
#     for filename in os.listdir('../data/vasp-file/M-N-C/%s_zip' % task):
#         unzip_dir(os.path.join('../data/vasp-file/M-N-C/%s_zip' % task, filename), '../data/vasp-file/M-N-C/%s/%s' % (task, filename[:-4]))
#
# # Analysis of PDOS of 4 N atoms and central metal
# for filename in tqdm(os.listdir('../data/vasp-file/M-N-C/dos'), desc="Analyzing DOS data"):
#     M = filename[:-4]
#     if M in AM:
#         obt = "s"
#     elif M in AEM:
#         obt = "p"
#     elif M in TM:
#         obt= "d"
#     elif M in MGM:
#         obt = "p"
#     elif M in LnM:
#         obt = "d"
#
#     # import data
#     doscar = Doscar(filename=f"../data/vasp-file/M-N-C/dos/{filename}/DOSCAR",
#                     ispin=2,
#                     lmax=3,
#                     lorbit=11,
#                     read_pdos=True)
#     e_range = [-30, 30]
#     pdos_obt = {45: "p", 46:"p", 47:"p", 48:"p", 49:obt}
#     color_list = ["#bad6ea", "#88BEDC", "#539DCC", "#2A7AB9", "#ce4459"]
#     atom_name = ["N1", "N2", "N3", "N4", M]
#     pdos_list = []
#     for key in pdos_obt.keys():
#         idx = key
#         obt = pdos_obt[key]
#         up = doscar.pdos_sum([idx - 1], spin='up', l=obt)
#         down = doscar.pdos_sum([idx - 1], spin='down', l=obt)
#
#         # energy coordiantion
#         energies = doscar.energy - doscar.efermi  # E - E_F
#
#         #  integral range E_F-e_range[0] eV to E_F+e_range[1] eV
#         emask = (energies >= e_range[0]) & (energies <= e_range[1])
#         x = energies[emask]
#         y_up = up[emask]
#         y_down = down[emask]
#
#         pdos = [x, y_up, y_down]
#         pdos_list.append(pdos)
#
#     # plotting
#     plt.figure(dpi=300)
#
#     all_y_up = []
#     all_y_down = []
#
#     for i in range(len(pdos_list)):
#         atom_pdos = pdos_list[i]
#         x = atom_pdos[0]
#         y_up = atom_pdos[1]
#         y_down = -atom_pdos[2]
#
#         plt.plot(x, y_up, color=color_list[i], label = atom_name[i], zorder=5-i, alpha=0.5, linewidth=1)
#         plt.plot(x, y_down, color=color_list[i], zorder=5-i, alpha=0.5, linewidth=1)
#
#         all_y_up.extend(y_up)
#         all_y_down.extend(y_down)
#
#     extend = 0.5
#     y_min_actual = min(min(all_y_up), min(all_y_down))
#     y_max_actual = max(max(all_y_up), max(all_y_down))
#     plt.ylim(y_min_actual - extend, y_max_actual + extend)
#
#     plt.axvline(x=0, color="grey", linewidth=2, linestyle="--", alpha=0.7)
#
#     plt.xlabel("E - E$_f$ (eV)")
#     plt.ylabel("PDOS(eV)")
#     plt.title("%s orbital of %s" % (obt, M))
#
#     if e_range[0] != -np.inf and e_range[1] != np.inf:
#         plt.xlim(e_range[0], e_range[1])
#
#     plt.grid(True, linestyle="--", alpha=0.8)
#     plt.legend(loc="upper left", prop={'size': 12})
#     plt.savefig("../figures/dos/%s_%s.png" % (M, obt))
#     plt.close()

# summarize the calculated data
MNC_dict = {}
col_name = [# from ../data/vasp-file/M
            "single atom energy/eV",
            # from atom bulk energy.xlsx
            "E(bulk)/eV",
            # from ../data/vasp-file/M-N-C/energy
            "M-N-C energy/ev",
            "average distance of M-N bond/A",
            "average angle of M-N-C/degree",
            "CM1",
            "CM2",
            # from ../data/vasp-file/M-N-C/dos
            "band center/eV",
            "band width/eV",
            # from mendeleev package
            "atomic number",
            "atomic wight/g mol-1",
            "atomic_radius/pm",
            "covalent_radius_cordero/pm",
            "heat_of_formation",
            "molar_heat_capacity",
            "vdw_radius",
            "zeff",
            "group number",
            # from atom potential.xlsx
            "common valence",
            "U_diss_std_acid/V",
            "U_diss_std_base/E",
            # label, calculated by the data above
            "E_b/eV",
            "E_f/eV",
            "U_diss_acid/V",
            "U_diss_base/V",
            "stable pH"]

# initialization
for M in Metals:
    MNC_dict[M] = []
    for i in range(len(col_name)):
        MNC_dict[M].append("-")

# from ../data/vasp-file/M
for filename in os.listdir('../data/vasp-file/M'):
    if filename != "energy_zip":
        e = GetEnergy(f"../data/vasp-file/M/{filename}/OUTCAR")
        MNC_dict[filename][0] = e

# from atom bulk energy.xlsx
e_bulk = pd.read_excel("../data/atom-table/bulk energy.xlsx")
e_bulk = e_bulk.set_index('element')['Fit-partial(eV)'].to_dict()
for M in MNC_dict.keys():
    MNC_dict[M][1] = e_bulk[M]

# from ../data/vasp-file/M-N-C/energy
for filename in tqdm(os.listdir('../data/vasp-file/M-N-C/energy'), desc="Processing Energy Data"):
    e = GetEnergy(f"../data/vasp-file/M-N-C/energy/{filename}/OUTCAR")
    M = element(filename[:-4])

    # energy for MNC
    MNC_dict[M.symbol][2] = e

    # average distance of M-N bond
    dis_sum = 0
    for id in range(45, 49):
        dis_sum += GetDistance(f"../data/vasp-file/M-N-C/energy/{filename}/CONTCAR", id, 49)

    MNC_dict[M.symbol][3] = dis_sum / 4

    # average angle of M-N bond
    ang1 = GetAngle(f"../data/vasp-file/M-N-C/energy/{filename}/CONTCAR", idx1=48, idx2=49, idx3=45)
    ang2 = GetAngle(f"../data/vasp-file/M-N-C/energy/{filename}/CONTCAR", idx1=46, idx2=49, idx3=47)
    MNC_dict[M.symbol][4] = (ang1 + ang2) / 2

    # CM1
    MNC_dict[M.symbol][5] = 0.5 * pow(M.atomic_number, 2.4)

    # CM2
    cm2_sum = 0
    for id in range(45, 49):
        cm2_sum += M.atomic_number * element("N").atomic_number / GetDistance(f"../data/vasp-file/M-N-C/energy/{filename}/CONTCAR", id, 49)
    MNC_dict[M.symbol][6] = cm2_sum / 4

# from ../data/vasp-file/M-N-C/dos
for filename in tqdm(os.listdir('../data/vasp-file/M-N-C/dos'), desc="Processing DOS data"):
    M = filename[:-4]

    if M in AM:
        obt = "s"
    if M in AEM:
        obt = "p"
    if M in TM:
        obt= "d"
    if M in MGM:
        obt = "p"
    if M in LnM:
        obt = "d"

    dbc, width = GetBandCenter(DOSCAR_path=f"../data/vasp-file/M-N-C/dos/{filename}/DOSCAR",
                                idx=49,
                                orbital=obt,
                                e_range=[-30, 30])

    MNC_dict[M][7] = dbc
    MNC_dict[M][8] = width

# from mendeleev package
for M in tqdm(MNC_dict.keys(), desc="Generating mendeleev feature"):
    M = element(M)
    MNC_dict[M.symbol][9] = M.atomic_number
    MNC_dict[M.symbol][10] = M.atomic_weight
    MNC_dict[M.symbol][11] = M.atomic_radius
    MNC_dict[M.symbol][12] = M.covalent_radius_cordero
    MNC_dict[M.symbol][13] = M.heat_of_formation
    MNC_dict[M.symbol][14] = M.molar_heat_capacity
    MNC_dict[M.symbol][15] = M.vdw_radius
    MNC_dict[M.symbol][16] = M.zeff()
    if M.group_id is not None:
        MNC_dict[M.symbol][17] = M.group_id
    else:
        MNC_dict[M.symbol][17] = 3 # for lanthanide metals

# from atom potential.xlsx
potential = pd.read_excel("../data/atom-table/potential.xlsx", sheet_name="U_diss")
valence = potential.set_index('element')['valence'].to_dict()
U_diss_std_acid = potential.set_index('element')['U_diss_acid'].to_dict()
U_diss_std_base = potential.set_index('element')['U_diss_base'].to_dict()

for M in tqdm(MNC_dict.keys(), desc="Reading Potential.xlsx"):
    MNC_dict[M][18] = valence[M]
    MNC_dict[M][19] = U_diss_std_acid[M]
    MNC_dict[M][20] = U_diss_std_base[M]

# label, calculated by the data above
E_N4C = GetEnergy("../data/vasp-file/common-molecules/energy/N-C/OUTCAR")

for M in tqdm(MNC_dict.keys(), desc="Calculating stability parameters"):
    if MNC_dict[M][2] != "-":
        E_b = MNC_dict[M][2] - E_N4C - MNC_dict[M][0]
        E_f = MNC_dict[M][2] - E_N4C - MNC_dict[M][1]
        U_diss_acid = U_diss_std_acid[M] - E_f / valence[M] + 0.0592 * 0
        U_diss_base = U_diss_std_base[M] - E_f / valence[M] + 0.0592 * 14

        MNC_dict[M][21] = E_b
        MNC_dict[M][22] = E_f
        MNC_dict[M][23] = U_diss_acid
        MNC_dict[M][24] = U_diss_base
        stable_pH = []
        if U_diss_acid >= 0 and U_diss_base >= 0:
            MNC_dict[M][25] = "Both"
        elif U_diss_acid >= 0 and U_diss_base <= 0:
            MNC_dict[M][25] = "Acid"
        elif U_diss_base >= 0 and U_diss_acid <= 0:
            MNC_dict[M][25] = "Base"
        elif U_diss_base <= 0 and U_diss_acid <= 0:
            MNC_dict[M][25] = "None"

MNC_df = pd.DataFrame(MNC_dict).T
MNC_df.columns = col_name
MNC_df.to_excel("../data/M-N-C data set.xlsx")
