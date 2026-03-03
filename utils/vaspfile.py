import os
import re
import ase.io.vasp
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import mendeleev
from utils.doscar import *
from scipy.integrate import simps
from pymatgen.io.vasp import Outcar

# 1.VASP input files
# INCAR
def GetINCAR(template_path: str,
             output_path: str,
             mat_name:str):
    # read file
    with open(template_path, 'r+', encoding='utf-8') as f:
        lines = f.readlines()

    # Modify job_name in INCAR
    modified_lines = []
    for line in lines:
        if line.replace(" ", "").strip().startswith('SYSTEM'):
            eq_index = line.find('=')
            if eq_index != -1:
                modified_line = line[:eq_index + 1] + f' {mat_name}\n'
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)
        else:
            modified_lines.append(line)

    # output file
    with open(output_path, 'w+', encoding='utf-8') as f:
        f.writelines(modified_lines)


#KPOINT
def GetKPOINT(template_path: str,
             output_path: str):
    # read the file
    with open(template_path, 'r+', encoding='utf-8') as f:
        lines = f.readlines()

    # output file
    with open(output_path, 'w+', encoding='utf-8') as f:
        f.writelines(lines)


# POSCAR
def GetPOSCAR(template_path: str,
              output_path: str,
              replace_pair: dict,
              ):
    cell = ase.io.vasp.read_vasp(file=template_path).copy()

    # separate index and element symbol
    index_replacements = {}
    symbol_replacements = {}

    for key, new_symbol in replace_pair.items():
        if isinstance(key, int):
            index_replacements[key] = new_symbol
        elif isinstance(key, str):
            symbol_replacements[key] = new_symbol

    # element replacement
    if symbol_replacements:
        symbols = cell.get_chemical_symbols()
        for i, old_symbol in enumerate(symbols):
            if old_symbol in symbol_replacements:
                if i not in index_replacements:
                    cell[i].symbol = symbol_replacements[old_symbol]

    # index replacement
    for idx, new_symbol in index_replacements.items():
        if 0 < idx <= len(cell):
            cell[idx-1].symbol = new_symbol
        else:
            raise ValueError(f"Atom index: {idx} out of range (1-{len(cell)})")

    ase.io.write(output_path, cell, format='vasp', direct=True, vasp5=True)

    return 0


# POTCAR
def GetPOTCAR(poscar_path: str,
              output_path: str,
              potential_db_path: str):

    cell = ase.io.vasp.read_vasp(file=poscar_path).copy()
    elements = set()
    with open(output_path, 'w+') as potcar:
        for element in cell.get_chemical_symbols():
            if element not in elements:
                elements.add(element)
                potcar_path = os.path.join(potential_db_path, element, 'POTCAR')
                if not os.path.exists(potcar_path):
                    try:
                        if os.path.isdir(os.path.join(potential_db_path, element)):
                            var_dir = os.path.join(potential_db_path, element)
                        else:
                            all_subdirs = [d for d in os.listdir(potential_db_path)
                                           if os.path.isdir(os.path.join(potential_db_path, d))]
                            target_dirs = [d for d in all_subdirs if d.startswith("%s_" % element)]
                            var_dir = target_dirs[0]
                        potcar_path = os.path.join(potential_db_path, var_dir, 'POTCAR')
                    except FileNotFoundError:
                        print("POTCAR does not exists for %s" % element)
                with open(potcar_path, 'r') as potcar_part:
                    potcar.write(potcar_part.read())


# Get ISPIN for INCAR file
def GetISPIN(atom: mendeleev.element):
    ec = atom.ec
    unpaired_electrons = 0 # Counting of unpaired electrons

    # Parsing orbital
    for orbital in ec.conf:
        orbital_type = orbital[1]
        electron_count = ec.conf[orbital]

        if orbital_type == 's':
            orbitals = 1
        elif orbital_type == 'p':
            orbitals = 3
        elif orbital_type == 'd':
            orbitals = 5
        elif orbital_type == 'f':
            orbitals = 7

        if electron_count <= orbitals:
            unpaired_in_orbital = electron_count
        else:
            unpaired_in_orbital = max(0, 2 * orbitals - electron_count)

        unpaired_electrons += unpaired_in_orbital

    # Output
    if unpaired_electrons > 0:
        return 2
    else:
        return 1


# 2.file zip and unzip
def zip_dir(dir_origin, dir_target):
    if not os.path.exists(dir_origin):
        print("Input Directory: %s not exists" % dir_origin)
        return 0

    if not dir_target.endswith('.zip'):
        dir_target += '.zip'

    output_dir = os.path.dirname(dir_target)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with zipfile.ZipFile(dir_target, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(dir_origin):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dir_origin)
                z.write(file_path, arcname)


def unzip_dir(dir_origin, dir_target):
    if not os.path.exists(dir_origin):
        print("Input File: %s not exists" % dir_origin)
        return 0
    if not os.path.exists(dir_target):
        os.makedirs(dir_target)
    if ".zip" not in dir_origin:
        print("Warning: %s is not zipped" % dir_origin)
        return 0
    with zipfile.ZipFile('%s' % dir_origin, 'r') as z:
        z.extractall(dir_target)


# 3.VASP output file processing
# check the optimization
def CheckOpt(OUTCAR_path):
    if not os.path.exists(OUTCAR_path):
        print("The structure have not been optimized.")
        return False

    # Check optimization task is converge or not
    ck = open(OUTCAR_path).read()
    if "reached required accuracy - stopping structural energy minimisation" not in ck:
        print("The optimized structure is not reached required accuracy in %s" % OUTCAR_path)
        return False

    return True


# Get Energy from OUTCAR file
def GetEnergy(OUTCAR_path):
    with open(OUTCAR_path, 'r') as file:
        lines = file.readlines()

    # find line "energy(sigma->0)"from back
    target_line = None
    for line in reversed(lines):
        if "energy(sigma->0)" in line:
            target_line = line.strip()
            break
    if target_line is None:
        return "-"

    # get energy value through re
    pattern = r"energy\(sigma->0\)\s*=\s*(-?\d+\.\d+)"
    match = re.search(pattern, target_line)
    if match:
        return float(match.group(1))
    else:
        return "-"


# Get the distance of 2 atoms using CONTAR file
def GetDistance(CONTCAR_path: str, # The path of CONTCAR file
                idx1: int, # the index of atom 1
                idx2: int # the index of atom 2
                ):
    cell = ase.io.vasp.read_vasp(file=CONTCAR_path).copy()

    if idx1 > len(cell) or idx2 > len(cell) or idx1 <= 0 or idx2 <= 0:
        raise ValueError(f"Index out of range (1-{len(cell)})")

    # Calculation of index
    distance = cell.get_distance(idx1-1, idx2-1, mic=True)

    return distance

# Get the angle of 3 atoms using CONTAR file
def GetAngle(CONTCAR_path: str,
             idx1: int,
             idx2: int,
             idx3: int):
    cell = ase.io.vasp.read_vasp(file=CONTCAR_path).copy()
    n_atoms = len(cell)
    indices = [idx1, idx2, idx3]
    for idx in indices:
        if idx < 1 or idx > n_atoms:
            raise ValueError(f"Index {idx} out of range (1-{n_atoms})")
    angle = cell.get_angle(idx1-1, idx2-1, idx3-1, mic=True)
    return angle

# DOS analysis and claculation of band center
def GetBandCenter(DOSCAR_path: str, #  path of the DOSCAR file
                  idx: int, # the atom to be analyzed
                  orbital: int = "d", # The orbital type of the atom
                  e_range: list = [-np.inf, np.inf], # The energy range of the E - Ef
                  plot_path: str = None, # Save the PDOS plot at plot_path
    **kwargs):
    """
    :return: The energy of the band center relative to the Fermi level (eV)
    """

    # check the input file
    if not os.path.exists(DOSCAR_path):
        raise FileNotFoundError("%s not exists" % DOSCAR_path)

    # Check the orbital type
    if orbital not in ["s", "p", "d", "f"]:
        raise TypeError("The orbital is not support.")
    if e_range[0] > e_range[1]:
        raise ValueError("emin should smaller than emax.")

    # import data
    doscar = Doscar(filename=DOSCAR_path,
                    ispin=2,
                    lmax=3,
                    lorbit=11,
                    read_pdos=True)

    up = doscar.pdos_sum([idx-1], spin='up', l=orbital)
    down = doscar.pdos_sum([idx-1], spin='down', l=orbital)

    # energy coordiantion
    energies = doscar.energy - doscar.efermi  # E - E_F

    #  integral range E_F-e_range[0] eV to E_F+e_range[1] eV
    emask = (energies >= e_range[0]) & (energies <= e_range[1])
    x = energies[emask]
    y_up = up[emask]
    y_down = down[emask]

    # band center
    dbc_up = simps(y_up * x, x) / simps(y_up, x)
    dbc_down = simps(y_down * x, x) / simps(y_down, x)
    dbc = (dbc_up + dbc_down) / 2

    # band width
    moment2_up = simps(y_up * (x - dbc_up) ** 2, x) / simps(y_up, x)
    moment2_down = simps(y_down * (x - dbc_down) ** 2, x) / simps(y_down, x)
    width_up = np.sqrt(moment2_up)
    width_down = np.sqrt(moment2_down)
    width = (width_up + width_down) / 2

    if np.isnan(dbc) and np.isnan(width):
        print("Warning: the band is not formed in %s orbital." % orbital)

    # Show the PDOS plot
    if plot_path is not None:
        plt.figure(dpi=300)
        plt.plot(x, y_up, color="#D69D98", label="up-spin")
        plt.plot(x, -y_down, color="#9FBBD5", label="down-spin")
        extend = 0.5
        plt.axvline(x=0, ymin=min(-y_down) - 5*extend, ymax=max(y_up) + 5*extend, color="grey", linewidth=2, linestyle="--", alpha=0.5)
        if not np.isnan(dbc_up) and not np.isnan(dbc_down):
            plt.plot([dbc_up, dbc_up], [0, max(y_up) + extend], linestyle="--", color="#BA3E45", label="E$_d$=%.3feV" % dbc_up)
            plt.plot([dbc_down, dbc_down], [0, min(-y_down) - extend], linestyle="--", color="#3A4B6E", label="E$_d$=%.3feV" % dbc_down)
        plt.xlabel("E - E$_f$ (eV)")
        plt.ylabel("PDOS(eV)")
        plt.title("%s orbital of ")
        plt.ylim([min(-y_down) - 5*extend, max(y_up) + 5*extend])
        if e_range[0] != -np.inf and e_range[1] != np.inf:
            plt.xlim(e_range[0], e_range[1])
        if kwargs["plot_title"] is not None:
            plt.title(kwargs["plot_title"], fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.8)
        plt.legend(loc="upper left", prop={'size': 8})
        plt.savefig(plot_path)
        plt.close()

    # return e_avg
    return dbc, width

