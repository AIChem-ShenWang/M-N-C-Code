import os
import re
import shutil
import sys
import warnings
warnings.filterwarnings("ignore")

import mendeleev
from tqdm import tqdm

import numpy as np
import pandas as pd

from mendeleev import element

# The package sits one level above this script, so make it importable whatever
# directory the script is started from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vaspfile import *

# 1. List of metals, grouped by family
# AM  alkali metal
AM = ["Li", "Na", "K", "Rb", "Cs"]
# AEM alkaline-earth metal
AEM = ["Be", "Mg", "Ca", "Sr", "Ba"]
# TM  transition metal
TM = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
      "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
      "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"]
# MGM main-group metal
MGM = ["Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]
# LnM lanthanide metal
LnM = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]

Metals = AM + AEM + TM + MGM + LnM

# 2. VASP input file generation
potential_db_path="../data/5.4.4 VASP POTCAR/PBE/potpaw_PBE"

# Potentials that differ from the element name
POTCAR_VARIANT = {
    "Na": "Na_pv", "K": "K_sv", "Rb": "Rb_sv", "Cs": "Cs_sv",
    "Ca": "Ca_sv", "Sr": "Sr_sv", "Ba": "Ba_sv",
    "Sc": "Sc_sv", "Ti": "Ti_sv", "V": "V_sv", "Cr": "Cr_pv", "Mn": "Mn_pv",
    "Y": "Y_sv", "Zr": "Zr_sv", "Nb": "Nb_sv", "Mo": "Mo_sv", "Ru": "Ru_pv", "Rh": "Rh_pv",
    "Hf": "Hf_pv", "Ta": "Ta_pv", "W": "W_pv",
    "Ga": "Ga_d", "Ge": "Ge_d", "In": "In_d", "Sn": "Sn_d",
    "Tl": "Tl_d", "Pb": "Pb_d", "Bi": "Bi_d"
}

# DFT+U of 3d and 4f orbitals, as LDAUL, LDAUU, LDAUJ, LMAXMIX
LDAU_3D = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni"]
# Sm is absent: its runs are carried over from the reference set that used no on-site
# term, so it gets neither DFT+U nor a starting spin.
LDAU_4F = ["Ce", "Pr", "Nd", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm"]
LDAU_3D_PARAM = (2, 4.0, 0.0, 4)
LDAU_4F_PARAM = (3, 5.0, 0.0, 6)

# Starting spin of the metal for the compound, for elements whose free-atom Hund's rule
# does not match the valence of the potential actually used. Ho sits in M-N4-C as Ho3+,
# 4f10, whose four unpaired electrons are not the three of the free 4f11 6s2 atom; the
# free-atom count settles in a different local minimum. It only feeds MAGMOM, never the
# NUPDOWN of the isolated atom, which is a constraint and must match the parity of the
# electron count.
SPIN_OVERRIDE = {"Ho": 4}

# Metals whose runs carry neither a starting spin nor a spin constraint. Sm is taken
# over from the reference set as it stands, so VASP picks the spin on its own.
NO_SPIN_TAGS = {"Sm"}


def GetLDAU(M: str):
    if M in LDAU_3D:
        return LDAU_3D_PARAM
    elif M in LDAU_4F:
        return LDAU_4F_PARAM
    else:
        return None

# INCAR and KPOINTS template file names of each stage
TASK_TEMPLATE = {
    "opt": ("INCAR_OPT", "KPOINT_OPT"),
    "energy": ("INCAR_ENERGY", "KPOINT_ENERGY"),
    "dos": ("INCAR_DOS", "KPOINT_ENERGY"),
}

# 3. Layout of the calculation directories
# The whole M-N-C chain runs in one work/ directory, so the WAVECAR and CHGCAR of a
# material are reused in place as the chain goes opt -> energy -> dos. Once a stage
# is over its essentials are collected into the directory named after it.
WORK_DIR = "work"

M_dir = "../data/vasp-file/M"
MNC_dir = "../data/vasp-file/M-N-C"

# Collected out of work/ once a stage is over. The DOS stage keeps its DOSCAR too,
# which the PDOS analysis reads.
STAGE_KEEP = ("CONTCAR", "INCAR", "KPOINTS", "OSZICAR", "OUTCAR", "POSCAR", "POTCAR")
STAGE_EXTRA = {"dos": ("DOSCAR",)}

# The stage each stage consumes, and which must therefore be read before it starts.
PREVIOUS_STAGE = {"energy": "opt", "dos": "energy"}

# Energy window around E_F, in eV, as (lower, upper), shared by the band-center
# descriptor and the PDOS figures. It follows the convention of the d-band center
# literature and stays in the valence region only: the upper edge stops at +5 eV,
# because above it the d projector of a free atom picks up high-lying unoccupied
# weight that has nothing to do with bonding, and that alone swings the center of Sm
# from 1.3 eV to 21 eV. The lower edge sits at -10 eV, so the semicore s/p of the
# alkali and alkaline earth metals below it is left out on purpose as well; the
# descriptor is meant to describe the states that take part in bonding.
E_WINDOW = (-10.0, 5.0)


def WorkRoot():
    """The directory the whole chain runs in, where VASP is submitted."""
    return os.path.join(MNC_dir, WORK_DIR)


def StageRoot(stage: str):
    """The directory collecting the essentials of one finished stage."""
    return os.path.join(MNC_dir, stage)


def MaterialDirs(task_dir: str):
    """Sorted material directories of one task.

    Only directories are listed, so helper files and partially staged copies next to
    them are never mistaken for a material.
    """
    if not os.path.isdir(task_dir):
        return []
    return sorted(name for name in os.listdir(task_dir)
                  if not name.startswith(".")
                  and os.path.isdir(os.path.join(task_dir, name)))


def IsFinished(OUTCAR_path: str):
    """Whether a run reached its normal end.

    A killed or crashed run leaves an OUTCAR without the closing accounting block, so
    the energy it holds belongs to an unconverged iteration.
    """
    if not os.path.isfile(OUTCAR_path):
        return False
    with open(OUTCAR_path, "r", errors="ignore") as f:
        return any("General timing" in line for line in f)


def HasConverged(OUTCAR_path: str):
    """Whether a relaxation reached its force criterion (EDIFFG).

    A run can end normally without it, when NSW runs out while the last structure is
    still moving.
    """
    with open(OUTCAR_path, "r", errors="ignore") as f:
        return any("reached required accuracy" in line for line in f)


def TagValue(file_path: str, tag: str):
    """The integer a VASP file gives for one input tag, or None if it is absent."""
    pattern = re.compile(r"^\s*%s\s*=\s*(-?\d+)" % tag, re.IGNORECASE)
    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                return int(match.group(1))
    return None


def OutcarStage(OUTCAR_path: str):
    """Which stage an OUTCAR came from, read off the parameters VASP echoes back."""
    if not os.path.isfile(OUTCAR_path):
        return None
    return StageOf(TagValue(OUTCAR_path, "IBRION"), TagValue(OUTCAR_path, "LORBIT"))


def StageOf(ibrion, lorbit):
    """The stage a pair of IBRION and LORBIT values belongs to.

    Only the relaxation moves ions, so IBRION > 0 identifies opt. The two static runs
    are told apart by LORBIT, which only the DOS stage sets. A finished but unconverged
    relaxation is otherwise indistinguishable from a static run.
    """
    if ibrion is None:
        return None
    if ibrion > 0:
        return "opt"
    return "dos" if (lorbit or 0) >= 10 else "energy"


def IncarStage(incar_path: str):
    """Which stage an INCAR is written for, from the parameters it sets.

    The counterpart of OutcarStage() for inputs, so a directory already prepared for a
    stage is recognised rather than written over again.
    """
    if not os.path.isfile(incar_path):
        return None
    return StageOf(TagValue(incar_path, "IBRION"), TagValue(incar_path, "LORBIT"))


def SkipReason(src_dir: str, require_convergence: bool):
    """Why a material cannot move on to the next stage yet, or None if it can.

    A relaxation must also have reached its force criterion; a static run has none.
    """
    outcar = os.path.join(src_dir, "OUTCAR")
    if not os.path.isfile(outcar):
        return "no OUTCAR, never submitted"
    if not IsFinished(outcar):
        return "still running or crashed (no General timing)"
    if require_convergence and not HasConverged(outcar):
        return "ended without reaching the required accuracy"
    return None


def WriteINCAR(stage: str,
               material_dir: str,
               M,
               species: list):
    """(Re)build the INCAR of one material from the template of `stage`.

    MAGMOM only matters for the relaxation, later stages start from the WAVECAR of
    the previous one. The DFT+U block belongs to every stage.

    LDAUL / LDAUU / LDAUJ hold one value per species in POSCAR order, hence `species`.
    """
    incar_path = os.path.join(material_dir, "INCAR")
    GetINCAR(template_path="../data/template/%s" % TASK_TEMPLATE[stage][0],
             output_path=incar_path,
             mat_name="%s-N-C" % M.symbol)

    if stage == "opt" and M.symbol not in NO_SPIN_TAGS:
        magmom_M = float(SPIN_OVERRIDE.get(M.symbol, CountUnpaired(M)))
        SetINCARTags(incar_path, {"MAGMOM": "44*0.0 4*0.0 %s" % magmom_M})

    ldau = GetLDAU(M.symbol)
    if ldau is not None:
        SetLDAU(incar_path, ldau, species=species, metal=M.symbol)


def GenerateMInputs():
    """VASP inputs of the isolated metal atoms (../data/vasp-file/M).

    These only have an energy calculation, no relaxation and no DOS run.
    """
    if not os.path.exists(M_dir):
        os.makedirs(M_dir)

    for i in tqdm(range(len(Metals)), desc="Generating M VASP input files"):
        M = element(Metals[i])
        M_path = os.path.join(M_dir, M.symbol)
        if not os.path.exists(M_path):
            os.mkdir(M_path)

        # POSCAR first: the species order comes from it
        poscar_path = os.path.join(M_path, "POSCAR")
        GetPOSCAR(template_path="../data/template/POSCAR_ATOM",
                  output_path=poscar_path,
                  replace_pair={1:M.symbol})
        species = GetSpecies(poscar_path)

        incar_path = os.path.join(M_path, "INCAR")
        GetINCAR(template_path="../data/template/INCAR_ENERGY_ATOM",
                 output_path=incar_path,
                 mat_name=M.symbol)
        SetINCARTags(incar_path, {"ISPIN": GetISPIN(M)})
        # The free-atom rule, not SPIN_OVERRIDE: NUPDOWN is a constraint, and an odd
        # electron count only admits an odd difference.
        hund = CountUnpaired(M)
        if hund > 0 and M.symbol not in NO_SPIN_TAGS:
            SetINCARTags(incar_path, {"MAGMOM": hund, "NUPDOWN": hund})
        ldau = GetLDAU(M.symbol)
        if ldau is not None:
            SetLDAU(incar_path, ldau, species=species, metal=M.symbol)

        GetKPOINT(template_path="../data/template/KPOINT_SM",
                 output_path=os.path.join(M_path, "KPOINTS"))
        GetPOTCAR(poscar_path=poscar_path,
                  output_path=os.path.join(M_path, "POTCAR"),
                  potential_db_path=potential_db_path,
                  potential_map=POTCAR_VARIANT)


def HasInputs(material_dir: str):
    """Whether a material already holds a complete set of VASP inputs."""
    return all(os.path.isfile(os.path.join(material_dir, name))
               for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR"))


def GenerateWorkInputs(stage: str):
    """Set work/ up for the first stage, generating its inputs from the templates.

    A material that already holds its inputs keeps them, so the templates can neither
    overwrite a finished run nor disturb a running one.
    """
    task_dir = WorkRoot()
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    kpoint_template = TASK_TEMPLATE[stage][1]
    created = 0

    for i in tqdm(range(len(Metals)), desc="Generating M-N-C VASP input files for %s" % stage):
        M = element(Metals[i])
        MNC_path = os.path.join(task_dir, "%s-N-C" % M.symbol)

        if HasInputs(MNC_path):
            continue
        if not os.path.exists(MNC_path):
            os.mkdir(MNC_path)

        # POSCAR first: the species order comes from it
        poscar_path = os.path.join(MNC_path, "POSCAR")
        GetPOSCAR(template_path="../data/template/POSCAR_MNC",
                  output_path=poscar_path,
                  replace_pair={49:M.symbol})
        species = GetSpecies(poscar_path)

        WriteINCAR(stage, MNC_path, M, species)

        GetKPOINT(template_path="../data/template/%s" % kpoint_template,
                 output_path=os.path.join(MNC_path, "KPOINTS"))
        GetPOTCAR(poscar_path=poscar_path,
                  output_path=os.path.join(MNC_path, "POTCAR"),
                  potential_db_path=potential_db_path,
                  potential_map=POTCAR_VARIANT)
        created += 1

    return created


def CopyEssentials(stage: str, src_dir: str, dst_dir: str):
    """Copy the essentials of one finished material into `dst_dir`.

    A finished run writes all of STAGE_KEEP, so a missing one is an error rather than
    something to skip: an incomplete archive would otherwise only surface when the
    data set is built. STAGE_EXTRA is optional.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for name in STAGE_KEEP:
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            raise FileNotFoundError("no %s in %s" % (name, src_dir))
        shutil.copyfile(src, os.path.join(dst_dir, name))
    for name in STAGE_EXTRA.get(stage, ()):
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(dst_dir, name))


def IsArchived(stage: str, name: str):
    """Whether the essentials of `name` have already been collected in `stage`."""
    return os.path.isfile(os.path.join(StageRoot(stage), name, "OUTCAR"))


def ReportOutcome(verb: str, stage: str, total: int, done, skipped, failed, target: str):
    """Print what one pass over work/ managed to do."""
    print()
    print("%d of %d material(s) %s %s -> %s:" % (len(done), total, verb, stage, target))
    if done:
        print("    " + " ".join(done))
    if skipped:
        print("Skipped %d material(s):" % len(skipped))
        for name, reason in skipped:
            print("    %-9s %s" % (name, reason))
    if failed:
        print("Failed %d material(s):" % len(failed))
        for name, reason in failed:
            print("    %-9s %s" % (name, reason))


def PrepareWorkStage(stage: str,
                     material_dir: str,
                     M):
    """Turn a finished material of work/ into the input of `stage`.

    INCAR and KPOINTS are rebuilt from the template of `stage`, the relaxed CONTCAR
    becomes the POSCAR, and the stale POSCAR / OUTCAR are dropped. WAVECAR and CHGCAR
    stay in place, so the next stage continues from the previous wavefunction
    (ISTART = 1).
    """
    poscar_path = os.path.join(material_dir, "POSCAR")
    outcar_path = os.path.join(material_dir, "OUTCAR")
    contcar_path = os.path.join(material_dir, "CONTCAR")

    # WriteINCAR needs the species order, which comes from the structure: whichever of
    # the two files an interrupted run left in place.
    structure_path = poscar_path if os.path.isfile(poscar_path) else contcar_path
    species = GetSpecies(structure_path)
    WriteINCAR(stage, material_dir, M, species)
    GetKPOINT(template_path="../data/template/%s" % TASK_TEMPLATE[stage][1],
              output_path=os.path.join(material_dir, "KPOINTS"))

    # The relaxed structure takes over; a static run writes no CONTCAR, so there the
    # POSCAR already is the structure. The OUTCAR goes first, so an interruption
    # cannot leave a directory that still looks like a finished run.
    if os.path.isfile(outcar_path):
        os.remove(outcar_path)
    if os.path.isfile(contcar_path):
        if os.path.isfile(poscar_path):
            os.remove(poscar_path)
        os.rename(contcar_path, poscar_path)


def ReadStage(stage: str):
    """Collect the finished runs of `stage` out of work/ into their archive.

    A work directory's stage is read off its OUTCAR, so results of a later stage are
    never filed under an earlier one. The OUTCAR is consumed once filed, which is what
    tells StartStage() that the material may move on.
    """
    work_root = WorkRoot()
    materials = MaterialDirs(work_root)
    if not materials:
        print("No material directory found in %s, nothing to do." % work_root)
        return 0

    # only a relaxation has a force criterion to reach
    require_convergence = (stage == "opt")

    dst_root = StageRoot(stage)
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    skipped, failed, done = [], [], []
    for name in tqdm(materials, desc="Reading %s" % stage):
        work_dir = os.path.join(work_root, name)
        outcar = os.path.join(work_dir, "OUTCAR")

        run_stage = OutcarStage(outcar)
        if run_stage != stage:
            skipped.append((name, "%s run" % (run_stage or "no OUTCAR")))
            continue

        reason = SkipReason(work_dir, require_convergence)
        if reason is not None:
            skipped.append((name, reason))
            continue

        staging = os.path.join(dst_root, ".%s.tmp" % name)
        try:
            # assembled next to the archive, so a partial copy is never mistaken for
            # a complete one
            if os.path.isdir(staging):
                shutil.rmtree(staging)
            CopyEssentials(stage, work_dir, staging)
            dst_dir = os.path.join(dst_root, name)
            if os.path.isdir(dst_dir):
                shutil.rmtree(dst_dir)
            os.rename(staging, dst_dir)
            # consumed: the run is filed, work/ is free for the next stage
            os.remove(outcar)
        except Exception as e:      # one bad material must not abort the whole batch
            # a half written copy must not be left behind
            if os.path.isdir(staging):
                shutil.rmtree(staging)
            failed.append((name, "%s: %s" % (type(e).__name__, e)))
            continue
        done.append(name)

    ReportOutcome("read", stage, len(materials), done, skipped, failed, dst_root)
    return len(done)


def StartStage(stage: str):
    """Set the work directories up for `stage`, from the structure they hold.

    A material is prepared only once its previous run has been read, and only if it is
    not prepared already. An OUTCAR in the way means a calculation is running or is
    waiting to be read, and rebuilding the inputs would destroy it.
    """
    work_root = WorkRoot()
    materials = MaterialDirs(work_root)
    if not materials:
        print("No material directory found in %s, nothing to do." % work_root)
        return 0

    previous = PREVIOUS_STAGE.get(stage)

    skipped, failed, done = [], [], []
    for name in tqdm(materials, desc="Preparing %s" % stage):
        work_dir = os.path.join(work_root, name)

        run_stage = OutcarStage(os.path.join(work_dir, "OUTCAR"))
        if run_stage is not None:
            skipped.append((name, "%s run in the way, not read yet" % run_stage))
            continue

        if IncarStage(os.path.join(work_dir, "INCAR")) == stage:
            skipped.append((name, "already prepared"))
            continue

        if previous is not None and not IsArchived(previous, name):
            skipped.append((name, "%s not read yet" % previous))
            continue

        try:
            PrepareWorkStage(stage, work_dir, element(name[:-4]))
        except Exception as e:      # one bad material must not abort the whole batch
            failed.append((name, "%s: %s" % (type(e).__name__, e)))
            continue
        done.append(name)

    ReportOutcome("start", stage, len(materials), done, skipped, failed, work_root)
    return len(done)


def SplitStage(stage):
    """Split the DOSCAR of every material of the stage into per-atom files.

    The materials that already hold the files are skipped, so the call can be
    repeated as new runs land.
    """
    stage_dir = '../data/vasp-file/M-N-C/%s' % stage
    if not os.path.exists(stage_dir):
        raise FileNotFoundError("%s does not exist" % stage_dir)

    todo = []
    for name in sorted(os.listdir(stage_dir)):
        if not os.path.isfile(os.path.join(stage_dir, name, "DOSCAR")):
            continue
        if os.path.exists(os.path.join(stage_dir, name, "DOS0")):
            continue
        todo.append(name)

    if not todo:
        print("The DOSCAR files of %s are already split." % stage)
        return

    for name in tqdm(todo, desc="Splitting the DOSCAR of %s" % stage):
        SplitDOSCAR(os.path.join(stage_dir, name))


def BuildDataset():
    """Analyse the finished runs and write the data set.

    Needs every stage at once, so it is not part of the chain.
    """
    # The PDOS is read one site at a time from the per-atom files, which keeps
    # the whole DOSCAR out of memory.
    SplitStage("dos")

    # PDOS of the four N atoms and the central metal
    for filename in tqdm(os.listdir('../data/vasp-file/M-N-C/dos'), desc="Analyzing DOS data"):
        M = filename[:-4]
        if M in AM:
            obt = "s"
        elif M in AEM:
            obt = "p"
        elif M in TM:
            obt= "d"
        elif M in MGM:
            obt = "p"
        elif M in LnM:
            obt = "d"

        doscar = SplitDoscar(dos_dir=f"../data/vasp-file/M-N-C/dos/{filename}",
                             ispin=2)
        e_range = list(E_WINDOW)
        pdos_obt = {45: "p", 46:"p", 47:"p", 48:"p", 49:obt}
        color_list = ["#bad6ea", "#88BEDC", "#539DCC", "#2A7AB9", "#ce4459"]
        atom_name = ["N1", "N2", "N3", "N4", M]
        pdos_list = []
        for key in pdos_obt.keys():
            idx = key
            obt = pdos_obt[key]
            up = doscar.pdos_sum([idx - 1], spin='up', l=obt)
            down = doscar.pdos_sum([idx - 1], spin='down', l=obt)

            # shift to the Fermi level
            energies = doscar.energy - doscar.efermi  # E - E_F

            # window kept around E_F
            emask = (energies >= e_range[0]) & (energies <= e_range[1])
            x = energies[emask]
            y_up = up[emask]
            y_down = down[emask]

            pdos = [x, y_up, y_down]
            pdos_list.append(pdos)

        # plotting
        plt.figure(dpi=300)

        all_y_up = []
        all_y_down = []

        for i in range(len(pdos_list)):
            atom_pdos = pdos_list[i]
            x = atom_pdos[0]
            y_up = atom_pdos[1]
            y_down = -atom_pdos[2]

            plt.plot(x, y_up, color=color_list[i], label = atom_name[i], zorder=5-i, alpha=0.5, linewidth=1)
            plt.plot(x, y_down, color=color_list[i], zorder=5-i, alpha=0.5, linewidth=1)

            all_y_up.extend(y_up)
            all_y_down.extend(y_down)

        extend = 0.5
        y_min_actual = min(min(all_y_up), min(all_y_down))
        y_max_actual = max(max(all_y_up), max(all_y_down))
        plt.ylim(y_min_actual - extend, y_max_actual + extend)

        plt.axvline(x=0, color="grey", linewidth=2, linestyle="--", alpha=0.7)

        plt.xlabel("E - E$_f$ (eV)")
        plt.ylabel("PDOS(eV)")
        plt.title("%s orbital of %s" % (obt, M))

        if e_range[0] != -np.inf and e_range[1] != np.inf:
            plt.xlim(e_range[0], e_range[1])

        plt.grid(True, linestyle="--", alpha=0.8)
        plt.legend(loc="upper left", prop={'size': 12})
        plt.savefig("../figures/dos/%s_%s.png" % (M, obt))
        plt.close()

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
                "out-of-plane displacement of M/A",
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
                # derived from the columns above
                "E_b/eV",
                "E_f/eV",
                "U_diss_acid/V",
                "U_diss_base/V",
                "stable pH"]

    for M in Metals:
        MNC_dict[M] = []
        for i in range(len(col_name)):
            MNC_dict[M].append("-")

    # from ../data/vasp-file/M
    for filename in os.listdir('../data/vasp-file/M'):
        outcar = f"../data/vasp-file/M/{filename}/OUTCAR"
        if not os.path.isfile(outcar):
            continue
        if filename not in MNC_dict:    # extra runs kept beside the data set
            continue
        MNC_dict[filename][0] = GetEnergy(outcar)

    # from atom bulk energy.xlsx
    e_bulk = pd.read_excel("../data/atom-table/bulk energy.xlsx")
    e_bulk = e_bulk.set_index('element')['Fit-partial(eV)'].to_dict()
    for M in MNC_dict.keys():
        MNC_dict[M][1] = e_bulk[M]

    # from ../data/vasp-file/M-N-C/energy
    for filename in tqdm(os.listdir('../data/vasp-file/M-N-C/energy'), desc="Processing Energy Data"):
        e = GetEnergy(f"../data/vasp-file/M-N-C/energy/{filename}/OUTCAR")
        M = element(filename[:-4])

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

        # out-of-plane displacement of the metal from the N4 plane
        MNC_dict[M.symbol][5] = GetOutOfPlane(
            f"../data/vasp-file/M-N-C/energy/{filename}/CONTCAR",
            idx_M=49, idx_N=[45, 46, 47, 48])

        # Coulomb matrix of the M-N4 site.  Only the atoms of the first
        # coordination shell enter the matrix, the metal plus its four N, so the
        # matrix has the same size for every element:
        #   diagonal     E_i     = 0.5 * Z_i ** 2.4          (atomic energy)
        #   off-diagonal M_ij    = Z_i * Z_j / |R_i - R_j|   (Coulomb repulsion)
        # Only the upper triangle is kept, and the two blocks are collapsed to
        # their averages: CM1 is the average diagonal element and CM2 the average
        # off-diagonal element.
        contcar = f"../data/vasp-file/M-N-C/energy/{filename}/CONTCAR"
        z_of = {49: M.atomic_number}                      # the central metal
        for id in range(45, 49):
            z_of[id] = element("N").atomic_number

        site = sorted(z_of)                               # the five shell atoms
        off_diagonal, diagonal = [], []
        for i, id_i in enumerate(site):
            diagonal.append(0.5 * pow(z_of[id_i], 2.4))
            for id_j in site[i + 1:]:
                off_diagonal.append(z_of[id_i] * z_of[id_j] / GetDistance(contcar, id_i, id_j))

        MNC_dict[M.symbol][6] = float(np.mean(diagonal))
        MNC_dict[M.symbol][7] = float(np.mean(off_diagonal))

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

        dbc, width = GetBandCenter(dos_dir=f"../data/vasp-file/M-N-C/dos/{filename}",
                                   idx=49,
                                   orbital=obt,
                                   e_range=list(E_WINDOW))

        MNC_dict[M][8] = dbc
        MNC_dict[M][9] = width

    # from mendeleev package
    for M in tqdm(MNC_dict.keys(), desc="Generating mendeleev feature"):
        M = element(M)
        MNC_dict[M.symbol][10] = M.atomic_number
        MNC_dict[M.symbol][11] = M.atomic_weight
        MNC_dict[M.symbol][12] = M.atomic_radius
        MNC_dict[M.symbol][13] = M.covalent_radius_cordero
        MNC_dict[M.symbol][14] = M.heat_of_formation
        MNC_dict[M.symbol][15] = M.molar_heat_capacity
        MNC_dict[M.symbol][16] = M.vdw_radius
        MNC_dict[M.symbol][17] = M.zeff()
        if M.group_id is not None:
            MNC_dict[M.symbol][18] = M.group_id
        else:
            MNC_dict[M.symbol][18] = 3 # lanthanides have no group

    # from atom potential.xlsx
    potential = pd.read_excel("../data/atom-table/potential.xlsx", sheet_name="U_diss")
    valence = potential.set_index('element')['valence'].to_dict()
    U_diss_std_acid = potential.set_index('element')['U_diss_acid'].to_dict()
    U_diss_std_base = potential.set_index('element')['U_diss_base'].to_dict()

    for M in tqdm(MNC_dict.keys(), desc="Reading Potential.xlsx"):
        MNC_dict[M][19] = valence[M]
        MNC_dict[M][20] = U_diss_std_acid[M]
        MNC_dict[M][21] = U_diss_std_base[M]

    # labels, derived from the data above
    E_N4C = GetEnergy("../data/vasp-file/N-C/energy/OUTCAR")

    for M in tqdm(MNC_dict.keys(), desc="Calculating stability parameters"):
        if MNC_dict[M][2] != "-":
            E_b = MNC_dict[M][2] - E_N4C - MNC_dict[M][0]
            E_f = MNC_dict[M][2] - E_N4C - MNC_dict[M][1]
            U_diss_acid = U_diss_std_acid[M] - E_f / valence[M] + 0.0592 * 0
            U_diss_base = U_diss_std_base[M] - E_f / valence[M] + 0.0592 * 14

            MNC_dict[M][22] = E_b
            MNC_dict[M][23] = E_f
            MNC_dict[M][24] = U_diss_acid
            MNC_dict[M][25] = U_diss_base
            stable_pH = []
            if U_diss_acid >= 0 and U_diss_base >= 0:
                MNC_dict[M][26] = "Both"
            elif U_diss_acid >= 0 and U_diss_base <= 0:
                MNC_dict[M][26] = "Acid"
            elif U_diss_base >= 0 and U_diss_acid <= 0:
                MNC_dict[M][26] = "Base"
            elif U_diss_base <= 0 and U_diss_acid <= 0:
                MNC_dict[M][26] = "None"

    MNC_df = pd.DataFrame(MNC_dict).T
    MNC_df.columns = col_name
    MNC_df.to_excel("../data/M-N-C data set.xlsx")


# 4. Entry point
USAGE = """
usage: python "dataset generator.py" <system> <stage> <action>

  system  M       the isolated metal atoms
          M-N-C   the M-N4-C systems

  The M-N-C chain runs entirely inside data/vasp-file/M-N-C/work, which is where
  VASP is submitted. Each stage is used twice:

  start   write the inputs of the stage into work/, ready for submission
  read    copy the finished results out of work/ into the directory of the stage

  So a stage is submitted as:  <stage> start -> sbatch work/submit.sh -> <stage> read

  stage   M-N-C: opt      the relaxation
                 energy   the total energy of the relaxed structure
                 dos      the projected DOS
                          split  cut the DOSCAR into one file per atom
                 dataset  analyse the PDOS and write the data set
          M:     energy   the isolated-atom inputs, start only
""".strip("\n")

# stage -> actions, per system.
VALID_MODES = {
    "M": {"energy": ["start"]},
    "M-N-C": {
        "opt": ["start", "read"],
        "energy": ["start", "read"],
        "dos": ["start", "split", "read"],
        "dataset": ["start"],
    },
}


def ParseArguments(argv):
    """Read <system> <stage> <action> from the command line and validate them."""
    if len(argv) != 4:
        print(USAGE)
        raise SystemExit(2)

    system, stage, action = argv[1], argv[2], argv[3]
    if system not in VALID_MODES:
        print("Unknown system: %s" % system)
        print(USAGE)
        raise SystemExit(2)
    if stage not in VALID_MODES[system]:
        print("%s does not have the stage %s, it only has: %s"
              % (system, stage, ", ".join(VALID_MODES[system])))
        raise SystemExit(2)
    if action not in VALID_MODES[system][stage]:
        print("%s %s does not have the action %s, it only has: %s"
              % (system, stage, action, ", ".join(VALID_MODES[system][stage])))
        raise SystemExit(2)
    return system, stage, action


def main(argv):
    system, stage, action = ParseArguments(argv)

    if system == "M":
        GenerateMInputs()
    elif stage == "dataset":
        BuildDataset()
    elif action == "start":
        # opt is generated from the templates, later stages from the structure the
        # material already holds
        if stage == "opt":
            GenerateWorkInputs(stage)
        else:
            StartStage(stage)
    elif action == "split":
        SplitStage(stage)
    else:
        ReadStage(stage)


if __name__ == "__main__":
    main(sys.argv)
