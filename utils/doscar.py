# Ref: https://vasppy.readthedocs.io/en/latest/_modules/vasppy/doscar.html

import os

import numpy as np
import pandas as pd
from scipy.integrate import simps
import datetime
import time


def pdos_column_names(lmax, ispin):
    if lmax == 2:
        names = ['s', 'p_y', 'p_z', 'p_x', 'd_xy', 'd_yz', 'd_z2-r2', 'd_xz', 'd_x2-y2']
        # names = [ 's', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2' ]
    elif lmax == 3:
        names = ['s', 'p_y', 'p_z', 'p_x', 'd_xy', 'd_yz', 'd_z2-r2', 'd_xz', 'd_x2-y2',
                 'f_y(3x2-y2)', 'f_xyz', 'f_yz2', 'f_z3', 'f_xz2', 'f_z(x2-y2)', 'f_x(x2-3y2)']
        # names = [ 's', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2','f-3',
        #         'f-2', 'f-1', 'f0', 'f1', 'f2', 'f3']
    else:
        raise ValueError('lmax value not supported')

    if ispin == 2:
        all_names = []
        for n in names:
            all_names.extend(['{}_up'.format(n), '{}_down'.format(n)])
    else:
        all_names = names
    all_names.insert(0, 'energy')
    return all_names


class Doscar:
    '''
    Contains all the data in a VASP DOSCAR file, and methods for manipulating this.
    '''

    number_of_header_lines = 6

    def __init__(self, filename, ispin=2, lmax=2, lorbit=11, spin_orbit_coupling=False, read_pdos=True, species=None):
        '''
        Create a Doscar object from a VASP DOSCAR file.
        Args:
            filename (str): Filename of the VASP DOSCAR file to read.
            ispin (optional:int): ISPIN flag.
                Set to 1 for non-spin-polarised or 2 for spin-polarised calculations.
                Default = 2.
            lmax (optional:int): Maximum l angular momentum. (d=2, f=3). Default = 2.
            lorbit (optional:int): The VASP LORBIT flag. (Default=11).
            spin_orbit_coupling (optional:bool): Spin-orbit coupling (Default=False).
            read_pdos (optional:bool): Set to True to read the atom-projected density of states (Default=True).
            species (optional:list(str)): List of atomic species strings, e.g. [ 'Fe', 'Fe', 'O', 'O', 'O' ].
                Default=None.
        '''
        self.filename = filename
        self.ispin = ispin
        self.lmax = lmax
        self.spin_orbit_coupling = spin_orbit_coupling
        if self.spin_orbit_coupling:
            raise NotImplementedError('Spin-orbit coupling is not yet implemented')
        self.lorbit = lorbit
        self.pdos = None
        self.species = species
        self.read_header()
        self.read_total_dos()
        if read_pdos:
            try:
                self.read_projected_dos()
            except:
                raise
        # if species is set, should check that this is consistent with the number of entries in the
        # projected_dos dataset

    @property
    def number_of_channels(self):
        if self.lorbit == 11:
            return {2: 9, 3: 16}[self.lmax]
        raise NotImplementedError

    def read_header(self):
        self.header = []
        with open(self.filename, 'r') as file_in:
            for i in range(Doscar.number_of_header_lines):
                self.header.append(file_in.readline())
        self.process_header()

    def process_header(self):
        self.number_of_atoms = int(self.header[0].split()[0])
        self.number_of_data_points = int(self.header[5].split()[2])
        self.efermi = float(self.header[5].split()[3])

    def read_total_dos(self):  # assumes spin_polarised
        start_to_read = Doscar.number_of_header_lines
        df = pd.read_csv(self.filename,
                         skiprows=start_to_read,
                         nrows=self.number_of_data_points,
                         sep='\s+',
                         names=['energy', 'up', 'down', 'int_up', 'int_down'],
                         index_col=False)
        self.energy = df.energy.values
        df.drop('energy', axis=1)
        self.tdos = df

    def read_atomic_dos_as_df(self, atom_number):  # currently assume spin-polarised, no-SO-coupling, no f-states
        assert atom_number > 0 & atom_number <= self.number_of_atoms
        start_to_read = Doscar.number_of_header_lines + atom_number * (self.number_of_data_points + 1)
        df = pd.read_csv(self.filename,
                         skiprows=start_to_read,
                         nrows=self.number_of_data_points,
                         sep='\s+',
                         names=pdos_column_names(lmax=self.lmax, ispin=self.ispin),
                         index_col=False)
        return df.drop('energy', axis=1)

    def read_projected_dos(self):
        """
        Read the projected density of states data into """
        pdos_list = []
        for i in range(self.number_of_atoms):
            df = self.read_atomic_dos_as_df(i + 1)
            pdos_list.append(df)
        # self.pdos  =   pdos_list
        self.pdos = np.vstack([np.array(df) for df in pdos_list]).reshape(
            self.number_of_atoms, self.number_of_data_points, self.number_of_channels, self.ispin)

    def pdos_select(self, atoms=None, spin=None, l=None, m=None):
        """
        Returns a subset of the projected density of states array.
        """
        valid_m_values = {'s': [],
                          'p': ['x', 'y', 'z'],
                          'd': ['xy', 'yz', 'z2-r2', 'xz', 'x2-y2'],
                          'f': ['y(3x2-y2)', 'xyz', 'yz2', 'z3', 'xz2', 'z(x2-y2)', 'x(x2-3y2)']}
        if not atoms:
            atom_idx = list(range(self.number_of_atoms))
        else:
            atom_idx = atoms
        to_return = self.pdos[atom_idx, :, :, :]
        if not spin:
            spin_idx = list(range(self.ispin))
        elif spin == 'up':
            spin_idx = [0]
        elif spin == 'down':
            spin_idx = [1]
        elif spin == 'both':
            spin_idx = [0, 1]
        else:
            raise ValueError
        to_return = to_return[:, :, :, spin_idx]

        if not l:
            channel_idx = list(range(self.number_of_channels))
        elif l == 's':
            channel_idx = [0]
        elif l == 'p':
            if not m:
                channel_idx = [1, 2, 3]
            else:
                channel_idx = [i for i, v in enumerate(valid_m_values['p']) if v in m]
        elif l == 'd':
            if not m:
                channel_idx = [4, 5, 6, 7, 8]
            else:
                channel_idx = [i for i, v in enumerate(valid_m_values['d']) if v in m]
        elif l == 'f':
            if not m:
                channel_idx = [9, 10, 11, 12, 13, 14, 15]
            else:
                channel_idx = [i for i, v in enumerate(valid_m_values['f']) if v in m]
        else:
            raise ValueError

        return to_return[:, :, channel_idx, :]

    def pdos_sum(self, atoms=None, spin=None, l=None, m=None):
        return np.sum(self.pdos_select(atoms=atoms, spin=spin, l=l, m=m), axis=(0, 2, 3))


# 2. Per-atom files: split the DOSCAR once, then read single sites from them
# The split_dos.ksh distributed with VASP keeps only the first columns of every
# block and needs an external "vp" helper for the coordinate comment, so it
# cannot serve the d and f projections used here. SplitDOSCAR writes the whole
# block instead and needs nothing but the DOSCAR.

def _ChannelIndex(l, m, lmax):
    """Column positions of one orbital, counted without the energy column."""
    if lmax not in (2, 3):
        raise ValueError("lmax value not supported")
    names = pdos_column_names(lmax=lmax, ispin=1)[1:]

    if l is None:
        return list(range(len(names)))
    if l == 's':
        return [0]
    if l not in ('p', 'd', 'f'):
        raise ValueError("Unknown orbital: %s" % l)
    if m is None:
        return [i for i, name in enumerate(names) if name.split('_')[0] == l]

    # m is a substring of the label, e.g. "z2" of "d_z2-r2"
    if isinstance(m, str):
        m = [m]
    return [i for i, name in enumerate(names)
            if name.split('_')[0] == l and any(k in name.split('_', 1)[-1] for k in m)]


def SplitDOSCAR(dos_dir, efermi=None):
    """Write DOS0 (total) and DOS1..DOSn (projected) beside the DOSCAR.

    DOS0 holds "energy up down int_up int_down". Every other file holds the
    energy followed by the spin channels of each projection, in the column
    order given by pdos_column_names. All energies are shifted so that the
    Fermi level sits at zero.
    """
    lines = open(os.path.join(dos_dir, "DOSCAR")).readlines()

    natom = int(lines[0].split()[0])
    nedos = int(lines[5].split()[2])
    if efermi is None:
        efermi = float(lines[5].split()[3])

    energy = np.array([float(line.split()[0]) for line in lines[6:6 + nedos]]) - efermi

    total = np.array([line.split() for line in lines[6:6 + nedos]], dtype=float)
    np.savetxt(os.path.join(dos_dir, "DOS0"),
               np.column_stack([energy, total[:, 1:]]), fmt="%15.8E")

    # Each projected block is preceded by the one-line header that DOSCAR
    # repeats in front of every atom.
    first = 6 + nedos + 1
    for i in range(1, natom + 1):
        start = first + (i - 1) * (nedos + 1)
        block = np.array([line.split() for line in lines[start:start + nedos]], dtype=float)
        np.savetxt(os.path.join(dos_dir, "DOS%d" % i),
                   np.column_stack([energy, block[:, 1:]]), fmt="%15.8E")

    return natom, nedos


class SplitDoscar:
    """Per-atom view of a DOSCAR previously split by SplitDOSCAR.

    Sites are read one at a time and kept, so a caller that needs a few atoms
    never builds the whole natom x nedos x nchannel x ispin array.

    Args:
        dos_dir (str): Directory holding DOSCAR and its DOS0..DOSn files.
        ispin (int): 1 for unpolarised, 2 for spin polarised.
    """

    def __init__(self, dos_dir, ispin=2):
        self.dos_dir = dos_dir
        self.ispin = ispin
        # SplitDOSCAR already shifted the energies, so E_F reads as zero here.
        self.efermi = 0.0
        self.energy = np.loadtxt(os.path.join(dos_dir, "DOS0"), usecols=0)
        self.nedos = self.energy.size

        self.natom = 0
        while os.path.exists(os.path.join(dos_dir, "DOS%d" % (self.natom + 1))):
            self.natom += 1

        first_row = np.loadtxt(os.path.join(dos_dir, "DOS1"), max_rows=1)
        self.nchannel = (first_row.size - 1) // ispin
        self.lmax = {9: 2, 16: 3}.get(self.nchannel)
        self._sites = {}

    def site(self, atom):
        """(nedos, nchannel, ispin) arrays of one site, atom numbered from 0."""
        if atom not in self._sites:
            data = np.loadtxt(os.path.join(self.dos_dir, "DOS%d" % (atom + 1)))
            self._sites[atom] = data[:, 1:].reshape(self.nedos, self.nchannel, self.ispin)
        return self._sites[atom]

    def pdos_select(self, atoms=None, spin=None, l=None, m=None):
        if not atoms:
            atoms = list(range(self.natom))
        to_return = np.stack([self.site(atom) for atom in atoms])

        if not spin:
            spin_idx = list(range(self.ispin))
        elif spin == 'up':
            spin_idx = [0]
        elif spin == 'down':
            spin_idx = [1]
        elif spin == 'both':
            spin_idx = [0, 1]
        else:
            raise ValueError("Unknown spin: %s" % spin)

        channel_idx = _ChannelIndex(l, m, self.lmax)
        return to_return[:, :, channel_idx, :][:, :, :, spin_idx]

    def pdos_sum(self, atoms=None, spin=None, l=None, m=None):
        return np.sum(self.pdos_select(atoms=atoms, spin=spin, l=l, m=m), axis=(0, 2, 3))