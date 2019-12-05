#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

import numpy as np

from ard_gsm.mol import MolGraph


class QChemError(Exception):
    pass


class QChem(object):
    """
    `mol` is an RDKit molecule with Hs already added. It should already
    contain the 3D geometry of the lowest-energy conformer.
    Alternatively, it can be a MolGraph object with coordinates.

    The methods of this class have only been validated for Q-Chem 5.1 DFT
    calculations.
    """

    def __init__(self, mol=None, config_file=None, logfile=None):
        self.mol = mol  # RDKit molecule with Hs added or MolGraph

        if config_file is None:
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       os.pardir,
                                       'config',
                                       'qchem.opt_freq')
        with open(config_file) as f:
            self.config = [line.strip() for line in f]

        self.logfile = logfile
        if logfile is None:
            self.log = None
        else:
            with open(logfile) as f:
                self.log = f.read().splitlines()
                for line in self.log:
                    if 'fatal error' in line:
                        raise QChemError(f'Q-Chem job {logfile} had an error!')

    def make_input(self, path, charge=0, multiplicity=1, mem=None, comment=None):
        if isinstance(self.mol, MolGraph):
            symbols = self.mol.get_symbols()
            coords = self.mol.get_coords()
        else:
            symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
            coords = self.mol.GetConformers()[0].GetPositions()
        self.make_input_from_coords(path, symbols, coords, charge=charge, multiplicity=multiplicity,
                                    mem=mem, comment=comment)

    def make_input_from_coords(self, path, symbols, coords, charge=0, multiplicity=1, mem=None, comment=None):
        config = self.config[:]
        if comment is not None:
            config.insert(0, comment + '\n')

        for i, line in enumerate(config):
            if line.startswith('$molecule'):
                cblock = [f'{symbol}  {xyz[0]: .10f}  {xyz[1]: .10f}  {xyz[2]: .10f}'
                          for symbol, xyz in zip(symbols, coords)]
                cblock.insert(0, f'{charge} {multiplicity}')
                config[(i+1):(i+1)] = cblock
                break  # If there are more than 1 molecule block, only fill the first one

        if mem is not None:
            # Memory specified in MB
            config = insert_into_qcinput(config, f'MEM_TOTAL                 {mem:d}', '$rem')

        with open(path, 'w') as f:
            for line in config:
                f.write(line + '\n')

    def get_energy(self, first=False):
        if first:
            iterable = self.log
        else:
            iterable = reversed(self.log)
        for line in iterable:
            if 'total energy' in line:  # Double hybrid methods
                return float(line.split()[-2])
            elif 'energy in the final basis set' in line:  # Other DFT methods
                return float(line.split()[-1])
        else:
            raise QChemError(f'Energy not found in {self.logfile}')

    def get_geometry(self, first=False):
        if first:
            iterable = range(len(self.log))
        else:
            iterable = reversed(range(len(self.log)))
        for i in iterable:
            line = self.log[i]
            if 'Standard Nuclear Orientation' in line:
                symbols, coords = [], []
                for line in self.log[(i+3):]:
                    if '----------' not in line:
                        data = line.split()
                        symbols.append(data[1])
                        coords.append([float(c) for c in data[2:]])
                    else:
                        return symbols, np.array(coords)
        else:
            raise QChemError(f'Geometry not found in {self.logfile}')

    def get_moments_of_inertia(self):
        for line in reversed(self.log):
            if 'Eigenvalues --' in line:
                inertia = [float(i) * 0.52917721092**2.0 for i in line.split()[-3:]]  # Convert to amu*angstrom^2
                if inertia[0] == 0.0:  # Linear rotor
                    inertia = np.sqrt(inertia[1]*inertia[2])
                return inertia

    def get_frequencies(self):
        freqs = []
        for line in reversed(self.log):
            if 'Frequency' in line:
                freqs.extend([float(f) for f in reversed(line.split()[1:])])
            elif 'VIBRATIONAL ANALYSIS' in line:
                freqs.reverse()
                return np.array(freqs)
        else:
            raise QChemError(f'Frequencies not found in {self.logfile}')

    def get_normal_modes(self):
        modes = []
        for i in reversed(range(len(self.log))):
            line = self.log[i]
            if 'Raman Active' in line:
                mode1, mode2, mode3 = [], [], []
                for line in self.log[(i+2):]:
                    if 'TransDip' not in line:
                        vals = line.split()[1:]
                        mode1.append([float(v) for v in vals[:3]])
                        mode2.append([float(v) for v in vals[3:6]])
                        mode3.append([float(v) for v in vals[6:]])
                    else:
                        modes.extend([np.array(mode3), np.array(mode2), np.array(mode1)])
                        break
            elif 'VIBRATIONAL ANALYSIS' in line:
                modes.reverse()
                return modes
        else:
            raise QChemError(f'Normal modes not found in {self.logfile}')

    def get_zpe(self):
        for line in reversed(self.log):
            if 'Zero point vibrational energy' in line:
                return float(line.split()[-2]) / 627.5095  # Convert to Hartree
        else:
            raise QChemError(f'ZPE not found in {self.logfile}')

    def get_charge(self):
        for i, line in enumerate(self.log):
            if '$molecule' in line:
                return int(self.log[i+1].strip().split()[0])
        else:
            raise QChemError(f'Charge not found in {self.logfile}')

    def get_multiplicity(self):
        for i, line in enumerate(self.log):
            if '$molecule' in line:
                return int(self.log[i+1].strip().split()[-1])
        else:
            raise QChemError(f'Multiplicity not found in {self.logfile}')

    def get_comment(self):
        for i, line in enumerate(self.log):
            if 'User input' in line:
                comment = self.log[i+2].strip()
                if comment and comment not in {'$molecule', '$rem'}:
                    return comment
        else:
            raise QChemError(f'No comment found in {self.logfile}')


def insert_into_qcinput(inp_file, s, pattern, first_only=False):
    inp_file = inp_file[:]
    acc = 1
    for i, line in enumerate(inp_file[:]):
        if line.startswith(pattern):
            inp_file.insert(i + acc, s)
            if first_only:
                break
            acc += 1
    return inp_file
