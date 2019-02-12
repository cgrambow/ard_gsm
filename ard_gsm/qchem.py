#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from six.moves import xrange

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
                        raise QChemError('Q-Chem job {} had an error!'.format(logfile))

    def make_input(self, path, charge=0, multiplicity=1):
        if isinstance(self.mol, MolGraph):
            symbols = self.mol.get_symbols()
            coords = self.mol.get_coords()
        else:
            symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
            coords = self.mol.GetConformers()[0].GetPositions()
        self.make_input_from_coords(path, symbols, coords, charge=charge, multiplicity=multiplicity)

    def make_input_from_coords(self, path, symbols, coords, charge=0, multiplicity=1):
        for i, line in enumerate(self.config):
            if line.startswith('$molecule'):
                cblock = ['{0}  {1[0]: .10f}  {1[1]: .10f}  {1[2]: .10f}'.format(symbol, xyz)
                          for symbol, xyz in zip(symbols, coords)]
                cblock.insert(0, '{} {}'.format(charge, multiplicity))
                self.config[(i+1):(i+1)] = cblock
                break  # If there are more than 1 molecule block, only fill the first one

        with open(path, 'w') as f:
            for line in self.config:
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
            raise QChemError('Energy not found in {}'.format(self.logfile))

    def get_geometry(self, first=False):
        if first:
            iterable = xrange(len(self.log))
        else:
            iterable = reversed(xrange(len(self.log)))
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
            raise QChemError('Geometry not found in {}'.format(self.logfile))

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
            raise QChemError('Frequencies not found in {}'.format(self.logfile))

    def get_normal_modes(self):
        modes = []
        for i in reversed(xrange(len(self.log))):
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
            raise QChemError('Normal modes not found in {}'.format(self.logfile))

    def get_zpe(self):
        for line in reversed(self.log):
            if 'Zero point vibrational energy' in line:
                return float(line.split()[-2]) / 627.5095  # Convert to Hartree
        else:
            raise QChemError('ZPE not found in {}'.format(self.logfile))

    def get_multiplicity(self):
        for i, line in enumerate(self.log):
            if '$molecule' in line:
                return int(self.log[i+1].strip().split()[-1])
        else:
            raise QChemError('Multiplicity not found in {}'.format(self.logfile))
