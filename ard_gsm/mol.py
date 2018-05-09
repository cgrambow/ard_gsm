#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

import numpy as np


class MolData(object):
    """
    Class to parse molecular data from the QM9 dataset.
    """
    def __init__(self, path=None):
        # Geometries and energies at B3LYP/6-31G(2df,p) level of
        # theory if parsed directly from QM9 dataset

        # Structure, partial charges, and frequencies
        self.natoms = None            # Row 0
        self.elements = None          # Row 2,...,natoms+1
        self.coords = None            # Row 2,...,natoms+1 (Angstrom)
        self.mulliken_charges = None  # Row 2,...,natoms+1 (e)
        self.freqs = None             # Row natoms+2 (cm^-1)
        self.smiles = None            # Row natoms+3  As parsed from the optimized geo
        self.smiles2 = None           # Row natoms+3  Corresponding to input to B3LYP opt
        self.inchi = None             # Row natoms+4

        # Properties line (row 1)
        self.tag = None                       # Col 0
        self.index = None                     # Col 1
        self.rotational_consts = None         # Col 2-4 (GHz)
        self.dipole_mom = None                # Col 5 (Debye)
        self.isotropic_polarizability = None  # Col 6 (Bohr^3)
        self.homo = None                      # Col 7 (Ha)
        self.lumo = None                      # Col 8 (Ha)
        self.gap = None                       # Col 9 (Ha)
        self.electronic_extent = None         # Col 10 (Bohr^2)
        self.zpe = None                       # Col 11 (Ha)
        self.u0 = None                        # Col 12 (Ha)
        self.u298 = None                      # Col 13 (Ha)
        self.h298 = None                      # Col 14 (Ha)
        self.g298 = None                      # Col 15 (Ha)
        self.cv298 = None                     # Col 16 (Ha)

        # Other attributes
        self.file_name = None
        self.model_chemistry = None
        self.e0 = None  # (Ha)
        self.hf298 = None  # J/mol

        # Get structure
        if path is not None:
            self.parse_data(path)

    def copy(self):
        s = MolData()
        for attr, val in self.__dict__.iteritems():
            setattr(s, attr, val)
        return s

    def __str__(self):
        return '\n'.join('{}: {}'.format(k, v) for k, v in sorted(self.__dict__.iteritems()))

    def parse_data(self, path):
        """
        Read and parse an extended xyz file from the 134k dataset.
        """
        with open(path, 'r') as f:
            lines = f.readlines()

        self.natoms = int(lines[0].strip())
        props = lines[1].strip().split()
        xyz = np.array([line.split() for line in lines[2:(self.natoms+2)]])
        self.elements = list(xyz[:,0])

        try:
            self.coords = xyz[:,1:4].astype(np.float)
        except ValueError as e:
            if '*^' in str(e):  # Handle powers of 10
                coords_str = xyz[:,1:4]
                coords_flat = np.array([float(s.replace('*^', 'e')) for s in coords_str.flatten()])
                self.coords = coords_flat.reshape(coords_str.shape)
            else:
                raise

        try:
            self.mulliken_charges = xyz[:,4].astype(np.float)
        except ValueError as e:
            if '*^' in str(e):  # Handle powers of 10
                self.mulliken_charges = np.array([float(s.replace('*^', 'e')) for s in xyz[:,4]])
            else:
                raise

        self.freqs = np.array([float(col) for col in lines[self.natoms+2].split()])
        smiles = lines[self.natoms+3].split()
        self.smiles = smiles[1]  # This is the one corresponding to the relaxed B3LYP geometry
        self.smiles2 = smiles[0]
        self.inchi = lines[self.natoms+4].split()[1]

        self.tag = props[0]
        self.index = int(props[1])
        self.rotational_consts = np.array([float(c) for c in props[2:5]])
        self.dipole_mom = float(props[5])
        self.isotropic_polarizability = float(props[6])
        self.homo = float(props[7])
        self.lumo = float(props[8])
        self.gap = float(props[9])
        self.electronic_extent = float(props[10])
        self.zpe = float(props[11])
        self.u0 = float(props[12])
        self.u298 = float(props[13])
        self.h298 = float(props[14])
        self.g298 = float(props[15])
        self.cv298 = float(props[16])

        self.file_name = os.path.basename(path)
        self.model_chemistry = 'b3lyp/6-31g(2df,p)'
        self.e0 = self.u0 - self.zpe

    def contains_element(self, element):
        if element.lower() in {e.lower() for e in set(self.elements)}:
            return True
        else:
            return False
