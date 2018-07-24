#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable

_rdkit_periodic_table = GetPeriodicTable()


def smiles_to_rdkit(smi, gen_3d=True, nconf=100):
    """
    Convert smiles to RDKit molecule.
    Tries to generate the lowest-energy conformer.
    """
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)

    if gen_3d:
        cids = AllChem.EmbedMultipleConfs(mol, nconf, AllChem.ETKDG())

        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)

        energies = []
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=cid)
            ff.Minimize()
            energy = ff.CalcEnergy()
            energies.append(energy)

        energies = np.asarray(energies)
        min_energy_idx = np.argsort(energies)[0]

        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        min_conf = mol.GetConformer(cids[min_energy_idx])
        new_mol.AddConformer(min_conf, assignId=True)
        mol = new_mol

    return mol


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

    def to_rdkit(self, gen_3d=True, nconf=100):
        """
        Currently only uses SMILES to generate the 3D structure.
        Tries to generate the lowest-energy conformer.
        """
        return smiles_to_rdkit(self.smiles, gen_3d=gen_3d, nconf=nconf)


class Atom(object):
    """
    Represents an atom in a molecular graph.
    """

    def __init__(self, symbol=None, idx=None, coords=np.array([]), frozen=False):
        self.symbol = symbol
        self.idx = idx
        self.coords = coords
        self.frozen = frozen
        self.connections = {}

    def __str__(self):
        return '{}: {}'.format(self.idx, self.symbol)

    def __repr__(self):
        return '<Atom "{}">'.format(str(self))

    def copy(self):
        return Atom(
            symbol=self.symbol,
            idx=self.idx,
            coords=self.coords.copy(),
            frozen=self.frozen,
        )

    def get_cov_rad(self):
        return _rdkit_periodic_table.GetRcovalent(self.symbol)


class Connection(object):
    """
    Represents a connection in a molecular graph.

    Note: Equality and hash are only based on atom symbols and indices.
    """

    def __init__(self, atom1, atom2):
        self._atom1 = atom1
        self._atom2 = atom2
        self._make_order_invariant()

    def __str__(self):
        return '({})--({})'.format(str(self.atom1), str(self.atom2))

    def __repr__(self):
        return '<Connection "{}">'.format(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(str(self))

    def _make_order_invariant(self):
        # Ensure that atom ordering is consistent
        atoms = [self._atom1, self._atom2]
        atoms.sort(key=lambda a: a.symbol)
        if self._atom1.idx is not None or self._atom2.idx is not None:
            atoms.sort(key=lambda a: a.idx)
        self._atom1, self._atom2 = atoms

    @property
    def atom1(self):
        return self._atom1

    @property
    def atom2(self):
        return self._atom2

    @atom1.setter
    def atom1(self, val):
        self._atom1 = val
        self._make_order_invariant()

    @atom2.setter
    def atom2(self, val):
        self._atom2 = val
        self._make_order_invariant()

    def copy(self):
        return Connection(self.atom1, self.atom2)


class MolGraph(object):
    """
    Class to convert coordinates to a molecular graph
    and to generate driving coordinates.

    Note: Atom indices start at 1.
    """

    def __init__(self, atoms=None, symbols=None, coords=None, energy=None):
        self.atoms = atoms or []
        self.energy = energy

        if not self.atoms and symbols is not None:
            for idx, symbol in enumerate(symbols):
                atom = Atom(symbol=symbol, idx=idx+1)
                self.add_atom(atom)

        if coords is not None:
            self.set_coords(coords)

    def __iter__(self):
        for atom in self.atoms:
            yield atom

    def to_rmg_mol(self):
        import rmgpy.molecule.molecule as rmg_molecule

        rmg_atoms = [rmg_molecule.Atom(element=atom.symbol, coords=atom.coords) for atom in self]
        mapping = {atom: rmg_atom for atom, rmg_atom in zip(self.atoms, rmg_atoms)}
        rmg_bonds = [rmg_molecule.Bond(mapping[connection.atom1], mapping[connection.atom2])
                     for connection in self.get_all_connections()]
        rmg_mol = rmg_molecule.Molecule(atoms=rmg_atoms)
        for bond in rmg_bonds:
            rmg_mol.addBond(bond)

        return rmg_mol

    def perceive_smiles(self):
        """
        Using the geometry, perceive the corresponding SMILES
        with bond orders using Open Babel.
        """
        import pybel

        symbols = [atom.symbol for atom in self]
        coords = np.vstack([atom.coords for atom in self])
        cblock = ['{0}  {1[0]: .10f}  {1[1]: .10f}  {1[2]: .10f}'.format(s, c) for s, c in zip(symbols, coords)]
        xyz = str(len(symbols)) + '\n\n' + '\n'.join(cblock)
        mol = pybel.readstring('xyz', xyz)

        return mol.write('can').strip()

    def add_atom(self, atom):
        self.atoms.append(atom)
        atom.connections = {}
        return atom

    def add_connection(self, connection):
        if connection.atom1 not in self.atoms or connection.atom2 not in self.atoms:
            raise Exception('Cannot add connection between atoms not in the graph')
        else:
            connection.atom1.connections[connection.atom2] = connection
            connection.atom2.connections[connection.atom1] = connection
            return connection

    def get_all_connections(self):
        return {connection for atom in self.atoms for connection in atom.connections.itervalues()}

    def get_connection(self, atom1, atom2):
        if atom1 not in self.atoms or atom2 not in self.atoms:
            raise Exception('One or both of the specified atoms are not in this graph')

        try:
            return atom1.connections[atom2]
        except KeyError:
            raise Exception('The specified atoms are not connected in this graph')

    def remove_atom(self, atom):
        for atom2 in atom.connections:
            del atom2.connections[atom]
        atom.connections = {}
        self.atoms.remove(atom)

    def remove_connection(self, connection):
        if connection.atom1 not in self.atoms or connection.atom2 not in self.atoms:
            raise Exception('Cannot remove connection between atoms not in the graph')
        del connection.atom1.connections[connection.atom2]
        del connection.atom2.connections[connection.atom1]

    def copy(self, deep=False):
        other = MolGraph(energy=self.energy)
        atoms = self.atoms
        mapping = {}
        for atom in atoms:
            if deep:
                atom2 = other.add_atom(atom.copy())
                mapping[atom] = atom2
            else:
                connections = atom.connections
                other.add_atom(atom)
                atom.connections = connections
        if deep:
            for atom1 in atoms:
                for atom2 in atom1.connections:
                    connection = atom1.connections[atom2]
                    connection = connection.copy()
                    connection.atom1 = mapping[atom1]
                    connection.atom2 = mapping[atom2]
                    other.add_connection(connection)
        return other

    def merge(self, other):
        new = MolGraph()
        for atom in self.atoms:
            connections = atom.connections
            new.add_atom(atom)
            atom.connections = connections
        for atom in other.atoms:
            connections = atom.connections
            new.add_atom(atom)
            atom.connections = connections
        new.energy = self.energy + other.energy
        return new

    def split(self):
        new1 = self.copy()
        new2 = MolGraph()

        if len(self.atoms) == 0:
            return [new1]

        atoms_to_move = [self.atoms[-1]]
        idx = 0
        while idx < len(atoms_to_move):
            for atom2 in atoms_to_move[idx].connections:
                if atom2 not in atoms_to_move:
                    atoms_to_move.append(atom2)
            idx += 1

        if len(new1.atoms) == len(atoms_to_move):
            return [new1]

        for atom in atoms_to_move:
            new2.atoms.append(atom)
            new1.atoms.remove(atom)

        new = [new2]
        new.extend(new1.split())
        new.energy = None
        return new

    def sort_atoms(self):
        self.atoms.sort(key=lambda a: a.idx)

    def is_radical(self):
        """
        Determine whether or not the molecule is a radical based on the number
        of valence electrons for each atom. If the total number of valence
        electrons is odd, then it is a radical. This assumes that molecules
        with an even number of electrons are singlets. This method also assumes
        that none of the atoms are charged.
        """
        valence_electrons = {'H': 1, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 7, 'I': 7}
        symbols = [atom.symbol for atom in self]
        total_valence_electrons = sum(valence_electrons[s] for s in symbols)
        return bool(total_valence_electrons % 2)

    def is_isomorphic(self, other):
        """
        Test if self is isomorphic with other, ignoring atom indices.
        Requires RMG to do the isomorphism check.
        """
        self_rmg = self.to_rmg_mol()
        other_rmg = other.to_rmg_mol()
        return self_rmg.isIsomorphic(other_rmg)

    def set_coords(self, coords):
        """
        Set atom coordinates. Assumes coords are in same order as self.atoms.
        """
        try:
            coords = np.reshape(coords, (-1,3))
        except ValueError:
            raise Exception('Coordinates cannot be reshaped into matrix of size Nx3')
        assert len(coords) == len(self.atoms)

        for atom, xyz in zip(self.atoms, coords):
            atom.coords = xyz

    def infer_connections(self):
        """
        Delete connections and set them again based on coordinates.
        """
        atoms = self.atoms

        for atom in atoms:
            assert len(atom.coords) != 0

        for atom in atoms:
            for connection in atom.connections:
                self.remove_connection(connection)

        sorted_atoms = sorted(atoms, key=lambda a: a.coords[2])
        for i, atom1 in enumerate(sorted_atoms):
            for atom2 in sorted_atoms[(i+1):]:
                crit_dist = (atom1.get_cov_rad() + atom2.get_cov_rad() + 0.45)**2
                z_boundary = (atom1.coords[2] - atom2.coords[2])**2
                if z_boundary > 16.0:
                    break
                dist_sq = sum((atom1.coords - atom2.coords)**2)
                if dist_sq > crit_dist or dist_sq < 0.4:
                    continue
                else:
                    connection = Connection(atom1, atom2)
                    self.add_connection(connection)

    def is_atom_in_cycle(self, atom):
        return self._is_chain_in_cycle([atom])

    def _is_chain_in_cycle(self, chain):
        atom1 = chain[-1]
        for atom2 in atom1.connections:
            if atom2 is chain[0] and len(chain) > 2:
                return True
            elif atom2 not in chain:
                chain.append(atom2)
                if self._is_chain_in_cycle(chain):
                    return True
                else:
                    chain.remove(atom2)
        return False

    def label_equivalent_hydrogens(self):
        """
        Mark all equivalent hydrogens as frozen. For now, this assumes that the
        carbons they are attached to have 4 connections, which means this
        method does not yet work for radicals.
        """
        if self.is_radical():
            raise NotImplementedError('Cannot yet label equivalent hydrogens for radicals')
        for atom in self:
            if (atom.symbol.upper() == 'C'
                    and len(atom.connections) == 4
                    and not self.is_atom_in_cycle(atom)):
                first_hydrogen = True
                for atom2 in atom.connections:
                    if atom2.symbol.upper() == 'H':
                        if first_hydrogen:
                            first_hydrogen = False
                        else:
                            atom2.frozen = True
