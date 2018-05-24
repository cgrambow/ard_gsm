#!/usr/bin/env python
# -*- coding:utf-8 -*-

import itertools

from ard_gsm.mol import Connection
from config.limits import connection_limits


class ConnectionError(Exception):
    """
    For any invalid connection changes that occur in MolGraph.
    """
    pass


class DrivingCoords(object):
    def __init__(self, break_idxs=None, form_idxs=None):
        self._break_idxs = break_idxs or set()
        self._form_idxs = form_idxs or set()

        self.remove_duplicates()

    def __str__(self):
        s = 'NEW\n'
        for idxs in self._break_idxs:
            s += 'BREAK {0[0]} {0[1]}\n'.format(idxs)
        for idxs in self._form_idxs:
            s += 'ADD {0[0]} {0[1]}\n'.format(idxs)
        return s

    def __eq__(self, other):
        return str(self) == str(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(str(self))

    def remove_duplicates(self):
        self._break_idxs = {tuple(sorted(idxs)) for idxs in self._break_idxs}
        self._form_idxs = {tuple(sorted(idxs)) for idxs in self._form_idxs}

    def add_break_idxs(self, idxs):
        self._break_idxs.add(tuple(sorted(idxs)))

    def add_form_idxs(self, idxs):
        self._form_idxs.add(tuple(sorted(idxs)))


def generate_driving_coords(mol, maxbreak=3, maxform=3, maxchange=5, single_change=True):
    assert all(atom.idx is not None for atom in mol.atoms)
    driving_coords_set = set()

    # Enumerate all possible connections between atoms
    # and remove the ones for atoms that are already connected
    atoms = mol.atoms
    connections = mol.get_all_connections()
    all_possible_connections = [Connection(atom1, atom2)
                                for i, atom1 in enumerate(atoms)
                                for atom2 in atoms[(i+1):]]
    all_potential_new_connections = [connection for connection in all_possible_connections
                                     if connection not in connections]

    for nbreak in range(maxbreak+1):
        for nform in range(maxform+1):
            if nbreak == nform == 0:
                continue
            elif nbreak + nform > maxchange:
                continue
            elif not single_change and (nbreak + nform == 1):
                continue

            # Generate all possible combinations of connection changes
            potential_remove_connections_iter = itertools.combinations(connections, nbreak)
            potential_new_connections_iter = itertools.combinations(all_potential_new_connections, nform)
            potential_connection_changes = itertools.product(potential_remove_connections_iter,
                                                             potential_new_connections_iter)

            for connections_to_break, connections_to_form in potential_connection_changes:
                try:
                    change_connections(mol, connections_to_break, connections_to_form)
                except ConnectionError:
                    continue
                else:
                    break_idxs = [(c.atom1.idx, c.atom2.idx) for c in connections_to_break]
                    form_idxs = [(c.atom1.idx, c.atom2.idx) for c in connections_to_form]
                    driving_coords = DrivingCoords(break_idxs=break_idxs, form_idxs=form_idxs)
                    driving_coords_set.add(driving_coords)
                finally:
                    # Always restore connections for next molecule test
                    change_connections(mol, connections_to_form, connections_to_break, test_validity=False)

    return driving_coords_set


def change_connections(mol, connections_to_break, connections_to_form, test_validity=True):
    for connection in connections_to_break:
        mol.remove_connection(connection)
    for connection in connections_to_form:
        mol.add_connection(connection)

    if test_validity:
        # Only have to test the atoms involved in the changed connections
        for connection in connections_to_break:
            if not test_connection_validity(connection):
                raise ConnectionError('Breaking {} resulted in violation of connection limits'.format(connection))
        for connection in connections_to_form:
            if not test_connection_validity(connection):
                raise ConnectionError('Forming {} resulted in violation of connection limits'.format(connection))


def test_connection_validity(connection):
    atom1 = connection.atom1
    atom2 = connection.atom2
    atom1_ll, atom1_ul = connection_limits[atom1.symbol.upper()]
    atom2_ll, atom2_ul = connection_limits[atom2.symbol.upper()]
    if len(atom1.connections) < atom1_ll or len(atom1.connections) > atom1_ul:
        return False
    elif len(atom2.connections) < atom2_ll or len(atom2.connections) > atom2_ul:
        return False
    else:
        return True
