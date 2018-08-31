#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division

import numpy as np

from ard_gsm.util import get_dist_vecs


def group_reactions_by_products(reactions):
    """
    Given a dictionary of reactions, group the identical ones based on product
    identities and return a list of dictionaries, where each dictionary
    contains the reactions in that group.

    Note: Assumes that reactants and products already have connections assigned.
    """
    groups = []
    for num, rxn in reactions.iteritems():
        product = rxn[-1]
        for group in groups:
            product_in_group = group[list(group)[0]][-1]  # Product from "random" reaction in group
            if product.is_isomorphic(product_in_group):
                group[num] = rxn
                break
        else:
            groups.append({num: rxn})
    return groups


def group_reactions_by_connection_changes(reactions):
    """
    Given a dictionary of reactions, group the identical ones based on
    connection changes and return a list of dictionaries, where each dictionary
    contains the reactions in that group.

    Note: Assumes that reactants and products already have connections assigned.
    """
    connection_changes = {num: get_connection_changes(rxn[0], rxn[-1]) for num, rxn in reactions.iteritems()}
    groups = []
    for num, rxn in reactions.iteritems():
        for group in groups:
            if connection_changes[num] == connection_changes[list(group)[0]]:  # list(group)[0] is "first" key
                group[num] = rxn
                break
        else:
            groups.append({num: rxn})
    return groups


def get_connection_changes(mg1, mg2):
    """
    Get the connection changes given two molecular graphs. They are returned as
    two sets of tuples containing the connections involved in breaking and
    forming, respectively.
    """
    connections1 = mg1.get_all_connections()
    connections2 = mg2.get_all_connections()
    break_connections, form_connections = set(), set()
    for connection in connections1:
        if connection not in connections2:
            break_connections.add(connection)
    for connection in connections2:
        if connection not in connections1:
            form_connections.add(connection)
    return break_connections, form_connections


def normal_mode_analysis(reactant, product, ts, normal_mode):
    """
    Check if the TS is correct by identifying which bonds change in the
    reaction and checking if the bond length contributions of those
    bonds as obtained from the normal mode corresponding to the
    imaginary frequency of the TS are larger than those of other bonds.
    Requires that connections have been inferred for reactant, product,
    and TS. The normal mode should be provided as an array of Cartesian
    displacements.

    Note: IRC and normal mode analysis are actually quite likely to
    disagree because it can be very difficult to tell where the exact
    endpoints of a reaction are. Therefore, this function should be used
    with caution.
    """
    natoms = len(ts.atoms)
    dist_vecs = get_dist_vecs(ts.get_coords())

    # Project the normal mode displacements onto the distance vector and
    # take the norm of their difference to get the magnitude of the bond
    # length contribution.
    bond_variations = np.zeros((natoms, natoms))
    for i in range(natoms):
        for j in range(i+1, natoms):
            dvec = dist_vecs[:, i, j]
            d = np.dot(dvec, dvec)
            bond_variations[i, j] = abs(np.dot(normal_mode[i]-normal_mode[j], dvec)) / np.sqrt(d)
    bond_variations = np.maximum(bond_variations, bond_variations.T)  # Symmetrize

    broken, formed = get_connection_changes(reactant, product)
    changed = broken | formed

    # The Connection objects in `changed` do not use the Atom objects in
    # `ts`, so just extract the indices.
    changed_inds = {(connection.atom1.idx, connection.atom2.idx) for connection in changed}
    changed_bond_variations = [bond_variations[idx1-1, idx2-1] for idx1, idx2 in changed_inds]  # Atom inds start at 1

    for connection in ts.get_all_connections():
        idx1 = connection.atom1.idx
        idx2 = connection.atom2.idx
        if not (idx1, idx2) in changed_inds:
            bond_variation = bond_variations[idx1-1, idx2-1]
            if any(bond_variation > v for v in changed_bond_variations):
                return False

    return True
