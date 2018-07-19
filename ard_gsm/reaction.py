#!/usr/bin/env python
# -*- coding:utf-8 -*-


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
