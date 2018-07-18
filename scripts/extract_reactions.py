#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division

import argparse
import csv
import glob
import os
import re
import shutil

from ard_gsm.mol import MolGraph
from ard_gsm.util import iter_sub_dirs, read_xyz_file


def main():
    args = parse_args()

    pdir = args.out_dir
    if not os.path.exists(pdir):
        os.makedirs(pdir)
    num_regex = re.compile(r'\d+')
    stats_file = open(os.path.join(pdir, 'statistics.csv'), 'w')
    stats_writer = csv.writer(stats_file, quoting=csv.QUOTE_NONNUMERIC)
    stats_writer.writerow(
        ['ard_dir', 'ntotal',
         'nsuccess', 'fsuccess',
         'nunique', 'funique_success', 'funique_total',
         'nduplicate', 'fduplicate']
    )

    for gsm_sub_dir in iter_sub_dirs(args.gsm_dir):
        print('Extracting from {}...'.format(gsm_sub_dir))

        reactions = {}
        string_files = {}
        ntotal = 0
        for gsm_log in glob.iglob(os.path.join(gsm_sub_dir, 'gsm*.out')):
            ntotal += 1
            num = int(num_regex.search(os.path.basename(gsm_log)).group(0))
            string_file = os.path.join(gsm_sub_dir, 'stringfile.xyz{:04}'.format(num))
            string_files[num] = string_file

            if is_successful(gsm_log):
                xyzs = read_xyz_file(string_file, with_energy=True)
                string = [MolGraph(symbols=xyz[0], coords=xyz[1], energy=xyz[2]) for xyz in xyzs]
                reactant = string[0]
                product = string[-1]
                reactant.infer_connections()
                product.infer_connections()
                reactions[num] = string

        if args.group_by_connection_changes:
            reaction_groups = group_reactions_by_connection_changes(reactions)
        else:
            reaction_groups = group_reactions_by_products(reactions)

        # Statistics to do not take into consideration bus errors!
        nsuccess = len(reactions)
        if ntotal == 0 or nsuccess == 0:
            continue
        fsuccess = nsuccess / ntotal
        nunique = len(reaction_groups)
        funique_success = nunique / nsuccess
        funique_total = nunique / ntotal
        nduplicate = nsuccess - nunique
        fduplicate = nduplicate / nsuccess
        stats_writer.writerow(
            [os.path.basename(gsm_sub_dir), ntotal,
             nsuccess, fsuccess,
             nunique, funique_success, funique_total,
             nduplicate, fduplicate]
        )

        out_dir = os.path.join(pdir, os.path.basename(gsm_sub_dir))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        for group in reaction_groups:
            barriers = []
            for num, rxn in group.iteritems():
                reactant_energy = rxn[0].energy
                ts_energy = max(mol.energy for mol in rxn)
                barriers.append((num, ts_energy-reactant_energy))
            barriers.sort(key=lambda x: x[1])
            extracted_nums = [num for num, _ in barriers[:args.nextract]]
            for num in extracted_nums:
                shutil.copy(string_files[num], out_dir)

    stats_file.close()


def is_successful(gsm_log):
    """
    Success is defined as having converged to a transition state.
    """
    with open(gsm_log) as f:
        for line in reversed(f.readlines()):
            if '-XTS-' in line or '-TS-' in line:
                return True
    return False


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


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gsm_dir', metavar='GSMDIR', help='Path to directory containing GSM folders')
    parser.add_argument('out_dir', metavar='ODIR', help='Path to output directory')
    parser.add_argument('-n', '--nextract', type=int, default=4, metavar='N',
                        help='Number of duplicate reactions of the same type to extract (sorted by lowest barrier)')
    parser.add_argument('--group_by_connection_changes', action='store_true',
                        help='Use connection changes instead of product identities to distinguish reactions')
    return parser.parse_args()


if __name__ == '__main__':
    main()
