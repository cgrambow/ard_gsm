#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division

import argparse
import glob
import os
import re

import numpy as np

from ard_gsm.qchem import QChem, QChemError
from ard_gsm.mol import MolGraph
from ard_gsm.reaction import group_reactions_by_products, group_reactions_by_connection_changes, normal_mode_analysis
from ard_gsm.util import iter_sub_dirs


def main():
    args = parse_args()
    num_regex = re.compile(r'\d+')
    out_file = open(args.out_file, 'w')

    for ts_sub_dir in iter_sub_dirs(args.ts_dir):
        sub_dir_name = os.path.basename(ts_sub_dir)
        print('Extracting from {}...'.format(sub_dir_name))
        reactant_num = int(num_regex.search(sub_dir_name).group(0))
        reactant_file = os.path.join(args.reac_dir, 'molopt{}.log'.format(reactant_num))
        qr = QChem(logfile=reactant_file)

        # Check reactant frequencies
        if not valid_job(qr, freq_only=True, print_msg=False):
            raise Exception('Negative frequency for reactant in {}!'.format(reactant_file))

        reactant_energy = qr.get_energy() + qr.get_zpe()  # With ZPE
        reactant_symbols, reactant_coords = qr.get_geometry()
        reactant = MolGraph(symbols=reactant_symbols, coords=reactant_coords, energy=reactant_energy)
        reactant.infer_connections()  # Need this for reaction grouping later
        reactant_smiles = reactant.perceive_smiles()

        reactions = {}
        for ts_file in glob.iglob(os.path.join(ts_sub_dir, 'ts_optfreq*.out')):
            num = int(num_regex.search(os.path.basename(ts_file)).group(0))
            prod_file = os.path.join(args.prod_dir, sub_dir_name, 'prod_optfreq{:04}.out'.format(num))
            qp = QChem(logfile=prod_file)

            # Shouldn't have to check freqs, but do it just in case
            if not valid_job(qp, freq_only=True, print_msg=False):
                raise Exception('Negative frequency for product in {}!'.format(prod_file))

            product_energy = qp.get_energy() + qp.get_zpe()
            product_symbols, product_coords = qp.get_geometry()
            product = MolGraph(symbols=product_symbols, coords=product_coords, energy=product_energy)
            product.infer_connections()

            if not args.keep_isomorphic_reactions and reactant.is_isomorphic(product):
                print('Ignored {} because product is isomorphic with reactant'.format(prod_file))
                continue

            try:
                qts = QChem(logfile=ts_file)
            except QChemError as e:
                print(e)
                continue
            if not valid_job(qts, args.edist, args.gdist, ts=True):
                continue

            ts_energy = qts.get_energy() + qts.get_zpe()
            ts_symbols, ts_coords = qts.get_geometry()
            ts = MolGraph(symbols=ts_symbols, coords=ts_coords, energy=ts_energy)

            if args.check_normal_mode:
                ts.infer_connections()
                normal_mode = qts.get_normal_modes()[0]  # First one corresponds to imaginary frequency
                if not normal_mode_analysis(reactant, product, ts, normal_mode):
                    print('Ignored {} because of failed normal mode analysis'.format(ts_file))
                    continue

            reactions[num] = [reactant, ts, product]

        # Group duplicate reactions
        if args.group_by_connection_changes:
            reaction_groups = group_reactions_by_connection_changes(reactions)
        else:
            reaction_groups = group_reactions_by_products(reactions)

        # Extract the lowest barrier reaction from each group
        for group in reaction_groups:
            barriers = [(num, ts.energy - r.energy) for num, (r, ts, _) in group.iteritems()]
            extracted_num, barrier = min(barriers, key=lambda x: x[1])
            if barrier < 0.0:
                print('WARNING: Barrier for reaction {} in {} is negative!'.format(extracted_num, sub_dir_name))

            _, ts, product = group[extracted_num]
            product_smiles = product.perceive_smiles()
            barrier *= 627.5095  # Hartree to kcal/mol
            out_file.write('{}   {}   {}\n'.format(reactant_smiles, product_smiles, barrier))

            if args.include_reverse:
                # For reverse reactions, it's technically possible that some of
                # them are the same as already extracted reactions in a different
                # sub dir, but it's unlikely
                reverse_barrier = (ts.energy - product.energy) * 627.5095
                if reverse_barrier < 0.0:
                    print('WARNING: Barrier for reverse of reaction'
                          ' {} in {} is negative!'.format(extracted_num, sub_dir_name))
                out_file.write('{}   {}   {}\n'.format(product_smiles, reactant_smiles, reverse_barrier))

    out_file.close()


def valid_job(q, edist_max=None, gdist_max=None, ts=False, freq_only=False, print_msg=True):
    freqs = q.get_frequencies()
    nnegfreq = sum(1 for freq in freqs if freq < 0.0)

    if (ts and nnegfreq != 1) or (not ts and nnegfreq != 0):
        if print_msg:
            print('Ignored {} because of {} negative frequencies'.format(q.logfile, nnegfreq))
        return False

    if freq_only:
        return True

    assert edist_max is not None and gdist_max is not None

    edist = abs(q.get_energy() - q.get_energy(first=True)) * 627.5095
    geo = q.get_geometry()[1]
    gdiff = geo.flatten() - q.get_geometry(first=True)[1].flatten()
    gdist = np.sqrt(np.dot(gdiff, gdiff) / len(geo))

    if edist > edist_max:
        if print_msg:
            print('Ignored {} because of large energy change of {:.2f} kcal/mol'.format(q.logfile, edist))
        return False
    if gdist > gdist_max:
        if print_msg:
            print('Ignored {} because of large geometry change of {:.2f} Angstrom'.format(q.logfile, gdist))
        return False

    return True


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('reac_dir', help='Path to directory containing optimized reactant structures')
    parser.add_argument('prod_dir', help='Path to directory containing optimized product structures')
    parser.add_argument('ts_dir', help='Path to directory containing optimized TS structures')
    parser.add_argument('out_file', help='Path to output file')
    parser.add_argument('--include_reverse', action='store_true', help='Also extract reverse reactions')
    parser.add_argument('--edist', type=float, default=5.0,
                        help='Ignore TS files with energy differences (kcal/mol) larger than this')
    parser.add_argument('--gdist', type=float, default=1.0,
                        help='Ignore TS files with Cartesian RMSD (Angstrom) between first and last geometries'
                             ' larger than this')
    parser.add_argument('--check_normal_mode', action='store_true',
                        help='Perform a normal mode analysis to identify if the TS structure is correct (make sure to'
                             ' check the warnings in the normal_mode_analysis function before using this option')
    parser.add_argument('--group_by_connection_changes', action='store_true',
                        help='Use connection changes instead of product identities to distinguish reactions')
    parser.add_argument('--keep_isomorphic_reactions', action='store_true',
                        help='Consider reactions where the product is isomorphic with the reactant')
    return parser.parse_args()


if __name__ == '__main__':
    main()
