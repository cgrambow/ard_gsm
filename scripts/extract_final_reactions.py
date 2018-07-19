#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os
import re

import numpy as np

from ard_gsm.qchem import QChem, QChemError
from ard_gsm.mol import MolGraph
from ard_gsm.reaction import group_reactions_by_products, group_reactions_by_connection_changes
from ard_gsm.util import iter_sub_dirs, read_xyz_file


def main():
    args = parse_args()
    num_regex = re.compile(r'\d+')
    out_file = open(args.out_file, 'w')

    for gsm_dir in iter_sub_dirs(args.ts_dir):
        reactant_num = int(num_regex.search(os.path.basename(gsm_dir)).group(0))
        reactant_file = os.path.join(args.reactant_dir, 'molopt{}.log'.format(reactant_num))
        qr = QChem(logfile=reactant_file)

        # Check reactant frequencies
        reactant_freqs = qr.get_frequencies()
        if any(freq < 0.0 for freq in reactant_freqs):
            raise Exception('Negative frequency for reactant in {}!'.format(reactant_file))

        # Get reactant energy and SMILES
        reactant_energy = qr.get_energy() + qr.get_zpe()  # With ZPE
        reactant_symbols, reactant_coords = qr.get_geometry()
        reactant_mol = MolGraph(symbols=reactant_symbols, coords=reactant_coords, energy=reactant_energy)
        reactant_smiles = reactant_mol.perceive_smiles()

        reactions, ts_energies = {}, {}
        for ts_file in glob.iglob(os.path.join(gsm_dir, 'ts_optfreq*.out')):
            num = int(num_regex.search(os.path.basename(ts_file)).group(0))
            string_file = os.path.join(gsm_dir, 'stringfile.xyz{:04}'.format(num))

            # Check how much energy and geometry changed during TS optimization and save TS energy
            qts = QChem(logfile=ts_file)
            try:
                ts_e0 = qts.get_energy()
            except QChemError as e:
                if 'SCF failed to converge' in str(e):
                    print('Ignored {} because of SCF failure'.format(ts_file))
                    continue
                else:
                    raise
            ts_energy = ts_e0 + qts.get_zpe()  # Add ZPE

            edist = abs(ts_e0 - qts.get_energy(first=True)) * 627.5095
            ts_geo = qts.get_geometry()[1]
            gdiff = ts_geo.flatten() - qts.get_geometry(first=True)[1].flatten()
            gdist = np.sqrt(np.dot(gdiff, gdiff) / len(ts_geo))
            freqs = qts.get_frequencies()
            nnegfreq = sum(1 for freq in freqs if freq < 0.0)
            if edist > args.edist:
                print('Ignored {} because of large energy change of {:.2f} kcal/mol'.format(ts_file, edist))
                continue
            if gdist > args.gdist:
                print('Ignored {} because of large geometry change of {:.2f} Angstrom'.format(ts_file, gdist))
                continue
            if nnegfreq != 1:
                print('Ignored {} because of {} negative frequencies'.format(ts_file, nnegfreq))
                continue

            # Read string file, infer connections, and save reaction and product
            xyzs = read_xyz_file(string_file, with_energy=True)
            string = [MolGraph(symbols=xyz[0], coords=xyz[1], energy=xyz[2]) for xyz in xyzs]
            reactant = string[0]
            product = string[-1]
            reactant.infer_connections()
            product.infer_connections()

            if not args.keep_isomorphic_reactions and reactant.is_isomorphic(product):
                print('Ignored {} because product is isomorphic with reactant'.format(ts_file))
                continue

            ts_energies[num] = ts_energy
            reactions[num] = string

        # Group duplicate reactions
        if args.group_by_connection_changes:
            reaction_groups = group_reactions_by_connection_changes(reactions)
        else:
            reaction_groups = group_reactions_by_products(reactions)

        # Extract the lowest barrier reaction from each group
        for group in reaction_groups:
            ts_energies_in_group = [(num, ts_energies[num]) for num in group]
            extracted_num, ts_energy = min(ts_energies_in_group, key=lambda x: x[1])
            product = reactions[extracted_num][-1]

            # Write reaction to output file
            barrier = (ts_energy - reactant_energy) * 627.5095  # Write output in kcal/mol
            product_smiles = product.perceive_smiles()  # If product is not fully optimized, this might change
            out_file.write('{}   {}   {}\n'.format(reactant_smiles, product_smiles, barrier))

            # Write product input file(s) if desired
            if args.create_prod_jobs:
                fragments = product.split()
                for mol_idx, mol in enumerate(fragments):
                    mol.sort_atoms()
                    path = os.path.join(gsm_dir, 'prod_optfreq{:04}_{}.in'.format(extracted_num, mol_idx))
                    symbols = [atom.symbol for atom in mol]
                    coords = np.vstack([atom.coords for atom in mol])
                    multiplicity = 2 if mol.is_radical() else 1  # Assume that electrons are as paired as possible
                    q = QChem(config_file=args.config)
                    q.make_input_from_coords(path, symbols, coords, multiplicity=multiplicity)

    out_file.close()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('reactant_dir', help='Path to directory containing optimized reactant structures')
    parser.add_argument('ts_dir', help='Path to directory containing extracted string files and TS opt jobs')
    parser.add_argument('out_file', help='Path to output file')
    parser.add_argument('--edist', type=float, default=5.0,
                        help='Ignore TS files with energy differences (kcal/mol) larger than this')
    parser.add_argument('--gdist', type=float, default=1.0,
                        help='Ignore TS files with Cartesian RMSD (Angstrom) between first and last geometries'
                             ' larger than this')
    parser.add_argument('--group_by_connection_changes', action='store_true',
                        help='Use connection changes instead of product identities to distinguish reactions')
    parser.add_argument('--keep_isomorphic_reactions', action='store_true',
                        help='Consider reactions where the product is isomorphic with the reactant')
    parser.add_argument('--create_prod_jobs', action='store_true', help='Create product optfreq jobs')
    parser.add_argument(
        '--config', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.opt_freq'),
        help='Configuration file for product frequency (with opt) jobs in Q-Chem'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
