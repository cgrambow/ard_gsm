#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os
import re

from ard_gsm.mol import MolGraph
from ard_gsm.qchem import QChem, QChemError
from ard_gsm.reaction import group_reactions_by_products, group_reactions_by_connection_changes
from ard_gsm.util import iter_sub_dirs, read_xyz_file


def main():
    args = parse_args()
    num_regex = re.compile(r'\d+')

    for prod_sub_dir in iter_sub_dirs(args.prod_dir):
        sub_dir_name = os.path.basename(prod_sub_dir)

        out_dir = os.path.join(args.out_dir, sub_dir_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        elif not args.overwrite:
            continue

        print('Extracting from {}...'.format(sub_dir_name))

        reactions = {}
        for prod_file in glob.iglob(os.path.join(prod_sub_dir, 'prod_optfreq*.out')):
            num = int(num_regex.search(os.path.basename(prod_file)).group(0))
            string_file = os.path.join(args.gsm_dir, sub_dir_name, 'stringfile.xyz{:04}'.format(num))

            try:
                qp = QChem(logfile=prod_file)
            except QChemError as e:
                print(e)
                continue
            if any(freq < 0.0 for freq in qp.get_frequencies()):
                print('Ignored {} because of negative frequency'.format(prod_file))
                continue

            xyzs = read_xyz_file(string_file, with_energy=True)
            ts_xyz = max(xyzs[1:-1], key=lambda x: x[2])
            product_symbols, product_coords = qp.get_geometry()

            # Reactant and TS energies are based on string and are relative to the reactant
            reactant = MolGraph(symbols=xyzs[0][0], coords=xyzs[0][1], energy=xyzs[0][2])
            ts = MolGraph(symbols=ts_xyz[0], coords=ts_xyz[1], energy=ts_xyz[2])
            product = MolGraph(symbols=product_symbols, coords=product_coords)  # Don't bother assigning energy
            reactant.infer_connections()
            product.infer_connections()
            if not args.keep_isomorphic_reactions and reactant.is_isomorphic(product):
                print('Ignored {} because product is isomorphic with reactant'.format(prod_file))
                continue
            reactions[num] = [reactant, ts, product]

        if args.group_by_connection_changes:
            reaction_groups = group_reactions_by_connection_changes(reactions)
        else:
            reaction_groups = group_reactions_by_products(reactions)

        for group in reaction_groups:
            # Only consider TS energies instead of "barriers" b/c energies are relative to reactant
            reactions_in_group = group.items()  # Make list
            reactions_in_group.sort(key=lambda r: r[1][1].energy)  # Index 1 is TS

            for num, rxn in reactions_in_group[:args.nextract]:
                ts = rxn[1]
                path = os.path.join(out_dir, 'ts_optfreq{:04}.in'.format(num))
                qts = QChem(config_file=args.config)
                qts.make_input_from_coords(path, ts.get_symbols(), ts.get_coords())


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gsm_dir', metavar='GSMDIR', help='Path to directory containing GSM folders')
    parser.add_argument('prod_dir', metavar='PDIR', help='Path to directory containing folders with product opts')
    parser.add_argument('out_dir', metavar='ODIR', help='Path to output directory')
    parser.add_argument('-n', '--nextract', type=int, default=4, metavar='N',
                        help='Number of duplicate reactions of the same type to extract (sorted by lowest barrier)')
    parser.add_argument('--group_by_connection_changes', action='store_true',
                        help='Use connection changes instead of product identities to distinguish reactions')
    parser.add_argument('--keep_isomorphic_reactions', action='store_true',
                        help='Consider reactions where the product is isomorphic with the reactant')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite input files in existing directories')
    parser.add_argument(
        '--config', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.ts_opt_freq'),
        help='Configuration file for TS optfreq jobs in Q-Chem'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
