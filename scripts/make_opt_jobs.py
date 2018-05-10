#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import random

from ard_gsm.qchem import QChem
from ard_gsm.util import pickle_load


def main():
    args = parse_args()

    print('Loading data...')
    data = pickle_load(args.mol_data, compressed=True)

    if args.max_heavy > 0:
        data = [mol_data for mol_data in data if sum(1 for s in mol_data.elements if s != 'H') <= args.max_heavy]
    if args.random:
        random.shuffle(data)
    if args.num > 0:
        data = data[:args.num]

    print('Generating geometries and searching conformers...')
    mols = [mol_data.to_rdkit(nconf=args.nconf) for mol_data in data]
    names = [mol_data.file_name for mol_data in data]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print('Making input files...')
    for i, mol in enumerate(mols):
        q = QChem(mol, config_file=args.config)
        q.make_input(os.path.join(args.out_dir, 'molopt{}.in'.format(i)))

    with open(os.path.join(args.out_dir, 'names.txt'), 'w') as f:
        for i, name in enumerate(names):
            f.write('{}: {}\n'.format(i, name))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mol_data', metavar='FILE', help='Path to pickled and zipped list of MolData objects')
    parser.add_argument('out_dir', metavar='DIR', help='Path to output directory')
    parser.add_argument('--max_heavy', type=int, default=-1, metavar='H', help='Maximum number of heavy atoms')
    parser.add_argument('--num', type=int, default=-1, metavar='N', help='Number of molecules to choose from mol_data')
    parser.add_argument('--not_random', action='store_false', dest='random',
                        help='Select molecules in order instead of randomly')
    parser.add_argument('--nconf', type=int, default=100, metavar='C',
                        help='Number of conformers to generate for lowest-energy conformer search')
    parser.add_argument(
        '--config', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.opt_freq'),
        help='Configuration file for Q-Chem input files.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
