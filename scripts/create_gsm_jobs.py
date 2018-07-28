#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os
import re
import shutil

from ard_gsm.qchem import QChem, QChemError
from ard_gsm.mol import MolGraph
from ard_gsm.driving_coords import generate_driving_coords
from config.limits import connection_limits


def main():
    args = parse_args()
    config_qchem_start = args.config_qchem
    config_qchem_end = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.gsm.end')
    config_gsm = args.config_gsm
    config_gscreate = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'gsm.gscreate')

    pdir = args.out_dir
    if not os.path.exists(pdir):
        os.makedirs(pdir)
    num_regex = re.compile(r'\d+')

    with open(os.path.join(pdir, 'params.log'), 'w') as f:
        f.write('Connection limits:\n')
        for symbol in connection_limits:
            ll = connection_limits[symbol][0]
            ul = connection_limits[symbol][1]
            f.write('  {}: {}, {}\n'.format(symbol, ll, ul))
        f.write('maxbreak = {}\n'.format(args.maxbreak))
        f.write('maxform = {}\n'.format(args.maxform))
        f.write('maxchange = {}\n'.format(args.maxchange))
        f.write('single_change = {}\n'.format(not args.ignore_single_change))
        f.write('equiv_Hs = {}\n'.format(args.equiv_Hs))
        f.write('minbreak = {}\n'.format(args.minbreak))
        f.write('minform = {}\n'.format(args.minform))
        f.write('minchange = {}\n'.format(args.minchange))

    for log_idx, logfile in enumerate(glob.iglob(os.path.join(args.qlog_dir, '*.log'))):
        try:
            log = QChem(logfile=logfile)
        except QChemError as e:
            print(e)
            continue

        # Check frequencies
        try:
            freqs = log.get_frequencies()
        except QChemError as e:
            if 'not found' in str(e):
                print('Warning: Frequencies could not be found in {}'.format(logfile))
            else:
                raise
        else:
            if any(freq < 0.0 for freq in freqs):
                raise Exception('Negative frequency in {}! Not optimized'.format(logfile))

        symbols, coords = log.get_geometry()
        mol = MolGraph(symbols=symbols, coords=coords)
        mol.infer_connections()

        print('Making driving coordinates from {}'.format(logfile))
        driving_coords_set = generate_driving_coords(
            mol,
            maxbreak=args.maxbreak,
            maxform=args.maxform,
            maxchange=args.maxchange,
            single_change=not args.ignore_single_change,
            equiv_Hs=args.equiv_Hs,
            minbreak=args.minbreak,
            minform=args.minform,
            minchange=args.minchange
        )

        try:
            num = int(num_regex.search(os.path.basename(logfile)).group(0))
        except AttributeError:
            # Couldn't find number in filename
            num = log_idx

        out_dir = os.path.join(pdir, 'gsm{}'.format(num))
        scr_dir = os.path.join(out_dir, 'scratch')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(scr_dir):
            os.mkdir(scr_dir)

        shutil.copy(config_qchem_start, os.path.join(out_dir, 'qstart'))
        shutil.copy(config_qchem_end, os.path.join(out_dir, 'qend'))
        shutil.copy(config_gsm, os.path.join(out_dir, 'inpfileq'))
        shutil.copy(config_gscreate, os.path.join(out_dir, 'gscreate'))

        for idx, driving_coords in enumerate(driving_coords_set):
            isomers_file = os.path.join(scr_dir, 'ISOMERS{:04}'.format(idx))
            initial_file = os.path.join(scr_dir, 'initial{:04}.xyz'.format(idx))
            with open(isomers_file, 'w') as f:
                f.write(str(driving_coords))
            with open(initial_file, 'w') as f:
                f.write(str(len(symbols)) + '\n')
                f.write('\n')
                for symbol, xyz in zip(symbols, coords):
                    f.write('{0}  {1[0]: .10f}  {1[1]: .10f}  {1[2]: .10f}\n'.format(symbol, xyz))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('qlog_dir', metavar='QDIR', help='Path to directory containing geometry optimization outputs')
    parser.add_argument('out_dir', metavar='ODIR', help='Path to output directory')
    parser.add_argument('--maxbreak', type=int, default=3, metavar='B', help='Maximum number of connections to break')
    parser.add_argument('--maxform', type=int, default=3, metavar='F', help='Maximum number of connections to form')
    parser.add_argument('--maxchange', type=int, default=5, metavar='C', help='Maximum number of connections to change')
    parser.add_argument('--ignore_single_change', action='store_true',
                        help='Do not consider single connection changes (e.g., nbreak=1, nform=0)')
    parser.add_argument('--consider_equivalent_hydrogens', action='store_true', dest='equiv_Hs',
                        help='Create equivalent driving coordinates for the same reaction with different but '
                             'equivalent hydrogens, i.e., hydrogens attached to non-cyclic tetrahedral carbons')
    parser.add_argument('--minbreak', type=int, default=0, metavar='B', help='Minimum number of connections to break')
    parser.add_argument('--minform', type=int, default=0, metavar='F', help='Minimum number of connections to form')
    parser.add_argument('--minchange', type=int, default=1, metavar='F', help='Minimum number of connections to change')
    parser.add_argument(
        '--config_qchem', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.gsm.start'),
        help='Configuration file for Q-Chem calls in GSM'
    )
    parser.add_argument(
        '--config_gsm', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'gsm.inpfileq'),
        help='Settings for GSM calculations'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
