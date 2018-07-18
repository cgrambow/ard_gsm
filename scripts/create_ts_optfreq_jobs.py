#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os

from ard_gsm.qchem import QChem
from ard_gsm.util import iter_sub_dirs, read_xyz_file


def main():
    args = parse_args()

    for gsm_dir in iter_sub_dirs(args.string_dir):
        for string_file in glob.iglob(os.path.join(gsm_dir, 'stringfile.xyz*')):
            num = string_file[-4:]
            xyzs = read_xyz_file(string_file, with_energy=True)
            symbols, coords, _ = max(xyzs, key=lambda x: x[2])
            path = os.path.join(gsm_dir, 'optfreq{}.in'.format(num))
            q = QChem(config_file=args.config)
            q.make_input_from_coords(path, symbols, coords)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('string_dir', metavar='SDIR', help='Path to directory containing extracted string files')
    parser.add_argument(
        '--config', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.ts_opt_freq'),
        help='Configuration file for frequency (with opt) jobs in Q-Chem'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
