#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os
import re

from ard_gsm.qchem import QChem
from ard_gsm.util import iter_sub_dirs, read_xyz_file


def main():
    args = parse_args()
    num_regex = re.compile(r'\d+')

    for gsm_sub_dir in iter_sub_dirs(args.gsm_dir):
        out_dir = os.path.join(args.out_dir, os.path.basename(gsm_sub_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        elif not args.overwrite:
            continue

        print('Extracting from {}...'.format(gsm_sub_dir))
        for gsm_log in glob.iglob(os.path.join(gsm_sub_dir, 'gsm*.out')):
            num = int(num_regex.search(os.path.basename(gsm_log)).group(0))
            string_file = os.path.join(gsm_sub_dir, 'stringfile.xyz{:04}'.format(num))

            if is_successful(gsm_log):
                # Optimize van-der-Waals wells instead of separated products
                xyzs = read_xyz_file(string_file)
                path = os.path.join(out_dir, 'prod_optfreq{:04}.in'.format(num))
                q = QChem(config_file=args.config)
                q.make_input_from_coords(path, *xyzs[-1])


def is_successful(gsm_log):
    """
    Success is defined as having converged to a transition state.
    """
    with open(gsm_log) as f:
        for line in reversed(f.readlines()):
            if '-XTS-' in line or '-TS-' in line:
                return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gsm_dir', metavar='GSMDIR', help='Path to directory containing GSM folders')
    parser.add_argument('out_dir', metavar='ODIR', help='Path to output directory')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite input files in existing directories')
    parser.add_argument(
        '--config', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.opt_freq'),
        help='Configuration file for product optfreq jobs in Q-Chem'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
