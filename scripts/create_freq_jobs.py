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

    pdir = args.out_dir
    if not os.path.exists(pdir):
        os.makedirs(pdir)
    num_regex = re.compile(r'\d+')

    for gsm_sub_dir in iter_sub_dirs(args.gsm_dir):
        out_dir = os.path.join(pdir, os.path.basename(gsm_sub_dir))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        for tsfile in glob.iglob(os.path.join(gsm_sub_dir, 'scratch', 'tsq*.xyz')):
            num = int(num_regex.search(os.path.basename(tsfile)).group(0))
            gsm_log = os.path.join(gsm_sub_dir, 'gsm{}.out'.format(num))

            if is_successful(gsm_log, tsfile):
                q = QChem(config_file=args.config)
                symbols, coords = read_xyz_file(tsfile)[0]

                freq_path = os.path.join(out_dir, 'freq{:04}.in'.format(num))
                q.make_input_from_coords(freq_path, symbols, coords)


def is_successful(gsm_log, ts_file):
    """
    Success is defined as having a tightly converged transition state, "-XTS-".
    """
    if os.path.exists(ts_file):
        # Just because ts_file exists, doesn't mean job was successful
        with open(gsm_log) as f:
            for line in reversed(f.readlines()):
                if '-XTS-' in line:
                    return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gsm_dir', metavar='GSMDIR', help='Path to directory containing GSM folders')
    parser.add_argument('out_dir', metavar='ODIR', help='Path to output directory')
    parser.add_argument(
        '--config', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.freq'),
        help='Configuration file for frequency jobs in Q-Chem'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
