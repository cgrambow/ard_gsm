#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os


class QChem(object):
    """
    `mol` is an RDKit molecule with Hs already added. It should already
    contain the 3D geometry of the lowest-energy conformer. The options in the
    `qchem.in` config file should be set as desired.
    """

    def __init__(self, mol, config_file=None):
        self.mol = mol  # RDKit molecule with Hs added

        if config_file is None:
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.in')
        with open(config_file) as f:
            self.config = [line.strip() for line in f]

    def make_input(self, path):
        symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        coords = self.mol.GetConformers()[0].GetPositions()

        for i, line in enumerate(self.config):
            if line.startswith('$molecule'):
                cblock = ['{0}  {1[0]: .8f}  {1[1]: .8f}  {1[2]: .8f}'.format(symbol, xyz)
                          for symbol, xyz in zip(symbols, coords)]
                self.config[(i+2):(i+2)] = cblock

        with open(path, 'w') as f:
            for line in self.config:
                f.write(line + '\n')
