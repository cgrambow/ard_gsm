#!/usr/bin/env python
# -*- coding:utf-8 -*-

from collections import OrderedDict

import numpy as np

from ard_gsm.qchem import QChem, QChemError
from ard_gsm.mol import MolGraph, SanitizationError
from ard_gsm.reaction import Reaction, group_reactions_by_products, group_reactions_by_connection_changes
from ard_gsm.util import write_xyz_file


def parse_reaction(reactant, prod_file, ts_file,
                   keep_isomorphic=False, edist_max=5.0, gdist_max=1.0, normal_mode_check=False, soft_check=False):
    """
    Parse reaction given the reactant MolGraph, the product log file,
    and the TS log file. Return None if there was an error in the TS
    job, the job was not valid, or other criteria are not met.
    """
    product = qchem2molgraph(prod_file, return_none_on_err=True, freq_only=True, print_msg=False)
    if product is None:
        print('Ignored {} because of negative frequency or error'.format(prod_file))
        return None

    if not keep_isomorphic and reactant.is_isomorphic(product):
        print('Ignored {} because product is isomorphic with reactant'.format(prod_file))
        return None

    ts_qts = qchem2molgraph(ts_file, return_qobj=True, return_none_on_err=True,
                            edist_max=edist_max, gdist_max=gdist_max, ts=True)
    if ts_qts is None:
        return None
    ts, qts = ts_qts

    rxn = Reaction(reactant, product, ts, product_file=prod_file, ts_file=ts_file)

    # Negative barriers shouldn't occur because we're calculating them
    # based on reactant/product wells, but check this just in case
    if ts.energy - reactant.energy < 0.0:
        print('Ignored {} because of negative barrier'.format(ts_file))
        return None
    elif ts.energy - product.energy < 0.0:
        print('Ignored {} because of negative reverse barrier'.format(ts_file))
        return None

    if normal_mode_check:
        normal_mode = qts.get_normal_modes()[0]  # First one corresponds to imaginary frequency
        if not rxn.normal_mode_analysis(normal_mode, soft_check=soft_check):
            print('Ignored {} because of failed normal mode analysis'.format(ts_file))
            return None

    return rxn


def remove_duplicates(reactions, ndup=1, group_by_connection_changes=False, atommap=True, set_smiles=True):
    """
    Group all reactions and remove all duplicates in each group keeping
    only at most ndup identical reactions. Also set the product SMILES.
    """
    extracted_reactions = OrderedDict()
    extracted_smiles = {}

    # Group duplicate reactions
    if group_by_connection_changes:
        reaction_groups = group_reactions_by_connection_changes(reactions)
    else:
        reaction_groups = group_reactions_by_products(reactions)

    # Extract the ndup lowest barrier reactions from each group
    for group in reaction_groups:
        barriers = [(num, rxn.ts.energy - rxn.reactant.energy) for num, rxn in group.iteritems()]
        barriers.sort(key=lambda x: x[1])

        nextracted = 0
        for num, _ in barriers:
            rxn = group[num]
            if set_smiles:
                try:
                    rxn.product_smiles = rxn.product.perceive_smiles(atommap=atommap)
                except SanitizationError:
                    print('Ignored number {} because Smiles perception failed'.format(num))
                    continue

            extracted_reactions[num] = rxn
            nextracted += 1

            if nextracted >= ndup:
                break

    return extracted_reactions


def qchem2molgraph(logfile, return_qobj=False, return_none_on_err=False, **kwargs):
    """
    Convert a Q-Chem logfile to a MolGraph object. Return the QChem
    object in addition to the MolGraph if return_qobj is True. Catch
    QChemError if return_none_on_err is True and return None. Options
    in kwargs are passed to valid_job. If the job is not valid, also
    return None.
    """
    try:
        q = QChem(logfile=logfile)
    except QChemError as e:
        if return_none_on_err:
            print(e)
            return None
        else:
            raise

    if not valid_job(q, **kwargs):
        return None

    energy = q.get_energy() + q.get_zpe()  # With ZPE
    symbols, coords = q.get_geometry()
    mol = MolGraph(symbols=symbols, coords=coords, energy=energy)
    mol.infer_connections()

    if return_qobj:
        return mol, q
    else:
        return mol


def valid_job(q, edist_max=None, gdist_max=None, ts=False, freq_only=False, print_msg=True):
    freqs = q.get_frequencies()
    nnegfreq = sum(1 for freq in freqs if freq < 0.0)

    if (ts and nnegfreq != 1) or (not ts and nnegfreq != 0):
        if print_msg:
            print('{}: {} negative frequencies!'.format(q.logfile, nnegfreq))
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
            print('{}: Large energy change of {:.2f} kcal/mol!'.format(q.logfile, edist))
        return False
    if gdist > gdist_max:
        if print_msg:
            print('{}: Large geometry change of {:.2f} Angstrom!'.format(q.logfile, gdist))
        return False

    return True


def rxn2xyzfile(rxn, path):
    symbols = (mol.get_symbols() for mol in (rxn.reactant, rxn.ts, rxn.product))
    coords = (mol.get_coords() for mol in (rxn.reactant, rxn.ts, rxn.product))
    comments = [rxn.reactant_smiles, '', rxn.product_smiles]
    write_xyz_file(path, symbols, coords, comments=comments)
