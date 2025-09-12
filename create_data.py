import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict

import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MACCSkeys
import networkx as nx

from matrixs import MolMatrices
from utils import *

import torch.nn.functional as F


# normalize the dictionary
def dic_normalize(dic):

    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

# protein dictionary
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']
res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

# target graph node features
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

# get residue features
def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# drug sequence dictionary------------------------------------------------------
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

# map the symbol of atom to label
def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

# get all drug graph node features ------------------------------------------------------------------------
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

# map x to one-hot encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# map target sequence to label--------------------------------------------------------
def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x

#-create drug sequence, drug graph, target sequence , target graph


seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}  # encode alphabet from 1
seq_dict_len = len(seq_dict)
max_seq_len = 1000



# create train data and test data
def create_dataset(dataset,fold=0):
    # ------------------------------create  CSV file-------------------------------------------
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))  # len=5
    train_fold = [ee for e in train_fold for ee in e]  # davis len=25046
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))  # davis len=5010
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)  # davis len=68
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)  # davis len=442
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')  # davis len=68
    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    drug_finger = {}
    # get smiles sequence list
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)  # loading drugs
        drug_smiles.append(ligands[d])
    # get target sequence list
    for t in proteins.keys():
        prots.append(proteins[t])  # loading proteins
        prot_keys.append(t)


    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)  # affinity shape=(68 drug,442 prot)

    opts = ['train', 'test']

    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)  # not NAN
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]  # Train fold contains index information for both drug and target
        elif opt == 'test':
            rows, cols = rows[valid_fold], cols[valid_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_key,target_sequence,affinity,dip\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                ls += [cols[pair_ind]]
                f.write(','.join(map(str, ls)) + '\n')  # csv format
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)):', len(set(drugs)),'---len(set(prots)):', len(set(prots)))
    # all_prots += list(set(prots))
    print('finish',dataset,' csv file')

    compound_iso_smiles = drugs
    target_key = prot_keys
    # durg graph data
    smile_graph = {}

    # drug sequence data
    smile_tensor = {}
    for smile in compound_iso_smiles:
        smi_tensor = label_smiles(smile, 100, CHARISOSMISET)
        smile_tensor[smile] = smi_tensor
    print('finish drug sequence')
    # finger
    for smile in compound_iso_smiles:
        mol = MolFromSmiles(smile)
        fp = MACCSkeys.GenMACCSKeys(mol)
        fg = [int(_) for _ in fp.ToBitString()[1:]]
        drug_finger[smile] = fg

    df_train = pd.read_csv('data/' + dataset + '_train.csv')
    train_drugs, train_prots,  train_Y, train_dip = list(df_train['compound_iso_smiles']),list(df_train['target_sequence']),list(df_train['affinity']), list(df_train['dip'])
    # target sequence data
    train_prot_keys = np.asarray(list(df_train['target_key']))
    XT = [seq_cat(t) for t in train_prots]

    train_drugs, train_prots,  train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)

    df_test = pd.read_csv('data/' + dataset + '_test.csv')
    test_drugs, test_prots,  test_Y, test_dip = list(df_test['compound_iso_smiles']),list(df_test['target_sequence']),list(df_test['affinity']), list(df_test['dip'])
    test_prot_keys = np.asarray(list(df_test['target_key']))
    XT = [seq_cat(t) for t in test_prots]

    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT),np.asarray(test_Y)

    # make data PyTorch Geometric ready

    print('ready for train_data and test_data')
    train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, xt=train_prots,
                                y=train_Y, smile_tensor=smile_tensor)
    test_data = TestbedDataset(root='data', dataset=dataset+'_test', xd=test_drugs, xt=test_prots,
                               y=test_Y,smile_tensor=smile_tensor)
    print('finish train_data and test_data')

    return train_data , test_data
