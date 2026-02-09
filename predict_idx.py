
import sys
import numpy as np
from torch import nn
import matplotlib
matplotlib.use('TkAgg')
from predict_model import ModelNet
import torch


TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

#蛋白质
prot_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
prot_dict = {v:(i+1) for i,v in enumerate(prot_voc)}
prot_dict_len = len(prot_dict)
max_prot_len = 1000

#药物
drug_voc = "#%)(+-/.1032547698=A@CBEDGFIHKMLONPSRUTWVY[Z]\\acbedgfihmlonsruty"
drug_dict = {v:(i+1) for i,v in enumerate(drug_voc)}
drug_dict_len = len(drug_dict)
max_drug_len = 100

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

def prot_cat(prot):
  x = np.zeros(max_prot_len)
  for i, ch in enumerate(prot[:max_prot_len]):
    x[i] = prot_dict[ch]
  return x


def drug_cat(drug):
  x = np.zeros(max_drug_len)
  for i, ch in enumerate(drug[:max_drug_len]):
    x[i] = CHARISOSMISET[ch]
  return x

cuda_name = "cuda:0"
if len(sys.argv)>2:

    cuda_name = "cuda:" + str(int(sys.argv[2]))
print('cuda_name:', cuda_name)
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelNet().to(device)
model.load_state_dict(torch.load('model_ModelNet_davis.model'))
#model.load_state_dict(torch.load('model_ModelNet_kiba.model'))
model.eval()
loss_fn = nn.MSELoss()

# 构造输入数据

#1DRV
# drug = "CC(=O)C1=C[N+](=CC=C1)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)([O-])OP(=O)(O)OC[C@@H]3[C@H]([C@H]([C@@H](O3)N4C=NC5=C(N=CN=C54)N)O)O)O)O"
# target = "MHDANIRVAIAGAGGRMGRQLIQAALALEGVQLGAALEREGSSLLGSDAGELAGAGKTGVTVQSSLDAVKDDFDVFIDFTRPEGTLNHLAFCRQHGKGMVIGTTGFDEAGKQAIRDAAADIAIVFAANFSVGVNVMLKLLEKAAKVMGDYTDIEIIEAHHRHKVDAPSGTALAMGEAIAHALDKDLKDCAVYSREGHTGERVPGTIGFATVRAGDIVGEHTAMFADIGERLEITHKASSRMTFANGAVRSALWLSGKESGLFDMRDVLDLNNL"

# #1D2E
# drug = "C1=NC2=C(N1[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)O)O)O)N=C(NC2=O)N"
# target = "KPHVNVGTIGHVDHGKTTLTAAITKILAEGGGAKFKKYEEIDNAPEERARGITINAAHVEYSTAARHYAHTDCPGHADYVKNMITGTAPLDGCILVVAANDGPMPQTREHLLLARQIGVEHVVVYVNKADAVQDSEMVELVELEIRELLTEFGYKGEETPIIVGSALCALEQRDPELGLKSVQKLLDAVDTYIPVPTRDLEKPFLLPVESVYSIPGRGTVVTGTLERGILKKGDECEFLGHSKNIRTVVTGIEMFHKSLDRAEAGDNLGALVRGLKREDLRRGLVMAKPGSIQPHQKVEAQVYILTKEEGGRHKPFVSHFMPVMFSLTWDMACRIILPPGKELAMPGEDLKLTLILRQPMILEKGQRFTLRDGNRTIGTGLVTDTPAMTEEDKNIKW"

# # #1EC9
drug = "[C@H]([C@@H](C(=O)NO)O)([C@H](C(=O)[O-])O)O"
target = "MSSQFTTPVVTEMQVIPVAGHDSMLMNLSGAHAPFFTRNIVIIKDNSGHTGVGEIPGGEKIRKTLEDAIPLVVGKTLGEYKNVLTLVRNTFADRDAGGRGLQTFDLRTTIHVVTGIEAAMLDLLGQHLGVNVASLLGDGQQRSEVEMLGYLFFVGNRKATPLPYQSQPDDSCDWYRLRHEEAMTPDAVVRLAEAAYEKYGFNDFKLKGGVLAGEEEAESIVALAQRFPQARITLDPNGAWSLNEAIKIGKYLKGSLAYAEDPCGAEQGFSGREVMAEFRRATGLPTATNMIATDWRQMGHTLSLQSVDIPLADPHFWTMQGSVRVAQMCHEFGLTWGSHSNNHFDISLAMFTHVAAAAPGKITAIDTHWIWQEGNQRLTKEPFEIKGGLVQVPEKPGLGVEIDMDQVMKAHELYQKHGLGARDDAMGMQYLIPGWTFDNKRPCMVR"


drug = [drug_cat(drug)]
target = [prot_cat(target)]
drug = np.asarray(drug)
target = np.asarray(target)
drug = torch.LongTensor(drug).to(device)
print("diug形状")
print(drug.shape)
target = torch.LongTensor(target).to(device)

# 进行预测

model.eval()  # 切换到训练模式以进行反向传播
output, idx_xd_stage1, idx_xt_stage1, idx_xd_stage2, idx_xt_stage2, idx_xd_stage3, idx_xt_stage3 = model(drug, target)
print("drug的阶段一采样点")
print(idx_xd_stage1)
print("target的阶段一采样点")
print(idx_xt_stage1)

print("drug的阶段二采样点")
print(idx_xd_stage2)
print("target的阶段二采样点")
print(idx_xt_stage2)

print("drug的阶段三采样点")
print(idx_xd_stage3)
print("target的阶段三采样点")
print(idx_xt_stage3)

