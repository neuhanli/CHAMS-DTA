import os
import os.path as osp
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch
import torch.nn.functional as F
# 定义目标大小
target_size = 78


# 填充函数
def pad_to_target_size(matrix, rows_size, cols_size):
    # 获取当前矩阵的大小
    current_size = matrix.size(0)

    # 计算需要填充的行数和列数
    pad_rows = rows_size - current_size if rows_size > current_size else 0
    pad_cols = cols_size - current_size if cols_size > current_size else 0

    # 使用 pad 函数填充矩阵，pad 的参数是一个元组，格式为 (pad_left, pad_right, pad_top, pad_bottom)
    # 在这个例子中，我们只在底部和右侧填充，因为我们假设矩阵是方阵
    padded_matrix = F.pad(matrix, (0, pad_cols, 0, pad_rows), value=0)

    return padded_matrix

# 填充函数
def pad_to_1D_size(matrix, dim_size):
    # 获取当前矩阵的大小
    # 假设 matrix 是一个一维的 PyTorch 张量
    # 获取当前向量的大小
    current_size = matrix.size(0)

    # 计算需要填充的元素数量
    pad_elements = dim_size - current_size if current_size < dim_size else 0

    # 如果向量长度小于100，则在向量的末尾填充0
    if current_size < dim_size:
        padded_matrix = F.pad(matrix, (0, pad_elements), value=0)
    # 如果向量长度超过100，则截取前100个元素
    elif current_size > dim_size:
        padded_matrix = matrix[:dim_size]
    # 如果向量长度正好是100，则不需要任何操作
    else:
        padded_matrix = matrix

    return padded_matrix

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, smiles_string=None, transform=None,
                 pre_transform=None, smile_tensor=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.process(xd, xt, y, smile_tensor)


    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_DrugData.pt', self.dataset + '_TargetData.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)


    def process(self, xd, xt, y, smile_tensor):

        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        Drug_data_list = []
        Target_data_list = []
        data_len = len(xd)
        print('data_len:',data_len)

        for i in range(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            smile_ten = smile_tensor[smiles]


            DrugData = DATA.Data(
                                y=torch.FloatTensor([labels])
                                )
            DrugData.smileString = smiles
            DrugData.smiles = torch.LongTensor([smile_ten])

            TargetData = DATA.Data(
                                y=torch.FloatTensor([labels])
            )
            TargetData.target = torch.LongTensor([target])
            Drug_data_list.append(DrugData)
            Target_data_list.append(TargetData)

        if self.pre_filter is not None:
            Drug_data_list = [data for data in Drug_data_list if self.pre_filter(data)]
            Target_data_list = [data for data in Target_data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            Drug_data_list = [self.pre_transform(data) for data in Drug_data_list]
            Target_data_list = [self.pre_transform(data) for data in Target_data_list]
        print('Graph construction done. Saving to file.')
        self.DrugData = Drug_data_list
        self.TargetData = Target_data_list


    def __len__(self):
        return len(self.DrugData)

    def __getitem__(self, idx):
        return self.DrugData[idx], self.TargetData[idx]




def collate(data_list):

    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])

    return batchA, batchB

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))


def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

