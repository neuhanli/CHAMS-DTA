import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath
from transformer_xtd import SMILES_FASTAModel_xtd

import torch.nn.functional as F
class MLP_xd(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super(MLP_xd, self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        self.norm = nn.BatchNorm1d(in_channels, eps=1e-05)  # 使用更常见的eps值
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, 3, padding=1)
        self.drop = nn.Dropout(drop_rate)

        # 初始化权重
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out')
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x
class MLP_xt(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super(MLP_xt, self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        self.norm = nn.BatchNorm1d(in_channels, eps=1e-05)  # 使用更常见的eps值
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, 3, padding=1)
        self.drop = nn.Dropout(drop_rate)

        # 初始化权重
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out')
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x
class ConvolutionalAttention_xt(nn.Module):
    """
    The ConvolutionalAttention implementation
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        inter_channels (int, optional): The channels of intermediate feature.
        num_heads (int, optional): The num of heads in attention. Default: 8
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels=64,
                 num_heads=8):
        super(ConvolutionalAttention_xt, self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)

        self.kv = nn.Parameter(torch.zeros(inter_channels, in_channels, 3))
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 3))

    def _act_dn(self, x):
        x_shape = x.shape  # n,len,d

        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,len,d -> n,heads,len//heads,d
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)
        x = x.reshape([x_shape[0], self.inter_channels, -1])
        return x

    def forward(self, x):
        """
            x (Tensor): The input tensor. (n,len,dim)
        """
        x = self.norm(x)
        x1 = F.conv1d(
                x,
                self.kv,
                bias=None,
                stride=1,
                padding=1)
        x1 = self._act_dn(x1)
        x1 = F.conv1d(
                x1, self.kv.transpose(1, 0), bias=None, stride=1,
                padding=1)
        x3 = F.conv1d(
                x,
                self.kv3,
                bias=None,
                stride=1,
                padding=1)
        x3 = self._act_dn(x3)
        x3 = F.conv1d(
                x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=1)
        x = x1 + x3
        return x
class CABlock_xt(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.2):
        super(CABlock_xt, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate

        self.attn = ConvolutionalAttention_xt(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            inter_channels=64,
            num_heads=num_heads)
        self.mlp = MLP_xt(self.out_channels, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x_res = x
        x = x_res + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x

class ConvolutionalAttention_xd(nn.Module):
    """
    The ConvolutionalAttention implementation
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        inter_channels (int, optional): The channels of intermediate feature.
        num_heads (int, optional): The num of heads in attention. Default: 8
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels=64,
                 num_heads=8):
        super(ConvolutionalAttention_xd, self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)

        self.kv = nn.Parameter(torch.zeros(inter_channels, in_channels, 3))
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 3))

    def _act_dn(self, x):
        x_shape = x.shape  # n,len,dim
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,len,d -> n,heads,len//heads,d

        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)
        x = x.reshape([x_shape[0], self.inter_channels, -1])
        return x

    def forward(self, x):
        """
            x (Tensor): The input tensor. (n,len,dim)
        """
        x = self.norm(x)
        x1 = F.conv1d(
                x,
                self.kv,
                bias=None,
                stride=1,
                padding=1)
        x1 = self._act_dn(x1)
        x1 = F.conv1d(
                x1, self.kv.transpose(1, 0), bias=None, stride=1,
                padding=1)
        x3 = F.conv1d(
                x,
                self.kv3,
                bias=None,
                stride=1,
                padding=1)
        x3 = self._act_dn(x3)
        x3 = F.conv1d(
                x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=1)
        x = x1 + x3
        return x
class CABlock_xd(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.2):
        super(CABlock_xd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate

        self.attn = ConvolutionalAttention_xd(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            inter_channels=64,
            num_heads=num_heads)
        self.mlp = MLP_xd(self.out_channels, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x_res = x
        x = x_res + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x

class SelfAttention(nn.Module):
    """
    普通的单头自注意力机制。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 线性变换层（仅单头）
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

    def forward(self, x):
       
        B, C, L = x.shape
        # 将通道维度 C 交换到最后，便于和线性层匹配
        x = x.transpose(1, 2)  # [B, L, C]

        # 线性变换得到 Q, K, V
        q = self.query(x)  # [B, L, C]
        k = self.key(x)    # [B, L, C]
        v = self.value(x)  # [B, L, C]

        # 计算注意力分数
        # q, k 的倒数第1维都是 C，所以需要在最后一维进行转置 k -> [B, C, L]
        scores = torch.bmm(q, k.transpose(1, 2))  # [B, L, L]

        # 缩放因子
        d_k = q.size(-1)
        scores = scores / math.sqrt(d_k)

        # Softmax 归一化得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [B, L, L]

        return attn_weights

class CrossHybridAttention(nn.Module):
    """
    改造后的分层注意力模块：
    - 使用 CrossAttention1D 代替原本单序列的 SelfAttention。
    - 将 x 视为 Query, y 视为 Key/Value, 得到 cross_out 及注意力分数。
    - local_att 对 x 做局部卷积打分，再与 cross_out 对 x 提供的分数做门控融合。
    """
    def __init__(self, in_channels, embed_dim=128, num_heads=4):
        super().__init__()
        # 1) 跨模态注意力
        self.cross_att = CrossAttention1D(embed_dim=embed_dim, num_heads=num_heads)

        # 2) 局部注意力分支 -> 输出 [B, 1, L]
        self.local_att = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 3) 可学习门控 (MLP)，融合 cross_attention 的结果 与 局部打分
        #   这里我们与原 HybridAttention 类似，用线性 -> sigmoid
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        """
        x: [B, C, L] (如蛋白分支)
        y: [B, C, L] (如药物分支)
        返回: [B, L] 逐位置的注意力分数 (对 x 侧而言)
        """
        # -----------------------------
        # 1) 跨模态注意力
        # -----------------------------
        # cross_out: [B, C, L]   (x 的更新), cross_weights: [B, L, L]
        cross_out, cross_weights = self.cross_att(x, y)

        # cross_out 中各位置的特征可做简单压缩(例如 max-pool 以得到分数)
        # 这里演示按通道 max-pool => [B, L]
        cross_scores, _ = cross_weights.max(dim=-1)  # [B, L]


        # -----------------------------
        # 2) 局部注意力 (对 x 自身)
        # -----------------------------
        local_weights = self.local_att(x)       # [B, 1, L]
        local_scores = local_weights.squeeze(1) # [B, L]

        # -----------------------------
        # 3) 门控融合: cross_scores & local_scores
        # -----------------------------
        gating_input = torch.stack((cross_scores, local_scores), dim=-1)  # [B, L, 2]
        alpha = self.gate_mlp(gating_input).squeeze(-1)                  # [B, L], 取值(0,1)

        # 最终融合：fused_scores = α * cross_scores + (1 - α) * local_scores
        fused_scores = alpha * cross_scores + (1. - alpha) * local_scores

        return fused_scores

class F_Conv1d_xd(nn.Module):
    def __init__(
            self,
            in_channels: 128,

    ):
        super().__init__()

        # 最终卷积分支（也可以分多层）
        self.conv1_xd = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=2, stride=1)
        self.conv2_xd = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=1)
        self.conv3_xd = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=8, stride=1)

    def forward(self, x):

        # ----------------------
        # 最终卷积分支
        # ----------------------
        x = self.conv1_xd(x)
        x = self.conv2_xd(x)
        x = self.conv3_xd(x)

        return x

class F_Conv1d_xt(nn.Module):
    def __init__(
            self,
            in_channels: 128,

    ):
        super().__init__()

        # 最终卷积分支（也可以分多层）
        self.conv1_xt = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=2, stride=1)
        self.conv2_xt = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=1)
        self.conv3_xt = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=8, stride=1)

    def forward(self, x):

        # ----------------------
        # 最终卷积分支
        # ----------------------
        x = self.conv1_xt(x)
        x = self.conv2_xt(x)
        x = self.conv3_xt(x)

        return x




class MultiStageSampling(nn.Module):
    """
    两级筛选 & 分层注意力示例：
     (1) Stage1: attention_stage1 + Top-2k (k = num_samples)
     (2) 经过局部卷积/激活
     (3) Stage2: attention_stage2 + Top-k
     (4) 最终卷积分支
    """
    def __init__(
            self,
            in_channels: int,
            num_samples: int = 50,
            sequence_length: int = 1000
    ):
        super().__init__()
        self.num_samples = num_samples
        self.sequence_length = sequence_length

        # ----------------------
        # 分层的两个注意力模块
        # 可以相同，也可让 stage2 更深/更多 heads
        # ----------------------
        self.attention_stage =CrossHybridAttention(
            in_channels = in_channels,
        )


    def extract_patches_topk(self, x, attention_scores, k):
        """
        与原 extract_patches 类似，但这里可传入动态 k
        attention_scores: [B, L]
        x: [B, C, L]
        """
        B, C, L = x.shape

        # 仅输出来自 attention_scores 的最高 k 个索引
        # 若 x 里 attention_scores 是 [B, L]，需要先 squeeze
        _, idx = torch.topk(attention_scores, k, dim=-1)  # [B, k]
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1) # [B, C, k]
        patches = torch.gather(x, dim=2, index=idx_expanded)  # [B, C, k]

        return patches, idx

    def forward(self, x, y):
        """
        x: [B, C, L]
        """
        B, C, L = x.shape


        attn_scores_stage = self.attention_stage(x, y)   # [B, L]
        k = self.num_samples
        x, idx_stage = self.extract_patches_topk(x, attn_scores_stage, k)

        return x, idx_stage


class CrossAttention1D(nn.Module):
    """
    简易版 Cross-Attention 模块：
    输入 x、y 分别来自两种模态 (B, C, L)
    内部使用 nn.MultiheadAttention 做注意力交互
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        # PyTorch 自带多头注意力默认输入&输出: [batch_size, seq_len, embed_dim]
        # 而我们传入的 x, y 形状是 [B, C, L], 需要 permute 到 [B, L, C] 再做注意力
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x, y: [B, C, L]
        其中 x 对 y 做 query，返回 cross-attn 后的特征
        return:
            out -> [B, C, Lx],
            attn_weights -> [B, Lx, Ly]
        """
        Bx, Cx, Lx = x.shape
        By, Cy, Ly = y.shape
        assert Cx == Cy, "x,y 的通道数应一致 (embed_dim)，以便可用多头注意力"

        # 1) reshape => [B, L, C]
        x_ = x.permute(0, 2, 1)  # [B, Lx, C]
        y_ = y.permute(0, 2, 1)  # [B, Ly, C]

        # 2) Cross Attention: x_ 作为 query, y_ 作为 key/value
        attn_output, attn_weights = self.attn(query=x_, key=y_, value=y_)
        # attn_output: [B, Lx, C] , attn_weights: [B, Lx, Ly]

        # 3) 再把 attn_output 转回 [B, C, Lx]
        attn_output = attn_output.permute(0, 2, 1)

        return attn_output, attn_weights

# ------------------------------
# 新增一个“门控融合”模块
# ------------------------------
class StageFusion(nn.Module):
    """
    对 (curr, prev) 两个同维向量做可学习门控融合：
        alpha = sigmoid(Linear([curr; prev]))
        fused = alpha * curr + (1 - alpha) * prev
    """
    def __init__(self, feat_dim):
        super(StageFusion, self).__init__()
        self.linear = nn.Linear(feat_dim * 2, feat_dim)

    def forward(self, curr_feat, prev_feat):
        """
        curr_feat, prev_feat: [B, feat_dim] 形状
        """
        cat_feat = torch.cat([curr_feat, prev_feat], dim=1)  # [B, 2*feat_dim]
        alpha = torch.sigmoid(self.linear(cat_feat))         # [B, feat_dim]
        fused = alpha * curr_feat + (1 - alpha) * prev_feat  # [B, feat_dim]
        return fused

class ModelNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=32, output_dim=128, dropout=0.2):
        super(ModelNet, self).__init__()
        self.norm_mode = 'PN-SI'
        self.norm_scale = 1
        self.pooling_ratio = 0.5
        self.dropnode_rate = 0.2
        # SMILES embedding
        self.embedding_xd = nn.Embedding(78, 128)
        self.conv1_xd = nn.Conv1d(in_channels=128, out_channels=32, padding=0, kernel_size=2, stride=1)
        self.conv2_xd = nn.Conv1d(in_channels=32, out_channels=64, padding=0, kernel_size=4, stride=1)
        self.conv3_xd = nn.Conv1d(in_channels=64, out_channels=96, padding=0, kernel_size=8, stride=1)

        # drug embedding and RandomSamplingConv1d
        #stage1
        self.stage1_xd = MultiStageSampling(
            in_channels=128,
            num_samples=50,
            sequence_length=100
        )
        self.f_stage1_xd = F_Conv1d_xd(in_channels=128)
        self.cab_stage1_xd = CABlock_xd(in_channels=96, out_channels=96)

        #stage2
        self.stage2_xd = MultiStageSampling(
            in_channels=128,
            num_samples=25,
            sequence_length=100
        )
        self.f_stage2_xd = F_Conv1d_xd(in_channels=128)
        self.cab_stage2_xd = CABlock_xd(in_channels=96, out_channels=96)

        # stage3
        self.stage3_xd = MultiStageSampling(
            in_channels=128,
            num_samples=12,
            sequence_length=100
        )
        self.f_stage3_xd = F_Conv1d_xd(in_channels=128)
        self.cab_stage3_xd = CABlock_xd(in_channels=96, out_channels=96)


        # protein
        self.embedding_xt = nn.Embedding(26, 128)
        self.conv1_xt = nn.Conv1d(in_channels=128, out_channels=32, padding=0, kernel_size=8, stride=1)
        self.conv2_xt = nn.Conv1d(in_channels=32, out_channels=64, padding=0, kernel_size=16, stride=1)
        self.conv3_xt = nn.Conv1d(in_channels=64, out_channels=96, padding=0, kernel_size=24, stride=1)
        # Protein embedding and RandomSamplingConv1d
        #stage1
        self.stage1_xt = MultiStageSampling(
            in_channels=128,
            num_samples=500,
            sequence_length=1000
        )
        self.f_stage1_xt = F_Conv1d_xt(in_channels=128)
        self.cab_stage1_xt = CABlock_xt(in_channels=96, out_channels=96)
        #stage2
        self.stage2_xt = MultiStageSampling(
            in_channels=128,
            num_samples=250,
            sequence_length=1000
        )
        self.f_stage2_xt = F_Conv1d_xt(in_channels=128)
        self.cab_stage2_xt = CABlock_xt(in_channels=96, out_channels=96)

        # stage3
        self.stage3_xt = MultiStageSampling(
            in_channels=128,
            num_samples=100,
            sequence_length=1000
        )
        self.f_stage3_xt = F_Conv1d_xt(in_channels=128)
        self.cab_stage3_xt = CABlock_xt(in_channels=96, out_channels=96)

        # alignment
        self.cf_block_xd = CABlock_xd(in_channels=96, out_channels=96)
        self.cf_block_xt = CABlock_xt(in_channels=96, out_channels=96)


        # 各阶段特征融合模块 -------------------------------------------------
        # Drug分支特征融合
        self.fuse_xd_01 = StageFusion(96)  # 融合stage0和stage1
        self.fuse_xd_012 = StageFusion(96)  # 融合结果与stage2
        self.fuse_xd_0123 = StageFusion(96)  # 最终四阶段融合

        self.fuse_xdd_01 = StageFusion(96)
        self.fuse_xdd_012 = StageFusion(96)
        self.fuse_xdd_0123 = StageFusion(96)

        # Protein分支特征融合
        self.fuse_xt_01 = StageFusion(96)
        self.fuse_xt_012 = StageFusion(96)
        self.fuse_xt_0123 = StageFusion(96)

        self.fuse_xtt_01 = StageFusion(96)
        self.fuse_xtt_012 = StageFusion(96)
        self.fuse_xtt_0123 = StageFusion(96)


        # Fully connected layers
        self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, DrugData, TargetData):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Process drug data
        smiles = DrugData.smiles
        xd_embedding = self.embedding_xd(smiles).transpose(1, 2)
        # protein
        target = TargetData.target  # batch_size=256, seq_len=1000
        xt_embedding = self.embedding_xt(target).transpose(1, 2)

       #stage0
        #drug
        xd = self.conv1_xd(xd_embedding)
        xd = self.conv2_xd(xd)
        xd = self.conv3_xd(xd)
        xdd = xd

        xdd = self.cf_block_xd(xdd)
        xd, _ = torch.max(xd, -1)

        # max pool
        xdd = torch.squeeze(xdd)
        xdd, _ = torch.max(xdd, -1)

        #protein
        xt = self.conv1_xt(xt_embedding)
        xt = self.conv2_xt(xt)
        xt = self.conv3_xt(xt)
        xtt = xt

        xtt = self.cf_block_xt(xtt)
        xt, _ = torch.max(xt, -1)

        # max pool
        xtt = torch.squeeze(xtt)
        xtt, _ = torch.max(xtt, -1)

        #drug stage1
        xd_stage1, idx_xd = self.stage1_xd(xd_embedding, xt_embedding)
        xd_stage1_feature_conv = self.f_stage1_xd(xd_stage1)
        xd_stage1_feature = self.cab_stage1_xd(xd_stage1_feature_conv)
        xd_stage1_feature_conv, _ = torch.max(xd_stage1_feature_conv, -1)
        xd_stage1_feature, _ = torch.max(xd_stage1_feature, -1)



        # protein stage1
        xt_stage1, idx_xt = self.stage1_xt(xt_embedding, xd_embedding)
        xt_stage1_feature_conv = self.f_stage1_xt(xt_stage1)
        xt_stage1_feature = self.cab_stage1_xt(xt_stage1_feature_conv)
        xt_stage1_feature_conv, _ = torch.max(xt_stage1_feature_conv, -1)
        xt_stage1_feature, _ = torch.max(xt_stage1_feature, -1)


        #drug stage2
        xd_stage2, idx_xd = self.stage2_xd(xd_stage1, xt_stage1)
        xd_stage2_feature_conv = self.f_stage2_xd(xd_stage2)
        xd_stage2_feature = self.cab_stage2_xd(xd_stage2_feature_conv)
        xd_stage2_feature_conv, _ = torch.max(xd_stage2_feature_conv, -1)
        xd_stage2_feature, _ = torch.max(xd_stage2_feature, -1)



        # protein stage2
        xt_stage2, idx_xt = self.stage2_xt(xt_stage1, xd_stage1)
        xt_stage2_feature_conv = self.f_stage2_xt(xt_stage2)
        xt_stage2_feature = self.cab_stage2_xt(xt_stage2_feature_conv)
        xt_stage2_feature_conv, _ = torch.max(xt_stage2_feature_conv, -1)
        xt_stage2_feature, _ = torch.max(xt_stage2_feature, -1)


        #drug stage3
        xd_stage3, idx_xd = self.stage3_xd(xd_stage2, xt_stage2)
        xd_stage3_feature_conv = self.f_stage3_xd(xd_stage3)
        xd_stage3_feature_cab = self.cab_stage3_xd(xd_stage3_feature_conv)
        xd_stage3_feature_conv, _ = torch.max(xd_stage3_feature_conv, -1)
        xd_stage3_feature, _ = torch.max(xd_stage3_feature_cab, -1)


        # protein stage3
        xt_stage3, idx_xt = self.stage3_xt(xt_stage2, xd_stage2)
        xt_stage3_feature_conv = self.f_stage3_xt(xt_stage3)
        xt_stage3_feature_cab = self.cab_stage3_xt(xt_stage3_feature_conv)
        xt_stage3_feature_conv, _ = torch.max(xt_stage3_feature_conv, -1)
        xt_stage3_feature, _ = torch.max(xt_stage3_feature_cab, -1)
        #
        #
        #
        # Drug分支各阶段特征 -------------------------------------------------
        # Stage0特征
        xd_stage0 = xd  # [B,96]
        xdd_stage0 = xdd  # [B,96]

        # Stage1特征（池化后）
        xd_stage1 = xd_stage1_feature_conv  # [B,96]
        xdd_stage1 = xd_stage1_feature  # [B,96]

        # Stage2特征（池化后）
        xd_stage2 = xd_stage2_feature_conv  # [B,96]
        xdd_stage2 = xd_stage2_feature  # [B,96]

        # Stage3特征（池化后）
        xd_stage3 = xd_stage3_feature_conv  # [B,96]
        xdd_stage3 = xd_stage3_feature  # [B,96]
        #
        # # 四阶段融合 -------------------------------------------------------
        # xd特征融合
        fuse_xd_01 = self.fuse_xd_01(xd_stage1, xd_stage0)
        fuse_xd_012 = self.fuse_xd_012(xd_stage2, fuse_xd_01)
        fuse_xd_0123 = self.fuse_xd_0123(xd_stage3, fuse_xd_012)
        #
        # xdd特征融合
        fuse_xdd_01 = self.fuse_xdd_01(xdd_stage1, xdd_stage0)
        fuse_xdd_012 = self.fuse_xdd_012(xdd_stage2, fuse_xdd_01)
        fuse_xdd_0123 = self.fuse_xdd_0123(xdd_stage3, fuse_xdd_012)
        #
        # Protein分支各阶段特征 -------------------------------------------------
        # Stage0特征
        xt_stage0 = xt  # [B,96]
        xtt_stage0 = xtt  # [B,96]

        # Stage1特征（池化后）
        xt_stage1 = xt_stage1_feature_conv  # [B,96]
        xtt_stage1 = xt_stage1_feature  # [B,96]

        # Stage2特征（池化后）
        xt_stage2 = xt_stage2_feature_conv  # [B,96]
        xtt_stage2 = xt_stage2_feature  # [B,96]

        # Stage3特征（池化后）
        xt_stage3 = xt_stage3_feature_conv  # [B,96]
        xtt_stage3 = xt_stage3_feature  # [B,96]


        # 四阶段融合 -------------------------------------------------------
        # xt特征融合
        fuse_xt_01 = self.fuse_xt_01(xt_stage1, xt_stage0)
        fuse_xt_012 = self.fuse_xt_012(xt_stage2, fuse_xt_01)
        fuse_xt_0123 = self.fuse_xt_0123(xt_stage3, fuse_xt_012)

        # xtt特征融合
        fuse_xtt_01 = self.fuse_xtt_01(xtt_stage1, xtt_stage0)
        fuse_xtt_012 = self.fuse_xtt_012(xtt_stage2, fuse_xtt_01)
        fuse_xtt_0123 = self.fuse_xtt_0123(xtt_stage3, fuse_xtt_012)




        out = torch.cat((fuse_xd_0123, fuse_xdd_0123,  fuse_xt_0123, fuse_xtt_0123), 1)
        xc = self.fc1(out)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out_xtd = self.out(xc)

        return out_xtd, idx_xd, idx_xt
