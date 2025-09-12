ModelNet is a multi-stage cross-attention framework designed for predicting drug–target binding affinity. Molecular SMILES and protein sequences are first embedded into 128-dimensional latent representations. A three-stage cascaded sampler is then employed; at each stage, a CrossHybridAttention module integrates cross-modal and local attention to dynamically retain the most informative 1-D subsequences. Drug- and target-specific branches, each equipped with convolutional attention blocks (CABlock) and lightweight MLPs, progressively refine their respective features. Stage-wise outputs are adaptively fused via learnable gating before being fed into fully connected layers to yield a scalar binding-affinity score, enabling accurate and interpretable predictions.

"run"
python training.py
