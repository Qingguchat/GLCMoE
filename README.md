# MvDSCN
PyTorch Repo for “Global-and-Local Collaborative Mixture-of-Experts with Dual Contrastive Learning” (Deep Multi-View Clustering)

# Overview

We propose GLCMoE-DCL, a deep multi-view clustering framework that learns complementary and consistent semantics across views via a collaboration of Local and Global MoEs with dual contrastive learning:

* Local MoE (LMoE): a shared expert bank with view-specific routers (Top-K) produces view-level features and filters heterogeneity.

* Transformer Aggregator: fuses local view features into a compact local-fusion representation.

* Global MoE (GMoE): an attention-routed MoE on the concatenated feature space models high-order cross-view interactions to form a global representation.

* Dual Contrastive Learning (DCL):

-Local↔Local InfoNCE for cross-view consistency,

-Global↔Local-Fusion InfoNCE for cross-level alignment.

* Routing balance regularization prevents expert collapse.
At inference, the clustering embedding is

![GLCMoE](./GLCMoE.png)


# Requirements

* Python ≥ 3.9
* PyTorch (tested with 2.1+)
* numpy, scipy, scikit-learn, PyYAML

Quick install:
pip install -r requirements.txt


# Usage

*  Test by Released Result:
```bash
python test.py --config configBDGP.yaml --checkpoint model/BDGP.pth --runs 10 --outdir results_eval

```

*  Train Network:
```bash
python train.py --config configBDGP.yaml
```

