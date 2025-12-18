# MvDSCN
PyTorch Repo for “Global-and-Local Collaborative Mixture-of-Experts with Dual Contrastive Learning” (Deep Multi-View Clustering)

# Overview

We propose GLCMoE-DCL, a deep multi-view clustering framework that learns complementary and consistent semantics across views via a collaboration of Local and Global MoEs with dual contrastive learning:

* Local MoE (LMoE): The LMoE filters cross-view heterogeneity via parameters shared expert bank and preserves complementarity rich in fine-grained view-specific information by collaborative experts that are adaptively activated and reorganized via gating routing mechanism.

* Global MoE (GMoE): The GMoE with attention-based routing focuses on extracting cross-view consistency and capturing high-order cross-view interaction from unified multi-view feature space.

* Dual Contrastive Learning (DCL): The profound semantic alignment across views is achieved by DCL module from various levels, including coarse-grained granularity among local view-specific representations and fine-grained granularity between local fusion and global representation.

* The gating routing-balance regularizer is designed to alleviate the expert collapse and improve the availability of experts.

The overview framework of the proposed GLCMoE for DMvC task:

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

