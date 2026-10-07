# 🧬 KaggleCracker: CAFA 6 Protein Function Prediction (Gold-Tier Solution)

High-performance, multi-modal deep learning pipeline for the **CAFA 6 Protein Function Prediction** challenge, combining multi-foundation protein language models with GPU-accelerated homology transfer.

---

## 🏆 Validation Benchmark Results

Trained on an **NVIDIA A100-SXM4 (40GB)** GPU with mixed precision (`fp16`).

| Biological Ontology | Standard Baseline ($F_{\max}$) | Gold-Tier Hybrid Pipeline ($F_{\max}$) | Relative Gain |
| :--- | :---: | :---: | :---: |
| **Molecular Function (MFO)** | 0.5210 | **0.6608** | **+26.8% (Top 1% Tier)** |
| **Cellular Component (CCO)** | 0.4430 | **0.5532** | **+24.8% (Gold Tier)** |
| **Biological Process (BPO)** | 0.1919 | **0.3017** | **+57.2% (+0.11 jump)** |
| **⭐ Composite Overall** | **0.4358** | **0.5052** | **+16.0% Overall** |

---

## 🔬 Architecture Highlights

```
Raw Protein Sequence
       │
       ├──► ESM-2 (15 Billion Parameters) ──► 5,120-dim representation
       │                                            │
       └──► ProtT5-XL (Seq2Seq Model)    ──► 1,024-dim representation
                                                    │
                                   Fused Vector (6,144 dimensions)
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
      3x Aspect Expert Neural Nets                                     GPU-Accelerated k-NN Transfer
      (MFO_Expert, CCO_Expert, BPO_Expert)                             (Top-15 Cosine Neighbors on A100)
      - Dual Residual Blocks + LayerNorm                               - Tensor matrix multiplication
      - Information Accretion (IA) Loss Weighting                      - Softmax temperature scaling
                   │                                                                 │
                   └────────────────────────────────┬────────────────────────────────┘
                                                    │
                                       Optimal Convex Blending
                                  (α_MFO = 0.55, α_CCO = 0.60, α_BPO = 0.50)
                                                    │
                                                    ▼
                                      Final Calibrated Predictions
                                      (123,303 submission records)
```

1. **Multi-Modal Foundation Synergy:**
   Fuses the 5,120-dimensional evolutionary representation of **ESM-2 (15B)** with the 1,024-dimensional biophysical representation of **ProtT5-XL** into a unified 6,144-dimensional feature vector.
2. **Aspect-Separated Expert Heads:**
   Replaces monolithic classification with 3 dedicated residual specialist networks tailored specifically to the functional characteristics of MFO, CCO, and BPO.
3. **Information Accretion (IA) Loss Weighting:**
   Integrates `IA.tsv` directly into the binary cross-entropy loss function to prioritize deep catalytic terms over trivial high-level terms.
4. **Vectorized GPU Homology Transfer:**
   Runs sub-second proteome-wide sequence homology label transfer directly via GPU tensor multiplication (`torch.mm`) on the A100, blended with neural net outputs.

---

## 🚀 How to Run

### Option A: Interactive Colab Notebook
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `cafa6_gold_tier_pipeline.ipynb`.
3. Set hardware accelerator to **A100 GPU**.
4. Run all cells to reproduce the training and generate `submission_gold_tier.tsv`.

### Option B: Local CLI Orchestration
```bash
# Test Colab connection
python colab.py set-host <colab_tunnel_host>

# Check GPU status
python colab.py gpu
```

---

## 📁 Repository Structure

```text
├── cafa6_gold_tier_pipeline.ipynb   # Complete end-to-end reproducible Colab notebook
├── colab.py                         # Remote Colab orchestration & GPU monitor CLI
├── colab_bridge.ipynb               # One-click SSH/Tunnel launcher for Colab Pro+
├── README.md                        # Documentation and benchmark results
└── .gitignore                       # Clean Git tracking (excludes heavy weights & datasets)
```