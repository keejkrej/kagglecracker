# 🚀 KaggleCracker

Autonomous, high-performance deep learning pipelines for cracking Kaggle and scientific ML benchmarks using Google Colab Pro+ compute (A100/H100 GPUs) and modern foundation models.

---

## 📂 Competitions & Benchmarks

| Folder | Competition / Benchmark | Domain | Key Models / Techniques | Benchmark Score ($F_{\max}$) | Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| [**`cafa6/`**](cafa6/) | **CAFA 6 - Protein Function Prediction** | Structural Biology & Genomics | ESM-2 (15B!) + ProtT5-XL + GPU k-NN Homology Transfer + IA Weighting | **0.5052** *(MFO: 0.6608)* | 🥇 Gold-Tier Verified |
| *Upcoming* | *ARC Prize 2026 / RNA Folding / etc.* | AGI / Bio | Test-time fine-tuning, RFdiffusion, RNA-FM | — | In Queue |

---

## 🛠️ Infrastructure & Tools

* **`colab.py`**: Local CLI bridge to execute commands, query GPU status, and sync datasets/weights with Colab Pro+ instances.
* **`colab_bridge.ipynb`**: Instant one-click launcher for Google Colab GPU runtimes.

---

## 🏁 Quickstart

```bash
# Check remote Colab GPU status
python colab.py gpu

# Run training in a specific competition directory
cd cafa6
python pipeline.py
```