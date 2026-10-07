# ARC Prize 2026 Methodology & Technical Specifications

## 1. Problem Formulation

The Abstraction and Reasoning Corpus (ARC) evaluates few-shot general intelligence. An ARC task consists of $N_{\text{train}}$ demonstration input-output pairs:
$$\mathcal{D}_{\text{train}} = \{ (X_i, Y_i) \}_{i=1}^{N_{\text{train}}}$$
and $N_{\text{test}}$ test queries:
$$\mathcal{D}_{\text{test}} = \{ X_j^* \}_{j=1}^{N_{\text{test}}}$$
where $X, Y \in \{0, 1, \dots, 9\}^{H \times W}$ with $1 \le H, W \le 30$.

The goal is to find a latent program $\mathcal{P} \in \mathcal{L}_{\text{Python}}$ such that:
$$\forall (X_i, Y_i) \in \mathcal{D}_{\text{train}}, \quad \mathcal{P}(X_i) = Y_i$$
and predict the test output:
$$\hat{Y}_j^* = \mathcal{P}(X_j^*)$$

---

## 2. Dihedral Symmetry Group $D_4$ & Test-Time Augmentation

Grid reasoning problems frequently exhibit geometric invariance under the dihedral group $D_4$, which has order 8:
$$\mathcal{G}_{D_4} = \{ r_0, r_1, r_2, r_3, f_h, f_v, t, at \}$$
where:
- $r_k$: rotation by $k \cdot 90^\circ$
- $f_h, f_v$: horizontal and vertical reflection
- $t, at$: matrix transpose and anti-transpose

### Test-Time Invariance Theorem:
For any $g \in \mathcal{G}_{D_4}$, let $g(\mathcal{D})$ denote the transformed dataset where all grids are mapped by $g$. If a model synthesizes a program $\mathcal{P}'$ satisfying:
$$\mathcal{P}'(g(X_i)) = g(Y_i) \quad \forall i$$
then the true prediction under canonical orientation is recovered via:
$$\hat{Y}^* = g^{-1}(\mathcal{P}'(g(X^*)))$$
where $g^{-1}$ is the group inverse of $g$.

---

## 3. Test-Time Search & Programmatic Verification

Unlike standard language generation where hallucinated outputs cannot be easily detected, code generation for ARC allows **exact programmatic ground-truth verification**:

1. **Candidate Generation:** The LLM generates $K$ program candidates $\{ \mathcal{P}_1, \dots, \mathcal{P}_K \}$.
2. **Deterministic Pruning:** Any program $\mathcal{P}_k$ that raises a runtime exception, times out ($>2.0\text{s}$), or yields $\mathcal{P}_k(X_i) \ne Y_i$ for any $i \in \text{Train}$ is instantly pruned with 0 false positives.
3. **Execution Ensemble:** Only verified programs are executed on $X^*$. Because multiple distinct programs can pass the training pairs (the "induction problem"), we rank predictions by program frequency across diverse seeds, prompts, and augmentations.
4. **Top-2 Candidate Selection:** Kaggle allows two attempts per task. Selecting the two most confident distinct predictions yields a substantial boost in test-set solve rate.

---

## 4. Hardware Optimization on NVIDIA A100

- **BF16 / 4-bit Quantization:** Leveraging Unsloth's custom Triton kernels for 4-bit QLoRA reduces VRAM footprint to $<10\text{GB}$, allowing ultra-fast context processing with 4096 context length.
- **Batched Generation with vLLM:** High throughput generation achieves $>100\times$ speedup compared to standard HuggingFace pipelines, enabling broad test-time search over hundreds of candidate programs within seconds per task.
