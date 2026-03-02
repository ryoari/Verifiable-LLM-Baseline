# Open-Everything Verifiable LLM: Reproducibility Baseline

This repository documents the foundational engineering of a strictly deterministic LLM training pipeline, serving as a proof-of-concept for verifiable AI training.

## The Objective
To prove an AI model's epistemic lineage, we must eliminate all hardware and software entropy during training. If two researchers run the exact same pipeline, they must achieve bitwise-identical checkpoint hashes. 

## The Verification Roadmap: Why Start on CPU?
Modern GPUs optimize for maximum throughput by executing thousands of parallel threads and combining results via atomic additions. Because thread completion order is non-deterministic, the order of floating-point addition changes across runs. Due to non-associativity—where $(a+b)+c \neq a+(b+c)$—this hardware entropy cascades through the network, destroying determinism.

To solve this, we are isolating software entropy from hardware entropy in phases:

* **Phase 1: The CPU NanoGPT Baseline (Current)**
  By establishing bitwise reproducibility on a CPU first, we mathematically prove that our PyTorch seeds, data loaders, and Transformer operations (Self-Attention, LayerNorm) are perfectly locked down.
* **Phase 2: Hardware Drift Measurement (Next)**
  Introducing parallel execution primitives to measure initial hardware drift vectors against this CPU baseline.
* **Phase 3: Strict GPU Determinism**
  Enforcing `cuDNN.deterministic = True` and locking CUBLAS workspace configurations to force deterministic atomic operations.

## Phase 1 Execution Details
* **Architecture:** Deterministic port of Andrej Karpathy's NanoGPT (Self-Attention, dropout added for Failure Scenario 2).
* **Environment:** `torch>=2.10.0,<3.0`, strictly pinned seeds.
* **Dataset:** Deterministic character-level mapping with sequential dataloading.

### How to Verify
Run the automated reproducibility test suite:
```bash
python src/reproducibility_test.py
```

### Expected Outcome
The script runs a Prover and three independent Auditors to verify mathematically strict determinism and tamper-detection:
* **Scenario 1 (Clean Audit):** Passes segment replay bit-for-bit.
* **Scenario 2 (Bad Seed):** Detects environment tampering and fails the audit.
* **Scenario 3 (Noisy Weights):**
* Detects a $1 \times 10^{-10}$ parameter injection and accurately quantifies the trajectory drift ($\Delta$).

## Insight
 reproducibility requires saving not just weights and optimizer state, but the full RNG state.

#### A note on tooling: 
The architecture, experiments, and proofs in this repository are my own work. I used LLMs as a pair-programming aid — to accelerate implementation, debug PyTorch internals, and sharpen documentation. All design decisions, verification, and results are my own.