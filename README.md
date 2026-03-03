# Open-Everything Verifiable LLM: Reproducibility Baseline

This repository implements a strictly deterministic LLM training baseline designed to study and validate reproducible training pipelines for verifiable AI systems.

## The Objective
The objective is to eliminate controllable sources of software entropy during training and establish a reproducible CPU baseline where identical inputs produce bitwise-identical checkpoints.

Given identical configuration, data, and environment, repeated runs must produce bitwise-identical model checkpoints.

## The Verification Roadmap: Why Start on CPU?
Modern GPUs optimize for maximum throughput by executing thousands of parallel threads and combining results via atomic additions. Because thread completion order is non-deterministic, the order of floating-point addition changes across runs. Due to non-associativity—where $(a+b)+c \neq a+(b+c)$—This introduces small numerical drift across runs due to floating-point non-associativity.

To solve this, we are isolating software entropy from hardware entropy in phases:

* **Phase 1: The CPU NanoGPT Baseline (Current)**
  Establishing bitwise reproducibility on CPU allows us to validate that PyTorch seeds, RNG state serialization, and Transformer operations behave deterministically under controlled conditions.
* **Phase 2: Hardware Drift Measurement (Next)**
  Introducing parallel execution primitives to measure initial hardware drift vectors against this CPU baseline.
* **Phase 3: Strict GPU Determinism**
  Enforcing `cuDNN.deterministic = True` and locking CUBLAS workspace configurations to force deterministic atomic operations.

## Phase 1 Execution Details
* **Architecture:** Deterministic port of Andrej Karpathy's NanoGPT.
* **Algorithmic Entropy:** `dropout=0.1` is intentionally kept active during training. Dropout remains enabled (dropout=0.1) to demonstrate that algorithmic RNG state (e.g., dropout masks) is correctly synchronized across segmented replays.
* **Environment:** `torch==2.2.0`, strictly pinned seeds, and full Python/NumPy/Torch RNG state serialization.
* **Dataset:** Deterministic character-level mapping with sequential dataloading. 

##### NOTE: This dataset is a minimal deterministic placeholder to validate infrastructure. It is not representative of the final Wikipedia-scale pipeline.

### How to Verify
Run the automated reproducibility test suite:
```bash
python src/reproducibility_test.py
```

### Expected Outcome
The script runs a Prover and independent Auditors to verify mathematically strict determinism and tamper-detection:

#### Current falsifiability test suite:

* **Scenario 1**: proves the system can pass when everything is honest
* **Scenario 2**: proves it catches wrong randomness
* **Scenario 3**: proves it catches microscopic in-training tampering
* **Scenario 4**: proves it catches post-training weight substitution
* **Scenario 5**: Detects "at-rest" file corruption by verifying the embedded cryptographic seal upon checkpoint load.

## Insight
A key implementation detail is that reproducibility requires serialization of full RNG state (Python, NumPy, and Torch), not only model and optimizer weights.

#### A note on tooling: 
The architecture, experiments, and proofs in this repository are my own work. I used LLMs as a pair-programming aid — to accelerate implementation, debug PyTorch internals, and sharpen documentation. LLMs were used as a development aid (debugging, documentation, iteration). System design, verification logic, and experimental validation were independently implemented and evaluated.
