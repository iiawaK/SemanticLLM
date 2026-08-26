# SemanticLLM: Semantic-Channel Joint Optimization for 6G

Official implementation of the paper "Semantic-Channel Joint Optimization via Noise-Aware Decoding: An Information-Theoretic Approach to Robust Semantic Communication for 6G"

## Overview

This paper addresses the semantic-channel mismatch in 6G semantic communication through an information-theoretic lens. We derive a semantic capacity bound and propose a noise-aware decoding framework.

## Requirements

```
pip install scikit-learn matplotlib numpy tqdm
```

## Usage

```bash
python experiment_lite.py
```

This will:
1. Train noise-aware decoder with multi-SNR noise injection
2. Evaluate against digital baseline and random projection
3. Generate figures in `figs/`
4. Save results to `results.json`

## Results

| SNR | Noise-Aware (Ours) | Digital Baseline | Improvement |
|-----|-------------------|------------------|-------------|
| -10 dB | 0.169 | 0.042 | +303% |
| -5 dB | 0.274 | 0.053 | +418% |
| 0 dB | 0.374 | 0.094 | +298% |
| 5 dB | 0.442 | 0.228 | +94% |

## Citation

```bibtex
@article{zhu2026semantic,
  title={Semantic-Channel Joint Optimization via Noise-Aware Decoding},
  author={Zhu, Haonan},
  journal={Information Sciences},
  year={2026}
}
```

## License

MIT
