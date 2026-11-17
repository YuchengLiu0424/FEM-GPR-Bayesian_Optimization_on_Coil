# When Candidate Pools Miss Active Constraints: Evidence from Electromagnetic and Thermal Design

**Yucheng Liu and Chuncheng Zhao · Preprint, 2026**

**[Read the current paper (PDF, including numerical supplement)](https://yuchengliu0424.github.io/papers/constraint-aware-fem-design.pdf)** · [arXiv:2608.18903](https://arxiv.org/abs/2608.18903)

Finite candidate pools can miss high-value designs near intersecting active constraints, even when a surrogate is accurate. This study connects a geometric sampling analysis with controlled synthetic experiments and two finite-element design problems: a seven-variable electromagnetic coil and a six-variable heat spreader. It examines how acquisition search affects the use of a limited simulation budget.

## Main results

The matched comparison uses ten seeds per problem, with 25 initial observations and 50 further evaluations per run.

| Acquisition search | Coil response, mean (mT; higher is better) | Thermal resistance, mean (K/W; lower is better) |
|---|---:|---:|
| Finite-pool expected improvement (EI) | 0.9120 | 12.688 |
| Continuous constrained EI refinement | 0.9745 | 11.905 |
| Standard continuous LogEI | 0.9770 | 11.860 |

Standard continuous LogEI outperforms the tested EI refinement on nine coil seeds and all ten thermal seeds. Mesh and domain checks preserve the substantial continuous-search advantage over the pool, but do not resolve a physical ranking among closely performing continuous policies. The evidence supports improving candidate coverage and using stable continuous acquisition search; LogEI is an existing method, not a new optimizer introduced by this work.

For independent uniform candidates in a normalized box, the typical minimum sum of gaps to `k` active bounds scales as `(k! / N)^(1/k)` for pool size `N`. This analysis describes a coverage limitation; the manuscript distinguishes its sampling assumptions from the constrained sampling used in the FEM experiments.

## What this repository contains

This public repository currently contains the base electromagnetic Python–MPh–COMSOL workflow:

- `code/data/`: FEM sampling and automation, Gaussian-process surrogate modeling, optimization, validation, and supporting analysis outputs.
- `surrogate_simulation/`: the source COMSOL `.mph` model.

The current paper and its numerical supplement are available through the PDF link above. The complete code and trajectory archive for the expanded coil/thermal study has not yet been released in this repository. The base workflow should therefore not be treated as a complete reproduction package for every result in the current paper.

The source-model SHA-256 is `1c261222bc6281419b1b1d2b24e510c30acc6dbff83d4ceff3f10949d66cfcd3`.

## Software

The base workflow was run with COMSOL Multiphysics 6.3, Python 3.14.6, MPh 1.3.1, NumPy 2.4.3, SciPy 1.17.1, and scikit-learn 1.8.0. A licensed local COMSOL installation is required for scripts that perform new electromagnetic FEM solves. The independent thermal solver described in the paper does not require COMSOL, but is not included in this public release.

## Citation

The versionless arXiv link above is retained for the existing article and will resolve to the latest announced replacement.

```bibtex
@misc{liu2026candidatepools,
  title         = {When Candidate Pools Miss Active Constraints: Evidence from Electromagnetic and Thermal Design},
  author        = {Liu, Yucheng and Zhao, Chuncheng},
  year          = {2026},
  eprint        = {2608.18903},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.18903}
}
```

## License

No reuse license has been assigned yet. All rights are reserved until separate licenses for the code, data/model artifacts, and figures are selected.
