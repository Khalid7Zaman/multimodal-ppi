# TriWeave Phase C: sequence-space coverage of the synthetic candidate set

**Author:** Khalid Zaman, Research Assistant Professor (RAP).  **Supervisor:** Prof. Zhaoxi Sun, SUAT.
**Project:** TriWeave (multimodal-ppi).  2026-11-16.

This records the sequence-space coverage analysis the supervisor asked for, with numerical evidence. It
measures, in the model's own representation, whether the synthetic candidate proteins cover the sequence
space of the experimental proteins and whether they extend beyond it.

## Method

Each protein was embedded with the ESM-C backbone, which is the model's representation foundation, and the
per-residue features were mean-pooled into one vector per protein. All 20,534 experimental proteins and all
54,544 synthetic candidate proteins were embedded on a GPU. The vectors were reduced with principal component
analysis, and coverage and extension were measured on the top ten components, which capture 48.7 percent of
the variance. Scripts: `phase_c/embed_proteins.py` and `phase_c/coverage_analysis.py`. Figure:
`coverage_sequence_space.png`.

## Result

Coverage is strong. 99.4 percent of the experimental proteins fall inside the range of the synthetic set
across all ten components, and the distance from an experimental protein to its nearest synthetic protein is
about equal to the typical spacing between experimental proteins (a ratio of 1.06). So the synthetic set
densely covers the experimental sequence space.

Extension is partial. About 14 percent of the synthetic proteins lie beyond the experimental core, the 1 to 99
percentile box, so the synthetic set does reach some new regions. But overall it is somewhat more concentrated
than the experimental set, with a bounding-box volume ratio of 0.16, and it exceeds the experimental range on
only 3 of the 10 components. The reason is that the experimental affinity set spans an unusually wide
functional range, including antibody-antigen complexes, receptor-ligand pairs and some very long multi-chain
entries, which stretch the experimental range on several dimensions further than ordinary single proteins
reach.

Complementary evidence at the sequence-identity level points the same way from a different angle. At 30 percent
identity the synthetic set spans 20,335 clusters against the experimental 1,814, so it contains far more
distinct sequences, even though its spread in the model's representation is tighter.

## What this means

The STRING-based candidate set covers the experimental sequence space very well and adds a large number of
distinct sequences, which is what the synthetic data needs for breadth. It does not yet extend beyond the
experimental envelope along every dimension in the model's representation. To increase that extension, which is
the supervisor's stated goal, the candidate selection can be broadened, for example by adding more organisms
from STRING, or by selecting proteins chosen to be far from the experimental set in the representation space.
This connects directly to the importance analysis, which can score candidates by how much new sequence space
they add, so that selection favours proteins that extend the space rather than only cover it.

## Numbers at a glance

- Experimental proteins embedded: 20,534. Synthetic candidate proteins embedded: 54,544.
- Coverage: 99.4 percent of experimental proteins inside the synthetic range on every one of the top ten
  components. Nearest-synthetic to typical-experimental spacing ratio: 1.06.
- Extension: 14.3 percent of synthetic proteins beyond the experimental 1-99 percentile box. Dimensions fully
  covered by synthetic: 3 of 10. Bounding-box volume ratio synthetic over experimental: 0.16.
- Identity-level diversity (from block one and two): synthetic 20,335 clusters vs experimental 1,814.

## Reproduce

From `~/Projects/multimodal-ppi`:

    sbatch phase_c/embed_coverage.sbatch     # embeds both sets on a GPU, then runs the analysis

or, if the embeddings already exist, just the analysis on a login node:

    python phase_c/coverage_analysis.py --exp ~/Projects/ppi-data/phase_c/emb_experimental \
      --syn ~/Projects/ppi-data/phase_c/emb_synthetic --out ~/Projects/ppi-data/phase_c
