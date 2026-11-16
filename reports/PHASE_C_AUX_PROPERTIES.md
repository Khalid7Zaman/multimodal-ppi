# TriWeave Phase C: additional experimental properties that correlate with binding affinity

**Author:** Khalid Zaman, Research Assistant Professor (RAP).  **Supervisor:** Prof. Zhaoxi Sun, SUAT.
**Project:** TriWeave (multimodal-ppi).  2026-11-16.

This records the third of the three analyses the supervisor asked for before scaling up synthetic
generation, with numerical evidence. Point three is about other MEASURED experimental properties, not
computed values, that move together with binding affinity. If such properties exist and correlate with
affinity, they can be added as auxiliary prediction tasks. Training a shared model to predict several
related real measurements at once usually improves the main affinity prediction, and it reuses data that
already exists, so it is close to free.

## Method

The evidence comes from SKEMPI 2.0, a public benchmark of measured protein-protein binding energy,
kinetics and thermodynamics. For the same complexes it reports, from real experiments, the dissociation
constant (affinity), the association and dissociation rate constants kon and koff, and the calorimetric
enthalpy and entropy of binding. The file holds 7,085 measurements over 348 distinct complexes, where the
many rows of one complex are its single mutations.

Affinity was turned into a binding free energy with the standard relation dG = RT ln(Kd), using the
temperature recorded for each measurement. Two kinds of correlation were computed. Across complexes, each
complex is counted once, using its wild-type values, to see whether a measured property tracks affinity
from one complex to another. Across single mutations, every mutation is a separate measurement, and the
question is whether the change in a property upon mutation tracks the change in affinity. Correlations are
reported as both Pearson and Spearman, with the sample size, so small samples are visible.

Script: `phase_c/aux_properties.py`. Input `skempi_v2.csv` (downloaded on the cluster from the SKEMPI
site). Outputs `skempi_property_correlations.csv` and `aux_properties.png`. CPU only, seconds to run, and
it reads only the downloaded file and changes nothing in the project data.

## Result

The dissociation rate is the strongest measured correlate of affinity. Across the 88 distinct complexes
that report it, the logarithm of koff correlates with the binding free energy at Pearson 0.83 and Spearman
0.82. The association rate kon correlates at Pearson -0.75 over the same 88 complexes. So both rate
constants carry a large part of the affinity signal, koff positively, kon negatively, which matches the
physical picture that a tighter complex comes apart more slowly and tends to form faster.

The strongest and most data-rich evidence is at the mutation level. Over 1,844 single mutations with
measured kinetics, the change in log koff upon mutation explains the change in affinity at Pearson 0.95
and Spearman 0.89. The change in log kon tracks it more weakly, at Pearson -0.54. In plain terms, when a
mutation changes how tightly two proteins bind, that change shows up almost entirely in how fast they come
apart. This is a well-established result in the binding literature, and it is recovered here cleanly on a
large sample.

The thermodynamic terms are available for far fewer complexes, 37, and behave as expected. Enthalpy alone
correlates with affinity moderately, at Pearson 0.52 and Spearman 0.67, and the entropy term alone weakly,
at Pearson 0.33, because enthalpy and entropy trade off against each other across complexes, which is the
known enthalpy-entropy compensation. Taken together as enthalpy minus temperature times entropy they
reconstruct the free energy at Pearson 1.00, which is a consistency check that confirms the measured
calorimetry agrees with the measured affinity rather than an independent finding.

## What this means

There are real, measured properties that correlate strongly with affinity, so point three is answered with
numbers. The recommended auxiliary targets, in order of evidence and data availability, are the
dissociation rate koff first, then the association rate kon. Both correlate strongly with affinity, both
are measured for a useful number of complexes, and at the mutation level koff is an almost complete proxy
for the change in affinity. The calorimetric enthalpy and entropy are worth including as a secondary,
joint target where they are present, but the data is sparse, so they cannot carry a task on their own.

The practical way to use these is multi-task training with masking. The model keeps its affinity head and
gains small heads for koff and kon. Complexes that have a measured rate contribute a loss on that head,
and complexes that do not are simply masked out, so no data is wasted and the shared representation is
pulled toward the physics that governs affinity. The honest limitation is sample size. Only 88 complexes
report wild-type kinetics and only 37 report thermodynamics, so the auxiliary signal at the complex level
is thin. The mutation-level kinetic data is much larger at 1,844 points and is the stronger basis for an
auxiliary kinetic task. Beyond SKEMPI, the same properties can be gathered from PDBbind for affinity,
PROXiMATE for the thermodynamics of protein-protein mutations, ATLAS for T-cell receptor and peptide-MHC
affinities, and the Protein-Protein Affinity Benchmark, which widens the pool if a kinetic or thermodynamic
head is pursued.

## Numbers at a glance

- Source: SKEMPI 2.0, 7,085 measurements over 348 distinct complexes.
- Availability at the distinct-complex level: affinity 346, kon 88, koff 88, enthalpy 37, entropy 37.
- Across complexes (one per complex): log10(koff) vs dG Pearson 0.83, Spearman 0.82, n 88; log10(kon) vs
  dG Pearson -0.75, n 88; dH vs dG Pearson 0.52, Spearman 0.67, n 37; T*dS vs dG Pearson 0.33, n 37;
  (dH - T*dS) vs dG Pearson 1.00, n 37.
- Across mutations: d log10(koff) vs ddG Pearson 0.95, Spearman 0.89, n 1,844; d log10(kon) vs ddG
  Pearson -0.54, n 1,844.
- Recommended auxiliary targets: koff and kon (kinetic), with enthalpy and entropy as a secondary joint
  target where measured.

## Reproduce

Download once on the cluster:

    wget -O ~/Projects/ppi-data/skempi/skempi_v2.csv \
      https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv

Then from `~/Projects/multimodal-ppi` in the esmc environment:

    python phase_c/aux_properties.py \
      --csv ~/Projects/ppi-data/skempi/skempi_v2.csv \
      --out ~/Projects/ppi-data/phase_c
