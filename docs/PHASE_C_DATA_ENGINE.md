# TriWeave Phase C: the data engine

**Author:** Khalid Zaman, Research Assistant Professor (RAP).  **Supervisor:** Prof. Zhaoxi Sun, SUAT.
**Project:** TriWeave (repository: multimodal-ppi).  2026-11-02 (plan, revision 1).

This document scopes Phase C on paper before any code is written, so the plan can be checked against the
supervisor's direction first. Phase C builds the data engine. The model from Phase A is in place and the
freeze decision from Phase B is settled, so the next bottleneck is data. Experimental protein-protein
interaction labels are scarce, and the aim of Phase C is to produce a large, well-provenanced body of
training data that the fitting network can be preheated on in Phase D, before it is fine-tuned on the
experimental ground truth.

The work splits into three building blocks, matching points 1 to 4 of the supervisor's ten-point
direction: consolidate the experimental foundation, choose an interaction database to guide pair
selection, and generate synthetic labels from simulation and from distillation. Each block ends in a
concrete file and a short written note, so progress stays visible and reportable. The model stays
protein-only throughout, as in the current architecture. The small-molecule and screening direction
remains the deferred Phase F.

## 1. Block one: consolidate the experimental foundation

The experimental data is the ground truth the model is ultimately fine-tuned and judged on, so it is
assembled first and kept clean. The current sources are already in the project and need to be merged into
one consistent table rather than gathered afresh.

- PPB-Affinity: measured binding affinities (pKd) for roughly 6,500 complexes, with experimental
  structures from RCSB.
- RCSB structures: the experimental complexes used to build the 8 Angstrom contact graphs.
- Interface labels: the per-residue interface annotations used by the interface head.
- Bernett interaction pairs: the large leakage-free interaction benchmark of roughly 163,000 pairs.

The task is to build a single, versioned manifest keyed by protein pair. Each row records the two
sequences, whether an experimental structure is available, the pKd when measured, the interface label when
available, the interaction label, the original source, and a split assignment. Deduplication is done at the
level of the pair, and sequence identity clustering (for example with mmseqs2) is used so that the held-out
test pairs share no near-duplicate sequence with the training pairs. The Bernett splits already follow this
leakage-free rule, and the same clustering is applied across the merged table so that every later synthetic
pair can be checked against it.

The output of block one is a documented dataset manifest, a data card describing every field and its
source, and a count of how many pairs carry each kind of label. Nothing in this block needs a GPU.

## 2. Block two: an interaction database to guide pair selection

Synthetic pairs must not be drawn from all possible protein combinations, because the large majority of
protein pairs do not interact and random pairs would teach the model very little. A reported-interaction
database is used to choose plausible interacting pairs for the positive side, matched against sensibly
sampled non-interacting pairs for the negative side. Three established databases are candidates.

- STRING: the broadest, covering many organisms, with confidence-scored associations that combine several
  evidence types. A high-confidence physical subnetwork can be filtered out of it.
- BioGRID: a large curated set of physical and genetic interactions drawn from the literature.
- HuRI: a high-quality experimentally measured binary interactome for human proteins.

Following the supervisor's direction (meeting of 2026-11-05, Section 8), the scope is general rather than
species-specific: pairs are drawn across organisms, not restricted to human, because a general model is more
novel than a species-specific one. STRING, which spans many organisms, is the main positive source, with
HuRI and BioGRID as higher-confidence checks. The breadth of organism and sequence coverage is recorded
explicitly, since the supervisor asks that the synthetic data cover more sequence space than the experimental
data. Negative pairs are sampled to match the length and abundance profile of the positives, following the
same care against trivial shortcuts that the Bernett benchmark uses. Every candidate pair, positive or
negative, is screened against the experimental test clusters from block one and dropped if it is too similar,
so the synthetic data can never leak into the evaluation.

The output of block two is a candidate pair list with provenance for every pair, already de-leaked against
the experimental test set. This block also needs no GPU.

## 3. Block three: synthetic labels

The candidate pairs from block two are given approximate labels from two complementary and inexpensive
sources. They are kept as separate label columns, each with its own provenance and a weight, so that their
relative influence can be tuned in Phase D rather than fixed now.

### 3.1 Simulation-derived labels

Physical scoring produces approximate binding affinities. This is the group's established strength and
gives labels that are grounded in physics rather than in another model. Most candidate pairs have no
experimental structure, so a predicted complex structure is needed first. This connects directly to the
AlphaFold structure direction noted in Phase B: a predicted complex is built for the pair, and then a
physical method scores it.

- MM/GBSA gives a more physical estimate of binding free energy but needs a reasonable complex structure
  and is slower per pair.
- A docking scoring function is faster and rougher and can stand in where a full MM/GBSA pass is too
  expensive.

The honest caveat is that a predicted complex carries its own uncertainty, especially for weak or transient
interactions, and a physical score computed on an uncertain structure inherits that uncertainty. The plan
is therefore to start with a modest, carefully chosen subset where complex prediction is most reliable, to
record a confidence value alongside each simulation label, and to treat these labels as approximate targets
rather than as ground truth.

### 3.2 Distillation labels

Strong protein-protein interaction predictors released by other groups are run on the candidate pairs, and
their outputs are stored as soft training labels. This path needs no structure and no simulation, so it is
the cheapest and fastest way to label a large number of pairs, and it is the natural first batch. The
caveat recorded here is that distillation inherits the biases and the error modes of the teacher models, so
the choice of which models to distil from matters and is left as an open decision for the supervisor.

The output of block three is one or more synthetic label tables, each row carrying the pair, the label, the
method that produced it, and a confidence or weight.

## 4. First reportable milestone

The fastest path to a usable preheating set is distillation on a first batch of a few thousand candidate
pairs, because it needs neither structures nor simulation. That first batch is produced, checked for
leakage against the experimental test set, and handed to Phase D as the initial preheating corpus.
Simulation labels are then added for a smaller subset where predicted complexes are reliable. Delivering
the distillation batch first means Phase D can begin while the slower simulation pipeline is still being
built, and it gives an early, concrete result to report to the supervisor.

## 5. Open decisions for the supervisor

- Organism scope: begin with human proteins only, or include other organisms from the start.
- Simulation method and scale: MM/GBSA, a docking scoring function, or both, how many pairs to score, and
  on whose compute budget, given that the large training cards are often booked.
- Teacher models for distillation: which released predictors to distil from, and whether their licences
  allow this use.
- Label representation: an affinity regression target, an interaction probability, or both, and how to
  reconcile labels that come from different methods for the same pair.
- Weighting: how much the synthetic labels should count against the experimental labels in Phase D, and how
  the simulation and distillation labels should be weighted against each other.

## 6. Risks and how they are handled

- Predicted-complex uncertainty feeds into every simulation label. Handled by starting small, recording a
  per-label confidence, and keeping simulation labels as approximate targets.
- Teacher bias feeds into every distillation label. Handled by choosing teachers carefully and by keeping
  distillation and simulation as separate, independently weighted sources.
- Leakage between synthetic pairs and the experimental test set would quietly inflate every later result.
  Handled by clustering once in block one and screening every candidate pair against the test clusters in
  block two.
- Scope creep toward small molecules. Handled by keeping Phase C protein-only and leaving screening to the
  deferred Phase F.

## 7. What Phase C delivers into Phase D

At the end of Phase C the project has a consolidated experimental manifest with a data card, a de-leaked
candidate pair list, and synthetic label tables from distillation and from simulation, each with
provenance and weights. These are exactly the inputs the staged training in Phase D needs in order to
preheat the fitting network on the synthetic data and then fine-tune it on the experimental ground truth,
and to compare that two-stage regime against the auxiliary-task regime.

## 8. Supervisor decisions (meeting of 2026-11-05)

The five open decisions of Section 5 were put to the supervisor. His answers are recorded here and now
drive the work.

- Experimental affinity treatment: approved. The consolidation and the leakage-free re-split of the
  affinity data are accepted, so the affinity model is retrained on the clean split for honest numbers.
- Representation and freezing: confirmed. The language model (ESM-C) is the only pretrained part and is
  treated as the representation foundation, frozen in the downstream tasks. The honest-numbers retrain on
  the clean split is therefore run with the backbone frozen, matching the adopted design.
- Organism scope: general, not species-specific. The model and its data should cover all situations across
  organisms, because a general model is more novel than a human-only one. Block two is widened accordingly.
- Simulation data: a colleague (Yucheng Han) will help generate the MM/GBSA data. The supervisor holds a
  copy of the EnsPlex docking dataset and will try to bring it to the cluster in the coming weeks. So the
  simulation labels are a collaboration, not something built from scratch here.
- Distillation teachers: the supervisor will do his own literature review on which PPI predictor is current
  state of the art. The shortlist already gathered (sequence-only, released, offline: ProteinAffinityKD for
  affinity; D-SCRIPT or Topsy-Turvy, or SENSE-PPI, for interaction; PPB-Affinity baseline as a structure
  based cross-check) is kept as input to that review. No teacher is fixed yet.
- Label type: the synthetic labels can be affinity and, or other properties tied to the prediction targets.
- Scale and coverage: the synthetic data should be at least an order of magnitude larger than the
  experimental data, and should cover more sequence space than the experimental data.

The immediate task the supervisor assigned follows from the scale and coverage point: analyse the sequence
coverage of the current experimental data first, and prepare a sequence dataset for the synthetic data
generation. `phase_c/sequence_coverage.py` does the analysis and writes the seed sequence set; the full
candidate set at the required scale is then built in block two once the interaction database is chosen and
the simulation pipelines (MM/GBSA from the colleague, EnsPlex docking from the supervisor) are in place.
