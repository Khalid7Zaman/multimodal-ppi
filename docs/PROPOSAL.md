# A Multimodal Deep-Learning Framework for Predicting Protein-Protein Interactions and Their Drug-Relevant Properties

> Research Proposal, **v14**. Mirrored into the repo (`docs/PROPOSAL.md`) from
> `14. Research_Proposal_PPI_v14 (Start working on this project) .docx` so every session/tab can read it.
> Model name: **TriWeave**. Sequence backbone: **ESM-C** (selected over ESM-2 by a controlled ablation).
> v13 folded in the supervisor's data and training direction: frozen pretrained foundations with a
> trainable fitting network, synthetic data augmentation (simulation and distillation with guided pair
> selection), two training regimes to compare, and multi-task machinery. **v14** replaces Figure 1 with
> the unified main-architecture graphic (Part A: the model; Part B: how it is fed and trained) and reframes
> Section 6.1 to match. The data and training strategy is set out in full in `docs/ROADMAP.md`.
> Author: Khalid Zaman (RAP). Supervisor: Prof. Zhaoxi Sun, SUAT.
> Source `.docx` location on the laptop:
> `E:\Study\RAP-SUAT\Project and Meeting with PhD stds\Personal Projects\1. First Project\2. Protein-Protein-Interaction`

## Abstract

Protein-protein interactions organise almost everything a cell does, and disrupting them is a proven
route to new medicines. Predicting which proteins interact and how strongly they bind is therefore a
central goal of computational biology. Sequence-only models have advanced the field but they miss the
structural signal that governs binding, and the labelled data available to train them are limited. This
proposal develops **TriWeave**, a multimodal deep-learning model for protein-protein interaction. From the sequences of
two proteins it predicts whether they interact, how tightly they bind, and which residues form the
interface. The model combines three views of each protein. A protein language model reads the sequence. A
structural module draws on experimentally determined complex structures and measured binding affinities
from the PDBbind database. An evolutionary module adds conservation from sequence alignments. A
cross-attention layer then reasons across the pair and exposes the contacts it relies on. Binding affinity
and the interface it identifies are the drug-relevant outputs, since together they show whether an
interaction is a viable target and where to act on it. The model is benchmarked against leading
sequence-based predictors, and it stays complementary to the separate and harder task of predicting a full
bound complex, which the group already studies. To use the limited experimental data well, the
framework builds on pretrained representations kept frozen and augments the scarce measurements with
cheaper simulated and distilled training signals, as set out in the methodology.

## 1. Introduction and Background

Proteins rarely act alone. They work by binding to one another and so build the complexes and signalling
chains that keep a cell alive. When a binding surface fails the result is often disease. This is also what
makes these surfaces attractive drug targets, since a molecule that blocks the right interaction can
correct the fault. Mapping protein-protein interactions and understanding how tightly proteins bind has
therefore become central to both biology and medicine.

Measuring interactions in the laboratory is slow and expensive, and it is far from complete. Reference maps
of the human interactome remain unfinished [1]. Computation offers a way to fill the gaps. Protein language
models learn rich descriptions of proteins from millions of sequences [2,3,4], and the same ideas drive the
recent progress in structure prediction [5]. Building on this the PLM-interact model showed that encoding a
protein pair together beats encoding each protein alone [6]. Its authors also noted that adding structural
context should push the approach further.

That missing structural context matters, because binding happens in three dimensions. A protein's shape
decides which surface it presents and how well two partners fit. Evolution leaves a further clue, since
residues that sit at an interface tend to be conserved and to co-vary across related proteins [7]. Both
signals are largely absent from a sequence read on its own.

The PDBbind database gives a direct way to bring the structural signal in. It collects experimentally
determined complex structures together with their measured binding affinities [8]. This makes it a trusted
source of both the shapes of real complexes and the strength of real interactions, which is exactly what a
model of protein-protein binding needs to learn from. Related work has already used such data for
structure-based affinity prediction [9], and geometric models read binding surfaces directly to describe
how proteins meet [10]. The framework proposed here joins these structural signals with sequence and
evolution in a single model.

Scope matters here. Predicting the full three-dimensional structure of a bound complex from scratch is a
separate and very difficult problem, and the group already studies it with dedicated docking and ranking
methods [11,12]. The framework proposed here does not compete with that work. It predicts interaction,
binding strength and the interface, and it uses known structures as an input rather than trying to generate
them. This focus keeps the project achievable and aims it at the outputs that matter most for finding and
understanding drug targets.

## 2. Problem Statement

Current methods for predicting protein-protein interactions fall short in four linked ways. Models built
from sequence alone have no explicit view of structure. They learn correlations rather than the shapes and
contacts that cause binding. They also generalise poorly, and performance drops on distant species and on
proteins with no close relative in the training set. A third weakness is that most predictors give only a
yes-or-no answer. They report whether two proteins interact but not how strongly they bind and not which
residues form the interface, and both of these are what a drug programme needs. The fourth weakness is
data, since high-quality labelled interactions and measured affinities are scarce.

These gaps point to one unmet need. No current framework brings sequence, experimentally grounded structure
and evolutionary information together to predict, in a single model, whether two proteins interact, how
tightly they bind and where. This proposal sets out to fill that need.

## 3. Research Gap

The state of the art is easy to place. Sequence-only pair models such as PLM-interact capture how two
proteins relate but ignore structure and predict only interaction [6]. Structure-based methods predict
binding affinity or read interface geometry well [9,10], but they need a structure and they treat affinity
apart from interaction. Structure-prediction methods are advancing fast for complexes [11,12], yet
generating a bound structure is a separate task that this work is built to complement rather than repeat.

Two further gaps cut across the field. No framework grounds a protein-interaction model in the
experimentally measured structures and affinities that PDBbind provides. And no single model joins
sequence, structure and evolution to predict interaction, binding strength and interface residues together.
Closing these gaps is the aim of this proposal.

## 4. Aim, Hypothesis and Objectives

The **aim** is to design, build and evaluate a multimodal deep-learning framework that predicts
protein-protein interactions, their binding strength and their interface residues from sequence, grounded
in experimentally determined complex structures and measured affinities from PDBbind.

The central **hypothesis** is that joining protein-language-model representations with experimentally
grounded structure and evolutionary information will predict interactions, binding strength and interfaces
more accurately and more generally than a model trained on sequence alone.

The aim will be met through five **objectives**:

1. Assemble a training resource for protein-protein interaction from public interaction datasets together
   with the complex structures and measured affinities in PDBbind.
2. Develop a multimodal encoder that joins protein-language-model, structural and evolutionary
   representations of each protein.
3. Build a cross-attention module that models the relationship between two proteins and highlights the
   residues that drive it.
4. Predict interaction, binding affinity and interface residues within one multi-task model.
5. Benchmark the model against leading sequence-based predictors and provide clear, residue-level
   interpretability.

## 5. Model Inputs and Prediction Targets

To avoid any doubt about what the model does, its inputs and outputs are stated plainly. The required input
is the amino-acid sequence of two proteins. A three-dimensional structure may be supplied as an optional
extra input, and the structural signal for training is drawn from the experimental complexes in PDBbind.
The model returns:

- **Interaction.** Whether the two proteins bind.
- **Binding strength.** A quantitative estimate of affinity, learned from the measured values in PDBbind.
- **Interface residues.** The residues on each protein that form the contact surface.
- **Interpretability.** The contacts and residues behind each prediction.

The model does **not** predict the full three-dimensional bound structure. That task is handled separately
and complementarily within the group.

## 6. Methodology

### 6.1 The proposed architecture

Figure 1 shows the whole framework of the model, named **TriWeave**, in two parts. Part A is what the
model is made of, and Part B is how it is fed and trained. In Part A, two proteins are read by
representation foundations that are pretrained once and then kept frozen, giving three views of each
protein: the sequence through a protein language model, the structure as a residue contact graph from an
experimental or predicted structure, and the evolution through a multiple sequence alignment. A two-way
cross-attention fusion then lets the two proteins read each other, a trainable fitting network with
multi-task experts carries the shared representation, and the heads predict interaction, binding affinity
and interface residues. In Part B, the scarce experimental data is enlarged with inexpensive synthetic
data and the model is trained and compared under two regimes, as set out in Section 6.2 and Section 6.3.
The small-molecule side and the protein-ligand head shown in the figure are the later drug-screening
extension. The paragraphs below walk through Part A in order.

It begins at the simplest possible starting point. The sequences of two proteins arrive and nothing else is
required. A single shared encoder reads both, so Protein A and Protein B are treated the same way.

From here each protein is read three ways at once. The first reader is a protein language model that
captures the meaning carried in the sequence. The second is a structural module that turns an
experimentally determined complex structure from PDBbind into a residue contact graph, so the model sees
the real surface where two proteins meet [8]. The third looks across evolution and marks the positions that
nature has kept unchanged and that co-vary between partners [7]. By the time a protein leaves this stage it
is described by both its sequence and its shape.

The two descriptions then meet. Inside the cross-protein reasoning engine a cross-attention transformer
lets every residue of one protein look directly at every residue of the other. What it learns is drawn as
an interaction map that lights up the residues in contact, so a biologist can watch Protein A residue 145
reach for Protein B residue 320 and see why the model believes they bind.

That one shared understanding then answers three questions at once. It tells whether the two proteins
interact. It gives their binding strength as a real affinity, learned from the measured values in PDBbind.
And it names the residues that form the interface, so every answer comes with its reasons.

The last part of the framework is how the model is trained. It learns from experimentally grounded data,
taking interaction labels from public datasets and structures with measured affinities from PDBbind.

### 6.2 Data

The project draws on three public sources. Interaction labels come from established protein-protein
interaction datasets, including the sets released with PLM-interact and the leakage-free benchmark used to
test it fairly [6]. Complex structures and measured binding affinities come from PDBbind [8]. Evolutionary
information comes from multiple sequence alignments. Experimental structures and affinities keep the
training signal trustworthy and remain the ground truth, and where an experimental structure is not
available an AlphaFold-predicted structure can stand in so that coverage is not limited to solved complexes.

Experimental interaction data is scarce, so before the model meets the experimental ground truth the
training set is enlarged in two inexpensive and complementary ways. Physical simulation, through MM/GBSA or
docking scoring, produces approximate binding affinities for many protein pairs [9], and strong existing
interaction predictors are distilled by taking their predictions as extra labels. Both sources are guided
by a reported-interaction database, such as STRING, BioGRID or HuRI [15], so that the generated pairs are
plausible rather than random, since most protein pairs do not interact. These generated labels are cheap
and abundant, while the experimental measurements remain the ground truth for fine-tuning and for
evaluation.

### 6.3 Model and training

The three modules are built on pretrained foundation models that are used as frozen feature extractors, so
their general knowledge is kept and only a smaller fitting network is trained on top. Whether the sequence
backbone is better kept frozen or fine-tuned end to end is treated as an experimental question on this
task, since an earlier comparison on the small interaction data favoured fine-tuning. The sequence module
itself is built on a protein language model backbone applied to paired sequences [3,6]. Two candidate backbones, ESM-2 (650M) and ESM-C (600M), were compared under identical settings on the leakage-free benchmark, and ESM-C was selected because it gave the stronger ranking performance with fewer parameters. The structural module encodes
each complex from PDBbind as a residue contact graph and processes it with a graph network. The
evolutionary module encodes conservation from alignments. These three views are joined and passed to the
cross-attention module, and the shared output feeds the interaction, affinity and interface heads under one
combined objective. Training is supervised throughout, using interaction labels and the measured affinities
in PDBbind. Mixed precision and gradient checkpointing keep the full-size model within the memory of the
available GPUs.

How this extra data is best used is left open and settled by experiment rather than assumed. In a two-stage
regime the fitting network is first preheated on the simulated and distilled data and then fine-tuned on
the experimental data. In an alternative regime the simulated and distilled data are added as auxiliary
objectives and trained jointly with the experimental data. It is not taken for granted that one regime
beats the other, so both are built and compared on the same held-out experimental test set.

Because the model carries several tasks of very different sizes, the tasks can pull against one another
during training. Two families of methods address this and can be combined. Mixture-of-experts heads, such
as the multi-gate design [16] and its hierarchical and feature-decomposition variants [17], give the tasks
partly separate capacity, while gradient-projection methods [18] remove the conflicting part of one task's
gradient so that the tasks stop cancelling. The choice between these methods is one of the open questions
the project will test.

### 6.4 Evaluation

Interaction prediction is benchmarked against PLM-interact and its published baselines on the same leakage-free held-out
data [6], and is reported with the same metric set used in that work: the area under the precision-recall curve (AUPR), the area under the ROC curve (AUROC), precision, recall, F1 and accuracy, together with the Matthews correlation coefficient (MCC). Binding-affinity prediction is measured on held-out PDBbind complexes and compared with
established structure-based scoring approaches [9]. Interface predictions are checked against the contacts
seen in known complexes [10,12]. Ablation studies remove one module or one output at a time, so that the
contribution of structure and of evolution can each be seen clearly.

### 6.5 Implementation and phased plan

The project runs on the university GPU cluster in Python and PyTorch and stays under version control
throughout. It moves in phases (Table 1), and each phase ends in a working result that can be reported.

**Table 1. Phase-by-phase implementation plan.** Each phase ends in a working and reportable result.

| Phase | Goal | Outcome |
|---|---|---|
| 0 | Environment and baseline | Working environment, PLM-interact reproduced on a small dataset |
| 1 | Data assembly | Interaction datasets and PDBbind complexes prepared and aligned |
| 2 | Core interaction model | Sequence and cross-attention model trained and benchmarked against baselines |
| 3 | Structural and evolutionary modules | PDBbind structures and conservation joined with sequence. Measured gains |
| 4 | Affinity and interface heads | Binding-affinity and interface-residue predictions |
| 5 | Benchmarking and analysis | Full comparison, ablations and interpretability study |
| 6 | Dissemination | Manuscript and public code release |

## 7. Novelty and Contributions

The novelty lies in combining three things that have not been brought together for protein-protein
interaction. First the project grounds the model in the experimentally measured structures and affinities
of PDBbind, rather than in predicted structures, so its training signal is trustworthy [8]. Second it joins
sequence, structure and evolution in a single encoder and reasons across the pair with cross-attention, so
structure and interaction are learned together rather than apart. Third it predicts interaction, binding
strength and the interface in one interpretable model, and it does so as a partner to bound-structure
prediction rather than a rival. The result is not a single new module but a coherent system aimed at the
outputs that matter for finding and understanding protein-protein drug targets.

## 8. Significance of the Study

Grounding the model in experimental structures and affinities should make its predictions both more
accurate and more useful than sequence-only methods. By reporting binding strength and the interface
alongside interaction, the framework speaks straight to drug discovery, where blocking a specific protein
surface with a small molecule is already a proven therapeutic strategy [13]. Knowing how strongly two
proteins bind and which residues to target is precisely what turns an interaction into a candidate for
intervention.

The framework is also built to be used. It complements the group's structure-prediction work and draws on
shared, public data. Its reach extends to host-pathogen settings, where knowing how viral proteins seize
human ones matters for anticipating infection [14]. A model that predicts interaction, strength and
interface from sequence, backed by experimental structures, would be a practical tool for both basic
research and target discovery.

## 9. Expected Outcomes

The project should deliver a multimodal deep-learning model that predicts protein-protein interaction,
binding strength and interface residues better than sequence-only baselines. It should also deliver a
clean, reusable pipeline that pairs public interaction data with PDBbind structures and affinities. The
affinity predictions should agree with held-out measurements, and the interface predictions should match
known complexes. Alongside these the work will produce a residue-level interpretability layer and an openly
released codebase and trained models. Strong results in this field can never be promised in advance, so the
project is scoped such that every phase yields a self-contained result and the ambition of the final paper
matches the strength of the findings.

## 10. Positioning Against the Base Model

**Table 2. TriWeave (the proposed framework) set against PLM-interact** across input, data and prediction.

| Dimension | PLM-interact | TriWeave |
|---|---|---|
| Sequence representation | ESM-2 | ESM-C (selected over ESM-2) |
| Structural information | None | Experimental complexes (PDBbind) |
| Evolutionary information | None | Multiple sequence alignments |
| Pair reasoning | Joint pair encoder (full self-attention over the pair) | Cross-attention between separate per-protein encoders |
| Interaction prediction | Yes | Yes |
| Binding strength | No | Yes (from PDBbind affinities) |
| Interface prediction | No | Yes |
| Interpretability | Attention-based (not residue-level) | Residue-level interface + attention maps |
| Bound-structure prediction | No | No (complementary to group work) |

## References

1. Luck, K. et al. A reference map of the human binary protein interactome. *Nature* 580, 402-408 (2020).
2. Rives, A. et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *Proc. Natl Acad. Sci. USA* 118, e2016239118 (2021).
3. Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379, 1123-1130 (2023).
4. Hayes, T. et al. Simulating 500 million years of evolution with a language model. *Science* 387, 850-858 (2025).
5. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583-589 (2021).
6. Liu, D. et al. PLM-interact: extending protein language models to predict protein-protein interactions. *Nat. Commun.* 16, 9012 (2025).
7. Cong, Q. et al. Protein interaction networks revealed by proteome coevolution. *Science* 365, 185-189 (2019).
8. Liu, Z. et al. PDB-wide collection of binding data: current status of the PDBbind database. *Bioinformatics* 31, 405-412 (2015).
9. Su, Q. et al. Robust protein-ligand interaction modeling through integrating physical laws and geometric knowledge for absolute binding free energy calculation. *Chem. Sci.* 16, 5043-5057 (2025).
10. Gainza, P. et al. Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning. *Nat. Methods* 17, 184-192 (2020).
11. Bryant, P., Pozzati, G. & Elofsson, A. Improved prediction of protein-protein interactions using AlphaFold2. *Nat. Commun.* 13, 1265 (2022).
12. Humphreys, I. R. et al. Computed structures of core eukaryotic protein complexes. *Science* 374, eabm4805 (2021).
13. Vassilev, L. T. et al. In vivo activation of the p53 pathway by small-molecule antagonists of MDM2. *Science* 303, 844-848 (2004).
14. Stukalov, A. et al. Multilevel proteomics reveals host perturbations by SARS-CoV-2 and SARS-CoV. *Nature* 594, 246-252 (2021).
15. Szklarczyk, D. et al. The STRING database of protein-protein association networks and functional enrichment analyses. *Nucleic Acids Research* (2023).
16. Ma, J. et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. *KDD* 2018.
17. Wang, X. et al. HoME: Hierarchy of Multi-gate Experts for Multi-Task Learning at Kuaishou. *KDD* 2025.
18. Yu, T. et al. Gradient surgery for multi-task learning. *NeurIPS* 2020.
