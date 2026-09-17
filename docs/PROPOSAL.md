# A Multimodal Deep-Learning Framework for Predicting Protein–Protein Interactions and Their Drug-Relevant Properties

> Research Proposal — **v10**. Mirrored into the repo (`docs/PROPOSAL.md`) from
> `10. Research Proposal PPI v10.docx` so every session/tab can read it.
> Author: Khalid Zaman (RAP). Supervisor: Prof. Zhaoxi Sun, SUAT.
> Source `.docx` location on the laptop:
> `E:\Study\RAP-SUAT\Project and Meeting with PhD stds\Personal Projects\1. First Project\2. Protein-Protein-Interaction`

## Abstract

Protein–protein interactions organise almost everything a cell does, and disrupting them is a proven
route to new medicines. Predicting which proteins interact and how strongly they bind is therefore a
central goal of computational biology. Sequence-only models have advanced the field but they miss the
structural signal that governs binding, and the labelled data available to train them are limited. This
proposal develops a multimodal deep-learning model for protein–protein interaction. From the sequences of
two proteins it predicts whether they interact, how tightly they bind, and which residues form the
interface. The model combines three views of each protein. A protein language model reads the sequence. A
structural module draws on experimentally determined complex structures and measured binding affinities
from the PDBbind database. An evolutionary module adds conservation from sequence alignments. A
cross-attention layer then reasons across the pair and exposes the contacts it relies on. Binding affinity
and the interface it identifies are the drug-relevant outputs, since together they show whether an
interaction is a viable target and where to act on it. The model is benchmarked against leading
sequence-based predictors, and it stays complementary to the separate and harder task of predicting a full
bound complex, which the group already studies.

## 1. Introduction and Background

Proteins rarely act alone. They work by binding to one another and so build the complexes and signalling
chains that keep a cell alive. When a binding surface fails the result is often disease. This is also what
makes these surfaces attractive drug targets, since a molecule that blocks the right interaction can
correct the fault. Mapping protein–protein interactions and understanding how tightly proteins bind has
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
model of protein–protein binding needs to learn from. Related work has already used such data for
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

Current methods for predicting protein–protein interactions fall short in four linked ways. Models built
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
protein–protein interactions, their binding strength and their interface residues from sequence, grounded
in experimentally determined complex structures and measured affinities from PDBbind.

The central **hypothesis** is that joining protein-language-model representations with experimentally
grounded structure and evolutionary information will predict interactions, binding strength and interfaces
more accurately and more generally than a model trained on sequence alone.

The aim will be met through five **objectives**:

1. Assemble a training resource for protein–protein interaction from public interaction datasets together
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

Figure 1 tells the whole framework of the model in one view. Two protein sequences move through five
stages: an input layer; a multimodal encoder that joins sequence, structural and evolutionary
representations; a cross-attention reasoning engine that produces a residue-level interaction map; a
multi-task head that predicts interaction, binding affinity and interface residues; and training on
experimentally grounded data from PDBbind and public interaction datasets.

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

The project draws on three public sources. Interaction labels come from established protein–protein
interaction datasets, including the sets released with PLM-interact and the leakage-free benchmark used to
test it fairly [6]. Complex structures and measured binding affinities come from PDBbind [8]. Evolutionary
information comes from multiple sequence alignments. Using experimental structures and affinities keeps the
training signal trustworthy and removes any dependence on predicted structures.

### 6.3 Model and training

The sequence module fine-tunes an ESM-2 backbone on paired sequences [3,6]. The structural module encodes
each complex from PDBbind as a residue contact graph and processes it with a graph network. The
evolutionary module encodes conservation from alignments. These three views are joined and passed to the
cross-attention module, and the shared output feeds the interaction, affinity and interface heads under one
combined objective. Training is supervised throughout, using interaction labels and the measured affinities
in PDBbind. Mixed precision and gradient checkpointing keep the full-size model within the memory of the
available GPUs.

### 6.4 Evaluation

Interaction prediction is benchmarked against PLM-interact and its published baselines on shared held-out
data [6]. Binding-affinity prediction is measured on held-out PDBbind complexes and compared with
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
| 3 | Structural and evolutionary modules | PDBbind structures and conservation joined with sequence; measured gains |
| 4 | Affinity and interface heads | Binding-affinity and interface-residue predictions |
| 5 | Benchmarking and analysis | Full comparison, ablations and interpretability study |
| 6 | Dissemination | Manuscript and public code release |

## 7. Novelty and Contributions

The novelty lies in combining three things that have not been brought together for protein–protein
interaction. First the project grounds the model in the experimentally measured structures and affinities
of PDBbind, rather than in predicted structures, so its training signal is trustworthy [8]. Second it joins
sequence, structure and evolution in a single encoder and reasons across the pair with cross-attention, so
structure and interaction are learned together rather than apart. Third it predicts interaction, binding
strength and the interface in one interpretable model, and it does so as a partner to bound-structure
prediction rather than a rival. The result is not a single new module but a coherent system aimed at the
outputs that matter for finding and understanding protein–protein drug targets.

## 8. Significance of the Study

Grounding the model in experimental structures and affinities should make its predictions both more
accurate and more useful than sequence-only methods. By reporting binding strength and the interface
alongside interaction, the framework speaks straight to drug discovery, where blocking a specific protein
surface with a small molecule is already a proven therapeutic strategy [13]. Knowing how strongly two
proteins bind and which residues to target is precisely what turns an interaction into a candidate for
intervention.

The framework is also built to be used. It complements the group's structure-prediction work and draws on
shared, public data. Its reach extends to host–pathogen settings, where knowing how viral proteins seize
human ones matters for anticipating infection [14]. A model that predicts interaction, strength and
interface from sequence, backed by experimental structures, would be a practical tool for both basic
research and target discovery.

## 9. Expected Outcomes

The project should deliver a multimodal deep-learning model that predicts protein–protein interaction,
binding strength and interface residues better than sequence-only baselines. It should also deliver a
clean, reusable pipeline that pairs public interaction data with PDBbind structures and affinities. The
affinity predictions should agree with held-out measurements, and the interface predictions should match
known complexes. Alongside these the work will produce a residue-level interpretability layer and an openly
released codebase and trained models. Strong results in this field can never be promised in advance, so the
project is scoped such that every phase yields a self-contained result and the ambition of the final paper
matches the strength of the findings.

## 10. Positioning Against the Base Model

**Table 2. The proposed framework set against PLM-interact** across input, data and prediction.

| Dimension | PLM-interact | Proposed framework |
|---|---|---|
| Sequence representation | ESM-2 | ESM-2 / ESM-3 |
| Structural information | None | Experimental complexes (PDBbind) |
| Evolutionary information | None | Multiple sequence alignments |
| Pair reasoning | Limited | Cross-attention transformer |
| Interaction prediction | Yes | Yes |
| Binding strength | No | Yes (from PDBbind affinities) |
| Interface prediction | No | Yes |
| Interpretability | Limited | Residue-level |
| Bound-structure prediction | No | No (complementary to group work) |

## References

1. Luck, K. et al. A reference map of the human binary protein interactome. *Nature* 580, 402–408 (2020).
2. Rives, A. et al. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *Proc. Natl Acad. Sci. USA* 118, e2016239118 (2021).
3. Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379, 1123–1130 (2023).
4. Hayes, T. et al. Simulating 500 million years of evolution with a language model. *Science* 387, 850–858 (2025).
5. Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589 (2021).
6. Liu, D. et al. PLM-interact: extending protein language models to predict protein–protein interactions. *Nat. Commun.* 16, 9012 (2025).
7. Cong, Q. et al. Protein interaction networks revealed by proteome coevolution. *Science* 365, 185–189 (2019).
8. Liu, Z. et al. PDB-wide collection of binding data: current status of the PDBbind database. *Bioinformatics* 31, 405–412 (2015).
9. Su, Q. et al. Robust protein–ligand interaction modeling through integrating physical laws and geometric knowledge for absolute binding free energy calculation. *Chem. Sci.* 16, 5043–5057 (2025).
10. Gainza, P. et al. Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning. *Nat. Methods* 17, 184–192 (2020).
11. Bryant, P., Pozzati, G. & Elofsson, A. Improved prediction of protein–protein interactions using AlphaFold2. *Nat. Commun.* 13, 1265 (2022).
12. Humphreys, I. R. et al. Computed structures of core eukaryotic protein complexes. *Science* 374, eabm4805 (2021).
13. Vassilev, L. T. et al. In vivo activation of the p53 pathway by small-molecule antagonists of MDM2. *Science* 303, 844–848 (2004).
14. Stukalov, A. et al. Multilevel proteomics reveals host perturbations by SARS-CoV-2 and SARS-CoV. *Nature* 594, 246–252 (2021).
