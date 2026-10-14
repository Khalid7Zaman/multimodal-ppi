# Models and Weights: Pretrained Backbones and Fine-Tuned Checkpoints

**Author:** Khalid Zaman, Research Assistant Professor (RAP) **Supervisor:** Prof. Zhaoxi Sun, SUAT
**Project:** multimodal-ppi (Protein-Protein Interaction),   2026-10-14

This note records how the pretrained models are used in the project, which weights are original and
which are our own, and where each of them is kept. It is written so the choice can be explained and
defended directly.

## The starting point: pretrained protein language models

The sequence backbone of the model is a protein language model. We use two of them, ESM-2 (650
million parameters) and ESM-C (600 million parameters). Both were pretrained by their original
authors on hundreds of millions of protein sequences, so their weights already carry a general
understanding of protein sequence before we begin. We did not train a protein language model from
scratch. Building one of this size is far beyond the compute available to a single project, and it
is unnecessary when strong pretrained models are published and freely available.

We downloaded the original released versions of both models and stored them, unchanged, on the
cluster. These downloads are the exact weights as published. They are the fixed starting point for
every training run, and they are never overwritten.

## What we did: fine-tuning (transfer learning)

Rather than use the pretrained model as a frozen feature extractor, we fine-tuned it end to end on
our own task. In practice this means each training run loads a copy of the original pretrained
weights and then continues training them on our data: the Bernett interaction pairs for the
interaction model, and the PPB-Affinity complexes for the affinity and interface tasks. During
training the weights are adjusted so that the model becomes good at our specific problem, and the
adjusted weights are then saved as a new checkpoint. The original download is left exactly as it was.

End to end means that the backbone's own weights are updated during training, not only the small
prediction heads placed on top of it. This is a deliberate choice. It gives the model more freedom
to adapt to protein-protein interaction, at the cost of more compute than freezing the backbone
would need. In our experiments the fine-tuned model clearly outperforms the frozen alternative,
which is why we adopted it.

## Two sets of weights, both preserved

At the end there are two distinct sets of weights, and both are kept:

| Weights | What they are | Where they live |
|---|---|---|
| Original pretrained | ESM-2 and ESM-C exactly as released by their authors | Cluster, offline model folders. Never modified. |
| Our fine-tuned checkpoints | The models we trained on our task | `phase2/runs/.../best.pt` (interaction) and `phase3/runs/.../best.pt` (multimodal) |

For each training run we save the checkpoint that scored best on the validation split, and for the
repeated runs we keep all three seeds. These checkpoints are large binary files, so they remain on
the cluster and are deliberately kept out of the git repository. The code, the configuration and the
exact commands that produced them are all in the repository, so any checkpoint can be regenerated.

## Reproducibility

Every run fixes its random seed (0, 1 and 2 for the repeated runs), and the best checkpoint is saved
automatically. Because the seeds are fixed and the checkpoints are stored, a given model can be
reloaded and its reported numbers reproduced exactly. The repeated seeds are also what give the
results their error bars, reported as the mean and standard deviation across three runs.

## In one sentence

We use ESM-2 and ESM-C as pretrained backbones, keep their original released weights untouched, and
fine-tune a copy of them end to end on our interaction, affinity and interface data. The fine-tuned
weights are saved as separate checkpoints, so the original model and our trained model both exist
independently. This is standard transfer learning, the same approach used by the base model
PLM-interact, and it is chosen because it reuses the knowledge already learned from hundreds of
millions of sequences and is far more data-efficient and accurate than training a protein language
model from scratch.
