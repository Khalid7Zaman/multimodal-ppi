# Vendored teacher model: ProteinAffinityKD

Sequence-only protein-protein binding-affinity predictor (a sequence-based student distilled from a
structure-informed teacher). Used here as one distillation teacher for Phase C.

- Upstream: https://github.com/wajidarshad/ProteinAffinityKD
- Reference: Wajid et al., Investigating knowledge distillation through neural networks for protein
  binding affinity prediction. arXiv:2601.03704 (2026).
- Licence: MIT (upstream). These files are redistributed here only to make the cluster run reproducible
  without reaching GitHub.
- Files: student_with_diss_weights.pth (student weights), scaler_propy.pkl (feature StandardScaler),
  scaler_labels.pkl (label StandardScaler), key_list_proPy.npy (the fixed 1537 Propy feature keys).
- Output: binding free energy in kcal/mol (more negative = stronger binding).
- Inference wrapper: phase_c/teacher_affinity_kd.py
