#!/usr/bin/env python
"""
Phase 3 - PPB multi-task dataset: assemble everything the model needs, per complex.

For each PPB-Affinity complex (a row of ppb_<split>_iface.csv) this provides, for BOTH proteins
(receptor = "query", ligand = "text"):
  - ESM-2 token ids + attention mask            (sequence view)
  - a residue contact map, token-aligned         (structure view; zeros if no aligned structure)
  - per-residue conservation features, token-aligned  (evolution view; zeros if no MSA)
plus the targets:
  - pKd                                           (binding-affinity target)
  - per-residue interface labels, token-aligned   (interface target; -100 = ignore where unknown)

ALIGNMENT CONTRACT (matches model.py): every per-residue tensor is indexed to the SAME token
length L as the ESM tokens. ESM adds a start token (position 0) and an end token (position L-1);
residues occupy positions 1..n. Those special-token and padding positions carry zero structure,
zero evolution, and interface label -100, so the model treats them as "no info" - exactly as
struct_module / evo_features were designed to handle.

Reuses: struct_features (contact maps), evo_features (MSA conservation), collect_ppb_seqs.seq_id
(to find each protein's .a3m produced by msa_search.sh).

Quick self-test (works even before the full MSAs finish - evolution just falls back to zeros):
    python phase3/data.py --split train --n 3
"""
import os, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from struct_features import (load_model, get_residues, build_contact_map, chains_of)
from evo_features import features_for_a3m, FEAT_DIM
from collect_ppb_seqs import seq_id

DATA    = os.path.expanduser("~/Projects/ppi-data")
PPB     = os.path.join(DATA, "affinity", "ppb")
MSA_A3M = os.path.join(DATA, "msa", "full", "a3m")   # per-protein MSAs from msa_search.sh


def _contact_map_for(model, seq, chain_ids, threshold=8.0):
    """Contact map [n,n] for the chains, only if it lines up 1:1 with `seq`; else None."""
    if model is None:
        return None
    res = get_residues(model, chains_of(chain_ids))
    if len(res) != len(str(seq)):        # structure doesn't match sequence -> skip structure view
        return None
    A, _ = build_contact_map(res, threshold)
    return A


class PPBComplexData(Dataset):
    def __init__(self, split, tokenizer, max_length=512, contact_threshold=8.0, use_msa=True):
        self.df = pd.read_csv(os.path.join(PPB, f"ppb_{split}_iface.csv")).reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length
        self.max_res = max_length - 2          # room for start/end tokens
        self.thr = contact_threshold
        self.use_msa = use_msa

    def __len__(self):
        return len(self.df)

    def _one_protein(self, seq, contactA, iface_str):
        """Build token-aligned tensors for one protein."""
        seq = str(seq)
        sid = seq_id(seq)                       # id is from the FULL sequence (matches the .a3m)
        seq_t = seq[:self.max_res]              # truncate for tokens/features
        n = len(seq_t)

        enc = self.tok(seq_t, truncation=True, max_length=self.max_length)
        ids  = torch.tensor(enc["input_ids"], dtype=torch.long)
        mask = torch.tensor(enc["attention_mask"], dtype=torch.float)
        L = ids.shape[0]                        # token length (= n + 2 for ESM-2)

        # structure -> token-aligned contact map [L, L]
        adj = torch.zeros(L, L, dtype=torch.float)
        if contactA is not None and contactA.shape[0] > 0:
            m = min(n, contactA.shape[0])
            adj[1:1 + m, 1:1 + m] = torch.from_numpy(contactA[:m, :m])

        # evolution -> token-aligned conservation [L, 22]
        evo = torch.zeros(L, FEAT_DIM, dtype=torch.float)
        if self.use_msa:
            feats = features_for_a3m(os.path.join(MSA_A3M, f"{sid}.a3m"))
            if feats is not None and feats.shape[0] > 0:
                m = min(n, feats.shape[0])
                evo[1:1 + m] = torch.from_numpy(feats[:m])

        # interface labels -> token-aligned [L], -100 where unknown (start/end/pad or no label)
        iface = torch.full((L,), -100.0, dtype=torch.float)
        if isinstance(iface_str, str) and len(iface_str) > 0 and set(iface_str) <= set("01"):
            lab = np.frombuffer(iface_str[:n].encode(), dtype=np.uint8) - ord("0")
            m = min(n, len(lab))
            iface[1:1 + m] = torch.from_numpy(lab[:m].astype(np.float32))

        return {"ids": ids, "mask": mask, "adj": adj, "evo": evo, "iface": iface}

    def __getitem__(self, i):
        row = self.df.iloc[i]
        model = load_model(str(row["pdb"]))     # one structure load for both chains
        rec = self._one_protein(row["query"], _contact_map_for(model, row["query"], row["receptor_chains"], self.thr), row.get("iface_query", ""))
        lig = self._one_protein(row["text"],  _contact_map_for(model, row["text"],  row["ligand_chains"],   self.thr), row.get("iface_text", ""))
        pkd = float(row["pKd"]) if not pd.isna(row.get("pKd", np.nan)) else float("nan")
        return {"rec": rec, "lig": lig, "pkd": torch.tensor(pkd, dtype=torch.float)}


def _pad_protein(items):
    """Pad a list of single-protein dicts to the batch's max token length."""
    L = max(x["ids"].shape[0] for x in items)
    B = len(items)
    ids   = torch.zeros(B, L, dtype=torch.long)
    mask  = torch.zeros(B, L, dtype=torch.float)
    adj   = torch.zeros(B, L, L, dtype=torch.float)
    evo   = torch.zeros(B, L, FEAT_DIM, dtype=torch.float)
    iface = torch.full((B, L), -100.0, dtype=torch.float)
    for b, x in enumerate(items):
        l = x["ids"].shape[0]
        ids[b, :l] = x["ids"]; mask[b, :l] = x["mask"]
        adj[b, :l, :l] = x["adj"]; evo[b, :l] = x["evo"]; iface[b, :l] = x["iface"]
    return {"ids": ids, "mask": mask, "adj": adj, "evo": evo, "iface": iface}


def collate(batch):
    return {
        "rec": _pad_protein([b["rec"] for b in batch]),
        "lig": _pad_protein([b["lig"] for b in batch]),
        "pkd": torch.stack([b["pkd"] for b in batch]),
    }


def _self_test(split="train", n=3):
    from transformers import AutoTokenizer
    esm = os.path.expanduser("~/Projects/PLM-interact/offline/esm2_t12_35M_UR50D")
    tok = AutoTokenizer.from_pretrained(esm)
    ds = PPBComplexData(split, tok, max_length=512)
    print(f"{split}: {len(ds)} complexes")
    batch = collate([ds[i] for i in range(n)])
    for side in ("rec", "lig"):
        p = batch[side]
        struct = (p["adj"].sum(dim=(1, 2)) > 0).sum().item()
        evo = (p["evo"].sum(dim=(1, 2)) > 0).sum().item()
        iface = (p["iface"] >= 0).any(dim=1).sum().item()
        print(f"  {side}: ids {tuple(p['ids'].shape)}  adj {tuple(p['adj'].shape)}  evo {tuple(p['evo'].shape)}"
              f"  | with-structure {struct}/{n}  with-MSA {evo}/{n}  with-iface {iface}/{n}")
    print("  pKd:", batch["pkd"].tolist())
    print("\nOK: PPB multi-task dataset assembles sequence + structure + evolution + targets.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    _self_test(args.split, args.n)
