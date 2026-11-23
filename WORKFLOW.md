# Project Workflow Cheat-Sheet: multimodal-ppi

**Author:** Khalid Zaman, Research Assistant Professor (RAP).  **Supervisor:** Prof. Zhaoxi Sun, SUAT.

The project lives in **five places** that are kept in sync:

| Place | Location | Role |
|---|---|---|
| New laptop (DESKTOP-6FQV95D) | `D:\Study\RAP-SUAT\Project and Meeting with PhD stds\Personal Projects\1. First Project\2. Protein-Protein-Interaction\multimodal-ppi` | main working copy; reaches Gitee and GitHub |
| Old laptop (Dr-Zaman) | `E:\Study\RAP-SUAT\Project and Meeting with PhD stds\Personal Projects\1. First Project\2. Protein-Protein-Interaction\multimodal-ppi` | second working copy; reaches Gitee and GitHub |
| Cluster (Veda, user `zaman`) | `/home/zaman/Projects/multimodal-ppi` | where the code runs (GPU jobs via SLURM); reaches Gitee only |
| Gitee | https://gitee.com/khalid7zaman/multimodal-ppi | online copy #1 |
| GitHub | https://github.com/Khalid7Zaman/multimodal-ppi | online copy #2 |

Git is the "courier" that moves files between them. **Nothing moves until you run a command.**
The two laptops differ only in drive letter (D: and E:); git does not care about the drive letter.

---

## THE GOLDEN RULE

**Start every session with PULL. End every session with PUSH.**

---

## Start of a session (get the latest first)

```
git pull
```

Run it inside the project folder (the paths in the table above).

## Save your work (end of a session)

```
git add -A
git commit -m "describe what you did today"
git push
```

- **On a laptop:** `git push` updates BOTH Gitee and GitHub at once (`origin` has two push addresses).
- **On the cluster:** `git push` reaches Gitee only. The GitHub part times out, because the cluster
  network cannot reach github.com. This is expected.

## Getting cluster work onto GitHub

1. Cluster: `git push` (lands on Gitee).
2. Either laptop: `git pull` (fetches it from Gitee), then `git push` (sends it on to GitHub).

## Check where things stand

```
git status              # should say: up to date, working tree clean
git log --oneline -5    # your last 5 saved versions
git remote -v           # origin should show TWO (push) lines: gitee + github
```

---

## What goes online and what stays private

- **Online (Gitee and GitHub):** code, results, analysis reports and figures.
- **Laptops only:** the research proposal, the roadmap, the Phase C plan, related work, the archive and
  any other private planning material.
- `docs/PROPOSAL.md`, `docs/ROADMAP.md` and `docs/PHASE_C_DATA_ENGINE.md` are listed in `.gitignore`.
  They exist on both laptops but are never uploaded. **Never use `git add -f` on them.**
- When the proposal changes, copy the updated files to the other laptop yourself (USB or cloud folder);
  git will not move them.
- Copies of the repository made before the private files were removed must never be pushed. If an old
  copy turns up, delete it and download a fresh one with `git clone`.

## Credentials

- **Gitee:** username `khalid7zaman` (all lower case), password = your **Gitee token**.
- **GitHub:** username `Khalid7Zaman`, sign in through the browser window, or use your **GitHub token**
  as the password. Use the username, not an e-mail address.
- Laptops remember logins via **Windows Credential Manager** after the first sign-in.
- Cluster remembers logins via `git config credential.helper store` after the first push.
- **Never paste a token into a chat or share it.** If one leaks, regenerate it in the site's settings.

## One-time setup of the two-address push (already done; kept for new machines)

```
git remote set-url --add --push origin https://gitee.com/khalid7zaman/multimodal-ppi.git
git remote set-url --add --push origin https://github.com/Khalid7Zaman/multimodal-ppi.git
```

## Safety notes

- Big files (data, trained models) are ignored on purpose: see `.gitignore`.
- If a push is **rejected**, run `git pull` first, then `git push` again.
