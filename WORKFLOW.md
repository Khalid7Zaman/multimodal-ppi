# Project Workflow Cheat-Sheet — multimodal-ppi

Your project lives in **four places** that are kept in sync:

- **Cluster:** `~/Projects/multimodal-ppi` — where the code runs (GPU jobs via SLURM)
- **Laptop:** `E:\Study\RAP-SUAT\Project and Meeting with PhD stds\Personal Projects\1. First Project\2. Protein-Protein-Interaction\multimodal-ppi` — local copy
- **Gitee:**  https://gitee.com/khalid7zaman/multimodal-ppi — online backup #1
- **GitHub:** https://github.com/Khalid7Zaman/multimodal-ppi — online backup #2

Git is the "courier" that moves files between them. **Nothing moves until you run a command.**

> **One `git push` now updates BOTH Gitee and GitHub at once** (`origin` has two push URLs).
> You do NOT push to them separately anymore.

---

## THE GOLDEN RULE

**Start every session with PULL. End every session with PUSH.**

---

## Start of a session (get the latest first)

```
cd ~/Projects/multimodal-ppi        # cluster  (on the laptop, use the laptop path above)
git pull
```

## Save your work (end of a session)

```
git add -A
git commit -m "describe what you did today"
git push                            # goes to BOTH Gitee and GitHub
```

Change the words in quotes each time, e.g. `git commit -m "trained first small model"`.

## Check where things stand

```
git status              # should say: up to date, working tree clean
git log --oneline -5    # your last 5 saved versions
git remote -v           # origin should show TWO (push) lines: gitee + github
```

---

## Credentials

- **Gitee:**  username `khalid7zaman`, password = your **Gitee token**.
- **GitHub:** username `Khalid7Zaman`, password = your **GitHub token** (a Personal Access Token).
  - Laptop remembers it via **Windows Credential Manager** after the first sign-in.
  - Cluster remembers it via `git config credential.helper store` after the first push.
- **Never paste a token into a chat or share it.** If one leaks, regenerate it in the site's
  settings; the new one is picked up on the next sign-in (laptop) or the next push (cluster).

## One-time setup of the two-remote push (already done — kept here for reference / new machines)

```
git remote set-url --add --push origin https://gitee.com/khalid7zaman/multimodal-ppi.git
git remote set-url --add --push origin https://github.com/Khalid7Zaman/multimodal-ppi.git
```

## Safety notes

- Big files (data, trained models) are ignored on purpose — see `.gitignore`.
- If a push is **rejected**, run `git pull` first, then `git push` again.
