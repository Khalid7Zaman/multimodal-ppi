# Project Workflow Cheat-Sheet — multimodal-ppi

Your project lives in **three places** that must be kept in sync:

- **Cluster:** `~/Projects/multimodal-ppi` — where code runs
- **Gitee:** https://gitee.com/khalid7zaman/multimodal-ppi — online master + backup
- **Laptop:** `...\3. New project\multimodal-ppi` — local copy

Git is the "courier" that moves files between them. **Nothing moves until you run a command.**

---

## THE GOLDEN RULE

**Start every session with PULL. End every session with PUSH.**

---

## Start of a session (get the latest first)

```
cd ~/Projects/multimodal-ppi
git pull
```

## Save your work (end of a session)

```
cd ~/Projects/multimodal-ppi
git add -A
git commit -m "describe what you did today"
git push
```

Change the words inside the quotes each time, e.g. `git commit -m "trained first small model"`.

## Check where things stand

```
git status              # should say: up to date, working tree clean
git log --oneline -5    # your last 5 saved versions
```

---

## The same commands on the laptop

Open PowerShell, then:

```
cd "E:\Study\RAP-SUAT\Project and Meeting with PhD stds\Personal Projects\1. First Project\3. New project\multimodal-ppi"
git pull        # before you start
git add -A
git commit -m "..."
git push        # after you finish
```

---

## If it asks for username / password

- Username: `khalid7zaman`
- Password: your **Gitee token** (NOT your login password)

## Safety notes

- Never share your Gitee or GitHub token with anyone.
- Big files (data, trained models) are ignored on purpose — see `.gitignore`.
- If a push is **rejected**, run `git pull` first, then `git push` again.
