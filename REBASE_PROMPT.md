# Monthly upstream rebase — prompt

Paste the block below into Claude Code from `D:\source\open-webui`.
Read [`FORK.md`](FORK.md) first if you want the background yourself.

---

```
Rebase this open-webui fork onto the latest upstream. Read FORK.md first — it is
the manifest of every patch we carry, why each exists, and the condition under
which it should be deleted rather than rebased.

Ground rules:

- The PR branches are the source of truth: fix/no-double-split,
  feat/external-text-splitter, feat/loader-metadata, perf/reuse-embeddings.
  perf/reuse-embeddings stacks on fix/no-double-split — rebase that one first.
  deploy/gpa is regenerated from them, not edited directly.
- Upstream PRs target `dev`, so rebase onto origin/dev.
- Create a dated safety branch before touching anything, and tell me its name.
- Do not force-push anything. Do not touch backup/* branches.

Do this:

1. Fetch upstream. Report how far origin/dev moved and what it touched in the
   files we patch: routers/retrieval.py, config.py, Documents.svelte,
   retrieval/vector/main.py, retrieval/vector/dbs/pgvector.py,
   retrieval/vector/async_client.py.

2. BEFORE rebasing, check each patch against FORK.md's "Delete when" condition.
   Upstream has superseded our work twice already, so actually look:
   - Has upstream implemented an equivalent? Check the real code, not the
     changelog.
   - Has an upstream PR of ours merged? Check by content, not just by number.
   - Is any patch now dead code? The X-File-Metadata patch ran for months
     reading a key nothing ever wrote.
   Report a keep/drop recommendation per patch and WAIT for my decision before
   dropping anything.

3. Rebase each surviving PR branch onto origin/dev, one at a time. The recurring
   conflict is mechanical: our helper functions land at the same insertion point
   in routers/retrieval.py, and the filter_metadata/process_metadata import line.
   Keep both sides. If a conflict is anything other than that, stop and show me.

4. Rebuild deploy/gpa as a linear rebase of the four branches onto origin/dev,
   then re-apply the "docs: fork manifest" commit carrying FORK.md and
   REBASE_PROMPT.md on top. Verify with `git diff <old-deploy> deploy/gpa` — if
   the rebase preserved everything, that diff is empty apart from intended
   changes. Report it either way.

5. Verify. Do not report success without running these:
   - ruff format --check and ruff check --select=F on our changed .py files
     (install ruff into a scratch venv; it is not in the repo env)
   - npm run format  — must not touch OUR files. It WILL dirty four unrelated
     upstream files; revert those, do not commit them.
   - npm run i18n:parse — must add ZERO of our keys. It will add unrelated
     pre-existing drift; exclude that, do not commit it. FORK.md lists the
     drift keys as of the last rebase.
   - npm run build and npx vitest run
   - python -m py_compile on our changed .py files
   Node note: .npmrc pins Node <=22; on newer Node use
   `npm ci --engine-strict=false`.

6. Update FORK.md: line counts, upstream status per patch, the current
   known-red-CI list, and anything you learned. If a patch was dropped, move it
   to the "Removed" table with the reason.

7. Report: what moved upstream, what conflicted and how you resolved it, what
   verification actually ran and its result, and anything you could not verify.
   Flag any patch whose upstream case got weaker or stronger this month.

Never run `git stash pop` in this repo without checking `git stash list` first —
there are pre-existing stashes and popping the wrong one has happened before.
```

---

## After the rebase

Rebuild and push the image (see [`BUILD_AND_PUSH.md`](BUILD_AND_PUSH.md)):

```bash
git checkout deploy/gpa && docker build -t ghcr.io/arnold256/open-webui:$(git rev-parse --short HEAD) .
```

Then run the smoke checks listed at the end of `FORK.md` against the stack in
`D:\source\Parser`.

## When a PR merges upstream

Drop the branch, delete its section from `FORK.md`, add a row to the "Removed"
table, and rebuild `deploy/gpa` without it. That is the goal — this fork should
shrink to nothing.
