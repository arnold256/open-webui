# Monthly upstream rebase — prompt

Paste the block below into Claude Code from `D:\source\open-webui`. It covers the
whole cycle: sync upstream, rebase the patches, verify, and publish to both
remotes. Publishing is what deploys — pushing `deploy/gpa` to `devops` triggers
the Azure DevOps pipeline that builds the image into Harbor, and the platform
pipeline builds the deployed image from that.

Read [`FORK.md`](FORK.md) first if you want the background yourself.

**The short version**, when you do not want to paste the whole thing:

> Do the monthly rebase for this fork: follow REBASE_PROMPT.md end to end,
> including the publish step, and stop where it says to stop.

Claude still has to stop twice — once for the keep/drop decision on each patch,
once if a conflict is anything other than the known mechanical one — so this is
not unattended.

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
- The only force-push allowed is `--force-with-lease` onto our own PR branches
  and deploy/gpa on the GitHub fork in step 8, which a rebase makes unavoidable.
  Never force-push anything on `origin` (that is upstream) and never touch
  backup/* branches at all.

Do this:

1. Fetch every remote — origin (upstream), open-webui-(GPA) (our GitHub fork),
   devops (Azure DevOps, deploy/gpa only). Report how far origin/dev moved and
   what it touched in the files we patch: routers/retrieval.py, config.py,
   Documents.svelte, retrieval/vector/main.py, retrieval/vector/dbs/pgvector.py,
   retrieval/vector/async_client.py.

   Fast-forward the GitHub fork's main to upstream while you are there, so the
   fork does not sit visibly behind. If it will not fast-forward, leave it and
   say so — never force it.

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
   then re-apply the release-only commits on top — FORK.md's "Release-only
   commits" table lists them, currently the fork manifest docs, azure-pipelines.yml
   and the NODE_OPTIONS line in Dockerfile. Losing any of them breaks the release
   path while every local build keeps working, so check they are present rather
   than assuming the rebase carried them. Verify with
   `git diff <old-deploy> deploy/gpa` — if the rebase preserved everything, that
   diff is empty apart from intended changes. Report it either way.

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

8. Publish, once step 5 actually passed. Pushing is the release — there is no
   separate deploy step and no docker build by hand.

   - Push each surviving PR branch to `open-webui-(GPA)`. These were rebased, so
     they need `--force-with-lease`; that is expected on PR branches and only
     there.
   - Push `deploy/gpa` to BOTH `open-webui-(GPA)` and `devops`. The `devops`
     remote is Azure DevOps Server, and that push triggers the pipeline in
     `azure-pipelines.yml`, which builds the image and pushes it to Harbor as
     `registry.gpaeng.com.au/openwebui/open-webui-base:deploy-gpa` plus an
     immutable build-number tag.
   - Watch that build to completion and report its number and result. It is a
     long one: full npm build plus the Python layer. Do not report success from
     a queued build.
   - Then trigger `openwebui-platform CI` (the pipeline in the openwebui-platform
     repository) or tell me to. It builds the deployed image FROM the base you
     just published, and its promote stage writes the new tag into the gpaadlk3s
     overlay, which is what Fleet rolls out.

   If a push to `devops` is refused, say so rather than working around it — that
   remote is the deployment path and a silent fallback to the GHCR build means
   the cluster quietly keeps running the old image.

Never run `git stash pop` in this repo without checking `git stash list` first —
there are pre-existing stashes and popping the wrong one has happened before.
```

---

## After the rebase

Step 8 above already published it. What is left is confirming the thing that
actually runs changed:

```bash
git push open-webui-\(GPA\) deploy/gpa
git push devops deploy/gpa                 # this is the one that builds
```

Then run the smoke checks listed at the end of `FORK.md` against the deployed
stack, not just a local one. The image reports the commit it was built from —
the pipeline passes `BUILD_HASH` — so the About panel is the quickest proof the
rollout actually landed.

[`BUILD_AND_PUSH.md`](BUILD_AND_PUSH.md) covers building by hand, which is now
the fallback for when the agent is down rather than the normal path.

## When a PR merges upstream

Drop the branch, delete its section from `FORK.md`, add a row to the "Removed"
table, and rebuild `deploy/gpa` without it. That is the goal — this fork should
shrink to nothing.
