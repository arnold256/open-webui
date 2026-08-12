# Build & Push `open-webui` to GHCR

> **This is the fallback, not the release path.** Releasing is `git push devops
> deploy/gpa`, which triggers `open-webui-fork CI` and publishes to Harbor —
> see [`FORK.md`](FORK.md) § Build and release. Use the steps below when the
> build agent is unavailable, or when you want a local image to test against.
>
> An image published to GHCR **cannot be deployed as-is**. The package is
> private, and the Azure DevOps build agent has no GHCR credential, so the
> downstream `openwebui-platform CI` build fails on an anonymous 401. To ship a
> hand-built image, retag and push it into Harbor instead:
>
> ```powershell
> docker tag ghcr.io/arnold256/open-webui:$sha registry.gpaeng.com.au/openwebui/open-webui-base:deploy-gpa
> docker push registry.gpaeng.com.au/openwebui/open-webui-base:deploy-gpa
> ```

Steps to build the Docker image from this repo and push it to
`ghcr.io/arnold256/open-webui:latest`.

## Prerequisites

- **Docker Desktop is running** (check the whale icon in the system tray — it
  should say "Docker Desktop is running"). If not, start it and wait until it
  reports ready before continuing.
- A GitHub **classic** Personal Access Token with `write:packages` scope.
  - Create one at: https://github.com/settings/tokens/new
  - Scopes: tick `write:packages` (auto-selects `read:packages` + `repo`)
  - Copy the token immediately — it won't be shown again.
  - Note: fine-grained tokens don't work with GHCR; use a classic token.

> No local Python venv or `npm install` is required. The `Dockerfile`
> handles the frontend (`npm ci` + `npm run build`) and backend
> (`pip install -r backend/requirements.txt`) inside the image.

## 1. Login to GHCR (interactive)

From PowerShell in the repo root (`D:\source\open-webui`):

```powershell
docker login ghcr.io -u arnold256
```

When prompted for `Password:`, paste the PAT (input is hidden) and press Enter.

## 2. Build the image

```powershell
docker build -t ghcr.io/arnold256/open-webui:latest .
```

Optional `--build-arg` flags supported by the `Dockerfile`:

- `USE_CUDA=true` — bundle CUDA runtime deps
- `USE_OLLAMA=true` — bundle Ollama
- `USE_CUDA_VER=cu128` (default) / `cu126`
- `USE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `BUILD_HASH=<git sha>`

Example with build args + a commit-sha tag:

```powershell
$sha = git rev-parse --short HEAD
docker build `
  --build-arg BUILD_HASH=$sha `
  -t ghcr.io/arnold256/open-webui:latest `
  -t ghcr.io/arnold256/open-webui:$sha .
```

## 3. Push the image

```powershell
docker push ghcr.io/arnold256/open-webui:latest
```

If you also tagged with the sha:

```powershell
docker push ghcr.io/arnold256/open-webui:$sha
```

## 4. (Optional) Multi-arch build

```powershell
docker buildx create --use --name owui 2>$null
docker buildx build --platform linux/amd64,linux/arm64 `
  -t ghcr.io/arnold256/open-webui:latest --push .
```

## 5. (Optional) Make the package public

After the first push, manage visibility / access here:

https://github.com/users/arnold256/packages/container/open-webui/settings

## Troubleshooting

- **`docker: command not found` / build fails immediately** — Docker Desktop
  isn't running. Start it and retry.
- **`unauthorized` on push** — re-run `docker login ghcr.io -u arnold256`
  and confirm the PAT has `write:packages`.
- **Resolver / dependency errors during build** — clear the build cache:
  `docker builder prune -af` then rebuild.
- **Login error `password-stdin` on PowerShell** — the `echo $env:VAR | ...`
  pattern can fail in PowerShell; use the interactive login above instead.
