# Deploying the demo

Two routes. **Static is the recommended one** — it is free, instant, and the resulting demo is
arguably a better artefact than the server version.

---

## Route A — Static Space (free, recommended)

Hugging Face charges for Docker and Gradio Spaces; **Static Spaces are free for everyone**. That
constraint turned out to suit this project: the distilled guardrail is a logistic model over hashed
n-grams plus an isotonic step function, so the entire inference path is a sparse dot product. There
is nothing in it that needs a server.

`site/` therefore contains a JavaScript port of the student that produces **the same numbers as
scikit-learn** — `site/parity.test.js` asserts agreement to within 1e-6 on 30 fixtures exported from
Python, and it runs in CI. The LLM `/evaluate` path is dropped rather than faked, because a static
page cannot hold an API key.

What you gain over the Docker route: no cold start, no sleeping, $0 forever, and the line *"the
guardrail is so cheap it runs in your browser"*.

### Build

```bash
make site        # export model → assemble site/ → run the parity test
make site-serve  # preview at http://localhost:8000
```

### Create the Space

huggingface.co/new-space →

- **Owner:** `RugFace` · **Name:** `judgeguard`
- **SDK: Static**
- Public

### Push

`site/README.md` already carries the required frontmatter (`sdk: static`, `app_file: index.html`).
The Space wants those files at its repo *root*, so push the `site/` subtree:

```bash
git remote add space https://huggingface.co/spaces/RugFace/judgeguard
git subtree push --prefix site space main
```

If HF asks for credentials: username `RugFace`, password = an access token from
huggingface.co/settings/tokens with **write** scope.

To update later, rebuild and push the subtree again:

```bash
make site
git add -A && git commit -m "rebuild static site"
git subtree push --prefix site space main
```

If `git subtree push` ever rejects (it can after a force-push), rebuild the branch:

```bash
git push space `git subtree split --prefix site main`:main --force
```

### Verify

Open `https://RugFace-judgeguard.static.hf.space`, click through the five examples, and confirm:

- reference → **allow**
- hedging, omission, padded → **block**
- numeric swap → **allow**, at a probability identical to the reference (the documented blind spot)

Then put the URL in the GitHub README's `## Try it` section and in LinkedIn Featured.

---

## Route B — Docker Space (requires HF PRO, ~$9/month)

Only worth it if you want the **LLM judge path** (`POST /evaluate`) live as well. Everything is
already built; nothing changes in the code.

1. Create the Space with **SDK: Docker**, CPU basic.
2. Use `deploy/hf-space/README.md` (the Docker Space card, `app_port: 8080`) as the Space README.
3. Push the whole repo:

```bash
git remote add space https://huggingface.co/spaces/RugFace/judgeguard
git checkout -b hf-space
cp deploy/hf-space/README.md README.md
git add README.md && git commit -m "hf: docker space card"
git push space hf-space:main
git checkout main
```

4. Add keys under **Settings → Variables and secrets**: `JUDGEGUARD_PROVIDER_MODE=auto` as a
   *variable*, API keys as *secrets*. With no keys it still works, serving committed results.

Notes: `app_port: 8080` must match the Dockerfile's `EXPOSE`. Build takes 3–5 minutes; free Spaces
sleep after inactivity and wake in ~30 s.

---

## Route C — full app on another free host

If you want the FastAPI service live without paying HF, Render / Koyeb / Railway all have free
tiers and will build the existing `Dockerfile` directly. They sleep aggressively and change terms
often, and they are not where ML people browse — which is the main argument for Route A.
