# Deploying to Hugging Face Spaces

Spaces require a `README.md` whose YAML frontmatter configures the Space. That frontmatter would
look wrong on the GitHub repo page, so the Space gets its own README — `deploy/hf-space/README.md`
in this repo — and it is swapped in during the push.

## One-time setup

1. Create the Space: huggingface.co/new-space
   - **Owner:** RyanSingh0 · **Name:** `judgeguard`
   - **SDK: Docker** (not Gradio, not Streamlit) · **Blank** template
   - Hardware: **CPU basic (free)**. The guardrail is CPU-bound and 4 ms; you do not need a GPU.

2. Add your keys under **Settings → Variables and secrets** (optional — the demo works without
   them, serving committed results):

   | name | value |
   |---|---|
   | `JUDGEGUARD_PROVIDER_MODE` | `auto` (variable, not secret) |
   | `GEMINI_API_KEY` | secret |
   | `GROQ_API_KEY` | secret |
   | `OPENROUTER_API_KEY` | secret |

   `auto` means: live for any provider whose key is present, simulated for the rest. If you add no
   keys the Space still works end to end.

## Push

```bash
# from the repo root, once
git remote add space https://huggingface.co/spaces/RyanSingh0/judgeguard

# swap in the Space README, push, swap back
cp README.md /tmp/gh-readme.md
cp deploy/hf-space/README.md README.md
git add README.md && git commit -m "hf: space card"
git push space main

cp /tmp/gh-readme.md README.md
git add README.md && git commit -m "restore github readme"
git push origin main
```

If HF asks for credentials, use your username and an access token from
huggingface.co/settings/tokens (write scope).

### Cleaner alternative — a dedicated branch

Avoids the swap dance entirely:

```bash
git checkout -b hf-space
cp deploy/hf-space/README.md README.md
git add README.md && git commit -m "hf: space card"
git push space hf-space:main
git checkout main
```

Repeat the last three lines whenever you want to update the Space.

## Notes

- `app_port: 8080` in the frontmatter matches the Dockerfile's `EXPOSE 8080`. If you change one,
  change both.
- The image ships `results/` and `student_model.joblib`, so `/report`, `/summary`, the figures and
  the guardrail all work with no key, no volume and no network.
- Free Spaces sleep after inactivity and wake in ~30 s on the next request. Fine for a portfolio
  link; mention it if someone reports a slow first load.
- Build takes ~3–5 minutes. Watch the **Logs** tab; `/health` returning `{"status":"ok"}` means done.

## Verify after deploy

```bash
curl -s https://RyanSingh0-judgeguard.hf.space/health | jq
curl -s -X POST https://RyanSingh0-judgeguard.hf.space/guard \
  -H 'content-type: application/json' \
  -d '{"question":"Summarise the trial.","answer":"It is generally understood that this was a modest amount."}' | jq
```

Then put the URL in the GitHub README's `## Try it` section and in LinkedIn Featured.
