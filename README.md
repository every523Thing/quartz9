# quartz9 miner

Creates a Three.js asset from a prompt image via the witness: multiple first drafts, render-based selection, staged
revisions and a final choice, all inside the allowed calls and credits for each prompt.

Entry point: `main.py` -> `async generate(image, seed, witness) -> str | None`. The plan is in `miner404/versions.py`.

## Live run

```sh
uv venv --python 3.13 .venv-live
uv pip install --python .venv-live/bin/python -r requirements.txt
git clone --branch miner-runtime-v0.1.5 --depth 1 https://github.com/404-Repo/404-gen-subnet.git ../404-sdk
```

Create `witness.json` next to `main.py`, with `witness_url` set to `https://witness.subnet404.xyz` and
`proof_parameters` set to the absolute path of `404-sdk/witness-server/verifier/params`. Then:

```sh
.venv-live/bin/python main.py --prompts prompts.txt --seed PUBLISHED_SEED --wallet-name NAME --wallet-hotkey HOTKEY
```
