# Escape The Maze

This repo now has two layers:

- `training/`: all model training, evaluation, checkpoints, and export tooling
- repo root: static web app assets (`index.html`, `model.onnx`, `web_meta.json`, optional `maze.json`)

## Training / Native App (`training/main.py`)

`training/main.py` can run full training/eval or run inference-only native UI mode.

### Run training pipeline

```bash
cd training
python3 -m pip install -r requirements.txt
python3 main.py
```

### Run inference-only native UI

In `training/main.py`, set:

- `RunInferenceWithMainUI = True`

Then run:

```bash
cd training
python3 main.py
```

The native app solves mazes continuously, writes per-step SVG snapshots, and lets you click after solve/fail to generate another maze.

## Web App (root static site)

The web app is fully static (GitHub Pages-compatible) and runs ONNX inference in-browser using `onnxruntime-web`.

### Regenerate web assets from checkpoints

```bash
python3 training/export_to_web.py
```

This writes/refreshes at repo root:

- `index.html`
- `model.onnx`
- `web_meta.json`
- optional support files (`maze.json`, `unsolved.svg`)

### Run locally

```bash
python3 -m http.server 8000
```

Then open:

- `http://localhost:8000`

The web app behavior:

- maze centered on white background
- auto-solves at model speed
- no immediate backtracking (unless forced)
- plays a success sound on solve
- click after solve/fail to generate a new random maze# rat-cheese-model
