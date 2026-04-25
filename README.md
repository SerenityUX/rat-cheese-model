## My story so far

I started off by making an RL-based model.

I let it train on 100 mazes and loop through them.

I'm a few epochs in and seeing some progress but not much... (it spikes up and down)

I am going to run another experiment in parallel which is Supervised Imitation strategy.

Update after running Supervised Imitation on 10,000 mazes:

I switched to a maze-level metric (`maze_accuracy`) on 1,000 val mazes each epoch (not just next-step accuracy).

I trained with the updated supervised setup (masked legal actions + distance head + cosine lr + best-checkpoint saving).

The model climbed from about 35.7% at epoch 1 to 90.7% (907/1000) at best (`maze_supervised_policy_best.pt`).

I also tried DAgger phases after BC. It stayed around ~89.8%-90.6% and did not beat the 90.7% BC peak in that run. This is for the mazes intentionally held out of the dataset for the eval.

I then tried RL refine (REINFORCE) because the goal was to push higher, but early RL runs dropped from the pre-RL baseline and underperformed the BC best.

I was able to get up to 99% success rate on a random set of fresh 1,000 mazes generated, so I think the model in it's current form works pretty well :)

With proper RL, I could probably push it even higher.

Reproducibility update in `main.py`:

- Maze generation now uses a fixed seed (`MazeGenerationSeed`).
- Training uses fixed split/eval seeds.
- After supervised training, `main.py` refreshes `maze_supervised_policy_best_saved.pt` from the current run's `maze_supervised_policy_best.pt`.
- `main.py` runs a pre-RL random-maze check on that refreshed best checkpoint, then starts RL refine from that exact file.

## Current repo layout

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
