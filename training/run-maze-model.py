from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random

import torch


def _load_module(file_name, module_name):
    module_path = Path(__file__).with_name(file_name)
    spec = spec_from_file_location(module_name, module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_strategy_module(strategy):
    if strategy == "RL":
        return _load_module("train-a-small-maze-model.py", "train_a_small_maze_model")
    if strategy == "SupervisedImitation":
        return _load_module("train-supervised-imitation-maze-model.py", "train_supervised_imitation_maze_model")
    raise ValueError("strategy must be 'RL' or 'SupervisedImitation'")


def run_model(strategy="RL", mazes_dir="mazes", num_mazes=25, policy_path=None, eval_seed=42):
    """
    policy_path: optional path to a .pt checkpoint. If None, uses default per strategy.
    eval_seed: seed for which mazes are sampled (reproducible).
    """
    strategy_module = _load_strategy_module(strategy)
    if policy_path is not None:
        model_path = policy_path
    else:
        model_path = "maze_small_policy.pt" if strategy == "RL" else "maze_supervised_policy.pt"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    mazes = strategy_module.load_mazes(mazes_dir)
    contexts, max_rows, max_cols = strategy_module.prepare_contexts(mazes)
    if not contexts:
        raise ValueError("No maze contexts available")

    input_size = max_rows * max_cols * 6
    if strategy == "SupervisedImitation":
        model = strategy_module.SmallMazePolicy(
            input_size=input_size, max_rows=max_rows, max_cols=max_cols
        ).to(device)
    else:
        model = strategy_module.SmallMazePolicy(input_size=input_size).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    sample_count = min(num_mazes, len(contexts))
    rng = random.Random(eval_seed)
    eval_contexts = rng.sample(contexts, sample_count) if sample_count < len(contexts) else contexts

    solved = 0
    total_steps = 0
    print(f"[run] Strategy={strategy} model='{model_path}' device={device} eval_mazes={sample_count}")

    for idx, maze in enumerate(eval_contexts, start=1):
        pos = maze["start"]
        steps = 0
        max_steps = maze["max_steps"]
        finished = False

        while steps < max_steps:
            obs = strategy_module.get_observation(maze, pos).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(obs)
                logits = out[0] if isinstance(out, tuple) else out
                logits = logits.squeeze(0)
                valid_actions = []
                for action_id in range(4):
                    moved, _ = strategy_module.can_move(maze, pos, action_id)
                    if moved:
                        valid_actions.append(action_id)

                if valid_actions:
                    best_action = max(valid_actions, key=lambda a: float(logits[a].item()))
                    action = int(best_action)
                else:
                    action = int(torch.argmax(logits).item())

            moved, new_pos = strategy_module.can_move(maze, pos, action)
            pos = new_pos if moved else pos
            steps += 1

            if pos == maze["goal"]:
                finished = True
                break

        if finished:
            solved += 1
        total_steps += steps
        print(
            f"[run] maze={idx:03d}/{sample_count} solved={finished} steps={steps} "
            f"running_solve_rate={solved / idx:.3f}",
            flush=True,
        )

    solve_rate = solved / sample_count
    avg_steps = total_steps / sample_count
    print(f"[run] Done. solve_rate={solve_rate:.3f} avg_steps={avg_steps:.1f}")
    return {"solve_rate": solve_rate, "avg_steps": avg_steps, "sample_count": sample_count}


if __name__ == "__main__":
    run_model(strategy="RL", mazes_dir="mazes", num_mazes=25)
