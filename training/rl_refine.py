"""
REINFORCE fine-tuning of the supervised (+DAgger) maze policy. Optimizes
expected success with masked Categorical policy + entropy + baseline; no
inference-time search. Train episodes use train mazes only; val maze_accuracy
for checkpointing uses a fixed val split (same as imitation).
"""
import json
import random
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


def _load_supervised():
    p = Path(__file__).with_name("train-supervised-imitation-maze-model.py")
    spec = spec_from_file_location("maze_sup", p)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _holdout_mazes_from_pool(all_contexts, n, seed):
    """
    Fixed random holdout (not the train/val split) for pre-RL and progress reporting.
    """
    n = min(n, len(all_contexts))
    rng = random.Random(seed)
    return rng.sample(all_contexts, n) if n < len(all_contexts) else all_contexts


def _resolve_policy_in(policy_in):
    """Explicit .pt path, or first existing: _saved, best, supervised."""
    if policy_in is not None and str(policy_in).strip():
        p = Path(policy_in)
        if p.is_file():
            return p
        raise FileNotFoundError(f"policy_in not found: {p.resolve()}")
    for name in (
        "maze_supervised_policy_best_saved.pt",
        "maze_supervised_policy_best.pt",
        "maze_supervised_policy.pt",
    ):
        c = Path(name)
        if c.is_file():
            print(f"[rl_refine] using checkpoint: {c.resolve()}", flush=True)
            return c
    raise FileNotFoundError(
        "No checkpoint in cwd (tried maze_supervised_policy_best_saved.pt, ..._best.pt, ...policy.pt). "
        "Pass --policy path/to/model.pt"
    )


def _episode_grad(
    sup,
    model,
    ctx,
    device,
    step_penalty=0.0,
):
    """One sampled rollout. Returns (logp_sum, r_return, ent_sum) or None if trivial."""
    pos = ctx["start"]
    logps = []
    ents = []
    t = 0
    if pos == ctx["goal"]:
        return None
    for _ in range(ctx["max_steps"]):
        if pos == ctx["goal"]:
            break
        obs = sup.get_observation(ctx, pos).unsqueeze(0).to(device)
        logits, _ = model(obs)
        leg = sup._legal_action_mask(ctx, pos).unsqueeze(0).to(device)
        logits = sup._logits_legal_mask(logits, leg)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logps.append(dist.log_prob(a).squeeze(0))
        ents.append(dist.entropy().squeeze(0))
        action = int(a.view(-1)[0].item())
        moved, new_pos = sup.can_move(ctx, pos, action)
        pos = new_pos if moved else pos
        t += 1
    success = 1.0 if pos == ctx["goal"] else 0.0
    r = success - step_penalty * t
    if not logps:
        return None
    logp_sum = torch.stack(logps).sum()
    ent_sum = torch.stack(ents).sum()
    return logp_sum, r, ent_sum


def refine(
    mazes_dir="mazes",
    policy_in=None,
    policy_out="maze_rl_refined_policy.pt",
    best_path="maze_rl_refined_policy_best.pt",
    final_path="maze_policy_final.pt",
    val_split=0.1,
    split_seed=42,
    rollout_val_seed=100_043,
    maze_acc_val_sample=1000,
    pre_rl_holdout_mazes=1000,
    pre_rl_eval_seed=88_000,
    epochs=60,
    episodes_per_update=20,
    updates_per_epoch=40,
    lr=3e-4,
    weight_decay=1e-4,
    entropy_coef=0.04,
    step_penalty=0.0,
    baseline_ema=0.08,
    grad_clip=1.0,
    early_stop_maze_acc_patience=12,
    re_eval_holdout_every=2,
    meta_path="maze_rl_refined_policy_meta.json",
):
    print(
        "\n[rl_refine] Starting — first lines can take a while (loading 10k maze JSONs, then ~2k PRE-RL rollouts).\n",
        flush=True,
    )
    print(f"[rl_refine] cwd={Path.cwd().resolve()}", flush=True)
    policy_path = _resolve_policy_in(policy_in)
    print(f"[rl_refine] checkpoint={policy_path.resolve()}", flush=True)

    t0 = time.perf_counter()
    print("[rl_refine] loading train-supervised-imitation-maze-model.py ...", flush=True)
    sup = _load_supervised()
    print(f"[rl_refine] module loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[rl_refine] device={device}", flush=True)

    t0 = time.perf_counter()
    print(
        f"[rl_refine] loading all maze JSON from {mazes_dir!r} (many files — often 30–120s on disk)...",
        flush=True,
    )
    mazes = sup.load_mazes(mazes_dir)
    print(
        f"[rl_refine] loaded {len(mazes)} mazes in {time.perf_counter() - t0:.1f}s; building contexts...",
        flush=True,
    )
    contexts, max_rows, max_cols = sup.prepare_contexts(mazes)
    train_contexts, val_contexts = sup._split_by_maze(
        contexts, val_split=val_split, seed=split_seed
    )
    if not train_contexts:
        raise ValueError("No train mazes for RL refine")

    n_hold = min(pre_rl_holdout_mazes, len(contexts))
    holdout_contexts = _holdout_mazes_from_pool(
        contexts, n_hold, pre_rl_eval_seed
    )
    if len(holdout_contexts) < 10:
        raise ValueError("Not enough mazes for holdout evaluation")
    n_maze_eval = min(maze_acc_val_sample, len(val_contexts)) if maze_acc_val_sample > 0 else 0

    input_size = max_rows * max_cols * 6
    model = sup.SmallMazePolicy(
        input_size=input_size, max_rows=max_rows, max_cols=max_cols
    ).to(device)
    state = torch.load(str(policy_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    h_n = len(holdout_contexts)
    print(
        f"[rl_refine] PRE-RL greedy rollouts: {h_n} holdout mazes (then {n_maze_eval} val) — "
        f"often several minutes on MPS; not hung.\n",
        flush=True,
    )
    t0 = time.perf_counter()
    pre_maze, pre_solved, pre_tot = sup.evaluate_maze_solve_rate(
        model, holdout_contexts, device, sample_size=h_n, seed=0
    )
    print(
        f"[rl_refine] holdout PRE-RL eval done in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    pre_maze = pre_maze if pre_maze is not None else 0.0
    pre_val = None
    if n_maze_eval and val_contexts:
        t0 = time.perf_counter()
        print(f"[rl_refine] PRE-RL val rollouts ({n_maze_eval} mazes)...", flush=True)
        pre_val, pvs, pvt = sup.evaluate_maze_solve_rate(
            model, val_contexts, device, sample_size=n_maze_eval, seed=rollout_val_seed
        )
        print(
            f"[rl_refine] val PRE-RL eval done in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        if pre_val is not None:
            print(
                f"[rl_refine] PRE-RL (val, same 10% split, seed={rollout_val_seed}): "
                f"maze_accuracy={pre_val:.3f} ({pvs}/{pvt})",
                flush=True,
            )
    else:
        pvs, pvt = 0, 0
    if pre_val is None:
        pre_val = 0.0
    print(
        f"[rl_refine] PRE-RL (holdout {h_n} random mazes, pool=ALL, selection_seed={pre_rl_eval_seed}): "
        f"maze_accuracy={pre_maze:.3f} ({pre_solved}/{pre_tot})",
        flush=True,
    )
    print(
        f"[rl_refine] RL train uses only train mazes. Below: val + holdout vs these PRE-RL baselines.\n",
        flush=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    print(
        f"[rl_refine] device={device} train_mazes={len(train_contexts)} val_mazes={len(val_contexts)} "
        f"checkpoint={policy_path.name} episodes/batch={episodes_per_update} "
        f"updates/epoch={updates_per_epoch} val_rollout_n={n_maze_eval} "
        f"epochs={epochs} patience={early_stop_maze_acc_patience} "
        f"entropy={entropy_coef} lr={lr}\n"
    )

    baseline = 0.35
    best_maze = -1.0
    best_label = "init"
    no_improve = 0
    rng = random.Random(split_seed + 30_000)
    last_h_m, last_h_s, last_h_t = pre_maze, pre_solved, pre_tot

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for _u in range(updates_per_epoch):
            optimizer.zero_grad()
            mazes = [
                rng.choice(train_contexts) for _ in range(episodes_per_update)
            ]
            pol_losses = []
            for ctx in mazes:
                out = _episode_grad(
                    sup, model, ctx, device, step_penalty
                )
                if out is None:
                    continue
                logp, r, ent = out
                adv = r - baseline
                pol_losses.append(
                    -adv * logp - entropy_coef * ent
                )
                baseline = (1.0 - baseline_ema) * baseline + baseline_ema * r
            if not pol_losses:
                continue
            loss = torch.stack([p.view(()) for p in pol_losses]).mean()
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        scheduler.step()
        run_lr = scheduler.get_last_lr()[0]
        if epoch_losses:
            print(
                f"[rl_refine] epoch={epoch+1:03d}/{epochs} "
                f"loss~={sum(epoch_losses)/len(epoch_losses):.4f} "
                f"lr={run_lr:.2e} baseline~={baseline:.3f}",
                flush=True,
            )

        model.eval()
        maze_rate = None
        if n_maze_eval and val_contexts:
            maze_rate, sol, totn = sup.evaluate_maze_solve_rate(
                model, val_contexts, device, sample_size=n_maze_eval, seed=rollout_val_seed
            )
            if maze_rate is not None:
                val_boost = maze_rate - pre_val
                vs = "+" if val_boost >= 0 else ""
                re_hold = (
                    re_eval_holdout_every <= 1
                    or (epoch + 1) % re_eval_holdout_every == 0
                    or epoch == 0
                )
                if re_hold:
                    h_m, h_s, h_t = sup.evaluate_maze_solve_rate(
                        model, holdout_contexts, device, sample_size=h_n, seed=0
                    )
                    h_m = h_m if h_m is not None else 0.0
                    last_h_m, last_h_s, last_h_t = h_m, h_s, h_t
                    tag = ""
                else:
                    h_m, h_s, h_t = last_h_m, last_h_s, last_h_t
                    tag = " (holdout: last fresh eval)"
                h_boost = h_m - pre_maze
                hs = "+" if h_boost >= 0 else ""
                print(
                    f"[rl_refine]        val_maze_acc={maze_rate:.3f} ({sol}/{totn}) | "
                    f"vs pre-RL val {vs}{val_boost:.3f} | "
                    f"holdout={h_m:.3f} ({h_s}/{h_t}){tag} | "
                    f"vs pre-RL holdout {hs}{h_boost:.3f}",
                    flush=True,
                )
        else:
            maze_rate = None
            sol, totn = 0, 0
        if maze_rate is not None and maze_rate > best_maze + 1e-5:
            best_maze = maze_rate
            no_improve = 0
            best_label = f"rl_epoch_{epoch+1}"
            torch.save(model.state_dict(), best_path)
            print(
                f"[rl_refine]  -> new best maze_accuracy={best_maze:.3f} -> {best_path}",
                flush=True,
            )
        elif (
            maze_rate is not None
            and early_stop_maze_acc_patience
            and early_stop_maze_acc_patience > 0
        ):
            no_improve += 1
            if no_improve >= early_stop_maze_acc_patience:
                print(
                    f"[rl_refine] Early stopping (no val maze acc gain in "
                    f"{early_stop_maze_acc_patience} epochs). best={best_maze:.3f} at {best_label}"
                )
                break

    out_p, best_p, fin_p = Path(policy_out), Path(best_path), Path(final_path)
    if best_p.is_file() and best_maze >= 0.0:
        sd = torch.load(best_p, map_location="cpu")
        torch.save(sd, out_p)
        torch.save(sd, fin_p)
        print(
            f"\n[rl_refine] Wrote {out_p} and KILLER final {fin_p} "
            f"(best val maze_acc={best_maze:.3f} at {best_label})"
        )
    else:
        torch.save(model.state_dict(), out_p)
        torch.save(model.state_dict(), fin_p)
        print(f"\n[rl_refine] Wrote {out_p} and {fin_p} (last state)")

    Path(meta_path).write_text(
        json.dumps(
            {
                "strategy": "REINFORCE_refine",
                "policy_in": str(policy_path),
                "policy_out": str(out_p),
                "final_path": str(fin_p),
                "pre_rl_val_maze_accuracy": pre_val,
                "pre_rl_holdout_maze_accuracy": pre_maze,
                "pre_rl_holdout_n": h_n,
                "pre_rl_eval_seed": pre_rl_eval_seed,
                "best_val_maze_accuracy": best_maze if best_maze >= 0.0 else None,
                "best_at": best_label,
                "split_seed": split_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "final_path": str(fin_p),
        "best_maze": best_maze,
        "out_path": str(out_p),
        "pre_rl_val": pre_val,
        "pre_rl_holdout": pre_maze,
    }


if __name__ == "__main__":
    import argparse

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(description="REINFORCE refine (optional: use saved best checkpoint).")
    ap.add_argument(
        "--policy",
        default=None,
        help="Path to .pt (default: maze_supervised_policy_best_saved.pt, then best, then supervised).",
    )
    ap.add_argument("--mazes-dir", default="mazes")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--re-eval-holdout-every", type=int, default=2, help="1=every epoch (slower), 2=alternate")
    args = ap.parse_args()
    kwargs = {
        "mazes_dir": args.mazes_dir,
        "policy_in": args.policy,
        "re_eval_holdout_every": max(1, args.re_eval_holdout_every),
    }
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    refine(**kwargs)
