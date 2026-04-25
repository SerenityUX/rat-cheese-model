import json
import random
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


ACTIONS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}


def _logits_legal_mask(logits, legal_mask, neg=-1e4):
    """Set logits for illegal actions to -inf (approx); legal_mask (B,4) 1.0=legal, 0=wall."""
    return torch.where(
        legal_mask > 0.5,
        logits,
        torch.full_like(logits, neg),
    )


class SmallMazePolicy(nn.Module):
    def __init__(self, input_size, hidden_size=320, num_actions=4, max_rows=None, max_cols=None):
        super().__init__()
        if max_rows is None or max_cols is None:
            raise ValueError("SmallMazePolicy requires max_rows and max_cols for SupervisedImitation.")
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.channels = 6
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.flat_size = 96 * self.max_rows * self.max_cols
        self.head = nn.Sequential(
            nn.Linear(self.flat_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )
        self.dist_head = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, self.channels, self.max_rows, self.max_cols)
        h = self.encoder(x)
        flat = h.reshape(b, -1)
        return self.head(flat), self.dist_head(flat).squeeze(-1)


def load_mazes(mazes_dir="mazes"):
    mazes = []
    for json_path in sorted(Path(mazes_dir).glob("maze_*/maze.json")):
        mazes.append(json.loads(json_path.read_text(encoding="utf-8")))
    if not mazes:
        raise FileNotFoundError("No maze JSON files found. Generate mazes first.")
    return mazes


def build_wall_grid(maze):
    rows = maze["rows"]
    cols = maze["cols"]
    cell_size = maze["cell_size"]
    walls = [[{"N": False, "E": False, "S": False, "W": False} for _ in range(cols)] for _ in range(rows)]

    for (x1, y1), (x2, y2) in maze["lines"]:
        if y1 == y2:
            y = y1
            x = min(x1, x2)
            c = int(x // cell_size)
            if y == rows * cell_size:
                r = rows - 1
                walls[r][c]["S"] = True
            else:
                r = int(y // cell_size)
                walls[r][c]["N"] = True
                if r > 0:
                    walls[r - 1][c]["S"] = True
        elif x1 == x2:
            x = x1
            y = min(y1, y2)
            r = int(y // cell_size)
            if x == cols * cell_size:
                c = cols - 1
                walls[r][c]["E"] = True
            else:
                c = int(x // cell_size)
                walls[r][c]["W"] = True
                if c > 0:
                    walls[r][c - 1]["E"] = True

    return walls


def action_for_step(cur, nxt):
    dr = nxt[0] - cur[0]
    dc = nxt[1] - cur[1]
    for action, (adr, adc) in ACTIONS.items():
        if (dr, dc) == (adr, adc):
            return action
    raise ValueError(f"Invalid step in correct_path: {cur} -> {nxt}")


def get_observation(maze_ctx, pos):
    rows = maze_ctx["rows"]
    cols = maze_ctx["cols"]
    max_rows = maze_ctx["max_rows"]
    max_cols = maze_ctx["max_cols"]
    walls = maze_ctx["walls"]
    goal = maze_ctx["goal"]

    size = max_rows * max_cols
    north = [0.0] * size
    east = [0.0] * size
    south = [0.0] * size
    west = [0.0] * size
    pos_one_hot = [0.0] * size
    goal_one_hot = [0.0] * size

    for r in range(rows):
        for c in range(cols):
            idx = r * max_cols + c
            north[idx] = 1.0 if walls[r][c]["N"] else 0.0
            east[idx] = 1.0 if walls[r][c]["E"] else 0.0
            south[idx] = 1.0 if walls[r][c]["S"] else 0.0
            west[idx] = 1.0 if walls[r][c]["W"] else 0.0

    pos_idx = pos[0] * max_cols + pos[1]
    goal_idx = goal[0] * max_cols + goal[1]
    pos_one_hot[pos_idx] = 1.0
    goal_one_hot[goal_idx] = 1.0

    return torch.tensor(north + east + south + west + pos_one_hot + goal_one_hot, dtype=torch.float32)


def can_move(maze_ctx, pos, action):
    rows = maze_ctx["rows"]
    cols = maze_ctx["cols"]
    walls = maze_ctx["walls"]
    r, c = pos
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        return False, pos

    if action == 0 and walls[r][c]["N"]:
        return False, pos
    if action == 1 and walls[r][c]["E"]:
        return False, pos
    if action == 2 and walls[r][c]["S"]:
        return False, pos
    if action == 3 and walls[r][c]["W"]:
        return False, pos

    return True, (nr, nc)


def _legal_action_mask(ctx, pos):
    m = torch.zeros(4, dtype=torch.float32)
    for a in range(4):
        m[a] = 1.0 if can_move(ctx, pos, a)[0] else 0.0
    return m


def prepare_contexts(mazes):
    max_rows = max(m["rows"] for m in mazes)
    max_cols = max(m["cols"] for m in mazes)
    contexts = []
    for m in mazes:
        contexts.append(
            {
                "maze_id": m["maze_id"],
                "rows": m["rows"],
                "cols": m["cols"],
                "goal": tuple(m["end"]),
                "walls": build_wall_grid(m),
                "start": tuple(m["start"]),
                "max_rows": max_rows,
                "max_cols": max_cols,
                "max_steps": m["rows"] * m["cols"] * 2,
            }
        )
    return contexts, max_rows, max_cols


def _neighbors(ctx, pos):
    rows = ctx["rows"]
    cols = ctx["cols"]
    walls = ctx["walls"]
    r, c = pos
    out = []

    if r > 0 and not walls[r][c]["N"]:
        out.append((0, (r - 1, c)))
    if c < cols - 1 and not walls[r][c]["E"]:
        out.append((1, (r, c + 1)))
    if r < rows - 1 and not walls[r][c]["S"]:
        out.append((2, (r + 1, c)))
    if c > 0 and not walls[r][c]["W"]:
        out.append((3, (r, c - 1)))
    return out


def _goal_distances(ctx):
    goal = ctx["goal"]
    dists = {goal: 0}
    q = deque([goal])
    while q:
        cur = q.popleft()
        for _, nxt in _neighbors(ctx, cur):
            if nxt not in dists:
                dists[nxt] = dists[cur] + 1
                q.append(nxt)
    return dists


def _optimal_actions_for_pos(ctx, pos, dists):
    candidates = []
    for action, nxt in _neighbors(ctx, pos):
        if nxt in dists:
            candidates.append((dists[nxt], action))
    if not candidates:
        return []
    best_dist = min(x[0] for x in candidates)
    return [action for dist, action in candidates if dist == best_dist]


def _maze_state_labels(ctx):
    dists = _goal_distances(ctx)
    max_d = max(dists.values()) if dists else 1.0
    max_d = max(1.0, float(max_d))
    pairs = []
    for r in range(ctx["rows"]):
        for c in range(ctx["cols"]):
            pos = (r, c)
            if pos == ctx["goal"] or pos not in dists:
                continue
            actions = _optimal_actions_for_pos(ctx, pos, dists)
            if not actions:
                continue
            target = torch.zeros(4, dtype=torch.float32)
            for action in actions:
                target[action] = 1.0 / len(actions)
            d_norm = dists[pos] / max_d
            legal = _legal_action_mask(ctx, pos)
            pairs.append((pos, target, d_norm, legal))
    return pairs


def _split_by_maze(contexts, val_split=0.1, seed=42):
    rng = random.Random(seed)
    shuffled = contexts[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_split))
    val_contexts = shuffled[:val_count]
    train_contexts = shuffled[val_count:]
    if not train_contexts:
        train_contexts = val_contexts
    return train_contexts, val_contexts


def _estimate_state_coverage(contexts):
    counts = []
    total = 0
    for ctx in contexts:
        n = len(_maze_state_labels(ctx))
        counts.append(n)
        total += n
    return min(counts), max(counts), (sum(counts) / len(counts)), total


def _iter_batches(contexts, batch_size=256, shuffle=True):
    ctxs = contexts[:]
    if shuffle:
        random.shuffle(ctxs)

    x_buf = []
    y_buf = []
    d_buf = []
    leg_buf = []
    for ctx in ctxs:
        pairs = _maze_state_labels(ctx)
        if shuffle:
            random.shuffle(pairs)
        for pos, target, d_norm, legal in pairs:
            x_buf.append(get_observation(ctx, pos))
            y_buf.append(target)
            d_buf.append(torch.tensor(d_norm, dtype=torch.float32))
            leg_buf.append(legal)
            if len(x_buf) >= batch_size:
                yield torch.stack(x_buf), torch.stack(y_buf), torch.stack(d_buf), torch.stack(leg_buf)
                x_buf = []
                y_buf = []
                d_buf = []
                leg_buf = []

    if x_buf:
        yield torch.stack(x_buf), torch.stack(y_buf), torch.stack(d_buf), torch.stack(leg_buf)


def _rollout_solved_one(model, ctx, device):
    """Greedy rollout with valid-action masking (same idea as run-maze-model.py). Returns True if goal reached."""
    pos = ctx["start"]
    max_steps = ctx["max_steps"]
    for _ in range(max_steps):
        if pos == ctx["goal"]:
            return True
        obs = get_observation(ctx, pos).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(obs)
            logits = out[0].squeeze(0) if isinstance(out, tuple) else out.squeeze(0)
        valid = [a for a in range(4) if can_move(ctx, pos, a)[0]]
        if valid:
            action = max(valid, key=lambda a: float(logits[a].item()))
        else:
            action = int(torch.argmax(logits).item())
        moved, new_pos = can_move(ctx, pos, action)
        pos = new_pos if moved else pos
    return pos == ctx["goal"]


def evaluate_maze_solve_rate(model, contexts, device, sample_size=200, seed=0):
    """
    Fraction of mazes in a random sample that are fully solved by rollout.
    Returns (rate, solved_count, total_evaluated) or (None, 0, 0) if disabled/empty.
    """
    if sample_size <= 0 or not contexts:
        return None, 0, 0
    rng = random.Random(seed)
    n = min(sample_size, len(contexts))
    sample = rng.sample(contexts, n) if n < len(contexts) else contexts
    model.eval()
    solved = 0
    for ctx in sample:
        if _rollout_solved_one(model, ctx, device):
            solved += 1
    return solved / n, solved, n


def _write_meta(path, extra, input_size, max_rows, max_cols, split_seed, batch_size, lr, dist_loss_weight, model_path):
    meta = {
        "strategy": "SupervisedImitation",
        "input_size": input_size,
        "max_rows": max_rows,
        "max_cols": max_cols,
        "train_split_by_maze": True,
        "split_seed": split_seed,
        "batch_size": batch_size,
        "lr": lr,
        "dist_loss_weight": dist_loss_weight,
        "actions": {"0": "up", "1": "right", "2": "down", "3": "left"},
        "model_path": str(model_path),
        **extra,
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _batch_loss(
    model,
    xb,
    yb,
    d_target,
    leg,
    dist_weight,
):
    logits, d_pred = model(xb)
    logits = _logits_legal_mask(logits, leg)
    log_probs = F.log_softmax(logits, dim=1)
    loss_p = -(yb * log_probs).sum(dim=1).mean()
    loss_d = F.huber_loss(d_pred, d_target, reduction="mean", delta=0.2)
    loss = loss_p + dist_weight * loss_d
    preds = torch.argmax(logits, dim=1)
    hits = yb.gather(1, preds.unsqueeze(1)).squeeze(1) > 0
    return loss, int(hits.sum().item()), int(yb.shape[0]), loss_p.item(), loss_d.item()


def collect_dagger_pairs(
    model,
    train_contexts,
    device,
    n_mazes,
    max_pairs,
    seed,
):
    """
    DAgger: rollout the current policy on train mazes, label each visited state
    with the BFS-teacher (same as BC).
    """
    if not train_contexts or max_pairs <= 0:
        return []
    rng = random.Random(seed)
    n = min(n_mazes, len(train_contexts))
    sample = rng.sample(train_contexts, n) if n < len(train_contexts) else train_contexts
    model.eval()
    pairs = []
    print(
        f"[dagger]     collecting on-policy labels: rolling out policy on {n} train mazes "
        f"(silent stretches are normal)...",
        flush=True,
    )
    for mi, ctx in enumerate(sample):
        if mi > 0 and mi % 400 == 0:
            print(
                f"[dagger]     ... rollout maze {mi}/{n}, pairs collected so far {len(pairs)}",
                flush=True,
            )
        if len(pairs) >= max_pairs:
            break
        dists = _goal_distances(ctx)
        if not dists:
            continue
        max_d = max(1.0, float(max(dists.values())))
        pos = ctx["start"]
        for _ in range(ctx["max_steps"]):
            if pos == ctx["goal"]:
                break
            if pos not in dists:
                break
            actions = _optimal_actions_for_pos(ctx, pos, dists)
            if not actions:
                break
            target = torch.zeros(4, dtype=torch.float32)
            for a in actions:
                target[a] = 1.0 / len(actions)
            d_norm = dists[pos] / max_d
            leg = _legal_action_mask(ctx, pos)
            pairs.append(
                (
                    get_observation(ctx, pos),
                    target,
                    torch.tensor(d_norm, dtype=torch.float32),
                    leg,
                )
            )
            if len(pairs) >= max_pairs:
                break
            obs = get_observation(ctx, pos).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(obs)
                logits = out[0] if isinstance(out, tuple) else out
            logits = _logits_legal_mask(
                logits, leg.unsqueeze(0).to(device)
            )
            valid = [a for a in range(4) if can_move(ctx, pos, a)[0]]
            if not valid:
                break
            action = int(max(valid, key=lambda a: float(logits[0, a].item())))
            moved, new_pos = can_move(ctx, pos, action)
            pos = new_pos if moved else pos
    return pairs[:max_pairs]


def _stack_dagger_list(pairs):
    if not pairs:
        return None
    return (
        torch.stack([p[0] for p in pairs]),
        torch.stack([p[1] for p in pairs]),
        torch.stack([p[2] for p in pairs]),
        torch.stack([p[3] for p in pairs]),
    )


def _run_dagger_subepoch(
    model,
    static_ctx,
    Dstack,
    device,
    batch_size,
    dist_weight,
    optimizer,
    grad_clip,
):
    """One pass over full static BFS data + one shuffled pass over DAgger data."""
    model.train()
    tot_loss = 0.0
    tot = 0
    for xb, yb, d_t, leg in _iter_batches(static_ctx, batch_size=batch_size, shuffle=True):
        xb = xb.to(device)
        yb = yb.to(device)
        d_t = d_t.to(device)
        leg = leg.to(device)
        loss, bok, bcount, _, _ = _batch_loss(model, xb, yb, d_t, leg, dist_weight)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        tot_loss += float(loss.item()) * bcount
        tot += bcount

    if Dstack is not None:
        Dx, Dy, Dd, Dl = Dstack
        n = Dx.shape[0]
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            xb = Dx[idx].to(device)
            yb = Dy[idx].to(device)
            d_t = Dd[idx].to(device)
            leg = Dl[idx].to(device)
            loss, bok, bcount, _, _ = _batch_loss(
                model, xb, yb, d_t, leg, dist_weight
            )
            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            tot_loss += float(loss.item()) * bcount
            tot += bcount

    return tot_loss / max(1, tot)


def train(
    mazes_dir="mazes",
    epochs=20,
    batch_size=512,
    lr=7e-4,
    val_split=0.1,
    split_seed=42,
    maze_acc_val_sample=1000,
    dist_loss_weight=0.15,
    weight_decay=1e-4,
    grad_clip=1.0,
    early_stop_maze_acc_patience=7,
    rollout_eval_seed=None,
    best_path="maze_supervised_policy_best.pt",
    out_path="maze_supervised_policy.pt",
    dagger_phases=5,
    dagger_sub_epochs=3,
    dagger_mazes_per_phase=2000,
    dagger_max_pairs=120000,
    dagger_lr_factor=0.5,
):
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[supervised] Using device: {device}")
    print(f"[supervised] Loading mazes from '{mazes_dir}'...")
    mazes = load_mazes(mazes_dir)
    contexts, max_rows, max_cols = prepare_contexts(mazes)
    print(f"[supervised] Loaded {len(contexts)} mazes")
    train_contexts, val_contexts = _split_by_maze(contexts, val_split=val_split, seed=split_seed)
    train_min, train_max, train_avg, train_total = _estimate_state_coverage(train_contexts)
    val_min, val_max, val_avg, val_total = _estimate_state_coverage(val_contexts)
    print(
        "[supervised] State coverage from all reachable cells "
        f"(train min={train_min}, max={train_max}, avg={train_avg:.1f}, total={train_total}) "
        f"(val min={val_min}, max={val_max}, avg={val_avg:.1f}, total={val_total})"
    )

    eval_seed = rollout_eval_seed if rollout_eval_seed is not None else (split_seed + 100_001)
    input_size = max_rows * max_cols * 6
    model = SmallMazePolicy(input_size=input_size, max_rows=max_rows, max_cols=max_cols).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    n_maze_eval = min(maze_acc_val_sample, len(val_contexts)) if maze_acc_val_sample > 0 else 0
    print(
        f"[supervised] Maze split train={len(train_contexts)}, val={len(val_contexts)}, "
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}, AdamW_wd={weight_decay}, "
        f"cosine_anneal_T={epochs}, dist_loss_w={dist_loss_weight}, grad_clip={grad_clip}, "
        f"maze_rollout_val_mazes={n_maze_eval}, rollout_eval_seed={eval_seed} (fixed), "
        f"early_stop_patience_maze={early_stop_maze_acc_patience}, "
        f"dagger_phases={dagger_phases} (0=off), dagger_sub_epochs={dagger_sub_epochs}, "
        f"dagger_mazes/phase<={dagger_mazes_per_phase}, dagger_max_pairs={dagger_max_pairs}"
    )

    best_maze = -1.0
    best_label = "none"
    no_improve = 0
    out_p = Path(out_path)
    best_p = Path(best_path)
    run_epochs = 0
    last_maze_rate = None

    approx_train_batches = max(1, train_total // batch_size)
    train_progress_every = max(250, approx_train_batches // 8)

    for epoch in range(epochs):
        run_epochs = epoch + 1
        print(
            f"[supervised] epoch {run_epochs:03d}/{epochs}: train pass (~{approx_train_batches} batches, "
            f"then val, then {n_maze_eval} greedy rollouts — can take several minutes; not hung.",
            flush=True,
        )
        model.train()
        epoch_loss = 0.0
        correct = 0
        count = 0
        batch_i = 0

        for xb_cpu, yb_cpu, d_cpu, leg_cpu in _iter_batches(
            train_contexts, batch_size=batch_size, shuffle=True
        ):
            batch_i += 1
            xb = xb_cpu.to(device)
            yb = yb_cpu.to(device)
            d_t = d_cpu.to(device)
            leg = leg_cpu.to(device)

            loss, batch_ok, bcount, _, _ = _batch_loss(model, xb, yb, d_t, leg, dist_loss_weight)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item()) * bcount
            correct += batch_ok
            count += bcount
            if batch_i == 1 or batch_i % train_progress_every == 0:
                rloss = epoch_loss / max(1, count)
                racc = correct / max(1, count)
                print(
                    f"[supervised]     train batch {batch_i}  running_loss={rloss:.4f}  running_acc={racc:.3f}",
                    flush=True,
                )

        train_acc = correct / max(1, count)
        train_loss = epoch_loss / max(1, count)
        scheduler.step()

        print(f"[supervised]     train pass done ({batch_i} batches). val pass...", flush=True)
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for xvb_cpu, yvb_cpu, dvb_cpu, legvb_cpu in _iter_batches(
                val_contexts, batch_size=batch_size, shuffle=False
            ):
                xvb = xvb_cpu.to(device)
                yvb = yvb_cpu.to(device)
                dvb = dvb_cpu.to(device)
                legb = legvb_cpu.to(device)
                vloss, vok, vc, _, _ = _batch_loss(
                    model, xvb, yvb, dvb, legb, dist_loss_weight
                )
                val_loss_sum += float(vloss.item()) * vc
                val_correct += vok
                val_count += vc
            val_loss = val_loss_sum / max(1, val_count)
            val_acc = val_correct / max(1, val_count)

        print(
            f"[supervised]     val done. full-maze rollout on {n_maze_eval} val mazes (greedy)...",
            flush=True,
        )
        maze_acc_str = "n/a"
        maze_rate = None
        if maze_acc_val_sample and maze_acc_val_sample > 0 and val_contexts:
            n_eval = min(maze_acc_val_sample, len(val_contexts))
            maze_rate, solved_n, total_n = evaluate_maze_solve_rate(
                model,
                val_contexts,
                device,
                sample_size=n_eval,
                seed=eval_seed,
            )
            if maze_rate is not None:
                maze_acc_str = f"{maze_rate:.3f} ({solved_n}/{total_n})"
                last_maze_rate = maze_rate

        print(
            f"[supervised] epoch={run_epochs:03d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"maze_accuracy={maze_acc_str} lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        if maze_rate is not None and maze_rate > best_maze + 1e-5:
            best_maze = maze_rate
            best_label = f"bc_epoch_{run_epochs}"
            no_improve = 0
            torch.save(model.state_dict(), best_p)
            print(f"[supervised]  -> new best maze_accuracy={best_maze:.3f} saved to {best_p}", flush=True)
        elif maze_rate is not None and early_stop_maze_acc_patience and early_stop_maze_acc_patience > 0:
            no_improve += 1
            if no_improve >= early_stop_maze_acc_patience:
                print(
                    f"[supervised] Early stopping: no maze_accuracy gain for {early_stop_maze_acc_patience} epochs "
                    f"(best {best_maze:.3f} at {best_label})",
                    flush=True,
                )
                break

    if dagger_phases and dagger_phases > 0:
        if best_p.is_file() and best_maze >= 0.0:
            model.load_state_dict(torch.load(best_p, map_location=device))
        d_base_lr = lr * dagger_lr_factor
        print(
            f"\n[dagger] Start (policy from BC best). phases={dagger_phases}, "
            f"sub_epochs/phase={dagger_sub_epochs}, mazes/phase<={dagger_mazes_per_phase}, "
            f"max_pairs={dagger_max_pairs}, lr0={d_base_lr:.2e}\n",
            flush=True,
        )
        for ph in range(dagger_phases):
            pairs = collect_dagger_pairs(
                model,
                train_contexts,
                device,
                dagger_mazes_per_phase,
                dagger_max_pairs,
                split_seed + 17_000 + ph,
            )
            print(
                f"[dagger] phase {ph+1}/{dagger_phases}: collected {len(pairs)} on-policy state labels (teacher=BFS)",
                flush=True,
            )
            Ds = _stack_dagger_list(pairs) if pairs else None
            ph_lr = d_base_lr * (0.9**ph)
            opt_d = optim.AdamW(model.parameters(), lr=ph_lr, weight_decay=weight_decay)
            sched_d = optim.lr_scheduler.CosineAnnealingLR(
                opt_d, T_max=max(1, dagger_sub_epochs), eta_min=ph_lr * 0.01
            )
            for de in range(dagger_sub_epochs):
                avg = _run_dagger_subepoch(
                    model,
                    train_contexts,
                    Ds,
                    device,
                    batch_size,
                    dist_loss_weight,
                    opt_d,
                    grad_clip,
                )
                sched_d.step()
                print(
                    f"[dagger] phase {ph+1} subepoch {de+1}/{dagger_sub_epochs} mixed_loss~={avg:.4f} "
                    f"lr={sched_d.get_last_lr()[0]:.2e}",
                    flush=True,
                )

            n_eval = min(maze_acc_val_sample, len(val_contexts)) if maze_acc_val_sample > 0 else 0
            model.eval()
            if n_eval and val_contexts:
                maze_rate, solved_n, total_n = evaluate_maze_solve_rate(
                    model, val_contexts, device, sample_size=n_eval, seed=eval_seed
                )
                if maze_rate is not None:
                    print(
                        f"[dagger] phase {ph+1} maze_accuracy={maze_rate:.3f} "
                        f"({solved_n}/{total_n})",
                        flush=True,
                    )
                    if maze_rate > best_maze + 1e-5:
                        best_maze = maze_rate
                        best_label = f"dagger_phase_{ph+1}"
                        last_maze_rate = maze_rate
                        torch.save(model.state_dict(), best_p)
                        print(
                            f"[dagger]  -> new best maze_accuracy={best_maze:.3f} -> {best_p}",
                            flush=True,
                        )

    if best_p.is_file() and best_maze >= 0.0:
        best_sd = torch.load(best_p, map_location="cpu")
        torch.save(best_sd, out_p)
        print(
            f"\n[supervised] Wrote {out_p} from best checkpoint (maze_accuracy={best_maze:.3f}, "
            f"best_at={best_label})",
            flush=True,
        )
    else:
        torch.save(model.state_dict(), out_p)
        print(f"\n[supervised] Wrote {out_p} (last state)")

    _write_meta(
        Path("maze_supervised_policy_meta.json"),
        {
            "epochs_ran": run_epochs,
            "epochs_planned": epochs,
            "dagger_phases": dagger_phases,
            "best_maze_accuracy": best_maze if best_maze >= 0.0 else None,
            "best_maze_at": best_label if best_maze >= 0.0 else None,
            "last_maze_accuracy": last_maze_rate,
            "masked_legal_loss": True,
            "aux_dist_head": True,
            "dagger_enabled": bool(dagger_phases),
        },
        input_size,
        max_rows,
        max_cols,
        split_seed,
        batch_size,
        lr,
        dist_loss_weight,
        out_p,
    )
    print(f"[supervised] Saved metadata to maze_supervised_policy_meta.json")
    return {
        "out_path": str(out_p),
        "best_maze": best_maze,
        "best_label": best_label,
    }


if __name__ == "__main__":
    train()
