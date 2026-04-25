import json
import random
from collections import deque
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim


ACTIONS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}


class SmallMazePolicy(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, x):
        return self.net(x)


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


def neighbors_from_walls(maze_ctx, pos):
    rows = maze_ctx["rows"]
    cols = maze_ctx["cols"]
    walls = maze_ctx["walls"]
    r, c = pos
    out = []

    if r > 0 and not walls[r][c]["N"]:
        out.append((r - 1, c))
    if c < cols - 1 and not walls[r][c]["E"]:
        out.append((r, c + 1))
    if r < rows - 1 and not walls[r][c]["S"]:
        out.append((r + 1, c))
    if c > 0 and not walls[r][c]["W"]:
        out.append((r, c - 1))
    return out


def goal_distances(maze_ctx):
    goal = maze_ctx["goal"]
    dists = {goal: 0}
    q = deque([goal])
    while q:
        cur = q.popleft()
        for nxt in neighbors_from_walls(maze_ctx, cur):
            if nxt not in dists:
                dists[nxt] = dists[cur] + 1
                q.append(nxt)
    return dists


def prepare_contexts(mazes):
    max_rows = max(m["rows"] for m in mazes)
    max_cols = max(m["cols"] for m in mazes)
    contexts = []
    for m in mazes:
        path = [tuple(p) for p in m["correct_path"]]
        optimal_actions = {}
        for i in range(len(path) - 1):
            optimal_actions[path[i]] = action_for_step(path[i], path[i + 1])
        ctx = {
            "maze_id": m["maze_id"],
            "rows": m["rows"],
            "cols": m["cols"],
            "goal": tuple(m["end"]),
            "walls": build_wall_grid(m),
            "optimal_actions": optimal_actions,
            "start": tuple(m["start"]),
            "max_rows": max_rows,
            "max_cols": max_cols,
            "max_steps": m["rows"] * m["cols"] * 2,
            "area": m["rows"] * m["cols"],
        }
        ctx["goal_distances"] = goal_distances(ctx)
        contexts.append(ctx)
    return contexts, max_rows, max_cols


def train(
    mazes_dir="mazes",
    epochs=40,
    episodes_per_epoch=250,
    batch_size=128,
    gamma=0.97,
    lr=1e-3,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    target_update_every=200,
    progress_print_every=25,
    step_heartbeat_every=500,
):
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[train] Using device: {device}")
    print(f"[train] Loading mazes from '{mazes_dir}'...")
    mazes = load_mazes(mazes_dir)
    print(f"[train] Loaded {len(mazes)} maze files")
    contexts, max_rows, max_cols = prepare_contexts(mazes)
    print(f"[train] Prepared contexts. max_rows={max_rows}, max_cols={max_cols}")
    print(f"[train] Training config: epochs={epochs}, episodes_per_epoch={episodes_per_epoch}, batch_size={batch_size}")

    input_size = max_rows * max_cols * 6
    policy = SmallMazePolicy(input_size=input_size).to(device)
    target = SmallMazePolicy(input_size=input_size).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    replay = deque(maxlen=100_000)
    epsilon = epsilon_start
    step_count = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"[train] ---- epoch {epoch + 1}/{epochs} started ----", flush=True)
        total_reward = 0.0
        solved = 0

        progress = (epoch + 1) / epochs
        max_area_in_curriculum = int(64 + progress * (max_rows * max_cols - 64))
        eligible_contexts = [m for m in contexts if m["area"] <= max_area_in_curriculum]
        if not eligible_contexts:
            eligible_contexts = contexts
        print(
            f"[train] curriculum max_area={max_area_in_curriculum} "
            f"(eligible_mazes={len(eligible_contexts)}/{len(contexts)})",
            flush=True,
        )

        for episode_idx in range(episodes_per_epoch):
            maze = random.choice(eligible_contexts)
            pos = maze["start"]
            done = False
            steps = 0
            episode_reward = 0.0
            visited = {pos: 1}

            while not done and steps < maze["max_steps"]:
                state = get_observation(maze, pos)

                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        q_values = policy(state.unsqueeze(0).to(device))
                        action = int(torch.argmax(q_values, dim=1).item())

                moved, new_pos = can_move(maze, pos, action)
                expected_action = maze["optimal_actions"].get(pos)
                old_dist = maze["goal_distances"].get(pos, maze["rows"] * maze["cols"])
                new_dist = maze["goal_distances"].get(new_pos, maze["rows"] * maze["cols"])

                reward = -0.01
                if not moved:
                    reward -= 0.30
                elif expected_action is not None and action == expected_action:
                    reward += 0.25
                else:
                    reward -= 0.20

                # Dense shaping: reward getting closer to goal, punish moving away.
                reward += 0.08 * (old_dist - new_dist)

                # Small loop penalty for revisiting the same cell repeatedly.
                revisit_count = visited.get(new_pos, 0)
                if revisit_count > 0:
                    reward -= min(0.03 * revisit_count, 0.15)
                visited[new_pos] = revisit_count + 1

                if new_pos == maze["goal"]:
                    reward += 1.0
                    done = True
                    solved += 1

                next_state = get_observation(maze, new_pos)
                replay.append((state, action, reward, next_state, done))
                pos = new_pos
                episode_reward += reward
                steps += 1
                step_count += 1

                if len(replay) >= batch_size:
                    minibatch = random.sample(replay, batch_size)
                    states = torch.stack([x[0] for x in minibatch]).to(device)
                    actions = torch.tensor([x[1] for x in minibatch], dtype=torch.long, device=device)
                    rewards = torch.tensor([x[2] for x in minibatch], dtype=torch.float32, device=device)
                    next_states = torch.stack([x[3] for x in minibatch]).to(device)
                    dones = torch.tensor([x[4] for x in minibatch], dtype=torch.float32, device=device)

                    q_pred = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        q_next = target(next_states).max(dim=1).values
                        q_target = rewards + gamma * q_next * (1.0 - dones)

                    loss = nn.functional.mse_loss(q_pred, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if step_count % target_update_every == 0:
                    target.load_state_dict(policy.state_dict())

                if step_count % step_heartbeat_every == 0:
                    elapsed = time.time() - epoch_start
                    print(
                        f"[train] epoch={epoch + 1:03d} steps={step_count} "
                        f"episode={episode_idx + 1}/{episodes_per_epoch} elapsed_s={elapsed:.1f}",
                        flush=True,
                    )

            total_reward += episode_reward

            if (episode_idx + 1) % progress_print_every == 0 or (episode_idx + 1) == episodes_per_epoch:
                print(
                    f"[train] epoch={epoch + 1:03d} episode={episode_idx + 1}/{episodes_per_epoch} "
                    f"running_solve_rate={solved / (episode_idx + 1):.3f}",
                    flush=True,
                )

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        avg_reward = total_reward / episodes_per_epoch
        solve_rate = solved / episodes_per_epoch
        epoch_seconds = time.time() - epoch_start
        print(
            f"epoch={epoch + 1:03d} avg_reward={avg_reward:.3f} "
            f"solve_rate={solve_rate:.3f} epsilon={epsilon:.3f} epoch_s={epoch_seconds:.1f}",
            flush=True,
        )

    out_model = Path("maze_small_policy.pt")
    torch.save(policy.state_dict(), out_model)
    meta = {
        "input_size": input_size,
        "max_rows": max_rows,
        "max_cols": max_cols,
        "actions": {"0": "up", "1": "right", "2": "down", "3": "left"},
        "model_path": str(out_model),
    }
    Path("maze_small_policy_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] Saved model to {out_model}")
    print("[train] Saved metadata to maze_small_policy_meta.json")


if __name__ == "__main__":
    train()
