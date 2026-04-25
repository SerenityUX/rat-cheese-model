import json
import random
from collections import deque
from pathlib import Path


def _neighbors(r, c, rows, cols):
    dirs = [(-1, 0, "N", "S"), (1, 0, "S", "N"), (0, -1, "W", "E"), (0, 1, "E", "W")]
    for dr, dc, direction, opposite in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc, direction, opposite


def _build_maze(rows, cols, rng):
    walls = [[{"N": True, "E": True, "S": True, "W": True} for _ in range(cols)] for _ in range(rows)]
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    stack = [(0, 0)]
    visited[0][0] = True

    while stack:
        r, c = stack[-1]
        candidates = []
        for nr, nc, direction, opposite in _neighbors(r, c, rows, cols):
            if not visited[nr][nc]:
                candidates.append((nr, nc, direction, opposite))

        if not candidates:
            stack.pop()
            continue

        nr, nc, direction, opposite = rng.choice(candidates)
        walls[r][c][direction] = False
        walls[nr][nc][opposite] = False
        visited[nr][nc] = True
        stack.append((nr, nc))

    return walls


def _adjacency_from_walls(walls):
    rows = len(walls)
    cols = len(walls[0])
    graph = {}
    for r in range(rows):
        for c in range(cols):
            cur = (r, c)
            graph[cur] = []
            if not walls[r][c]["N"]:
                graph[cur].append((r - 1, c))
            if not walls[r][c]["S"]:
                graph[cur].append((r + 1, c))
            if not walls[r][c]["W"]:
                graph[cur].append((r, c - 1))
            if not walls[r][c]["E"]:
                graph[cur].append((r, c + 1))
    return graph


def _solve_path(graph, start, end):
    q = deque([start])
    prev = {start: None}
    while q:
        node = q.popleft()
        if node == end:
            break
        for nxt in graph[node]:
            if nxt not in prev:
                prev[nxt] = node
                q.append(nxt)

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _wall_lines(walls, cell_size):
    rows = len(walls)
    cols = len(walls[0])
    lines = []

    for r in range(rows):
        for c in range(cols):
            x = c * cell_size
            y = r * cell_size
            cell = walls[r][c]

            if cell["N"]:
                lines.append([[x, y], [x + cell_size, y]])
            if cell["W"]:
                lines.append([[x, y], [x, y + cell_size]])
            if r == rows - 1 and cell["S"]:
                lines.append([[x, y + cell_size], [x + cell_size, y + cell_size]])
            if c == cols - 1 and cell["E"]:
                lines.append([[x + cell_size, y], [x + cell_size, y + cell_size]])

    return lines


def _svg_maze(lines, width, height, out_file, path_points=None):
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white" />',
    ]

    for (x1, y1), (x2, y2) in lines:
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="2" />')

    if path_points:
        coords = " ".join(f"{x},{y}" for x, y in path_points)
        svg.append(
            f'<polyline points="{coords}" fill="none" stroke="red" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />'
        )
        sx, sy = path_points[0]
        ex, ey = path_points[-1]
        svg.append(f'<circle cx="{sx}" cy="{sy}" r="4" fill="green" />')
        svg.append(f'<circle cx="{ex}" cy="{ey}" r="4" fill="blue" />')

    svg.append("</svg>")
    Path(out_file).write_text("\n".join(svg), encoding="utf-8")


def _cell_path_to_pixel_centers(path_cells, cell_size):
    return [
        (c * cell_size + cell_size // 2, r * cell_size + cell_size // 2)
        for (r, c) in path_cells
    ]


def generate_maze_dataset(
    num,
    output_dir="mazes",
    min_size=8,
    max_size=14,
    cell_size=24,
    seed=None,
    verbose=True,
):
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    generated = []
    progress_step = max(1, num // 10) if num > 0 else 1

    if verbose:
        print(f"[maze-gen] Starting generation: num={num}, output_dir='{output_dir}'")

    for i in range(num):
        rows = rng.randint(min_size, max_size)
        cols = rng.randint(min_size, max_size)
        walls = _build_maze(rows, cols, rng)
        graph = _adjacency_from_walls(walls)
        start = (0, 0)
        end = (rows - 1, cols - 1)
        correct_path = _solve_path(graph, start, end)

        width = cols * cell_size
        height = rows * cell_size
        lines = _wall_lines(walls, cell_size)

        maze_id = f"maze_{i:06d}"
        maze_dir = out_root / maze_id
        maze_dir.mkdir(parents=True, exist_ok=True)

        unsolved_img = maze_dir / "unsolved.svg"
        solved_img = maze_dir / "solved.svg"
        json_file = maze_dir / "maze.json"

        _svg_maze(lines, width, height, unsolved_img)
        _svg_maze(lines, width, height, solved_img, _cell_path_to_pixel_centers(correct_path, cell_size))

        payload = {
            "maze_id": maze_id,
            "rows": rows,
            "cols": cols,
            "cell_size": cell_size,
            "start": list(start),
            "end": list(end),
            "lines": lines,
            "correct_path": [list(p) for p in correct_path],
            "images": {
                "unsolved": str(unsolved_img),
                "solved": str(solved_img),
            },
        }
        json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        generated.append(str(json_file))

        if verbose and ((i + 1) % progress_step == 0 or i + 1 == num):
            print(f"[maze-gen] Generated {i + 1}/{num} mazes")

    if verbose:
        print(f"[maze-gen] Done. Total generated: {len(generated)}")
    return generated
