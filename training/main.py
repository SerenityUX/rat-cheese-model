import shutil
import json
import random
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

ShouldTrainModel = True
TrainStrategy = "SupervisedImitation"  # "RL" or "SupervisedImitation"
ShouldRunModelAfterTraining = True
RunInferenceWithMainUI = True
NumMazesToGenerate = 10000
EvalNumMazes = 200
MazeGenerationSeed = 42
# After BC+DAgger, run REINFORCE refine and set maze_policy_final.pt to the best RL checkpoint.
RLEmazeRefine = True
SplitSeed = 42
PreRLEvalNumMazes = 500
PreRLEvalSeed = 424242


def _load_maze_module():
    module_path = Path(__file__).with_name("make-a-maze.py")
    spec = spec_from_file_location("make_a_maze", module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_train_module():
    module_path = Path(__file__).with_name("train-a-small-maze-model.py")
    spec = spec_from_file_location("train_a_small_maze_model", module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_supervised_train_module():
    module_path = Path(__file__).with_name("train-supervised-imitation-maze-model.py")
    spec = spec_from_file_location("train_supervised_imitation_maze_model", module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_module():
    module_path = Path(__file__).with_name("run-maze-model.py")
    spec = spec_from_file_location("run_maze_model", module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_strategy_module(strategy):
    if strategy == "RL":
        return _load_train_module()
    if strategy == "SupervisedImitation":
        return _load_supervised_train_module()
    raise ValueError("strategy must be 'RL' or 'SupervisedImitation'")


def _load_rl_refine():
    module_path = Path(__file__).with_name("rl_refine.py")
    spec = spec_from_file_location("rl_refine", module_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pick_policy_path():
    candidates = [
        Path("maze_policy_final.pt"),
        Path("maze_supervised_policy.pt"),
        Path("maze_supervised_policy_best_saved.pt"),
        Path("maze_supervised_policy_best.pt"),
        Path("maze_small_policy.pt"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No policy checkpoint found. Expected one of: "
        "maze_policy_final.pt, maze_supervised_policy.pt, maze_supervised_policy_best_saved.pt, "
        "maze_supervised_policy_best.pt, maze_small_policy.pt"
    )


def _infer_model_dims(strategy, policy_path, mazes_dir="mazes"):
    if strategy == "SupervisedImitation":
        meta = Path("maze_supervised_policy_meta.json")
        if meta.is_file():
            payload = json.loads(meta.read_text(encoding="utf-8"))
            mr = int(payload.get("max_rows", 0))
            mc = int(payload.get("max_cols", 0))
            if mr > 0 and mc > 0:
                return mr, mc

    strategy_module = _load_strategy_module(strategy)
    mazes = strategy_module.load_mazes(mazes_dir)
    _, max_rows, max_cols = strategy_module.prepare_contexts(mazes)
    return max_rows, max_cols


class MazeInferenceApp:
    def __init__(self, strategy="SupervisedImitation"):
        self.strategy = strategy
        self.strategy_module = _load_strategy_module(strategy)
        self.make_maze_module = _load_maze_module()
        self.policy_path = _pick_policy_path()
        self.max_rows, self.max_cols = _infer_model_dims(strategy, self.policy_path)
        self.device = (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.model = self._load_model().to(self.device)
        self.model.eval()
        self.runtime_dir = Path("runtime_ui_mazes")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.maze_raw = None
        self.maze_ctx = None
        self.pos = None
        self.path_cells = []
        self.prev_pos = None
        self.finished = False
        self.status = "Loading maze..."
        self.app = None
        self.window = None
        self.view = None
        self.timer_target = None
        self._request_new_maze = False
        self.current_svg_path = self.runtime_dir / "current.svg"
        self._step_idx = 0
        self.ui_size = 900.0
        self.steps_in_current_maze = 0
        self.max_steps_current_maze = 0
        self.total_mazes = 0
        self.solved_mazes = 0
        self.failed_mazes = 0

    def _load_model(self):
        input_size = self.max_rows * self.max_cols * 6
        if self.strategy == "SupervisedImitation":
            model = self.strategy_module.SmallMazePolicy(
                input_size=input_size, max_rows=self.max_rows, max_cols=self.max_cols
            )
        else:
            model = self.strategy_module.SmallMazePolicy(input_size=input_size)
        state = torch.load(self.policy_path, map_location=self.device)
        model.load_state_dict(state)
        return model

    def _generate_one_maze(self):
        seed = random.randint(0, 10_000_000)
        generated = self.make_maze_module.generate_maze_dataset(
            num=1,
            output_dir=str(self.runtime_dir),
            min_size=8,
            max_size=min(14, self.max_rows, self.max_cols),
            seed=seed,
            verbose=False,
        )
        json_path = Path(generated[0])
        return json.loads(json_path.read_text(encoding="utf-8"))

    def _build_ctx(self, maze):
        return {
            "maze_id": maze["maze_id"],
            "rows": maze["rows"],
            "cols": maze["cols"],
            "goal": tuple(maze["end"]),
            "walls": self.strategy_module.build_wall_grid(maze),
            "start": tuple(maze["start"]),
            "max_rows": self.max_rows,
            "max_cols": self.max_cols,
            "max_steps": maze["rows"] * maze["cols"] * 2,
        }

    def _predict_action(self):
        obs = self.strategy_module.get_observation(self.maze_ctx, self.pos).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(obs)
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits.squeeze(0)
        valid_actions = []
        for a in range(4):
            moved, nxt = self.strategy_module.can_move(self.maze_ctx, self.pos, a)
            if moved:
                valid_actions.append((a, nxt))
        if valid_actions:
            # Avoid immediate A->B->A oscillations unless backtracking is the only move.
            if self.prev_pos is not None:
                non_backtracking = [(a, nxt) for a, nxt in valid_actions if nxt != self.prev_pos]
                if non_backtracking:
                    return int(max(non_backtracking, key=lambda t: float(logits[t[0]].item()))[0])
            return int(max(valid_actions, key=lambda t: float(logits[t[0]].item()))[0])
        return int(torch.argmax(logits).item())

    def _svg_for_state(self):
        cell = int(self.maze_raw["cell_size"])
        maze_w = int(self.maze_raw["cols"]) * cell
        maze_h = int(self.maze_raw["rows"]) * cell
        pad = 16
        side = max(maze_w, maze_h) + pad * 2
        off_x = (side - maze_w) / 2
        off_y = (side - maze_h) / 2
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="{side}" viewBox="0 0 {side} {side}">',
            '<rect x="0" y="0" width="100%" height="100%" fill="white" />',
        ]
        for (x1, y1), (x2, y2) in self.maze_raw["lines"]:
            parts.append(
                f'<line x1="{x1 + off_x}" y1="{y1 + off_y}" x2="{x2 + off_x}" y2="{y2 + off_y}" stroke="black" stroke-width="2" />'
            )

        sr, sc = self.maze_ctx["start"]
        gr, gc = self.maze_ctx["goal"]
        sx, sy = (sc * cell + cell / 2 + off_x, sr * cell + cell / 2 + off_y)
        gx, gy = (gc * cell + cell / 2 + off_x, gr * cell + cell / 2 + off_y)
        parts.append(f'<circle cx="{sx}" cy="{sy}" r="5" fill="green" />')
        parts.append(f'<circle cx="{gx}" cy="{gy}" r="5" fill="blue" />')

        if len(self.path_cells) >= 2:
            coords = []
            for r, c in self.path_cells:
                coords.append(f"{c * cell + cell / 2 + off_x},{r * cell + cell / 2 + off_y}")
            parts.append(
                '<polyline points="'
                + " ".join(coords)
                + '" fill="none" stroke="red" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />'
            )

        pr, pc = self.pos
        px, py = (pc * cell + cell / 2 + off_x, pr * cell + cell / 2 + off_y)
        parts.append(f'<circle cx="{px}" cy="{py}" r="6" fill="#ff9800" />')
        parts.append("</svg>")
        return "\n".join(parts), side, side

    def _render(self, status):
        self.status = status
        svg_text, width, height = self._svg_for_state()
        self.current_svg_path.write_text(svg_text, encoding="utf-8")
        step_svg = self.runtime_dir / f"step_{self._step_idx:05d}.svg"
        step_svg.write_text(svg_text, encoding="utf-8")
        self._step_idx += 1
        if self.window is not None and self.view is not None:
            from Cocoa import NSMakeRect

            solve_rate = (self.solved_mazes / self.total_mazes) if self.total_mazes > 0 else 0.0
            title = (
                f"Escape The Maze - {status} | maze_steps={self.steps_in_current_maze}/{self.max_steps_current_maze} "
                f"| solved={self.solved_mazes} failed={self.failed_mazes} total={self.total_mazes} "
                f"solve_rate={solve_rate:.3f}"
            )
            self.window.setTitle_(title)
            self.view.setFrame_(NSMakeRect(0, 0, self.ui_size, self.ui_size))
            self.view.setNeedsDisplay_(True)

    def _advance_one_step(self):
        old_pos = self.pos
        action = self._predict_action()
        moved, nxt = self.strategy_module.can_move(self.maze_ctx, self.pos, action)
        self.pos = nxt if moved else self.pos
        self.prev_pos = old_pos
        if not self.path_cells or self.path_cells[-1] != self.pos:
            self.path_cells.append(self.pos)
        self.finished = self.pos == self.maze_ctx["goal"]

    def _start_new_maze(self):
        self.maze_raw = self._generate_one_maze()
        self.maze_ctx = self._build_ctx(self.maze_raw)
        self.pos = self.maze_ctx["start"]
        self.prev_pos = None
        self.path_cells = [self.pos]
        self.finished = False
        self.steps_in_current_maze = 0
        self.max_steps_current_maze = int(self.maze_ctx["max_steps"])
        self.total_mazes += 1
        self._render("Solving... one move per second")

    def request_new_maze(self):
        if self.finished:
            self._request_new_maze = True

    def _on_tick(self):
        if self.finished:
            if self._request_new_maze:
                self._request_new_maze = False
                self._start_new_maze()
            return
        self._advance_one_step()
        self.steps_in_current_maze += 1
        if self.finished:
            self.solved_mazes += 1
            self._render("Solved. Click maze to generate and solve a new one.")
        elif self.steps_in_current_maze >= self.max_steps_current_maze:
            self.finished = True
            self.failed_mazes += 1
            self._render("Failed (max steps reached). Click to generate next maze.")
        else:
            self._render("Solving... at model speed")

    def _draw_in_view(self):
        from Cocoa import NSBezierPath, NSColor, NSMakeRect, NSRectFill

        if self.maze_raw is None or self.maze_ctx is None or self.pos is None:
            return

        cell = int(self.maze_raw["cell_size"])
        maze_w = int(self.maze_raw["cols"]) * cell
        maze_h = int(self.maze_raw["rows"]) * cell
        bounds = self.view.bounds()
        off_x = (bounds.size.width - maze_w) / 2.0
        off_y = (bounds.size.height - maze_h) / 2.0
        NSColor.whiteColor().setFill()
        NSRectFill(bounds)

        NSColor.blackColor().setStroke()
        wall_path = NSBezierPath.bezierPath()
        wall_path.setLineWidth_(2.0)
        for (x1, y1), (x2, y2) in self.maze_raw["lines"]:
            wall_path.moveToPoint_((float(x1) + off_x, float(y1) + off_y))
            wall_path.lineToPoint_((float(x2) + off_x, float(y2) + off_y))
        wall_path.stroke()

        sr, sc = self.maze_ctx["start"]
        gr, gc = self.maze_ctx["goal"]
        sx, sy = (sc * cell + cell / 2 + off_x, sr * cell + cell / 2 + off_y)
        gx, gy = (gc * cell + cell / 2 + off_x, gr * cell + cell / 2 + off_y)

        NSColor.greenColor().setFill()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(sx - 5, sy - 5, 10, 10)).fill()
        NSColor.blueColor().setFill()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(gx - 5, gy - 5, 10, 10)).fill()

        if len(self.path_cells) >= 2:
            NSColor.redColor().setStroke()
            line = NSBezierPath.bezierPath()
            line.setLineWidth_(3.0)
            r0, c0 = self.path_cells[0]
            line.moveToPoint_((c0 * cell + cell / 2 + off_x, r0 * cell + cell / 2 + off_y))
            for r, c in self.path_cells[1:]:
                line.lineToPoint_((c * cell + cell / 2 + off_x, r * cell + cell / 2 + off_y))
            line.stroke()

        pr, pc = self.pos
        px, py = (pc * cell + cell / 2 + off_x, pr * cell + cell / 2 + off_y)
        NSColor.orangeColor().setFill()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(px - 6, py - 6, 12, 12)).fill()

    def run(self):
        try:
            import objc
            from Cocoa import (
                NSApp,
                NSApplication,
                NSBackingStoreBuffered,
                NSMakeRect,
                NSRectFill,
                NSTimer,
                NSView,
                NSWindow,
                NSWindowStyleMaskClosable,
                NSWindowStyleMaskMiniaturizable,
                NSWindowStyleMaskResizable,
                NSWindowStyleMaskTitled,
            )
            from Foundation import NSObject
        except Exception as exc:
            raise RuntimeError(
                "PyObjC is not installed. Install with: python3 -m pip install pyobjc"
            ) from exc

        print(
            f"[main-ui] Starting inference UI. strategy={self.strategy} "
            f"policy='{self.policy_path}' device={self.device}"
        )

        app_ref = self

        class MazeCanvasView(NSView):
            def isFlipped(self):
                return True

            def drawRect_(self, _rect):
                NSRectFill(self.bounds())
                app_ref._draw_in_view()

            def mouseDown_(self, _event):
                app_ref.request_new_maze()

        class TickTarget(NSObject):
            def tick_(self, _timer):
                app_ref._on_tick()

        self.app = NSApplication.sharedApplication()
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskResizable
            | NSWindowStyleMaskMiniaturizable
        )
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(100.0, 100.0, self.ui_size, self.ui_size),
            style,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_("Escape The Maze")

        self.view = MazeCanvasView.alloc().initWithFrame_(
            NSMakeRect(0.0, 0.0, self.ui_size, self.ui_size)
        )
        self.window.setContentView_(self.view)
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

        self._start_new_maze()
        self.timer_target = TickTarget.alloc().init()
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.0, self.timer_target, "tick:", None, True
        )

        self.app.run()


def generate_maze(num, seed=None):
    maze_module = _load_maze_module()
    return maze_module.generate_maze_dataset(
        num=num,
        output_dir="mazes",
        seed=seed,
        verbose=True,
    )


if __name__ == "__main__":
    if RunInferenceWithMainUI:
        MazeInferenceApp(strategy=TrainStrategy).run()
        raise SystemExit(0)

    print(f"[main] Generating mazes... count={NumMazesToGenerate}, seed={MazeGenerationSeed}")
    generated = generate_maze(NumMazesToGenerate, seed=MazeGenerationSeed)
    print(f"[main] Maze generation complete. Files created: {len(generated)}")

    final_policy_path = Path("maze_policy_final.pt")
    current_best_saved = Path("maze_supervised_policy_best_saved.pt")
    current_best = Path("maze_supervised_policy_best.pt")

    if ShouldTrainModel:
        print(f"[main] ShouldTrainModel=True -> starting training strategy='{TrainStrategy}'")
        if TrainStrategy == "RL":
            train_module = _load_train_module()
            train_module.train(mazes_dir="mazes")
        elif TrainStrategy == "SupervisedImitation":
            train_module = _load_supervised_train_module()
            train_result = train_module.train(
                mazes_dir="mazes",
                epochs=30,
                batch_size=512,
                val_split=0.1,
                split_seed=SplitSeed,
                maze_acc_val_sample=1000,
                early_stop_maze_acc_patience=7,
                dagger_phases=5,
                dagger_sub_epochs=3,
            )
            if current_best.is_file():
                shutil.copy2(current_best, current_best_saved)
                print(
                    f"[main] Refreshed {current_best_saved} from this run's {current_best} "
                    f"(best maze_accuracy={train_result.get('best_maze')})"
                )
            else:
                print(f"[main] WARNING: {current_best} not found after training")
        else:
            raise ValueError("TrainStrategy must be 'RL' or 'SupervisedImitation'")

        if TrainStrategy == "SupervisedImitation" and RLEmazeRefine:
            run_module = _load_run_module()
            pre_rl_policy = current_best_saved if current_best_saved.is_file() else Path("maze_supervised_policy.pt")
            print(
                f"[main] Pre-RL check on random mazes using {pre_rl_policy} "
                f"(num_mazes={PreRLEvalNumMazes}, eval_seed={PreRLEvalSeed})..."
            )
            run_module.run_model(
                strategy=TrainStrategy,
                mazes_dir="mazes",
                num_mazes=PreRLEvalNumMazes,
                eval_seed=PreRLEvalSeed,
                policy_path=str(pre_rl_policy),
            )
            print(
                "[main] RL refine: using this run's refreshed maze_supervised_policy_best_saved.pt "
                "(reproducible start), then PRE-RL val+holdout metrics, then long RL (see rl_refine.py)..."
            )
            rl = _load_rl_refine()
            rl.refine(
                mazes_dir="mazes",
                policy_in=str(current_best_saved),
                val_split=0.1,
                split_seed=SplitSeed,
                rollout_val_seed=SplitSeed + 100_001,
                maze_acc_val_sample=1000,
                pre_rl_eval_seed=88_000,
            )
        elif TrainStrategy == "SupervisedImitation" and not RLEmazeRefine:
            shutil.copy2("maze_supervised_policy.pt", final_policy_path)
            print(
                f"[main] RLEmazeRefine=False -> copied maze_supervised_policy.pt -> {final_policy_path}"
            )
    else:
        print("[main] ShouldTrainModel=False -> skipping model training")

    if ShouldRunModelAfterTraining and TrainStrategy == "SupervisedImitation":
        run_module = _load_run_module()
        p = final_policy_path
        if p.is_file():
            print(f"[main] Evaluating final policy: {p}")
        elif Path("maze_supervised_policy.pt").is_file():
            p = Path("maze_supervised_policy.pt")
            print(
                f"[main] {final_policy_path} not found; evaluating imitation weights {p} instead"
            )
        else:
            print(
                f"[main] No maze_policy_final.pt or maze_supervised_policy.pt; run_model may fail"
            )
        run_module.run_model(
            strategy=TrainStrategy,
            mazes_dir="mazes",
            num_mazes=EvalNumMazes,
            policy_path=str(p),
        )
    elif ShouldRunModelAfterTraining:
        print(f"[main] Running trained model with strategy='{TrainStrategy}'")
        run_module = _load_run_module()
        run_module.run_model(strategy=TrainStrategy, mazes_dir="mazes", num_mazes=EvalNumMazes)

    if TrainStrategy == "SupervisedImitation" and RLEmazeRefine:
        print(
            "[main] Artifacts: maze_supervised_policy.pt (imitation+DAgger best) -> "
            "maze_rl_refined_policy.pt / maze_policy_final.pt (RL). "
            "Aim: maximum val maze_accuracy; 1.0 is not guaranteed."
        )
