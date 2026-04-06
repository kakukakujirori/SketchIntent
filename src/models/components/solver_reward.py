import os
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

import sketchgraphs.data as datalib
from src.models.components.solver_feedback import SolverFeedback
from src.models.components.solver_worker import init_worker as _init_worker, evaluate_single as _evaluate_single_worker
from src.models.components.token_visualizer import visualize_constraint_sequence
from src.sketch_handlers.constants import ConstraintToken, CONSTRAINT_VAL_OFFSET
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer

_worker_solver_feedback: SolverFeedback | None = None


@dataclass
class SolverAsyncHandle:
    """Opaque handle returned by SolverReward.submit_async().

    Pass to SolverReward.collect_results() to retrieve (scores_cpu, metrics).
    GPU work can be performed between submit_async() and collect_results() so
    that the CPU solver runs concurrently with GPU logprob / ref-model forward.
    """
    futures:             list          # ProcessPoolExecutor Futures (キャッシュミス分)
    indices_to_evaluate: list[int]    # futures に対応するバッチインデックス
    cache_hits:          list[tuple]  # (batch_idx, cached_result) のリスト
    first_stop_indices:  list[int]    # 各サンプルの STOP トークン位置
    args_list:           list         # worker へ渡した (prim, cons, sketch) リスト（キャッシュ書き込み用）
    B:                   int          # バッチサイズ
    cons_shape:          tuple        # scores テンソルの形状


class SolverReward(torch.nn.Module):
    def __init__(
        self,
        context: SketchContext,
        reward_dof_reduction: float = 1.0,
        reward_fully_constrained: float = 1.0,
        penalty_conflicting: float = -1.0,
        penalty_redundant: float = -1.0,
        penalty_malformed: float = -0.5,
        penalty_deformed: float = -0.25,
        penalty_per_problematic_constraint: float = -1.0,
        penalty_num_total_constraints: float = -0.01,
        penalty_num_dimensional_constraints: float = -0.2,
        fully_constrained_dof: int = 2,
        distortion_threshold: float | None = None,
    ):
        super().__init__()
        self.sketch_tokenizer = SketchTokenizer(context)
        self.fully_constrained_dof = fully_constrained_dof

        assert all(v <= 0 for v in [penalty_conflicting, penalty_redundant, penalty_malformed, penalty_deformed, penalty_per_problematic_constraint, penalty_num_total_constraints, penalty_num_dimensional_constraints]), \
            f"{penalty_conflicting=}, {penalty_redundant=}, {penalty_malformed=}, {penalty_deformed=}, {penalty_per_problematic_constraint=}, {penalty_num_total_constraints=}, {penalty_num_dimensional_constraints=}"

        self.reward_dof_reduction = reward_dof_reduction
        self.reward_fully_constrained = reward_fully_constrained
        self.penalty_conflicting = penalty_conflicting
        self.penalty_redundant = penalty_redundant
        self.penalty_malformed = penalty_malformed
        self.penalty_deformed = penalty_deformed
        self.penalty_per_problematic_constraint = penalty_per_problematic_constraint
        self.penalty_num_total_constraints = penalty_num_total_constraints
        self.penalty_num_dimensional_constraints = penalty_num_dimensional_constraints

        # Execution and caching
        self.context = context
        self.distortion_threshold = distortion_threshold or 1/context.num_quantization_bins
        # Limit parallel workers: too many concurrent FreeCAD inits causes deadlocks.
        self.max_workers = min(8, max(1, (os.cpu_count() or 1) // 4))
        self._evaluation_cache = {}
        self.max_cache_size = 20000

        # Worker timeout per task (seconds). FreeCAD can occasionally hang.
        self.worker_timeout = 60

        # max_tasks_per_child: each worker is recycled after this many tasks to
        # prevent FreeCAD's internal memory leaks from accumulating indefinitely.
        # Set high to minimize respawn frequency (each spawn risks a deadlock).
        self._max_tasks_per_child = 512
        self._executor: ProcessPoolExecutor | None = None


    def _get_executor(self) -> ProcessPoolExecutor:
        """Return the persistent ProcessPoolExecutor, creating it if needed.

        Workers are spawned (not forked) to avoid CUDA/Qt context contamination,
        and are automatically recycled every `_max_tasks_per_child` calls to
        counteract FreeCAD's internal memory leaks.

        The initializer and worker function are imported from solver_worker.py,
        which is a torch-free module. This ensures spawned workers never trigger
        CUDA initialization.
        """
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(self.context, self.distortion_threshold),
                max_tasks_per_child=self._max_tasks_per_child,
            )
        return self._executor

    def __del__(self):
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
            self._executor = None

    def _get_dof_without_constraints(self, sketch: datalib.Sketch) -> int:
        dof = 0
        for entity in sketch.entities.values():
            if isinstance(entity, datalib.Point):
                dof += 2
            elif isinstance(entity, datalib.Line):
                dof += 4
            elif isinstance(entity, datalib.Circle):
                dof += 3
            elif isinstance(entity, datalib.Arc):
                dof += 5
            else:
                raise ValueError(f"Unknown entity type: {entity}")
        return dof

    def sort_intra_refs(self, cons_tokens: np.ndarray) -> np.ndarray:
        """
        Sort the refs of each constraint in the (predicted) constraint tokens.
        """
        if len(cons_tokens) == 0:
            return cons_tokens

        assert len(cons_tokens.shape) == 1, f"Expected 1D array, got {cons_tokens.shape}"

        prefix_token = cons_tokens[0:1] if cons_tokens[0] == ConstraintToken.Start else np.array([])
        body_tokens = cons_tokens[1:] if cons_tokens[0] == ConstraintToken.Start else cons_tokens

        if self.sketch_tokenizer.align_constraint_token_length_to > 0:
            cons_len = body_tokens.shape[0] // self.sketch_tokenizer.align_constraint_token_length_to
            body_tokens_reshaped = body_tokens[:cons_len * self.sketch_tokenizer.align_constraint_token_length_to].reshape(
                cons_len, self.sketch_tokenizer.align_constraint_token_length_to)
            for i in range(cons_len):
                cons_id, ref1, ref2, ref3 = body_tokens_reshaped[i].tolist()
                if cons_id in ConstraintToken._value2member_map_:
                    cons_type = ConstraintToken(cons_id).name
                    handler = self.sketch_tokenizer.context.get_constraint_handler(cons_type)
                    if handler is not None:
                        true_refs = tuple(r for r in [ref1, ref2, ref3] if r >= CONSTRAINT_VAL_OFFSET)
                        pad_refs = tuple(r for r in [ref1, ref2, ref3] if r < CONSTRAINT_VAL_OFFSET)
                        sorted_true_refs = handler._sort_ref_tokens(true_refs)
                        body_tokens_reshaped[i] = np.array([cons_id, *sorted_true_refs, *pad_refs])
                    else:
                        continue  # cons_type can be STOP or PAD
                else:
                    # Non-existing constraint type. We don't raise Exception here since it's outside the scope of this function
                    continue
            body_tokens = body_tokens_reshaped.flatten()
        else:
            raise NotImplementedError("IMPLEMENT IT!!!")

        return np.concatenate([prefix_token, body_tokens])

    # ---------------------------------------------------------------------- #
    #  Async API                                                              #
    #  Use submit_async() then collect_results() to overlap CPU solver with   #
    #  GPU computation in the training loop.                                  #
    # ---------------------------------------------------------------------- #

    def submit_async(
        self,
        prim_tokens: torch.Tensor,
        cons_tokens: torch.Tensor,
        sketch_original_list: list[datalib.Sketch] | None = None,
    ) -> SolverAsyncHandle:
        """Pre-process tokens, check cache, and submit solver jobs to the pool.

        Returns immediately with a SolverAsyncHandle — the solver workers run
        concurrently in the background.  Call collect_results(handle) to wait
        for completion and obtain (scores_cpu, metrics).

        Args:
            prim_tokens: (B, L_prim)  — may be on GPU or CPU
            cons_tokens: (B, L_cons)  — may be on GPU or CPU
            sketch_original_list: Optional list of original Sketch objects.
        """
        assert prim_tokens.size(0) == cons_tokens.size(0), \
            f"Batch size mismatch: {prim_tokens.shape=} != {cons_tokens.shape=}"
        B = prim_tokens.size(0)
        _sketch_list = sketch_original_list if sketch_original_list is not None else [None] * B
        assert B == len(_sketch_list), \
            f"Batch size mismatch: {prim_tokens.shape=} != {len(_sketch_list)=}"

        # Clear cache if it gets too large
        if len(self._evaluation_cache) > self.max_cache_size:
            self._evaluation_cache.clear()

        # Step A: Transfer tokens to CPU and pre-process
        prim_tokens_np = prim_tokens.detach().cpu().numpy()  # [B, L_prim]
        cons_tokens_np = cons_tokens.detach().cpu().numpy()  # [B, L_cons]

        args_list: list = []
        indices_to_evaluate: list[int] = []
        cache_hits: list[tuple] = []
        first_stop_indices: list[int] = []

        for i in range(B):
            prim = prim_tokens_np[i]
            cons = cons_tokens_np[i].copy()  # copy so we can mutate (PAD fill) safely
            sketch_original = _sketch_list[i]

            # Replace after-STOP tokens by PAD (to prevent false cache matching)
            stop_indices = (cons == ConstraintToken.Stop).nonzero()[0]
            first_stop_idx = int(stop_indices[0]) if len(stop_indices) > 0 else -1
            if first_stop_idx >= 0:
                cons[first_stop_idx + 1:] = ConstraintToken.Pad
            first_stop_indices.append(first_stop_idx)

            # tobytes() is O(1) — much faster than tuple(tolist()) for large arrays.
            cache_key = (prim.tobytes(), cons.tobytes())
            if cache_key in self._evaluation_cache:
                cache_hits.append((i, self._evaluation_cache[cache_key]))
            else:
                indices_to_evaluate.append(i)
                args_list.append((prim, cons, sketch_original))

        # Step B: Submit missing items to ProcessPoolExecutor (non-blocking)
        futures: list = []
        if args_list:
            executor = self._get_executor()
            futures = [executor.submit(_evaluate_single_worker, arg) for arg in args_list]

        return SolverAsyncHandle(
            futures=futures,
            indices_to_evaluate=indices_to_evaluate,
            cache_hits=cache_hits,
            first_stop_indices=first_stop_indices,
            args_list=args_list,
            B=B,
            cons_shape=tuple(cons_tokens.shape),
        )

    def collect_results(
        self,
        handle: SolverAsyncHandle,
    ) -> tuple[torch.Tensor, dict[str, Counter]]:
        """Wait for solver workers and compute reward scores.

        Blocks until all futures in *handle* have completed (or timed out).
        Returns (scores_cpu, metrics) where scores_cpu is a CPU float32 tensor
        of shape (B, L_cons).

        If called after GPU forward passes have already consumed most of the
        solver latency, this returns almost immediately.
        """
        # Collect future results
        evaluation_results: list[Any] = [None] * handle.B

        # Fill cache hits first
        for i, cached_res in handle.cache_hits:
            evaluation_results[i] = cached_res

        # Wait for submitted futures (with per-task timeout)
        if handle.futures:
            for fut_idx, f in enumerate(handle.futures):
                try:
                    res = f.result(timeout=self.worker_timeout)
                except FutureTimeoutError:
                    print(f"[SolverReward] Worker timeout ({self.worker_timeout}s) for batch index {i}")
                    res = None
                except Exception as e:
                    print(f"[SolverReward] Worker failed for batch index {i}: {e}")
                    res = None

                i = handle.indices_to_evaluate[fut_idx]
                evaluation_results[i] = res

                if res is not None:
                    # Only cache successful evaluations
                    prim, cons, _ = handle.args_list[fut_idx]
                    cache_key = (prim.tobytes(), cons.tobytes())
                    self._evaluation_cache[cache_key] = res

        # Step C: Compute reward scores from evaluation results
        scores = torch.zeros(handle.cons_shape, dtype=torch.float32)
        solver_dof_counter: Counter[int] = Counter()
        solver_status_counter: Counter[str] = Counter()

        for i in range(handle.B):
            res = evaluation_results[i]
            first_stop_idx = handle.first_stop_indices[i]

            if res is None:
                # Worker timed out or raised an exception; treat as evaluation failure.
                # Score stays 0, and we record a sentinel status for logging.
                solver_dof_counter[-1] += 1
                solver_status_counter["Timeout"] += 1
                continue

            solver_status, sketch, constraint_to_token_start, invalid_constraint_block_start = res

            # Categorize status for logging
            status_name = solver_status.Status
            if status_name == "Converged":
                if solver_status.DoF <= self.fully_constrained_dof:
                    status_name = "FC"
                else:
                    status_name = "Under"

            solver_dof_counter[solver_status.DoF] += 1
            solver_status_counter[status_name] += 1

            ################################################################
            # 1. Constraint-wise penalty
            ################################################################
            problematic_constraint_ids = set(
                solver_status.ConflictingConstraints +
                solver_status.RedundantConstraints +
                solver_status.MalformedConstraints
            )

            # Gather token start positions for all problematic constraints and malformed/invalid blocks.
            problematic_token_starts = sorted(set(
                [constraint_to_token_start[c_id] for c_id in problematic_constraint_ids] +
                list(invalid_constraint_block_start.keys())))

            if self.sketch_tokenizer.align_constraint_token_length_to > 0:
                problematic_token_ends = np.array(
                    [s + self.sketch_tokenizer.align_constraint_token_length_to - 1
                     for s in problematic_token_starts],
                    dtype=np.int64,
                )
            else:
                raise NotImplementedError("IMPLEMENT IT!!!")

            scores[i, problematic_token_ends] += self.penalty_per_problematic_constraint

            ################################################################
            # 2. Sequence-wise penalty / reward
            ################################################################
            seq_reward = 0.0

            if solver_status.Status != "Converged":
                # Invalid sequence
                invalid_penalties = []
                if solver_status.ConflictingConstraints: invalid_penalties.append(self.penalty_conflicting)
                if solver_status.RedundantConstraints:   invalid_penalties.append(self.penalty_redundant)
                if solver_status.MalformedConstraints:   invalid_penalties.append(self.penalty_malformed)

                # heaviest penalty as the sequence-level penalty
                seq_reward = min(invalid_penalties) if invalid_penalties else 0.0

            else:
                # Valid / converged sequence
                orig_dof = self._get_dof_without_constraints(sketch)
                dof_reduction_rate = (orig_dof - solver_status.DoF) / max(1, orig_dof - self.fully_constrained_dof)
                seq_reward = self.reward_dof_reduction * dof_reduction_rate

                if solver_status.DistortingConstraints:
                    seq_reward += self.penalty_deformed
                elif solver_status.DoF <= self.fully_constrained_dof:
                    seq_reward += self.reward_fully_constrained

                # dimension constraints penalty
                NumTotalConstraints = solver_status.NumRelationalConstraints + solver_status.NumDimensionalConstraints
                seq_reward += self.penalty_num_total_constraints * NumTotalConstraints
                seq_reward += self.penalty_num_dimensional_constraints * (
                    solver_status.NumDimensionalConstraints / max(1, NumTotalConstraints))

            ################################################################
            # 3. Assign the sequence reward at the STOP token position
            ################################################################
            scores[i, first_stop_idx] += seq_reward

        metrics: dict[str, Counter] = {
            "dof": solver_dof_counter,
            "status": solver_status_counter,
        }
        return scores, metrics

    # ---------------------------------------------------------------------- #
    #  Synchronous API (backward-compatible wrapper)                          #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        prim_tokens: torch.Tensor,
        cons_tokens: torch.Tensor,
        sketch_original_list: list[datalib.Sketch] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Counter]]:
        """Synchronous reward computation (submit + collect in one call).

        Kept for backward compatibility (used in validation_step etc.).
        For training, prefer calling submit_async() before GPU forward passes
        and collect_results() after, to overlap CPU solver with GPU compute.

        Args:
            prim_tokens: (B, L_prim)
            cons_tokens: (B, L_cons)
            sketch_original_list: Original sketch objects for precise prim values.

        Returns:
            scores: (B, L_cons) float tensor on the same device as cons_tokens.
            metrics: dict with 'dof' and 'status' Counters.
        """
        handle = self.submit_async(prim_tokens, cons_tokens, sketch_original_list)
        scores, metrics = self.collect_results(handle)
        return scores.to(cons_tokens.device), metrics
