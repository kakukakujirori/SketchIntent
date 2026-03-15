import numpy as np
import torch

import sketchgraphs.data as datalib
from src.models.components.solver_feedback import SolverFeedback
from src.models.components.token_visualizer import visualize_constraint_sequence
from src.sketch_handlers.constants import ConstraintToken, CONSTRAINT_VAL_OFFSET
from src.sketch_handlers.context import SketchContext
from src.sketch_handlers.sketch_tokenizer import SketchTokenizer


class SolverReward(torch.nn.Module):
    def __init__(
        self,
        context: SketchContext,
        penalty_conflicting: float = -1.0,
        penalty_redundant: float = -1.0,
        penalty_malformed: float = -0.5,
        penalty_deformed: float = -0.25,
        penalty_per_constraint: float = -1.0,
        fully_constrained_dof: int = 2,
        distortion_threshold: float | None = None,
    ):
        super().__init__()
        self.solver_feedback = SolverFeedback(context, distortion_threshold = distortion_threshold or 1/context.num_quantization_bins)
        self.sketch_tokenizer = SketchTokenizer(context)
        self.fully_constrained_dof = fully_constrained_dof

        assert all(v <= 0 for v in [penalty_conflicting, penalty_redundant, penalty_malformed, penalty_deformed, penalty_per_constraint]), \
            f"{penalty_conflicting=}, {penalty_redundant=}, {penalty_malformed=}, {penalty_deformed=}, {penalty_per_constraint=}"

        self.penalty_conflicting = penalty_conflicting
        self.penalty_redundant = penalty_redundant
        self.penalty_malformed = penalty_malformed
        self.penalty_deformed = penalty_deformed
        self.penalty_per_constraint = penalty_per_constraint

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
                cons_type = ConstraintToken(cons_id).name
                handler = self.sketch_tokenizer.context.get_constraint_handler(cons_type)
                if handler is not None:
                    true_refs = tuple(r for r in [ref1, ref2, ref3] if r >= CONSTRAINT_VAL_OFFSET)
                    pad_refs = tuple(r for r in [ref1, ref2, ref3] if r < CONSTRAINT_VAL_OFFSET)
                    sorted_true_refs = handler._sort_ref_tokens(true_refs)
                    body_tokens_reshaped[i] = np.array([cons_id, *sorted_true_refs, *pad_refs])
                else:
                    continue  # cons_type can be STOP or PAD
            body_tokens = body_tokens_reshaped.flatten()
        else:
            raise NotImplementedError("IMPLEMENT IT!!!")

        return np.concatenate([prefix_token, body_tokens])

    def forward(
        self,
        prim_tokens: torch.Tensor,
        cons_tokens: torch.Tensor,
        sketch_original_list: list[datalib.Sketch] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            prim_tokens: (B, L_prim)
            cons_tokens: (B, L_cons)
            sketch_original_list: Original sketch, used for precise primitive value assignment.
        """
        assert prim_tokens.size(0) == cons_tokens.size(0), f"Batch size mismatch: {prim_tokens.shape=} != {cons_tokens.shape=}"
        if sketch_original_list is None:
            sketch_original_list = [None] * prim_tokens.size(0)
        assert prim_tokens.size(0) == len(sketch_original_list), f"Batch size mismatch: {prim_tokens.shape=} != {len(sketch_original_list)=}"
        scores = torch.zeros_like(cons_tokens, dtype=torch.float32)

        for i, (prim, cons, sketch_original) in enumerate(zip(prim_tokens, cons_tokens, sketch_original_list)):
            prim, cons = prim.detach().cpu().numpy(), cons.detach().cpu().numpy()
            solver_status, sketch, invalid_cons_token_blocks = self.solver_feedback.evaluate_constraints(prim, cons, sketch_original, check_one_by_one=True)
            invalid_cons_token_blocks = np.array(invalid_cons_token_blocks)
            assert invalid_cons_token_blocks.size == 0 or invalid_cons_token_blocks.shape == (invalid_cons_token_blocks.shape[0], 4), f"{invalid_cons_token_blocks=}"

            ################################################################
            # 0. Replace after-STOP tokens by PAD (to prevent failse matching in Step 1)
            ################################################################
            stop_indices = (cons == ConstraintToken.Stop).nonzero()[0]
            first_stop_idx = stop_indices[0] if len(stop_indices) > 0 else -1
            if first_stop_idx >= 0:
                cons[first_stop_idx+1:] = ConstraintToken.Pad

            ################################################################
            # 1. Constraint-wise penalty
            ################################################################
            problematic_constraint_ids = set(solver_status.ConflictingConstraints + solver_status.RedundantConstraints + solver_status.MalformedConstraints)
            problematic_constraints = {c_id: sketch.constraints[c_id] for c_id in problematic_constraint_ids}
            problematic_sketch = datalib.Sketch(entities=sketch.entities, constraints=problematic_constraints)

            # NOTE: To deduce the corresponding token indices,
            #       we re-tokenize the problematic constraints, and
            #       search for the corresponding token subsequence in cons_tokens.
            problematic_constraint_tokens = self.sketch_tokenizer.tokenize(problematic_sketch, allow_invalid_constraints=True)['c_val']
            if len(problematic_constraint_tokens) > 0 and problematic_constraint_tokens[0] == ConstraintToken.Start:
                problematic_constraint_tokens = problematic_constraint_tokens[1:]
            while len(problematic_constraint_tokens) > 0 and problematic_constraint_tokens[-1] in (ConstraintToken.Pad, ConstraintToken.Stop):
                problematic_constraint_tokens = problematic_constraint_tokens[:-1]

            # Find matching
            cons_sorted = self.sort_intra_refs(cons)
            invalid_cons_token_blocks = self.sort_intra_refs(invalid_cons_token_blocks.flatten())
            cons_windows = np.lib.stride_tricks.sliding_window_view(cons, window_shape=self.sketch_tokenizer.align_constraint_token_length_to)  # (T) => (T, 4)

            if self.sketch_tokenizer.align_constraint_token_length_to > 0:
                problematic_constraint_tokens = problematic_constraint_tokens.reshape(-1, self.sketch_tokenizer.align_constraint_token_length_to)  # (K*4) => (K, 4)
                invalid_cons_token_blocks = invalid_cons_token_blocks.reshape(-1, self.sketch_tokenizer.align_constraint_token_length_to)
                problematic_constraint_tokens = np.concatenate([problematic_constraint_tokens, invalid_cons_token_blocks], axis=0)

                # cons_windows[:, None, :] shape: (T, 1, 4)
                # problematic_constraint_tokens[None, :, :] shape: (1, K, 4)
                # Comparison shape: (T, K, 4) -> all(axis=2) makes it (T, K)
                matches_batch = np.all(cons_windows[:, None, :] == problematic_constraint_tokens[None, :, :], axis=2)
                matched_cons_start_indices, matched_problematic_indices = np.where(matches_batch)
                if len(np.unique(matched_problematic_indices)) != len(problematic_constraint_tokens):
                    error_msg = "[SolverReward] ERROR: Failed to match problematic constraints to tokens:"
                    error_msg += f"\n{matched_cons_start_indices=}\n{matched_problematic_indices=}"

                    error_msg += "\n\nproblematic_constraint_tokens=["
                    for token_vis in visualize_constraint_sequence(problematic_constraint_tokens):
                        error_msg += f"\n{token_vis}"
                    error_msg += "]"

                    error_msg += f"\n\ncons_tokens={cons}"

                    error_msg += f"\n\ncons_tokens_vis={visualize_constraint_sequence(cons)}"

                    error_msg += "\n\nsketchgraphs_problematic_constraints=["
                    for c_id, c in problematic_constraints.items():
                        error_msg += f"\n{c_id}: {c}"
                    error_msg += "]"

                    raise ValueError(error_msg)

                matched_cons_end_indices = matched_cons_start_indices + self.sketch_tokenizer.align_constraint_token_length_to - 1
            else:
                raise NotImplementedError(f"IMPLEMENT IT!!!")

            scores[i, matched_cons_end_indices] += self.penalty_per_constraint

            ################################################################
            # 2. Sequence-wise penalty
            ################################################################
            seq_reward = 0.0

            if solver_status.Status != "Converged":
                # Invalid sequence
                invalid_penalties =[]
                if solver_status.ConflictingConstraints: invalid_penalties.append(self.penalty_conflicting)
                if solver_status.RedundantConstraints:   invalid_penalties.append(self.penalty_redundant)
                if solver_status.MalformedConstraints:   invalid_penalties.append(self.penalty_malformed)

                seq_reward = min(invalid_penalties) # heaviest penalty as the final penalty

            else:
                # valid sequence
                orig_dof = self._get_dof_without_constraints(sketch)
                seq_reward = (orig_dof - solver_status.DoF) / (orig_dof - self.fully_constrained_dof)

                # geometric distortion
                if solver_status.DistortingConstraints:
                    seq_reward += self.penalty_deformed

                # dimension constraints penalty
                NumTotalConstraints = solver_status.NumRelationalConstraints + solver_status.NumDimensionalConstraints
                if NumTotalConstraints == 0:
                    if not (np.all(cons[:2] == np.array([ConstraintToken.Stop, ConstraintToken.Pad])) or np.all(cons[:2] == np.array([ConstraintToken.Start, ConstraintToken.Stop]))):
                        raise ValueError(f"{visualize_constraint_sequence(cons)=}\n{solver_status=}")

                seq_reward -= NumTotalConstraints / len(sketch.entities)
                seq_reward -= solver_status.NumDimensionalConstraints / max(1, NumTotalConstraints)

            ################################################################
            # 3. Assign the reward
            ################################################################
            scores[i, first_stop_idx] += seq_reward

        return scores
