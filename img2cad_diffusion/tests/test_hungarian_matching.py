"""Tests for Hungarian matching loss implementation."""

import itertools
import pytest
import torch
import numpy as np

from img2cad_diffusion.losses import ConstraintLoss
from img2cad_diffusion.tokenization import CONSTRAINT_TOKEN_LEN


class TestHungarianMatching:
    """Tests for Hungarian matching correctness."""

    def test_cost_matrix_computation_matches_brute_force(self):
        """Verify cost matrix computation matches naive implementation."""
        loss_fn = ConstraintLoss(use_hungarian=True)

        # Small test case for brute force comparison
        B, N, block_size, V = 2, 4, CONSTRAINT_TOKEN_LEN, 20

        torch.manual_seed(42)
        pred_logits = torch.randn(B, N, block_size, V)
        gt_blocks = torch.randint(0, V, (B, N, block_size))
        input_blocks = gt_blocks.clone()  # For simplicity, input = GT
        mask_blocks = torch.rand(B, N, block_size) > 0.3  # ~70% masked

        # Compute using vectorized implementation
        cost_vectorized = loss_fn._compute_cost_matrix(
            pred_logits, gt_blocks, input_blocks, mask_blocks
        )

        # Compute using naive brute-force implementation
        cost_brute = self._compute_cost_brute_force(
            pred_logits, gt_blocks, input_blocks, mask_blocks, V
        )

        # Should match
        torch.testing.assert_close(cost_vectorized, cost_brute, rtol=1e-4, atol=1e-4)

    def _compute_cost_brute_force(self, pred_logits, gt_blocks, input_blocks, mask_blocks, V):
        """Naive O(B*N*N*4) implementation for reference."""
        import torch.nn.functional as F

        B, N, block_size, _ = pred_logits.shape
        cost = torch.zeros(B, N, N)

        for b in range(B):
            for i in range(N):
                # Build probability distribution for pred block i
                pred_probs = torch.zeros(block_size, V)
                for k in range(block_size):
                    if mask_blocks[b, i, k]:
                        pred_probs[k] = F.softmax(pred_logits[b, i, k], dim=-1)
                    else:
                        pred_probs[k, input_blocks[b, i, k]] = 1.0

                for j in range(N):
                    # Compute CE against GT block j
                    for k in range(block_size):
                        target_token = gt_blocks[b, j, k]
                        prob = pred_probs[k, target_token].clamp(min=1e-10)
                        cost[b, i, j] -= torch.log(prob)

        return cost

    def test_hungarian_matching_finds_optimal_assignment(self):
        """Verify Hungarian matching finds the same assignment as brute-force search."""
        loss_fn = ConstraintLoss(use_hungarian=True)

        # Small N so brute force is feasible (N! permutations)
        torch.manual_seed(123)
        B, N = 3, 5  # 5! = 120 permutations is manageable

        # Create random cost matrices
        cost_matrix = torch.rand(B, N, N) * 10

        # Get Hungarian solution
        hungarian_indices = loss_fn._hungarian_match(cost_matrix)
        hungarian_costs = self._compute_assignment_costs(cost_matrix, hungarian_indices)

        # Brute-force optimal solution
        brute_indices, brute_costs = self._brute_force_optimal_matching(cost_matrix)

        # Hungarian should find assignments with costs <= brute-force optimal (should be equal)
        for b in range(B):
            assert hungarian_costs[b] <= brute_costs[b] + 1e-6, \
                f"Sample {b}: Hungarian cost {hungarian_costs[b]:.4f} > brute force {brute_costs[b]:.4f}"

    def _compute_assignment_costs(self, cost_matrix, indices):
        """Compute total cost for given assignments."""
        B, N, _ = cost_matrix.shape
        costs = []
        for b in range(B):
            total = sum(cost_matrix[b, i, indices[b, i]] for i in range(N))
            costs.append(total.item())
        return costs

    def _brute_force_optimal_matching(self, cost_matrix):
        """Find optimal assignment by trying all N! permutations."""
        B, N, _ = cost_matrix.shape
        all_perms = list(itertools.permutations(range(N)))

        best_indices = []
        best_costs = []

        for b in range(B):
            min_cost = float('inf')
            min_perm = None

            for perm in all_perms:
                cost = sum(cost_matrix[b, i, perm[i]] for i in range(N))
                if cost < min_cost:
                    min_cost = cost
                    min_perm = perm

            best_indices.append(list(min_perm))
            best_costs.append(min_cost.item())

        return torch.tensor(best_indices), best_costs

    def test_hungarian_loss_consistent_with_vanilla_when_identity_match(self):
        """When optimal matching is identity, Hungarian and vanilla should give similar loss."""
        # This tests that when GT is in the "right" order, both methods work similarly

        torch.manual_seed(456)
        B, L, V = 4, 17, 20  # L = 1 (START) + 4 blocks * 4 tokens

        loss_fn_vanilla = ConstraintLoss(use_hungarian=False)
        loss_fn_hungarian = ConstraintLoss(use_hungarian=True)

        # Create logits that strongly predict specific targets (encourage identity match)
        target = torch.randint(0, V, (B, L))
        logits = torch.zeros(B, L, V)
        logits.scatter_(2, target.unsqueeze(-1), 10.0)  # High logit at target
        logits += torch.randn_like(logits) * 0.1  # Small noise

        mask = torch.ones(B, L, dtype=torch.bool)  # All masked
        reweighting = torch.ones(B, L)

        loss_vanilla = loss_fn_vanilla.forward(logits, target, mask, reweighting)
        loss_hungarian = loss_fn_hungarian.forward(logits, target, mask, reweighting)

        # Both should compute similar loss since optimal is identity
        # (Not exactly equal due to block structure handling)
        assert abs(loss_vanilla.item() - loss_hungarian.item()) < 0.5, \
            f"Vanilla: {loss_vanilla:.4f}, Hungarian: {loss_hungarian:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
