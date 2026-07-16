# ABOUTME: Unit tests for the game-state probe: shapes, NaN-masked loss gradients,
# ABOUTME: and denormalized error metrics on hand-built prediction/target pairs.
import math

import numpy as np
import torch

from probe_eval.probe import GameStateProbe, masked_mse, state_errors
from probe_eval.targets import ENTITY_DIM, POS_SCALE, TARGET_DIM


class TestProbe:
    def test_forward_shape(self):
        probe = GameStateProbe.for_latents(h=9, w=16, c=32)
        out = probe(torch.randn(7, 9 * 16 * 32))
        assert out.shape == (7, TARGET_DIM)


class TestMaskedMse:
    def test_ignores_nan_targets(self):
        pred = torch.zeros(2, 4)
        target = torch.tensor([[1.0, float("nan"), 1.0, float("nan")], [1.0, float("nan"), 1.0, 1.0]])
        assert float(masked_mse(pred, target)) == 1.0  # only the five valid 1.0s count

    def test_all_nan_gives_zero_with_grad(self):
        pred = torch.randn(3, 4, requires_grad=True)
        loss = masked_mse(pred, torch.full((3, 4), float("nan")))
        loss.backward()
        assert float(loss.detach()) == 0.0 and pred.grad is not None


class TestStateErrors:
    def test_known_ball_position_error(self):
        target = torch.zeros(1, TARGET_DIM)
        target[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(1)  # valid ball quat
        pred = target.clone()
        pred[0, 0] += 100.0 / POS_SCALE[0]  # 100 uu off on ball x
        errs = state_errors(pred, target)
        assert math.isclose(errs["ball_pos_uu"], 100.0, rel_tol=1e-4)
        assert math.isclose(errs["car_pos_uu"], 0.0, abs_tol=1e-4)

    def test_quat_geodesic_half_turn(self):
        target = torch.zeros(2, TARGET_DIM)
        pred = torch.zeros(2, TARGET_DIM)
        for e in range(5):
            base = e * ENTITY_DIM
            target[:, base + 3 : base + 7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
            pred[:, base + 3 : base + 7] = torch.tensor([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
        errs = state_errors(pred, target)  # 90° rotation about z
        assert math.isclose(errs["quat_geodesic_rad"], math.pi / 2, rel_tol=1e-4)

    def test_nan_entities_excluded(self):
        target = torch.zeros(1, TARGET_DIM)
        target[0, ENTITY_DIM:] = float("nan")  # all four cars invalid
        pred = torch.ones(1, TARGET_DIM)
        errs = state_errors(pred, target)
        assert not np.isnan(errs["ball_pos_uu"])
        assert np.isnan(errs["car_pos_uu"])
