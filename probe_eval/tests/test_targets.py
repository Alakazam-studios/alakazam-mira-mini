# ABOUTME: Unit tests for probe target extraction: layout, normalization, NaN masking
# ABOUTME: (demolition / missing rotation / frozen frames), quat canonicalization, alignment.
import numpy as np
import pytest

from probe_eval.targets import (
    BALL_VEL_SCALE,
    CAR_VEL_SCALE,
    ENTITY_DIM,
    POS,
    POS_SCALE,
    QUAT,
    TARGET_DIM,
    VEL,
    align_to_latents,
    canonical_quat,
    frame_target,
    sequence_targets,
)


def vec3(x=0.0, y=0.0, z=0.0):
    return {"x": x, "y": y, "z": z}


def quat(x=0.0, y=0.0, z=0.0, w=1.0):
    return {"x": x, "y": y, "z": z, "w": w}


def car(player_id, *, demolished=False, with_rotation=True, loc=None, vel=None):
    c = {
        "player_id": player_id,
        "team": player_id % 2,
        "location": loc or vec3(100.0 * player_id, 0.0, 17.0),
        "velocity": vel or vec3(0.0, CAR_VEL_SCALE / 2, 0.0),
        "attacker_player_id": 7 if demolished else -1,
    }
    if with_rotation:
        c["rotation"] = quat()
    return c


def frame(cars=None, ball_loc=None, ball_vel=None):
    return {
        "game": {"time_remaining": 300.0, "score_blue": 0, "score_orange": 0, "is_overtime": False},
        "ball": {
            "location": ball_loc or vec3(0.0, 0.0, 93.15),
            "velocity": ball_vel or vec3(BALL_VEL_SCALE / 2, 0.0, 0.0),
            "rotation": quat(),
            "angular_velocity": vec3(),
        },
        "cars": cars if cars is not None else [car(i) for i in (3, 1, 0, 2)],  # unsorted on purpose
    }


class TestFrameTarget:
    def test_shape_and_normalization(self):
        t = frame_target(frame())
        assert t.shape == (TARGET_DIM,)
        ball = t[:ENTITY_DIM]
        np.testing.assert_allclose(ball[POS], [0.0, 0.0, 93.15 / POS_SCALE[2]], atol=1e-6)
        np.testing.assert_allclose(ball[VEL], [0.5, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(ball[QUAT], [0, 0, 0, 1], atol=1e-6)

    def test_cars_sorted_by_player_id(self):
        t = frame_target(frame()).reshape(5, ENTITY_DIM)
        # car i sits at x = 100 * player_id, so sorted order reads 0, 100, 200, 300 on x.
        xs = t[1:, POS][:, 0] * POS_SCALE[0]
        np.testing.assert_allclose(xs, [0.0, 100.0, 200.0, 300.0], atol=1e-3)

    def test_demolished_car_is_nan(self):
        cars = [car(0), car(1, demolished=True), car(2), car(3)]
        t = frame_target(frame(cars=cars)).reshape(5, ENTITY_DIM)
        assert np.isnan(t[2]).all()  # entity index 2 = car with player_id 1
        assert not np.isnan(t[1]).any() and not np.isnan(t[3]).any()

    def test_missing_rotation_masks_only_quat(self):
        cars = [car(0, with_rotation=False), car(1), car(2), car(3)]
        t = frame_target(frame(cars=cars)).reshape(5, ENTITY_DIM)
        assert np.isnan(t[1, QUAT]).all()
        assert not np.isnan(t[1, POS]).any() and not np.isnan(t[1, VEL]).any()


class TestQuat:
    def test_canonicalization_collapses_double_cover(self):
        q = canonical_quat(quat(x=0.1, y=0.2, z=0.3, w=-0.5))
        assert q[3] >= 0
        np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-6)

    def test_degenerate_quat_is_nan(self):
        assert np.isnan(canonical_quat(quat(w=0.0))).all()


class TestSequenceTargets:
    def test_keep_mask_nans_whole_frames(self):
        seq = [frame(), frame(), frame()]
        t = sequence_targets(seq, keep=[True, False, True])
        assert not np.isnan(t[0]).any() and not np.isnan(t[2]).any()
        assert np.isnan(t[1]).all()

    def test_keep_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            sequence_targets([frame()], keep=[True, False])


class TestAlignment:
    def test_last_frame_of_each_window(self):
        per_frame = np.arange(8, dtype=np.float32).reshape(8, 1)
        out = align_to_latents(per_frame, temporal_stride=2)
        np.testing.assert_array_equal(out[:, 0], [1, 3, 5, 7])

    def test_trailing_partial_window_dropped(self):
        per_frame = np.arange(7, dtype=np.float32).reshape(7, 1)
        out = align_to_latents(per_frame, temporal_stride=2)
        np.testing.assert_array_equal(out[:, 0], [1, 3, 5])

    def test_stride_one_is_identity(self):
        per_frame = np.arange(5, dtype=np.float32).reshape(5, 1)
        np.testing.assert_array_equal(align_to_latents(per_frame, 1), per_frame)
