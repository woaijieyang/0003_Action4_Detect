import struct
import unittest
from unittest.mock import patch
import torch
from model.training.action4_loss_logging import accumulate_loss_logging


def old_loop(logged, losses, *, role, weights, counts, world_size,
             active_count=0, global_count=None):
    for key, value in losses.items():
        contribution = (weights.get(key, 1.) * world_size * value.detach() / counts[key]
                        if counts[key] > 0 else value.detach() * 0) if role == "A0" else sum([
                            weights.get(key, 1.) * value.detach()
                            * (world_size * active_count / global_count)])
        logged[key] = logged.get(key, 0.) + float(contribution)
    return logged


class LossLoggingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_bits(self, a, b):
        self.assertEqual(list(a), list(b))
        for key in a:
            self.assertEqual(struct.pack("!d", a[key]), struct.pack("!d", b[key]), key)

    def test_three_micros_mixed_dtypes_weights_counts_and_existing_order(self):
        for role in ("A0", "B0"):
            expected = {"existing": 1. / 3., "z": -0.}
            actual = dict(expected)
            for micro in range(3):
                losses = {key: torch.tensor(value, dtype=dtype, requires_grad=True)
                          for key, value, dtype in [
                              ("z", -(micro + 1) / 7, torch.float32),
                              ("a", 1.234567890123 + micro, torch.float64),
                              ("zero", -2.5, torch.float16),
                              ("default", 0.125 + micro, torch.bfloat16)]}
                kwargs = dict(role=role, weights={"z": .1, "a": 1.7, "zero": 0.},
                              counts={"z": 3., "a": 7., "zero": 0., "default": 2.},
                              world_size=2, active_count=micro, global_count=48)
                old_loop(expected, losses, **kwargs)
                self.assertIs(accumulate_loss_logging(actual, losses, **kwargs), actual)
                self.assert_bits(actual, expected)
                self.assertTrue(all(v.grad is None for v in losses.values()))

    def test_no_autograd_or_objective_mutation_and_one_host_transfer(self):
        x = torch.tensor(2., requires_grad=True)
        losses = {"x": x.square(), "y": x * 3}
        objective = losses["x"] + losses["y"]
        before = {k: v.detach().clone() for k, v in losses.items()}
        calls = []
        original = torch.Tensor.cpu
        def cpu(tensor, *args, **kwargs):
            calls.append((tensor.shape, tensor.requires_grad))
            return original(tensor, *args, **kwargs)
        with patch.object(torch.Tensor, "cpu", cpu):
            accumulate_loss_logging({}, losses, role="A0", weights={},
                                    counts={"x": 1, "y": 1}, world_size=2)
        self.assertEqual(calls, [(torch.Size([2]), False)])
        self.assertIsNone(x.grad)
        for key in losses:
            self.assertTrue(torch.equal(losses[key].detach(), before[key]))
        objective.backward()
        self.assertEqual(x.grad.item(), 7.)

    def test_zero_count_nonfinite_retained_not_silently_fixed(self):
        losses = {"bad": torch.tensor(float("inf"))}
        kwargs = dict(role="A0", weights={}, counts={"bad": 0}, world_size=2)
        self.assert_bits(old_loop({}, losses, **kwargs),
                         accumulate_loss_logging({}, losses, **kwargs))

    def test_empty_and_invalid_scalar(self):
        logged = {"keep": 3.}
        self.assertIs(accumulate_loss_logging(logged, {}, role="A0", weights={},
                                             counts={}, world_size=2), logged)
        with self.assertRaises(ValueError):
            accumulate_loss_logging({}, {"x": torch.ones(2)}, role="A0",
                                    weights={}, counts={"x": 1}, world_size=2)


if __name__ == "__main__":
    unittest.main()
