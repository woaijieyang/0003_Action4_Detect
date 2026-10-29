"""Pure CPU fake-signature tests, not SAM model execution."""
import unittest
import functools
from model.action4_sam_native_compat import make_init_state_bridge


class InitStateCompatTests(unittest.TestCase):
    def test_false_removed_other_fields_unchanged(self):
        calls = []
        def original(resource_path, offload_video_to_cpu=False, async_loading_frames=False):
            calls.append((resource_path, offload_video_to_cpu, async_loading_frames)); return "state"
        bridge = make_init_state_bridge(original)
        self.assertEqual(bridge(resource_path="region", offload_video_to_cpu=True,
                                async_loading_frames=False, offload_state_to_cpu=False), "state")
        self.assertEqual(calls, [("region", True, False)])

    def test_true_and_non_boolean_false_rejected_before_call(self):
        calls = []
        def original(resource_path): calls.append(resource_path)
        bridge = make_init_state_bridge(original)
        for value in (True, 0, None, "False"):
            with self.assertRaises(ValueError): bridge("region", offload_state_to_cpu=value)
        self.assertEqual(calls, [])

    def test_other_unknown_kwargs_not_filtered(self):
        def original(resource_path, offload_video_to_cpu=False): return resource_path
        bridge = make_init_state_bridge(original)
        with self.assertRaises(TypeError): bridge("region", offload_state_to_cpu=False, unknown=1)
        with self.assertRaises(TypeError): bridge("region", offload_state_to_cpu=False, video_loader_type="cv2")

    def test_original_support_keeps_argument(self):
        def original(resource_path, offload_state_to_cpu=False): return offload_state_to_cpu
        bridge = make_init_state_bridge(original)
        self.assertIs(bridge("region", offload_state_to_cpu=True), True)

    def test_omitted_kwarg_and_bound_method(self):
        class Model:
            def init_state(self, resource_path, offload_video_to_cpu=False): return (resource_path, offload_video_to_cpu)
        a, b = Model(), Model()
        a.init_state = make_init_state_bridge(a.init_state)
        self.assertEqual(a.init_state("a", offload_state_to_cpu=False), ("a", False))
        self.assertEqual(a.init_state("a"), ("a", False))
        with self.assertRaises(TypeError): b.init_state("b", offload_state_to_cpu=False)

    def test_original_error_propagates(self):
        def original(resource_path): raise RuntimeError("backend failed")
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            make_init_state_bridge(original)("region", offload_state_to_cpu=False)

    def test_decorated_original_signature_stays_strict(self):
        def original(resource_path, offload_video_to_cpu=False): return resource_path
        @functools.wraps(original)
        def decorated(*args, **kwargs): return original(*args, **kwargs)
        bridge = make_init_state_bridge(decorated)
        self.assertEqual(bridge("region", offload_state_to_cpu=False), "region")
        with self.assertRaises(TypeError): bridge("region", offload_state_to_cpu=False, unrelated=True)


if __name__ == "__main__": unittest.main()
