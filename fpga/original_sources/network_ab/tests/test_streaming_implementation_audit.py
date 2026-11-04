import unittest

from streaming_implementation_audit import audit


class StreamingImplementationAuditTest(unittest.TestCase):
    def test_real_oc8_padding_and_measured_cost_are_not_lost(self):
        result = audit()
        pw = result["one_serial_banked_pw_wrapper"]
        self.assertEqual(pw["commands"], 3362400)
        self.assertEqual(pw["last_term_lower_bound_cycles"], 372650400)
        self.assertGreater(pw["serial_measured_service_cycles_excluding_dispatch_gaps"],
                           pw["last_term_lower_bound_cycles"])
        head = next(layer for layer in pw["layers"] if layer["name"] == "rgb_head")
        self.assertEqual(head["oc8_commands"], 1280 * 720)
        self.assertEqual(head["store_words_if_reloaded_per_dot"], 1280 * 720 * 17)
        self.assertEqual(len(pw["layers"]), 11)

    def test_28_local_rq_slices_exceed_device(self):
        resource = audit()["replicating_28_independent_pw_slices"]
        self.assertEqual(resource["total_physical_dsps"], 168)
        self.assertFalse(resource["fits"])

    def test_padding_cost_keeps_historical_budget_as_assumption(self):
        result = audit()
        self.assertEqual(result["scenarios"][0]["cycles"], 4439524)
        rgb = result["rgb_oc8_padding_sensitivity"]
        self.assertEqual(rgb["valid_values"], 1280 * 720 * 3)
        self.assertEqual(rgb["padded_values"], 1280 * 720 * 8)
        self.assertEqual(rgb["frame_cycles_if_all_padding_computed_and_requantized"], 4631524)
        self.assertGreater(rgb["frame_cycles_if_padding_requant_skipped_only"], 4500000)


if __name__ == "__main__":
    unittest.main()
