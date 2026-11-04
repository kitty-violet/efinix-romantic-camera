import copy
import json
from pathlib import Path
import tempfile
import unittest

from network_ab.training.data import Sampler, load_content_manifest
from network_ab.training.runtime import EXPECTED_CONTRACT, HERE, configure_workspace, contract_guard, fingerprints, assert_fingerprints


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_workspace()

    def setUp(self):
        self.config = json.loads((HERE / "fp32_config.json").read_text(encoding="utf-8"))
        self.manifest = {"schema": "model-training-data-v1", "status": "approved", "images": [
            {"path": f"E:/contents/{split}.png", "sha256": f"image-{split}", "source_sha256": f"source-{split}",
             "group": split, "split": split, "role": "content", "height": 720, "width": 1280}
            for split in ("train", "val", "test")],
            "videos": [{"path": "E:/contents/test.mp4", "sha256": "source-test", "group": "test", "split": "test"}]}

    def load(self, manifest):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "中文数据清单.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            return load_content_manifest(path, check_files=False)

    def test_unique_contract(self):
        _, plan = contract_guard()
        self.assertEqual(plan["contract_sha256"], EXPECTED_CONTRACT)
        self.assertEqual(plan["trainable_parameter_count"], 11379)

    def test_wrong_contract_and_source_drift_rejected(self):
        with self.assertRaises(RuntimeError):
            contract_guard("0"*64)
        binding = fingerprints()
        binding["shared_files"]["contract_torch.py"] = "wrong"
        with self.assertRaises(RuntimeError):
            assert_fingerprints(binding)

    def test_source_disjoint_manifest(self):
        self.assertEqual(len(self.load(self.manifest)["images"]), 3)

    def test_leakage_group_source_and_duplicate_image(self):
        for key in ("group", "source_sha256", "sha256"):
            with self.subTest(key=key):
                manifest = copy.deepcopy(self.manifest)
                manifest["images"][1][key] = manifest["images"][0][key]
                with self.assertRaises(ValueError):
                    self.load(manifest)

    def test_missing_split_and_missing_video(self):
        for key in ("images", "videos"):
            manifest = copy.deepcopy(self.manifest)
            manifest[key].pop()
            with self.assertRaises(ValueError):
                self.load(manifest)

    def test_holdout_video_cannot_overlap_train(self):
        self.manifest["videos"][0]["sha256"] = "source-train"
        with self.assertRaises(ValueError):
            self.load(self.manifest)

    def test_draft_style_and_wrong_size_rejected(self):
        for scope, key, value in (("top", "status", "draft"), ("row", "role", "style"), ("row", "height", 480)):
            manifest = copy.deepcopy(self.manifest)
            target = manifest if scope == "top" else manifest["images"][0]
            target[key] = value
            with self.assertRaises(ValueError):
                self.load(manifest)

    def test_sampler_resume_across_epoch_and_augmentations(self):
        sampler = Sampler(self.manifest["images"], [{"sha256": "style-a"}, {"sha256": "style-b"}], self.config)
        for step in range(5):
            sampler.next_spec(step)
        state = sampler.state_dict()
        expected = [sampler.next_spec(step) for step in range(5, 50)]
        sampler.load_state_dict(state)
        self.assertEqual(expected, [sampler.next_spec(step) for step in range(5, 50)])

    def test_full_frame_and_crop_bounds(self):
        sampler = Sampler(self.manifest["images"], [{"sha256": "style"}], self.config)
        self.assertEqual(sampler.next_spec(999)["crop"], [0, 0, 720, 1280])
        for step in range(200):
            y, x, h, w = sampler.next_spec(step)["crop"]
            self.assertTrue(0 <= y <= 720-h and 0 <= x <= 1280-w)
            self.assertEqual(h % 4, 0)
            self.assertEqual(w % 4, 0)


if __name__ == "__main__":
    unittest.main()
