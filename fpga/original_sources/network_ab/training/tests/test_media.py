"""Exercise the actual HEVC 10-bit decoder, frame selection and VFR timestamps."""
from pathlib import Path
import subprocess
import tempfile
import unittest

import imageio_ffmpeg
import numpy as np

from network_ab.training.media import decode_frames, probe_video
from network_ab.training.runtime import configure_workspace


class MediaRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = configure_workspace()
        cls.temporary = tempfile.TemporaryDirectory(dir=root / "tmp")
        cls.video = Path(cls.temporary.name) / "中文_10bit_HLG_VFR.mkv"
        # Uniform frames expose decoder-created black bottom rows unambiguously.
        frames = np.stack([np.full((1080, 1920, 3), value, np.uint8) for value in (80, 120, 180, 210)])
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", "10", "-i", "pipe:0",
            "-vf", "settb=1/1000,setpts='if(lt(N,2),N,2+(N-2)*3)*100+30'",
            "-fps_mode", "passthrough", "-enc_time_base", "1:1000",
            "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "pools=2:frame-threads=1:log-level=error:colorprim=9:transfer=18:colormatrix=9",
            "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc",
            str(cls.video)]
        subprocess.run(cmd, input=frames.tobytes(), capture_output=True, check=True, timeout=60)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_hevc_bottom_rows_and_vfr_frame_identity(self):
        meta = probe_video(self.video)
        self.assertEqual(meta["color_tags"]["color_trc"], "arib-std-b67")
        self.assertAlmostEqual(meta["timestamp_origin_seconds"], .03, places=6)
        whole = list(decode_frames(self.video))
        self.assertEqual([n for n, _, _ in whole], [0, 1, 2, 3])
        np.testing.assert_allclose([t for _, t, _ in whole], [0, .1, .2, .5], atol=1e-6)
        selected = list(decode_frames(self.video, indices=[1, 3]))
        for n, pts, frame in selected:
            self.assertEqual(pts, whole[n][1])
            np.testing.assert_array_equal(frame, whole[n][2])
            self.assertGreater(frame[-3:].mean(), 10)
            self.assertLess(abs(float(frame[-3:].mean())-float(frame[100:103].mean())), 1)

    def test_invalid_and_missing_selected_frames_fail(self):
        with self.assertRaises(ValueError):
            list(decode_frames(self.video, indices=[3, 1]))
        with self.assertRaises(RuntimeError):
            list(decode_frames(self.video, indices=[99]))


if __name__ == "__main__":
    unittest.main()
