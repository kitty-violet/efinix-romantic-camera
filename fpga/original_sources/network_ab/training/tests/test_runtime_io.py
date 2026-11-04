"""Exercise Windows file-sharing contention and unrelated fatal I/O errors."""
import ctypes
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from network_ab.training.runtime import atomic_replace, configure_workspace, write_json

configure_workspace()


class AtomicWriteTests(unittest.TestCase):
    def test_nontransient_permission_failure_is_not_swallowed(self):
        with patch("os.replace", side_effect=PermissionError("not a Windows sharing error")) as replace:
            with self.assertRaises(PermissionError):
                atomic_replace("missing", "missing-target")
            self.assertEqual(replace.call_count, 1)

    @unittest.skipUnless(os.name == "nt", "Windows share modes")
    def test_actual_windows_reader_lock_recovers_without_partial_json(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "progress.json"
            write_json(path, {"step": 1})
            # FILE_SHARE_READ|FILE_SHARE_WRITE intentionally omit FILE_SHARE_DELETE.
            handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            released = threading.Event()
            def release():
                time.sleep(0.2)
                kernel.CloseHandle(handle)
                released.set()
            worker = threading.Thread(target=release)
            worker.start()
            try:
                original = os.replace
                with patch("os.replace", wraps=original) as replace:
                    write_json(path, {"step": 2, "payload": "complete"*500})
                    self.assertGreaterEqual(replace.call_count, 2)
            finally:
                worker.join()
            self.assertTrue(released.is_set())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"step": 2, "payload": "complete"*500})

    def test_ui_progress_share_failure_does_not_abort_training(self):
        from network_ab.training.train import write_progress
        error = PermissionError("simulated persistent UI reader lock")
        error.winerror = 32
        with patch("network_ab.training.train.write_json", side_effect=error):
            with patch("builtins.print") as warning:
                write_progress("progress.json", {"step": 3})
                warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
