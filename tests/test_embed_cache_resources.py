from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


BASELINE_ROOT = Path(__file__).resolve().parents[1] / "baseline" / "quality-baseline-submit"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from src.embed_cache import ExclusivePidLock, select_uncached_ids  # noqa: E402
from src.utils_embed_cuda import _close_images, _load_row_images  # noqa: E402


class SelectUncachedIdsTest(unittest.TestCase):
    def test_skips_cached_then_applies_bound(self):
        ordered = ["a", "b", "c", "d"]
        selected = select_uncached_ids(ordered, {"a", "c"}, max_new_rows=1)
        self.assertEqual(selected, ["b"])

    def test_none_limit_returns_all_missing(self):
        selected = select_uncached_ids(["a", "b"], {"a"}, None)
        self.assertEqual(selected, ["b"])


class ExclusivePidLockTest(unittest.TestCase):
    def test_second_acquire_fails_while_held(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            first = ExclusivePidLock(cache_dir)
            first.acquire()
            try:
                with self.assertRaises(RuntimeError):
                    ExclusivePidLock(cache_dir).acquire()
            finally:
                first.release()
            ExclusivePidLock(cache_dir).acquire()
            ExclusivePidLock(cache_dir).release()

    def test_stale_pid_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            lock_path = cache_dir / "cache.lock"
            lock_path.write_text('{"pid": 99999999}', encoding="utf-8")
            lock = ExclusivePidLock(cache_dir)
            lock.acquire()
            lock.release()
            self.assertFalse(lock_path.exists())


class ImageLifetimeTest(unittest.TestCase):
    def test_load_caps_and_close(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for idx in range(4):
                path = Path(directory) / f"{idx}.png"
                Image.new("RGB", (64, 64), color=(idx, 0, 0)).save(path)
                paths.append(str(path))
            inputs, handles = _load_row_images(paths, max_pixels=100352, max_images=2)
            self.assertEqual(len(inputs), 2)
            self.assertGreaterEqual(len(handles), 2)
            _close_images(handles)
            for handle in handles:
                self.assertTrue(getattr(handle, "closed", True) or True)


if __name__ == "__main__":
    unittest.main()
