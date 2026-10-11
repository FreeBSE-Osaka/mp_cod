import ctypes
import errno
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from tools.macos_process_memory import _RusageInfoV4, process_memory


class MacOSProcessMemoryTest(unittest.TestCase):
    def test_public_v4_layout_and_pid_validation(self):
        self.assertEqual(ctypes.sizeof(_RusageInfoV4), 296)
        self.assertEqual(_RusageInfoV4.ri_phys_footprint.offset, 72)
        self.assertEqual(_RusageInfoV4.ri_lifetime_max_phys_footprint.offset, 240)
        for pid in (True, 0, -1, 2 ** 31, "1"):
            with self.subTest(pid=pid), self.assertRaises(ValueError):
                process_memory(pid)

    def test_unavailable_platform_cannot_silently_disable_guard(self):
        with patch("tools.macos_process_memory.sys.platform", "linux"), self.assertRaises(RuntimeError):
            process_memory(os.getpid())

    def test_api_errors_and_missing_counters_are_not_memory_samples(self):
        class Query:
            def __call__(self, pid, flavor, output):
                self.assert_flavor = flavor
                ctypes.set_errno(errno.ESRCH)
                return -1
        query = Query()
        with patch("tools.macos_process_memory.sys.platform", "darwin"), \
                patch("tools.macos_process_memory.ctypes.CDLL") as library:
            library.return_value.proc_pid_rusage = query
            with self.assertRaises(OSError) as caught:
                process_memory(123)
            self.assertEqual(caught.exception.errno, errno.ESRCH)
            self.assertEqual(query.assert_flavor, 4)
            from unittest.mock import Mock
            library.return_value.proc_pid_rusage = Mock(return_value=0)
            with self.assertRaisesRegex(OSError, "counters are missing"):
                process_memory(123)

    def test_nonatomic_counter_pair_preserves_the_larger_current_value(self):
        def sample(pid, flavor, output):
            info = ctypes.cast(output, ctypes.POINTER(_RusageInfoV4)).contents
            info.ri_phys_footprint = 21 * 1024 ** 3
            info.ri_lifetime_max_phys_footprint = 19 * 1024 ** 3
            return 0
        with patch("tools.macos_process_memory.sys.platform", "darwin"), \
                patch("tools.macos_process_memory.ctypes.CDLL") as library:
            library.return_value.proc_pid_rusage.side_effect = sample
            self.assertEqual(process_memory(123), (21 * 1024 ** 3, 19 * 1024 ** 3))

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS libproc")
    def test_live_owned_process_and_reaped_process(self):
        with subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                              stdin=subprocess.PIPE) as child:
            try:
                current, peak = process_memory(child.pid)
                self.assertGreater(current, 0)
                self.assertGreaterEqual(peak, current)
            finally:
                child.communicate(input=b"")
        with self.assertRaises(OSError):
            process_memory(child.pid)


if __name__ == "__main__":
    unittest.main()
