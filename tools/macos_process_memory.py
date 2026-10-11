"""Read a macOS process's physical footprint without spawning a diagnostic tool."""
import ctypes
import os
import sys


class _RusageInfoV4(ctypes.Structure):
    # Match the public SDK's sys/resource.h, including trailing fields for ABI safety.
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
            "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


def process_memory(pid):
    """Return current and lifetime-peak physical bytes; never substitute RSS or zero."""
    if type(pid) is not int or not 0 < pid <= 2 ** 31 - 1:
        raise ValueError("pid must be a positive signed 32-bit integer")
    if sys.platform != "darwin":
        raise RuntimeError("Physical-footprint measurement requires macOS")
    query = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pid_rusage
    query.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    query.restype = ctypes.c_int
    info = _RusageInfoV4()
    ctypes.set_errno(0)
    if query(pid, 4, ctypes.byref(info)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(pid))
    current, peak = int(info.ri_phys_footprint), int(info.ri_lifetime_max_phys_footprint)
    # XNU gathers the peak before the current value, so this pair is not atomic.
    if current <= 0 or peak <= 0:
        raise OSError(f"Physical-footprint counters are missing: current={current}, peak={peak}")
    return current, peak
