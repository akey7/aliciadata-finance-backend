"""
Peak memory of a backend run, for sizing the machine it runs on, from the
operating system's resource usage counters (the standard library's resource
module; macOS and Linux only).

Two peaks are available, each the largest resident set size (physical RAM)
of one process:

- the main process (RUSAGE_SELF);
- the largest child process that has finished (RUSAGE_CHILDREN), which in a
  backend run is the largest ARIMA worker. ProcessPoolExecutor joins its
  workers before training returns, so they have all finished by the time the
  peaks are read.

Neither is the combined peak of all processes running at once, which these
counters cannot give. The main process and the workers run at the same time,
so main + workers × largest worker is reported as an upper bound for it.

ru_maxrss is in bytes on macOS and in kibibytes on Linux; maxrss_bytes()
converts both to bytes. Everything here is pure except measure_peak_memory().
"""

import resource
import sys
from dataclasses import dataclass

BYTES_PER_KIB = 1024
BYTES_PER_MIB = 1024**2


def maxrss_bytes(maxrss: int, platform: str = sys.platform) -> int:
    """
    A ru_maxrss value in bytes: macOS ("darwin") reports bytes, Linux and
    the other Unix systems kibibytes. Pure function.

    Parameters
    ----------
    maxrss : int
        The ru_maxrss field of resource.getrusage().

    platform : str, optional
        The sys.platform value the number came from. Defaults to this one.

    Returns
    -------
    int
        The same peak, in bytes.
    """
    return maxrss if platform == "darwin" else maxrss * BYTES_PER_KIB


@dataclass(frozen=True)
class PeakMemory:
    """
    Peak resident set sizes of one run, in bytes.

    Attributes
    ----------
    main_bytes : int
        Peak of the main process.

    largest_worker_bytes : int
        Peak of the largest finished child process (an ARIMA worker), or 0
        if none ran.

    workers : int
        ARIMA worker processes that ran at the same time as the main
        process: 0 when ARIMA was cached or trained in the main process.
    """

    main_bytes: int
    largest_worker_bytes: int
    workers: int

    @property
    def upper_bound_bytes(self) -> int:
        """
        Upper bound for the combined peak of all processes: main_bytes +
        workers × largest_worker_bytes. The true combined peak is at most
        this, and lower when the workers' peaks differ or do not coincide.
        """
        return self.main_bytes + self.workers * self.largest_worker_bytes


def measure_peak_memory(workers: int) -> PeakMemory:
    """
    Read the peaks of this process and of its finished children so far.

    Parameters
    ----------
    workers : int
        ARIMA worker processes that ran (see PeakMemory.workers).

    Returns
    -------
    PeakMemory
        The peaks, in bytes.
    """
    return PeakMemory(
        main_bytes=maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        largest_worker_bytes=maxrss_bytes(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        ),
        workers=workers,
    )


def _mib(n_bytes: int) -> str:
    return f"{n_bytes / BYTES_PER_MIB:.1f} MiB"


def format_peak_memory(peak: PeakMemory) -> str:
    """
    Lines describing the peaks, for printing at the end of a run. Pure
    function.

    Parameters
    ----------
    peak : PeakMemory
        The peaks to describe.

    Returns
    -------
    str
        One line per peak, in MiB (1 MiB = 1,048,576 bytes), without a
        trailing newline. Without workers, the main process's peak is the
        whole run's.
    """
    lines = [
        "Peak memory (resident set size):",
        f"  main process: {_mib(peak.main_bytes)}",
    ]
    if peak.workers == 0:
        lines.append("  no ARIMA worker processes ran, so this is the whole run's peak")
        return "\n".join(lines)
    lines += [
        f"  largest of {peak.workers} ARIMA worker processes: "
        f"{_mib(peak.largest_worker_bytes)}",
        f"  upper bound for all processes together (main + {peak.workers} × "
        f"largest worker): {_mib(peak.upper_bound_bytes)}",
    ]
    return "\n".join(lines)
