"""
Tests for aliciadata_finance_backend.memory_usage: the ru_maxrss unit
conversion on macOS and Linux, the upper bound for all processes, the
printed report, and reading the counters (with resource.getrusage replaced
by a fake, so the results do not depend on the test process).
"""

from types import SimpleNamespace

import pytest

from aliciadata_finance_backend import memory_usage
from aliciadata_finance_backend.memory_usage import (
    BYTES_PER_MIB,
    PeakMemory,
    format_peak_memory,
    maxrss_bytes,
    measure_peak_memory,
)


def test_maxrss_is_bytes_on_macos():
    assert maxrss_bytes(267_010_048, platform="darwin") == 267_010_048


@pytest.mark.parametrize("platform", ["linux", "freebsd14"])
def test_maxrss_is_kibibytes_elsewhere(platform):
    assert maxrss_bytes(260_752, platform=platform) == 260_752 * 1024


def test_upper_bound_adds_every_worker_to_the_main_process():
    peak = PeakMemory(
        main_bytes=300 * BYTES_PER_MIB,
        largest_worker_bytes=200 * BYTES_PER_MIB,
        workers=2,
    )
    assert peak.upper_bound_bytes == 700 * BYTES_PER_MIB


def test_upper_bound_without_workers_is_the_main_process():
    peak = PeakMemory(main_bytes=300 * BYTES_PER_MIB, largest_worker_bytes=0, workers=0)
    assert peak.upper_bound_bytes == 300 * BYTES_PER_MIB


def test_report_with_workers_shows_every_peak_in_mib():
    peak = PeakMemory(
        main_bytes=int(254.6 * BYTES_PER_MIB),
        largest_worker_bytes=200 * BYTES_PER_MIB,
        workers=2,
    )
    report = format_peak_memory(peak)
    assert "main process: 254.6 MiB" in report
    assert "largest of 2 ARIMA worker processes: 200.0 MiB" in report
    assert "main + 2 × largest worker): 654.6 MiB" in report
    assert not report.endswith("\n")


def test_report_without_workers_says_the_main_peak_is_the_whole_run():
    peak = PeakMemory(main_bytes=100 * BYTES_PER_MIB, largest_worker_bytes=0, workers=0)
    report = format_peak_memory(peak)
    assert "main process: 100.0 MiB" in report
    assert "whole run's peak" in report
    assert "worker processes:" not in report
    assert "upper bound" not in report


def test_measure_reads_self_and_children_and_converts(monkeypatch):
    usage = {
        memory_usage.resource.RUSAGE_SELF: 300,
        memory_usage.resource.RUSAGE_CHILDREN: 200,
    }
    monkeypatch.setattr(
        memory_usage.resource,
        "getrusage",
        lambda who: SimpleNamespace(ru_maxrss=usage[who]),
    )
    monkeypatch.setattr(
        memory_usage,
        "maxrss_bytes",
        lambda maxrss: maxrss * BYTES_PER_MIB,
    )
    peak = measure_peak_memory(workers=3)
    assert peak == PeakMemory(
        main_bytes=300 * BYTES_PER_MIB,
        largest_worker_bytes=200 * BYTES_PER_MIB,
        workers=3,
    )
