"""Report generation helpers."""

from nf_agent.reports.benchmark_report import BenchmarkReportConfig, build_benchmark_report
from nf_agent.reports.v6e_profile import V6EProfileReportConfig, build_v6e_profile_report

__all__ = [
    "BenchmarkReportConfig",
    "V6EProfileReportConfig",
    "build_benchmark_report",
    "build_v6e_profile_report",
]
