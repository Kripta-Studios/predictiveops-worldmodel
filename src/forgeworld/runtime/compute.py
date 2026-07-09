"""Hardware-aware compute planning for CPU/GPU industrial experiments."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


DevicePreference = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class GpuDevice:
    index: int
    name: str
    total_vram_gb: float | None = None
    backend: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_vram_gb": self.total_vram_gb,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class ComputeProfile:
    cpu_threads: int
    system_ram_gb: float | None
    gpus: tuple[GpuDevice, ...] = ()
    platform: str = field(default_factory=platform.platform)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])

    @property
    def has_cuda(self) -> bool:
        return any(gpu.backend == "cuda" for gpu in self.gpus)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_threads": self.cpu_threads,
            "system_ram_gb": self.system_ram_gb,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "has_cuda": self.has_cuda,
            "platform": self.platform,
            "python_version": self.python_version,
        }


@dataclass(frozen=True)
class WorkerPlan:
    device: str
    cpu_workers: int
    dataloader_workers: int
    baseline_parallel_jobs: int
    seed_parallel_jobs: int
    gpu_parallel_jobs: int
    gpu_memory_fraction: float | None
    precision: str
    smoke_mode: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "cpu_workers": self.cpu_workers,
            "dataloader_workers": self.dataloader_workers,
            "baseline_parallel_jobs": self.baseline_parallel_jobs,
            "seed_parallel_jobs": self.seed_parallel_jobs,
            "gpu_parallel_jobs": self.gpu_parallel_jobs,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "precision": self.precision,
            "smoke_mode": self.smoke_mode,
            "notes": list(self.notes),
        }


def _detect_system_ram_gb() -> float | None:
    if platform.system().lower() == "windows":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return round(status.ullTotalPhys / (1024**3), 2)
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    return round((pages * page_size) / (1024**3), 2)


def _detect_gpus_with_torch() -> tuple[GpuDevice, ...]:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return ()
    try:
        if not torch.cuda.is_available():
            return ()
        devices: list[GpuDevice] = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                GpuDevice(
                    index=index,
                    name=str(properties.name),
                    total_vram_gb=round(properties.total_memory / (1024**3), 2),
                )
            )
        return tuple(devices)
    except Exception:
        return ()


def _detect_gpus_with_nvidia_smi() -> tuple[GpuDevice, ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return ()
    if completed.returncode != 0:
        return ()
    devices: list[GpuDevice] = []
    for index, raw_line in enumerate(completed.stdout.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        name, _, memory_mb = line.rpartition(",")
        try:
            total_vram_gb = round(float(memory_mb.strip()) / 1024, 2)
        except ValueError:
            total_vram_gb = None
        devices.append(GpuDevice(index=index, name=name.strip(), total_vram_gb=total_vram_gb))
    return tuple(devices)


def detect_compute_profile() -> ComputeProfile:
    """Detect the local compute profile with optional CUDA discovery."""

    gpus = _detect_gpus_with_torch() or _detect_gpus_with_nvidia_smi()
    return ComputeProfile(
        cpu_threads=os.cpu_count() or 1,
        system_ram_gb=_detect_system_ram_gb(),
        gpus=gpus,
    )


def plan_workers(
    profile: ComputeProfile,
    *,
    smoke_mode: bool = False,
    device_preference: DevicePreference = "auto",
    requested_cpu_workers: int | None = None,
) -> WorkerPlan:
    """Build a conservative parallel execution plan.

    The plan reserves some CPU capacity for the OS and GPU driver, limits GPU
    concurrency to one job per device by default, and keeps a CPU-only smoke mode.
    """

    if smoke_mode:
        return WorkerPlan(
            device="cpu",
            cpu_workers=min(2, profile.cpu_threads),
            dataloader_workers=0,
            baseline_parallel_jobs=1,
            seed_parallel_jobs=1,
            gpu_parallel_jobs=0,
            gpu_memory_fraction=None,
            precision="float32",
            smoke_mode=True,
            notes=("smoke_mode_for_ci_or_low_resource_debugging",),
        )

    cpu_workers = requested_cpu_workers or max(1, profile.cpu_threads - 2)
    cpu_workers = min(cpu_workers, profile.cpu_threads)
    dataloader_workers = min(8, max(1, profile.cpu_threads // 4))
    baseline_parallel_jobs = min(8, max(1, profile.cpu_threads // 4))
    seed_parallel_jobs = min(3, max(1, profile.cpu_threads // 8))

    notes: list[str] = []
    if device_preference == "cpu":
        device = "cpu"
        gpu_jobs = 0
        gpu_memory_fraction = None
        precision = "float32"
        notes.append("cuda_disabled_by_request")
    elif profile.has_cuda:
        device = "cuda:0"
        gpu_jobs = len(profile.gpus)
        gpu_memory_fraction = 0.85
        precision = "mixed_precision_auto"
        notes.append("run_one_training_job_per_gpu_unless_memory_profile_proves_more_is_safe")
    elif device_preference == "cuda":
        device = "cpu"
        gpu_jobs = 0
        gpu_memory_fraction = None
        precision = "float32"
        notes.append("cuda_requested_but_not_detected")
    else:
        device = "cpu"
        gpu_jobs = 0
        gpu_memory_fraction = None
        precision = "float32"

    if profile.system_ram_gb is not None and profile.system_ram_gb <= 32:
        notes.append("keep_large_datasets_memory_mapped_or_streamed_on_32gb_ram")

    return WorkerPlan(
        device=device,
        cpu_workers=cpu_workers,
        dataloader_workers=dataloader_workers,
        baseline_parallel_jobs=baseline_parallel_jobs,
        seed_parallel_jobs=seed_parallel_jobs,
        gpu_parallel_jobs=gpu_jobs,
        gpu_memory_fraction=gpu_memory_fraction,
        precision=precision,
        smoke_mode=False,
        notes=tuple(notes),
    )


def write_compute_report(path: Path, profile: ComputeProfile, plan: WorkerPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"profile": profile.to_dict(), "plan": plan.to_dict()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
