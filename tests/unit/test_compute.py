from __future__ import annotations

from forgeworld.runtime.compute import ComputeProfile, GpuDevice, plan_workers


def test_worker_plan_uses_cuda_and_parallel_cpu_for_local_workstation() -> None:
    profile = ComputeProfile(
        cpu_threads=32,
        system_ram_gb=32.0,
        gpus=(GpuDevice(index=0, name="NVIDIA GeForce RTX 5070 Ti", total_vram_gb=12.0),),
    )

    plan = plan_workers(profile)

    assert plan.device == "cuda:0"
    assert plan.cpu_workers == 30
    assert plan.dataloader_workers == 8
    assert plan.baseline_parallel_jobs == 8
    assert plan.seed_parallel_jobs == 3
    assert plan.gpu_parallel_jobs == 1
    assert plan.gpu_memory_fraction == 0.85
    assert "keep_large_datasets_memory_mapped_or_streamed_on_32gb_ram" in plan.notes


def test_smoke_mode_forces_small_cpu_plan() -> None:
    profile = ComputeProfile(
        cpu_threads=32,
        system_ram_gb=32.0,
        gpus=(GpuDevice(index=0, name="NVIDIA GeForce RTX 5070 Ti", total_vram_gb=12.0),),
    )

    plan = plan_workers(profile, smoke_mode=True)

    assert plan.device == "cpu"
    assert plan.cpu_workers == 2
    assert plan.dataloader_workers == 0
    assert plan.gpu_parallel_jobs == 0
    assert plan.smoke_mode


def test_cuda_request_without_gpu_falls_back_with_note() -> None:
    profile = ComputeProfile(cpu_threads=8, system_ram_gb=16.0, gpus=())

    plan = plan_workers(profile, device_preference="cuda")

    assert plan.device == "cpu"
    assert "cuda_requested_but_not_detected" in plan.notes
