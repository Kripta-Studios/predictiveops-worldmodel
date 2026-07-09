# Runtime Compute Profile

ForgeWorld uses hardware detection and explicit compute profiles instead of
hard-coded training assumptions.

For the local workstation described by the project owner:

- 32 CPU threads;
- 32 GB system RAM;
- one NVIDIA CUDA GPU with 12 GB VRAM.

The default plan is conservative:

- reserve two CPU threads for the OS and GPU driver;
- use up to eight data-loading workers;
- parallelize simple baselines and development seeds on CPU;
- run one neural training job per 12 GB GPU unless measured memory use proves
  that more concurrency is safe;
- keep every training/evaluation command capable of CPU smoke mode.

Generate the detected profile:

```powershell
python scripts/system/print_compute_profile.py --write-report outputs/system/compute_profile.json
```

Force smoke mode:

```powershell
python scripts/system/print_compute_profile.py --smoke
```

The profile is execution evidence only. It is not benchmark evidence and does
not support performance claims without recorded timings and matched protocols.
