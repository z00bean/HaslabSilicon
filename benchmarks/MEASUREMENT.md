# Hardware inference measurement protocol

This is the reporting contract for future HASLAB FPGA and silicon results. No current repository result qualifies as a hardware benchmark. Functional simulator command counts and DMA byte totals are useful planning data, but they are not elapsed time, measured memory traffic, or energy.

For each result, publish a machine-readable record and reproduction instructions containing:

- Git revision; model and weight hashes; export, compiler, package, bitstream, and firmware hashes; ABI and numerical profile versions.
- Board or die revision, device, memory configuration, clock, voltage, cooling, host, power supply, tool versions, and build commands.
- Workload, dataset split, input resolution, batch size, precision, preprocessing, postprocessing, and exactly which operations run on host versus accelerator. Declare every fallback.
- Correctness: layerwise mismatch counts against the golden path, detection mAP50–95 (or the workload's agreed metric), and the exact evaluation command.
- Timing: separate preprocessing, transfers, accelerator, and postprocessing; report end-to-end input-to-result latency distributions (at least median, p95, and p99), throughput under stated concurrency, warmup, sample count, and measurement clock.
- Power: instrument location and calibration; idle and active power; whole-board and accelerator-rail readings when accessible; sample rate, averaging window, ambient conditions, sustained-run duration, temperature, and throttling. State unavailable fields explicitly.
- Derived energy per inference from integrated measured joules over completed inferences, with idle-inclusive and incremental figures distinguished. Publish raw traces or sufficient aggregates to recompute the calculation.
- Resource and cost context: LUT/DSP/BRAM or die area, external memory capacity and measured traffic, and board/package price at the recorded date if a cost claim is made.

Compare systems only on the same model artifact, accuracy target, input, batch, workload partition, and measurement boundary. If those conditions differ, report separate results and explain the difference; do not produce a speedup or efficiency ratio. For model-specific hardware, include the cost and time of a model change when making a programmability or flexibility claim. Label estimates, simulator projections, FPGA measurements, and silicon measurements separately.
