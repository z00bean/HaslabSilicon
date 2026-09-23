# FPGA integration

Board-specific wrappers, constraints, tool settings, and reproducible build instructions will live here. Keep portable accelerator sources under `hardware/rtl/`.

No board has been selected and no FPGA project, bitstream, timing result, or resource claim exists.

## Selection order

Do not purchase a board solely from nominal LUT, DSP, or memory counts. First run small synthesis probes for the HASLAB SRAMs, INT8 MAC lane, DMA/AXI path, and debug logic on at least two plausible targets. Record tool and license requirements along with resource and timing results. The selected board must leave enough margin for integration and observability, not merely fit the arithmetic core.

The current leading vision candidate is the [AMD Kria KV260](https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html) because it combines a Zynq UltraScale+ MPSoC, 4 GB DDR, substantial programmable-logic resources, USB cameras, direct camera interfaces, display output, and an embedded Linux host. It is a candidate rather than a decision; its x86-64 Windows/Linux vendor-tool and licensing requirements must be acceptable and its measured synthesis results must justify the larger device.

The current constrained university/reference candidate is the [Digilent Zybo Z7-20](https://digilent.com/shop/zybo-z7-zynq-7000-arm-fpga-soc-development-board/). It provides a smaller Zynq-7020 fabric, 1 GB DDR3L, a direct Pcam/MIPI connector, HDMI, Ethernet, and onboard programming. It is useful for exposing resource and bandwidth pressure, but it offers much less accelerator margin than the KV260. The Z7-10 is not the intended HASLAB vision target.

## Development-host split

Portable lint, unit simulation, differential tests, and most RTL iteration should run with open-source tools on macOS and Linux. Verilator is the primary planned simulator; Icarus Verilog can provide a second implementation for compatible tests. Board-specific simulation, synthesis, place-and-route, embedded Linux generation, and programming use the selected vendor flow on a supported x86-64 Linux or Windows host. Do not make a Mac-only vendor-tool assumption part of the project.

## Bring-up order

1. Run the binary conformance corpus against portable RTL simulation.
2. Synthesize target probes and select the board from recorded evidence.
3. Bring up reset, registers, local SRAM, DMA, and stored-image inference.
4. Compare layer boundaries and full COCO accuracy on the actual command path.
5. Add a USB camera first when it shortens integration; add a direct sensor path when its latency or deployment value is measurable.
6. Publish camera-to-box latency, sustained throughput, dropped frames, thermals, power, transfers, and accuracy under `benchmarks/MEASUREMENT.md`.
