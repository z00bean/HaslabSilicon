# Runtime tests

The current tests compile license-clear synthetic Conv-SiLU work, load and execute it through the simulator runtime, and compare output bytes with the independent numerical model. They also cover FIFO submission accounting, corrupt package rejection, exact input binding, stale reset-generation tokens, and structured arithmetic-fault completion. The pinned two-layer benchmark separately executes and compares all 614,400 retained values.
