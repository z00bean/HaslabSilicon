# Runtime tests

The current tests compile a license-clear synthetic Conv-SiLU tile, load and execute it through the simulator runtime, and compare every output byte with the independent numerical model. They also cover FIFO submission accounting, corrupt package rejection, exact input binding, stale reset-generation tokens, and structured arithmetic-fault completion. The pinned full-layer benchmark separately executes and compares all 409,600 values.
