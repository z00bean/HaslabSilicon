# Runtime tests

The current tests compile a license-clear synthetic Conv-SiLU tile, load and execute it through the simulator runtime, and compare every output byte with the independent numerical model. They also cover corrupt package rejection, exact input binding, stale reset-generation tokens, and structured arithmetic-fault completion.
