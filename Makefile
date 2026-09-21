.DEFAULT_GOAL := help

.PHONY: help check format lint test sim conformance fpga

help:
	@printf '%s\n' 'HASLAB repository foundation' \
	  '' \
	  'Available now:' \
	  '  make check  Validate repository metadata' \
	  '  make test   Run reference, simulator, infrastructure tests and conformance fixtures' \
	  '  make sim    Run the command-simulator unit tests' \
	  '  make conformance  Validate ABI/corpus integrity and run stored command fixtures' \
	  '' \
	  'Reserved for future implementation:' \
	  '  make format | lint | fpga'

check:
	@./scripts/check-repository.sh

test:
	@PYTHONPATH=reference python3 -m unittest discover -s reference/tests -p 'test_*.py' -v
	@PYTHONPATH=reference:simulation python3 -m unittest discover -s simulation/tests -p 'test_*.py' -v
	@$(MAKE) conformance

conformance:
	@python3 -m conformance.generate
	@PYTHONPATH=reference:simulation:. python3 -m unittest discover -s conformance/tests -p 'test_*.py' -v
	@PYTHONPATH=reference:simulation:. python3 -m conformance.runner

sim:
	@PYTHONPATH=reference:simulation python3 -m unittest discover -s simulation/tests -p 'test_*.py' -v

format lint fpga:
	@printf '%s\n' 'This target is reserved; HASLAB implementation tooling has not been added yet.'
	@false
