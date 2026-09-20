.DEFAULT_GOAL := help

.PHONY: help check format lint test sim fpga

help:
	@printf '%s\n' 'HASLAB repository foundation' \
	  '' \
	  'Available now:' \
	  '  make check  Validate repository metadata' \
	  '  make test   Run all Python reference/simulator unit tests' \
	  '  make sim    Run the command-simulator unit tests' \
	  '' \
	  'Reserved for future implementation:' \
	  '  make format | lint | fpga'

check:
	@./scripts/check-repository.sh

test:
	@PYTHONPATH=reference python3 -m unittest discover -s reference/tests -p 'test_*.py' -v
	@PYTHONPATH=reference:simulation python3 -m unittest discover -s simulation/tests -p 'test_*.py' -v

sim:
	@PYTHONPATH=reference:simulation python3 -m unittest discover -s simulation/tests -p 'test_*.py' -v

format lint fpga:
	@printf '%s\n' 'This target is reserved; HASLAB implementation tooling has not been added yet.'
	@false
