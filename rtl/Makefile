# Imhotep RTL. Run inside WSL after `. rtl/tools/env.sh`, or from Windows with rtl/tools/wsl.sh:
#   make -C rtl test          all block tests (Verilator + cocotb), bit-exact against rtl/model
#   make -C rtl test-pdm      one block
#   make -C rtl quality       PDM signal quality + VAD accuracy from the reference models
#   make -C rtl synth         Yosys + OpenSTA against GF180 (see syn/)
BLOCKS ?= pdm vad flash uart mac ram

.PHONY: test quality synth activity clean $(addprefix test-,$(BLOCKS))

test: $(addprefix test-,$(BLOCKS))
	@python tools/summary.py $(BLOCKS)

$(addprefix test-,$(BLOCKS)): test-%:
	@$(MAKE) --no-print-directory -C tb -f block.mk BLOCK=$* run

quality:
	python model/pdm_quality.py
	python model/vad_eval.py

# switching activity from RTL simulation (Verilator toggle coverage) for the power estimate
activity:
	cd tb && rm -f coverage.dat && $(MAKE) --no-print-directory -f block.mk BLOCK=pdm VARIANT=cov one && mv coverage.dat results/pdm_cov.dat
	cd tb && rm -f coverage.dat && $(MAKE) --no-print-directory -f block.mk BLOCK=vad VARIANT=cov one && mv coverage.dat results/vad_cov.dat
	python syn/activity.py

synth:
	python syn/synth.py
	python syn/pnr.py

clean:
	rm -rf tb/sim_build tb/results tb/__pycache__ model/__pycache__
