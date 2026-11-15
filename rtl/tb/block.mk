# One cocotb run per (block, variant). Called from rtl/Makefile: make -C tb -f block.mk BLOCK=pdm run
SIM            ?= verilator
TOPLEVEL_LANG  ?= verilog
SRC            := $(abspath ../src)

# block -> top module, sources, variants (extra Verilator -G parameter sets)
TOP_pdm    := pdm_frontend
SRCS_pdm   := pdm_frontend.sv pdm_clkgen.sv cic_decim.sv fir_decim2.sv dc_block.sv clk_gate.sv
VARS_pdm   := div1 div4 div16
ARGS_pdm_div1  := -GDIV=1
# toggle-coverage builds used only by `make activity` (switching activity for the power estimate)
ARGS_pdm_cov := -GDIV=1 --coverage-toggle --coverage-max-width 4096
ARGS_vad_cov := --coverage-toggle --coverage-max-width 4096
ARGS_pdm_div4  := -GDIV=4
ARGS_pdm_div16 := -GDIV=16

TOP_vad    := vad
SRCS_vad   := vad.sv clk_gate.sv
VARS_vad   := default

TOP_flash  := flash_tb
SRCS_flash := spi_flash_reader.sv ../tb/flash_model.sv ../tb/flash_tb.sv
VARS_flash := quad octal quad32 half2
ARGS_flash_common := --timing --timescale 1ps/1ps
ARGS_flash_quad   := -GIOW=4 $(ARGS_flash_common)
ARGS_flash_octal  := -GIOW=8 $(ARGS_flash_common)
ARGS_flash_quad32 := -GIOW=4 -GAW=32 $(ARGS_flash_common)
ARGS_flash_half2  := -GIOW=8 -GHALF=2 $(ARGS_flash_common)

TOP_uart   := uart_tx
SRCS_uart  := uart_tx.sv
VARS_uart  := default

TOP_mac    := mac_lane
SRCS_mac   := mac_lane.sv
VARS_mac   := w8 w4 w8p w4p
ARGS_mac_w8  := -GWW=8
ARGS_mac_w4  := -GWW=4
ARGS_mac_w8p := -GWW=8 -GPIPE=1
ARGS_mac_w4p := -GWW=4 -GPIPE=1

TOP_ram    := latch_ram
SRCS_ram   := latch_ram.sv clk_gate.sv
VARS_ram   := latch flop
ARGS_ram_latch := -GWORDS=64 -GWIDTH=8 -GFLOPS=0
ARGS_ram_flop  := -GWORDS=48 -GWIDTH=12 -GFLOPS=1

VARIANT ?= default

.PHONY: run
run:
	@for v in $(VARS_$(BLOCK)); do \
	  $(MAKE) --no-print-directory -f block.mk BLOCK=$(BLOCK) VARIANT=$$v one || exit 1; \
	done

.PHONY: one
one:
	@mkdir -p results
	@$(MAKE) --no-print-directory -f block.mk BLOCK=$(BLOCK) VARIANT=$(VARIANT) sim \
	  > results/$(BLOCK)_$(VARIANT).log 2>&1; \
	  grep -E "TESTS=|PASS=|FAIL=|Error|%Error" results/$(BLOCK)_$(VARIANT).log | tail -3; true

ifneq ($(filter sim,$(MAKECMDGOALS)),)
COCOTB_TOPLEVEL     := $(TOP_$(BLOCK))
COCOTB_TEST_MODULES := test_$(BLOCK)
VERILOG_SOURCES     := $(addprefix $(SRC)/,$(SRCS_$(BLOCK)))
SIM_BUILD           := $(or $(MARIGOLD_BUILD),.)/sim_build/$(BLOCK)_$(VARIANT)
COCOTB_RESULTS_FILE := results/$(BLOCK)_$(VARIANT).xml
EXTRA_ARGS          += -I$(SRC) -Wno-fatal -Wno-WIDTH -Wno-UNUSED -O2 $(ARGS_$(BLOCK)_$(VARIANT))
export VARIANT
ifeq ($(VARIANT),cov)
export COCOTB_TEST_FILTER := activity_stimulus
endif
export PYTHONPATH := $(abspath ../model):$(abspath .):$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
endif
