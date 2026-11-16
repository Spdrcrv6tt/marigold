# Wake spotter cocotb tests (Verilator), bit-exact against sw/wake/golden. Run from Windows:
#   bash rtl/wake/tb/run.sh            (all variants)       or inside WSL: make -C rtl/wake/tb VARIANT=plain
# Variants: plain (no stalls), stall (ROM and config-port grant stalls). Vectors: rtl/wake/tb/vectors/<case>/.
SIM           ?= verilator
TOPLEVEL_LANG ?= verilog
VARIANT       ?= plain
WK            := $(abspath ..)
COCOTB_TOPLEVEL     := wake_tb
COCOTB_TEST_MODULES := test_wake
VERILOG_SOURCES     := $(abspath ../../src/clk_gate.sv) $(WK)/wk_mem.sv $(WK)/wk_fe.sv $(WK)/wk_nn.sv $(WK)/wk_dp.sv $(WK)/mg_wake.sv $(abspath wake_tb.sv)
SIM_BUILD           := $(or $(MARIGOLD_BUILD),.)/sim_build/wake_$(VARIANT)
COCOTB_RESULTS_FILE := results/wake_$(VARIANT)_$(WAKE_CASE)_$(WAKE_CORRUPT).xml
WAKE_CASE        ?= breeze
WAKE_CORRUPT     ?= 0
WAKE_EXPECT_FAIL ?= 0
MEMD_plain := $(WK)/mem/
MEMD_stall := $(WK)/mem/
MEMD_flip  := $(abspath results)/mem_flip/
ARGS_plain := -GROM_STALL=0 -GCFG_STALL=0
ARGS_stall := -GROM_STALL=1 -GCFG_STALL=1
ARGS_flip  := -GROM_STALL=0 -GCFG_STALL=0
ARGS_debug := -GROM_STALL=0 -GCFG_STALL=0 +define+WAKE_DEBUG
MEMD_debug := $(WK)/mem/
EXTRA_ARGS += -Wno-fatal -Wno-WIDTH -Wno-UNUSED -O2 -GMEMDIR='"$(MEMD_$(VARIANT))"' $(ARGS_$(VARIANT))
TRACE := $(abspath results)/trace_$(WAKE_CASE)_$(VARIANT)_$(WAKE_CORRUPT).txt
PLUSARGS += +TRACE=$(TRACE) +CFG=$(abspath vectors)/$(WAKE_CASE)/cfg.mem
export WAKE_CASE WAKE_CORRUPT WAKE_EXPECT_FAIL
export WAKE_TRACE := $(TRACE)
export VARIANT
export PYTHONPATH := $(abspath .):$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
