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
MEMD_v2plain := $(WK)/mem2/
MEMD_v2stall := $(WK)/mem2/
MEMD_v2flip  := $(abspath results)/mem2_flip/
V1 := -GLANES=4 -GC=96 -GNOUT=40 -GGATE=0 -GPREROLL=0
V2 := -GLANES=16 -GC=240 -GNOUT=48 -GGATE=1 -GPREROLL=10
ARGS_plain := -GROM_STALL=0 -GCFG_STALL=0 $(V1)
ARGS_stall := -GROM_STALL=1 -GCFG_STALL=1 $(V1)
ARGS_flip  := -GROM_STALL=0 -GCFG_STALL=0 $(V1)
ARGS_v2plain := -GROM_STALL=0 -GCFG_STALL=0 $(V2)
ARGS_v2stall := -GROM_STALL=1 -GCFG_STALL=1 $(V2)
ARGS_v2flip  := -GROM_STALL=0 -GCFG_STALL=0 $(V2)
VECDIR := $(if $(findstring v2,$(VARIANT)),vectors2,vectors)
ARGS_debug := -GROM_STALL=0 -GCFG_STALL=0 +define+WAKE_DEBUG
MEMD_debug := $(WK)/mem/
EXTRA_ARGS += -Wno-fatal -Wno-WIDTH -Wno-UNUSED -O2 -GMEMDIR='"$(MEMD_$(VARIANT))"' $(ARGS_$(VARIANT))
TRACE := $(abspath results)/trace_$(WAKE_CASE)_$(VARIANT)_$(WAKE_CORRUPT).txt
PLUSARGS += +TRACE=$(TRACE) +CFG=$(abspath $(VECDIR))/$(WAKE_CASE)/cfg.mem
export WAKE_CASE WAKE_CORRUPT WAKE_EXPECT_FAIL
export WAKE_VECDIR := $(VECDIR)
export WAKE_TRACE := $(TRACE)
export VARIANT
export PYTHONPATH := $(abspath .):$(PYTHONPATH)
include $(shell cocotb-config --makefiles)/Makefile.sim
