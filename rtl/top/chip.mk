# Whole-chip Verilator build of marigold_top with the harness rtl/top/tb/mg_main.cpp (run in WSL with
# rtl/core/tools/env.sh). make -f rtl/top/chip.mk sim IMG=<flash.hex> [BSUF=x]; make -f rtl/top/chip.mk smoke
ROOT  := /mnt/d/marigold
B     := $(MARIGOLD_BUILD)/chip$(BSUF)
SRC   := $(ROOT)/rtl/src
CORE  := $(ROOT)/rtl/core
TOP   := $(ROOT)/rtl/top
LLVM  ?= $(HOME)/.imhotep/core-llvm/bin
IMG   ?= $(B)/smoke.hex
ROMB  ?= 4194304
RTL   := $(TOP)/marigold_top.sv $(TOP)/mg_ctl.sv $(TOP)/mg_capture.sv $(TOP)/mg_wake_stub.sv $(TOP)/mg_cfg_load.sv \
         $(TOP)/mg_cfg_ram.sv $(TOP)/mg_match.sv $(TOP)/mg_msg.sv $(TOP)/mg_gpio.sv $(TOP)/mg_i2c.sv \
         $(CORE)/mg_core.sv $(CORE)/imh_cpu.sv $(CORE)/imh_mem.sv $(CORE)/imh_mac.sv $(CORE)/imh_vec.sv $(CORE)/imh_fan.sv \
         $(SRC)/pdm_frontend.sv $(SRC)/pdm_clkgen.sv $(SRC)/cic_decim.sv $(SRC)/fir_decim2.sv $(SRC)/dc_block.sv \
         $(SRC)/vad.sv $(SRC)/clk_gate.sv $(SRC)/latch_ram.sv $(SRC)/mac_lane.sv $(SRC)/uart_tx.sv          $(ROOT)/rtl/wake/mg_wake.sv $(ROOT)/rtl/wake/wk_fe.sv $(ROOT)/rtl/wake/wk_nn.sv $(ROOT)/rtl/wake/wk_dp.sv          $(ROOT)/rtl/wake/wk_mem.sv
STUB  ?= 0
VFLAGS := --cc --exe --build -j 8 -O3 --x-assign fast --x-initial fast -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-PINMISSING \
          -Wno-TIMESCALEMOD -I$(SRC) -I$(CORE) --top-module marigold_top -CFLAGS -O3 -MAKEFLAGS OPT_FAST=-O3 \
          -GROM_BYTES=$(ROMB) -GWAKE_STUB=$(STUB)
RVFLAGS := --target=riscv32-unknown-elf -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin -nostdlib -mno-relax \
           -msmall-data-limit=0 -Wall

.PHONY: sim smoke
sim:
	mkdir -p $(B)
	verilator $(VFLAGS) -GROM_INIT=\"$(IMG)\" --Mdir $(B)/obj -o sim $(RTL) $(TOP)/tb/mg_main.cpp > $(B)/build.log 2>&1 || \
	  { grep -E "rror" -A4 $(B)/build.log | head -40; exit 1; }

smoke:
	mkdir -p $(B)
	$(LLVM)/clang $(RVFLAGS) -fuse-ld=lld -Wl,-T,$(CORE)/fw/link.ld -o $(B)/smoke.elf $(CORE)/fw/start.S $(TOP)/tb/fw_smoke.c
	$(LLVM)/llvm-objcopy -O binary $(B)/smoke.elf $(B)/smoke.bin
	python3 -c "import sys; fw=open('$(B)/smoke.bin','rb').read(); img=b'IMH1'+len(fw).to_bytes(4,'little')+fw; open('$(B)/smoke.img','wb').write(img)"
	od -An -v -tx1 -w1 $(B)/smoke.img > $(B)/smoke.hex
	$(LLVM)/llvm-size $(B)/smoke.elf
