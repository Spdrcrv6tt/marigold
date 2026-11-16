# Imhotep core + chip: firmware, image, simulations.
# Run in WSL:  bash rtl/core/tools/wsl.sh 'make -C rtl/top <target>'
#   make host           PC build of the firmware (MAC array as a C function)        -> $(B)/fw_host
#   make image          RV32IM firmware + ROM/flash image                           -> $(IMG)/flash.bin, flash.hex
#   make sim-rom        Verilator: whole chip, weights in on-die ROM, PDM in        -> $(B)/rom/sim
#   make sim-flash      Verilator: whole chip, weights in external quad SPI flash   -> $(B)/flash/sim
#   make sim-cov        sim-rom with toggle coverage (switching activity for power)  -> $(B)/cov/sim
TAG     ?= w8
LANES   ?= 16
USE_VEC ?= 1
RAM     ?= 4194304
THREADS ?= 1
CAPAW   ?= 16
RATIO   ?= 16
FWDEFS  ?=
UARTDIV := $(shell expr $(RATIO) \* 3072000 / 115200)
ROOT    := /mnt/d/marigold
IMG     := $(ROOT)/data/core/$(TAG)
# unit tests (test-mac, test-fan) compile against layout.h of two Poinsetta images, copied as fixtures (rtl/PROVENANCE.md)
UIMG    ?= $(ROOT)/rtl/core/tb/img/w8
UTSIMG  ?= $(ROOT)/rtl/core/tb/img/ts-w4c8r
B       := $(CORE_BUILD)
FW      := $(ROOT)/rtl/core/fw
CORE    := $(ROOT)/rtl/core
SRC     := $(ROOT)/rtl/src
TB      := $(ROOT)/rtl/top/tb
RVFLAGS := --target=riscv32-unknown-elf -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin -nostdlib \
           -mno-relax -msmall-data-limit=0 -DLANES=$(LANES) -DUSE_VEC=$(USE_VEC) $(FWDEFS) -I$(IMG) -Wall -Wno-unused
RTL     := $(CORE)/imh_cpu.sv $(CORE)/imh_mem.sv $(CORE)/imh_mac.sv $(CORE)/imh_vec.sv $(CORE)/imh_fan.sv $(CORE)/imh_core.sv $(SRC)/mac_lane.sv \
           $(SRC)/uart_tx.sv $(SRC)/spi_flash_reader.sv
CHIP    := $(RTL) $(ROOT)/rtl/top/imh_capture.sv $(ROOT)/rtl/top/imhotep_top.sv $(SRC)/pdm_frontend.sv $(SRC)/pdm_clkgen.sv \
           $(SRC)/cic_decim.sv $(SRC)/fir_decim2.sv $(SRC)/dc_block.sv $(SRC)/vad.sv $(TB)/imh_flash_model.sv
VFLAGS  := --cc --exe --build -j 8 -O3 --x-assign fast --x-initial fast -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-PINMISSING \
           -Wno-TIMESCALEMOD -I$(SRC) -I$(CORE) --top-module tb_chip -CFLAGS -O3 -MAKEFLAGS OPT_FAST=-O3 -GLANES=$(LANES) -GRAM_BYTES=$(RAM) \
           -GIMAGE=\"$(B)/flash.hex\" -GIMG_BYTES=$(IMGB) -GCAP_AW=$(CAPAW) -GUART_DIV=$(UARTDIV) $(if $(filter-out 1,$(THREADS)),--threads $(THREADS))

.PHONY: host image sim-rom sim-flash sim-cov test-mac
host:
	gcc -O2 -DHOST -DLANES=$(LANES) -DUSE_VEC=$(USE_VEC) -Wall -Wno-unused -I$(IMG) -o $(B)/fw_host $(FW)/model.c $(FW)/hal_host.c

image:
	$(LLVM)/clang $(RVFLAGS) -fuse-ld=lld -Wl,-T,$(FW)/link.ld -o $(B)/fw.elf $(FW)/start.S $(FW)/hal_rv.c $(FW)/model.c
	$(LLVM)/llvm-objcopy -O binary $(B)/fw.elf $(B)/fw.bin
	$(LLVM)/llvm-objdump -d $(B)/fw.elf > $(B)/fw.lst
	$(LLVM)/llvm-size $(B)/fw.elf
	python3 $(ROOT)/rtl/core/tools/catimage.py $(B)/fw.bin $(IMG)/data.bin $(B)/flash.bin
	od -An -v -tx1 -w1 $(B)/flash.bin > $(B)/flash.hex
	cp $(B)/fw.bin $(B)/flash.bin $(B)/flash.hex $(IMG)/

sim-rom:
	verilator $(VFLAGS) -GWSRC_ROM=1 --Mdir $(B)/rom -o sim $(TB)/tb_chip.sv $(CHIP) $(TB)/sim_main.cpp
# toggle-coverage build of the ROM chip for the power estimate (rtl/core/syn/power_core.py)
sim-cov:
	verilator $(VFLAGS) --coverage-toggle -GWSRC_ROM=1 --Mdir $(B)/cov -o sim $(TB)/tb_chip.sv $(CHIP) $(TB)/sim_main.cpp
sim-flash:
	verilator $(VFLAGS) -GWSRC_ROM=0 --Mdir $(B)/flash -o sim $(TB)/tb_chip.sv $(CHIP) $(TB)/sim_main.cpp

# unit test of the MAC array and the per-element unit against their C references (fw/hal_host.c)
test-mac:
	for L in 16 8 4; do \
	  verilator --cc --exe --build -j 8 -O3 --x-assign fast --x-initial fast -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-PINMISSING \
	    -I$(SRC) -I$(CORE) --top-module unit_tb -GLANES=$$L -CFLAGS "-O2 -DHOST -DHOST_LIB -DLANES=$$L -I$(FW) -I$(UIMG)" \
	    --Mdir $(B)/unit$$L -o test_units $(CORE)/tb/unit_tb.sv $(CORE)/imh_mac.sv $(CORE)/imh_vec.sv $(CORE)/imh_mem.sv $(SRC)/mac_lane.sv \
	    $(CORE)/tb/test_mac.cpp $(FW)/hal_host.c > $(B)/unit$$L.build.log 2>&1 || { grep -E "rror" -A4 $(B)/unit$$L.build.log | head -40; exit 1; }; \
	  $(B)/unit$$L/test_units 4000 7 || exit 1; \
	done
	verilator --cc --exe --build -j 8 -O3 --x-assign fast --x-initial fast -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-PINMISSING \
	  -I$(SRC) -I$(CORE) --top-module unit_tb -GLANES=16 -CFLAGS "-O2 -DHOST -DHOST_LIB -DLANES=16 -I$(FW) -I$(UTSIMG)" \
	  --Mdir $(B)/unitts -o test_units $(CORE)/tb/unit_tb.sv $(CORE)/imh_mac.sv $(CORE)/imh_vec.sv $(CORE)/imh_mem.sv $(SRC)/mac_lane.sv \
	  $(CORE)/tb/test_mac.cpp $(FW)/hal_host.c > $(B)/unitts.build.log 2>&1 || { grep -E "rror" -A4 $(B)/unitts.build.log | head -40; exit 1; }
	$(B)/unitts/test_units 12000 11

# unit test of the fan-out array against its C reference (fw/hal_host.c fan_go); the second run with a full-rate
# weight source prints the array's MAC efficiency
test-fan:
	verilator --cc --exe --build -j 8 -O3 --x-assign fast --x-initial fast -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-PINMISSING \
	  -I$(SRC) -I$(CORE) --top-module fan_tb -GLANES=16 -GRAM_BYTES=131072 -CFLAGS "-O2 -DHOST -DHOST_LIB -DLANES=16 -I$(FW) -I$(UIMG)" \
	  --Mdir $(B)/fan -o test_fan $(CORE)/tb/fan_tb.sv $(CORE)/imh_fan.sv $(CORE)/imh_mem.sv $(SRC)/mac_lane.sv \
	  $(CORE)/tb/test_fan.cpp $(FW)/hal_host.c > $(B)/fan.build.log 2>&1 || { grep -E "rror" -A4 $(B)/fan.build.log | head -40; exit 1; }
	$(B)/fan/test_fan 400 7
	$(B)/fan/test_fan 60 11 full
	$(B)/fan/test_fan 7 3 shapes

# ---- Poinsetta streaming firmware (rtl/core/fw/stream.c) with the format-3 image (mkimage_ts.py data)
TSIMG := $(ROOT)/data/core/ts-$(TAG)
IMGB  ?= 33554432
host-ts:
	gcc -O2 -DHOST -DLANES=$(LANES) $(FWDEFS) -Wall -Wno-unused -I$(TSIMG) -I$(FW) -o $(B)/fw_host_ts $(FW)/stream.c $(FW)/hal_host.c
image-ts:
	$(LLVM)/clang -I$(TSIMG) $(RVFLAGS) -fuse-ld=lld -Wl,-T,$(FW)/link.ld -o $(B)/fw_ts.elf $(FW)/start.S $(FW)/hal_rv.c $(FW)/stream.c
	$(LLVM)/llvm-objcopy -O binary $(B)/fw_ts.elf $(B)/fw.bin
	$(LLVM)/llvm-size $(B)/fw_ts.elf
	python3 $(ROOT)/rtl/core/tools/catimage.py $(B)/fw.bin $(TSIMG)/data.bin $(B)/flash.bin
	od -An -v -tx1 -w1 $(B)/flash.bin > $(B)/flash.hex

# ---- Marigold intent firmware (rtl/core/fw/marigold.c) with the mg image (rtl/core/tools/mkimage_mg.py data)
#   make TAG=randtest host-mg   PC build (fan-out array, MAC array, per-element unit as C functions)  -> $(B)/fw_host_mg
#       $(B)/fw_host_mg data.bin seg.raw events.txt [trace] [ram_bytes] [blob] [end flags]; check: rtl/core/tools/check_mg.py
#   make TAG=randtest image-mg  RV32IM build (32 KB PMEM, link_mg.ld) + ROM image                   -> $(B)/flash.hex
MGIMG := $(ROOT)/data/core/mg-$(TAG)
.PHONY: host-mg image-mg mg-data
mg-data: $(MGIMG)/layout.h
$(MGIMG)/layout.h: $(ROOT)/data/golden/mg-$(TAG).npz $(ROOT)/rtl/core/tools/mkimage_mg.py
	python3 $(ROOT)/rtl/core/tools/mkimage_mg.py data --tag $(TAG)
host-mg: $(MGIMG)/layout.h
	gcc -O2 -DHOST -DLANES=$(LANES) $(FWDEFS) -Wall -Wno-unused -I$(MGIMG) -I$(FW) -o $(B)/fw_host_mg $(FW)/marigold.c $(FW)/hal_host.c
image-mg: $(MGIMG)/layout.h
	$(LLVM)/clang -I$(MGIMG) $(RVFLAGS) -ffunction-sections -fdata-sections -fuse-ld=lld -Wl,--gc-sections \
	  -Wl,-T,$(FW)/link_mg.ld -o $(B)/fw_mg.elf $(FW)/start.S $(FW)/hal_rv.c $(FW)/marigold.c
	$(LLVM)/llvm-objcopy -O binary $(B)/fw_mg.elf $(B)/fw_mg.bin
	$(LLVM)/llvm-objdump -d $(B)/fw_mg.elf > $(B)/fw_mg.lst
	$(LLVM)/llvm-size -A $(B)/fw_mg.elf | grep -E "^\.(text|bss|data) "
	python3 $(ROOT)/rtl/core/tools/catimage.py $(B)/fw_mg.bin $(MGIMG)/data.bin $(B)/flash.bin \
	  $$(sed -n 's/^#define DATA_BASE \([0-9]*\)u$$/\1/p' $(MGIMG)/layout.h)
	od -An -v -tx1 -w1 $(B)/flash.bin > $(B)/flash.hex
