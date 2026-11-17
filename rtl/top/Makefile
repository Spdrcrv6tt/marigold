# Marigold core: unit tests of the MAC arrays and the per-element unit, firmware and ROM image.
# Run in WSL:  bash rtl/core/tools/wsl.sh 'make -C rtl/top <target>'   (the whole chip: rtl/top/chip.mk)
#   make test-mac / test-fan    MAC array + per-element unit / fan-out array against their C references
#   make TAG=<tag> image-mg     RV32IM firmware + ROM image of a model export   (section below)
# Poinsetta's whole-chip and streaming-decoder targets were removed (review R15); their sources are in
# rtl/ref/poinsetta and their history in Poinsetta commit 1719528 (rtl/PROVENANCE.md).
TAG     ?= w8
LANES   ?= 16
USE_VEC ?= 1
RAM     ?= 4194304
THREADS ?= 1
CAPAW   ?= 16
RATIO   ?= 16
FWDEFS  ?=
ROOT    := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
IMG     := $(ROOT)/data/core/$(TAG)
# unit tests (test-mac, test-fan) compile against layout.h of the model whose LUT tables the committed
# rtl/core/imh_rom.svh holds (tag read from its first line; layout.h made from weights/ if missing);
# another export: UIMG=UTSIMG=$(ROOT)/data/core/mg-<tag>
ROMTAG  := $(shell sed -n '1s/.*from mg-\([A-Za-z0-9_]*\)\.npz.*/\1/p' $(ROOT)/rtl/core/imh_rom.svh)
UIMG    ?= $(ROOT)/data/core/mg-$(ROMTAG)
UTSIMG  ?= $(UIMG)
UFIX     = test -f $(UIMG)/layout.h || python3 $(ROOT)/rtl/core/tools/mkimage_mg.py data --tag $(ROMTAG) --golden-dir $(ROOT)/weights
B       := $(CORE_BUILD)
FW      := $(ROOT)/rtl/core/fw
CORE    := $(ROOT)/rtl/core
SRC     := $(ROOT)/rtl/src
TB      := $(ROOT)/rtl/top/tb
RVFLAGS := --target=riscv32-unknown-elf -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin -nostdlib \
           -mno-relax -msmall-data-limit=0 -DLANES=$(LANES) -DUSE_VEC=$(USE_VEC) $(FWDEFS) -I$(IMG) -Wall -Wno-unused

.PHONY: test-mac test-fan
# unit test of the MAC array and the per-element unit against their C references (fw/hal_host.c)
test-mac:
	$(UFIX)
	for L in 16 8 4; do \
	  rm -rf $(B)/unit$$L; verilator --cc --exe --build -j 8 -O3 --x-assign fast --x-initial fast -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-PINMISSING \
	    -I$(SRC) -I$(CORE) --top-module unit_tb -GLANES=$$L -CFLAGS "-O2 -DHOST -DHOST_LIB -DLANES=$$L -I$(FW) -I$(UIMG)" \
	    --Mdir $(B)/unit$$L -o test_units $(CORE)/tb/unit_tb.sv $(CORE)/imh_mac.sv $(CORE)/imh_vec.sv $(CORE)/imh_mem.sv $(SRC)/mac_lane.sv \
	    $(CORE)/tb/test_mac.cpp $(FW)/hal_host.c > $(B)/unit$$L.build.log 2>&1 || { grep -E "rror" -A4 $(B)/unit$$L.build.log | head -40; exit 1; }; \
	  $(B)/unit$$L/test_units 4000 7 || exit 1; \
	done
	rm -rf $(B)/unitts; verilator --cc --exe --build -j 8 -O3 --x-assign fast --x-initial fast -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-PINMISSING \
	  -I$(SRC) -I$(CORE) --top-module unit_tb -GLANES=16 -CFLAGS "-O2 -DHOST -DHOST_LIB -DLANES=16 -I$(FW) -I$(UTSIMG)" \
	  --Mdir $(B)/unitts -o test_units $(CORE)/tb/unit_tb.sv $(CORE)/imh_mac.sv $(CORE)/imh_vec.sv $(CORE)/imh_mem.sv $(SRC)/mac_lane.sv \
	  $(CORE)/tb/test_mac.cpp $(FW)/hal_host.c > $(B)/unitts.build.log 2>&1 || { grep -E "rror" -A4 $(B)/unitts.build.log | head -40; exit 1; }
	$(B)/unitts/test_units 12000 11

# unit test of the fan-out array against its C reference (fw/hal_host.c fan_go); the second run with a full-rate
# weight source prints the array's MAC efficiency
test-fan:
	$(UFIX)
	rm -rf $(B)/fan; verilator --cc --exe --build -j 8 -O3 --x-assign fast --x-initial fast -Wno-fatal -Wno-WIDTH -Wno-UNUSED -Wno-PINMISSING \
	  -I$(SRC) -I$(CORE) --top-module fan_tb -GLANES=16 -GRAM_BYTES=131072 -CFLAGS "-O2 -DHOST -DHOST_LIB -DLANES=16 -I$(FW) -I$(UIMG)" \
	  --Mdir $(B)/fan -o test_fan $(CORE)/tb/fan_tb.sv $(CORE)/imh_fan.sv $(CORE)/imh_mem.sv $(SRC)/mac_lane.sv \
	  $(CORE)/tb/test_fan.cpp $(FW)/hal_host.c > $(B)/fan.build.log 2>&1 || { grep -E "rror" -A4 $(B)/fan.build.log | head -40; exit 1; }
	$(B)/fan/test_fan 400 7
	$(B)/fan/test_fan 60 11 full
	$(B)/fan/test_fan 7 3 shapes

# ---- Marigold intent firmware (rtl/core/fw/marigold.c) with the mg image (rtl/core/tools/mkimage_mg.py data)
#   make TAG=randtest host-mg   PC build (fan-out array, MAC array, per-element unit as C functions)  -> $(B)/fw_host_mg
#       $(B)/fw_host_mg data.bin seg.raw events.txt [trace] [ram_bytes] [blob] [end flags]; check: rtl/core/tools/check_mg.py
#   make TAG=randtest image-mg  RV32IM build (32 KB PMEM, link_mg.ld) + ROM image                   -> $(B)/flash.hex
#   A test export outside data/golden (rtl/core/tools/mkrand_mg.py) into its own build directory, e.g.
#   make TAG=rand3 MGGOLD=$(ROOT)/data/core/mg-rand3 B=$CORE_BUILD/r3 host-mg image-mg
MGIMG  := $(ROOT)/data/core/mg-$(TAG)
MGGOLD ?= $(ROOT)/data/golden
.PHONY: host-mg image-mg mg-data
mg-data: $(MGIMG)/layout.h
$(MGIMG)/layout.h: $(MGGOLD)/mg-$(TAG).npz $(ROOT)/rtl/core/tools/mkimage_mg.py
	python3 $(ROOT)/rtl/core/tools/mkimage_mg.py data --tag $(TAG) --golden-dir $(MGGOLD)
host-mg: $(MGIMG)/layout.h
	mkdir -p $(B)
	gcc -O2 -DHOST -DLANES=$(LANES) $(FWDEFS) -Wall -Wno-unused -I$(MGIMG) -I$(FW) -o $(B)/fw_host_mg $(FW)/marigold.c $(FW)/hal_host.c
image-mg: $(MGIMG)/layout.h
	mkdir -p $(B)
	$(LLVM)/clang -I$(MGIMG) $(RVFLAGS) -ffunction-sections -fdata-sections -fuse-ld=lld -Wl,--gc-sections \
	  -Wl,-T,$(FW)/link_mg.ld -o $(B)/fw_mg.elf $(FW)/start.S $(FW)/hal_rv.c $(FW)/marigold.c
	$(LLVM)/llvm-objcopy -O binary $(B)/fw_mg.elf $(B)/fw_mg.bin
	$(LLVM)/llvm-objdump -d $(B)/fw_mg.elf > $(B)/fw_mg.lst
	$(LLVM)/llvm-size -A $(B)/fw_mg.elf | grep -E "^\.(text|bss|data) "
	python3 $(ROOT)/rtl/core/tools/catimage.py $(B)/fw_mg.bin $(MGIMG)/data.bin $(B)/flash.bin \
	  $$(sed -n 's/^#define DATA_BASE \([0-9]*\)u$$/\1/p' $(MGIMG)/layout.h)
	od -An -v -tx1 -w1 $(B)/flash.bin > $(B)/flash.hex
