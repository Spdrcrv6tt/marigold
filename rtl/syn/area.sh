#!/bin/bash
# Logic area of Marigold's blocks with Yosys and the NanGate45 typical library (FreePDK45, Apache 2.0, read-only
# from ~/.poinsetta/pdk/orfs). Memories are inferred, listed with their bits and removed before mapping (they are
# counted as SRAM or ROM at 28 nm densities in reports/p1-core.md). Run in WSL after rtl/tools/env.sh:
#   bash rtl/syn/area.sh  -> $MARIGOLD_BUILD/syn/<top>.stat.txt, <top>.mem.txt
set -e
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LIB=$HOME/.poinsetta/pdk/orfs/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
O=$MARIGOLD_BUILD/syn; mkdir -p $O
S="$R/src"; C="$R/core"; T="$R/top"; W="$R/wake"
CORE="$C/mg_core.sv $C/imh_cpu.sv $C/imh_mem.sv $C/imh_mac.sv $C/imh_vec.sv $C/imh_fan.sv $S/mac_lane.sv $S/clk_gate.sv $S/latch_ram.sv"
CTL="$T/mg_ctl.sv $T/mg_cfg_load.sv $T/mg_cfg_ram.sv $T/mg_match.sv $T/mg_msg.sv $T/mg_gpio.sv $T/mg_i2c.sv $S/uart_tx.sv $S/latch_ram.sv $S/clk_gate.sv"
FE="$S/pdm_frontend.sv $S/pdm_clkgen.sv $S/cic_decim.sv $S/fir_decim2.sv $S/dc_block.sv $S/clk_gate.sv"
run() {   # top, files...
  top=$1; shift
  cat > $O/$top.ys <<YS
read_verilog -sv -DSYNTHESIS -I$S -I$C $*
$PRE
hierarchy -check -top $top
proc; flatten; opt -purge
memory -nomap
tee -o $O/$top.mem.txt select -list t:\$mem_v2
tee -a $O/$top.mem.txt stat
delete t:\$mem_v2
synth -top $top -flatten
dfflibmap -liberty $LIB
abc -liberty $LIB
opt_clean
tee -o $O/$top.stat.txt stat -liberty $LIB
YS
  timeout 3000 yosys -q -l $O/$top.log $O/$top.ys > /dev/null 2>&1 || echo "yosys $top failed (log $O/$top.log)"
  grep -E "Chip area" $O/$top.stat.txt | tail -1 | sed "s/^/$top: /"
}
run mg_ctl $CTL &
PRE="chparam -set AW 8 -set BW 4 mg_capture" run mg_capture $T/mg_capture.sv $S/clk_gate.sv &   # logic only: 16 banks, ring shrunk
run pdm_frontend $FE &
run vad $S/vad.sv $S/clk_gate.sv &
PRE="chparam -set MEMDIR \"$W/mem/\" mg_wake" run mg_wake $W/mg_wake.sv $W/wk_fe.sv $W/wk_nn.sv $W/wk_dp.sv $W/wk_mem.sv $S/clk_gate.sv &
run mg_core $CORE &
wait
