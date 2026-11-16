. /mnt/d/marigold/rtl/tools/env.sh
cd /mnt/d/marigold/rtl/wake/mem
yosys -q -l ../syn/gates.log -s ../syn/gates.ys > /dev/null 2>&1
grep -E "cells|_" ../syn/gates.log | tail -30
