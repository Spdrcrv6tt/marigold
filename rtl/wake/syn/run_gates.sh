H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); M=$(cd "$H/../../.." && pwd)
. "$M/rtl/tools/env.sh"
sed "s#/mnt/d/marigold#$M#g" "$H/gates.ys" > "${MARIGOLD_BUILD}/wake_gates.ys"
cd "$M/rtl/wake/mem"
yosys -q -l ../syn/gates.log -s "${MARIGOLD_BUILD}/wake_gates.ys" > /dev/null 2>&1
grep -E "cells|_" ../syn/gates.log | tail -30
