"""Print a pass/fail table from the cocotb result files and exit non-zero on any failure."""
import glob
import os
import sys
import xml.etree.ElementTree as ET

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tb", "results")
blocks = sys.argv[1:]
bad = 0
total = 0
for b in blocks:
    files = sorted(glob.glob(os.path.join(here, b + "_*.xml")))
    if not files:
        print("%-8s NO RESULTS (build failed? see tb/results/%s_*.log)" % (b, b))
        bad += 1
        continue
    for f in files:
        root = ET.parse(f).getroot()
        for tc in root.iter("testcase"):
            total += 1
            failed = any(c.tag in ("failure", "error") for c in tc)
            bad += failed
            print("%-22s %-34s %s" % (os.path.basename(f)[:-4], tc.get("name"), "FAIL" if failed else "pass"))
print("%d tests, %d failed" % (total, bad))
sys.exit(1 if bad else 0)
