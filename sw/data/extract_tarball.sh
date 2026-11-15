#!/bin/bash
# Extract a dataset tarball into its own directory, check the file count against the
# tar listing, and delete the tarball only when tar succeeded and the counts match.
# usage: extract_tarball.sh <tarball> <destdir> <logprefix> [members...]
set -u
tb="$1"; dest="$2"; log="$3"; shift 3
tar -xvzf "$tb" -C "$dest" "$@" > "$log.list" 2> "$log.err"; rc=$?
n_list=$(grep -v '/$' "$log.list" | wc -l)
n_disk=$(find "$dest" -type f ! -name "$(basename "$tb")" | wc -l)
echo "tar_rc=$rc listed_files=$n_list disk_files=$n_disk" | tee "$log.count"
if [ "$rc" = 0 ] && [ "$n_list" = "$n_disk" ] && [ "$n_list" -gt 0 ] && [ "${KEEP_TARBALL:-0}" != 1 ]; then
  rm -f "$tb" && echo "deleted $tb" | tee -a "$log.count"
else
  echo "kept $tb" | tee -a "$log.count"
fi
