#!/usr/bin/env bash
# inspect_kernel.sh - resource usage + ISA dump for one kernel source file.
#
#   inspect_kernel.sh cuda <file.cu>  <sm_XY[a|f]>  [extra nvcc flags...]
#   inspect_kernel.sh hip  <file.hip> <gfxNNNN>     [extra hipcc flags...]
#
# Output goes to ./inspect_out/. Prints the key numbers (registers, spills, shared memory / LDS,
# occupancy hints) and leaves the full SASS / GCN assembly on disk for reading.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  sed -n '2,9p' "$0"
  exit 1
fi

backend="$1"; src="$2"; arch="$3"; shift 3
out="inspect_out"
mkdir -p "$out"
base="$(basename "${src%.*}")_${arch}"

case "$backend" in
  cuda)
    command -v nvcc >/dev/null || { echo "nvcc not found in PATH" >&2; exit 2; }
    sm="${arch#sm_}"                 # e.g. 90a
    cubin="$out/${base}.cubin"
    echo "== nvcc ${arch}: compile + ptxas resource usage =="
    nvcc -std=c++20 -O3 --expt-relaxed-constexpr -lineinfo \
         -gencode "arch=compute_${sm},code=sm_${sm}" \
         -Xptxas=-v -cubin "$src" -o "$cubin" "$@" 2>&1 | tee "$out/${base}.ptxas.txt"
    echo
    echo "== cuobjdump resource usage =="
    cuobjdump -res-usage "$cubin" | tee "$out/${base}.resusage.txt"
    echo
    echo "== SASS written to $out/${base}.sass =="
    cuobjdump -sass "$cubin" > "$out/${base}.sass"
    echo "Quick instruction census (hot-loop sanity):"
    for pat in 'LDG\.E\.128' 'LDGSTS' 'UTMALDG' 'HGMMA' 'UTCHMMA|UTCQMMA|UTCOMMA' 'HMMA|IMMA' \
               'LDSM' 'STL|LDL' 'BAR\.SYNC' 'MUFU'; do
      printf '  %-28s %s\n' "$pat" "$(grep -Ec "$pat" "$out/${base}.sass" || true)"
    done
    echo "(STL/LDL > 0 means local-memory traffic: check spills in the ptxas output above.)"
    ;;
  hip)
    command -v hipcc >/dev/null || { echo "hipcc not found in PATH" >&2; exit 2; }
    asm="$out/${base}.s"
    echo "== hipcc ${arch}: kernel-resource-usage remarks =="
    hipcc -std=c++23 -O3 --offload-arch="$arch" -Rpass-analysis=kernel-resource-usage \
          --offload-device-only -c "$src" -o "$out/${base}.o" "$@" 2>&1 | tee "$out/${base}.resource.txt"
    echo
    echo "== ISA written to $asm =="
    hipcc -std=c++23 -O3 --offload-arch="$arch" --offload-device-only -S "$src" -o "$asm" "$@"
    echo "Quick instruction census (hot-loop sanity):"
    for pat in 'global_load_dwordx4|buffer_load_dwordx4' 'global_load_lds|buffer_load.*lds' \
               'v_mfma' 'v_wmma|v_swmmac' 'ds_read_b128' 'ds_write_b128' 'ds_read_b64_tr' \
               'scratch_' 's_waitcnt vmcnt\(0\)' 's_barrier'; do
      printf '  %-34s %s\n' "$pat" "$(grep -Ec "$pat" "$asm" || true)"
    done
    echo "(scratch_* > 0 means spills; check .vgpr_count/.agpr_count/.private_segment_fixed_size.)"
    ;;
  *)
    echo "backend must be 'cuda' or 'hip'" >&2
    exit 1
    ;;
esac
