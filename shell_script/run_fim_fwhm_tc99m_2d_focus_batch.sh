#!/bin/bash
set -euo pipefail


# CUDA 12.9
# nvcc -O3 -use_fast_math --extended-lambda -arch=sm_90 -o fim_tc99m_2D_focus fim_profile_fwhm_Tc99m_100mmFOV_focus40d5_2D.cu -lcurand -lcublas


# CUDA 12.4
# nvcc -O3 -use_fast_math --extended-lambda -arch=sm_90 \
#  -I"$CONDA_PREFIX/include" \
#  -I"$CONDA_PREFIX/targets/x86_64-linux/include" \
#  -L"$CONDA_PREFIX/lib" \
#  -L"$CONDA_PREFIX/targets/x86_64-linux/lib" \
#  -o fim_tc99m_2D_focus_100mmFOV \
#  cuda_script/fim_profile_fwhm_Tc99m_2D.cu \
#  -lcurand -lcublas

CUBE_DIR="crystal_geometry_agent_design"
MAP_DIR="crystal_geometry_agent_design"
OUT_DIR="/datastore01/user-storage/huitian/SCI_SPECT/fim_profile_fwhm"
ALL_FWHM_BIN="fim_fwhm/fim_fwhm_agent.bin"

N_DEVICES=4 

mkdir -p "$OUT_DIR"

start_ts=$(date +%s)

declare -a cubes=()
declare -a tags=()
declare -a maps=()

mapfile -t cubes < <(ls "$CUBE_DIR"/cube_pos_*.txt 2>/dev/null | sort || true)

for cube in "${cubes[@]}"; do
    base=$(basename "$cube")          # cube_pos_<TAG>.txt
    tag=${base#cube_pos_}             # <TAG>.txt
    tag=${tag%.txt}                   # <TAG>
    map="${MAP_DIR}/map_${tag}.txt"

    tags+=("$tag")
    maps+=("$map")
done

N=${#tags[@]}
echo "Found $N geometry designs to process."
echo "Using $N_DEVICES GPUs (0..$((N_DEVICES-1)))."

# Truncate/create the aggregated binary file
: > "$ALL_FWHM_BIN"

run_worker() {
    local worker_id=$1
    local dev=$worker_id
    local i tag cube map fwhm_tmp

    for (( i=0; i<N; ++i )); do
        # Distribute jobs round-robin over GPUs
        if (( i % N_DEVICES != worker_id )); then
            continue
        fi

        tag=${tags[$i]}
        map=${maps[$i]}
        cube="${CUBE_DIR}/cube_pos_${tag}.txt"
        fwhm_tmp="${OUT_DIR}/fwhm_${tag}.bin"

        echo "[GPU $dev] Processing tag '$tag'..."
        ./executables/fim_tc99m_2D_focus_120mmFOV "$cube" "$map" "$fwhm_tmp" "$dev"
    done
}

for (( d=0; d<N_DEVICES; ++d )); do
    run_worker "$d" &
done

    wait
    echo "All GPU workers finished. Concatenating FWHM outputs in canonical order..."

    for ((i=0; i<N; ++i)); do
        tag=${tags[$i]}
        fwhm_tmp="${OUT_DIR}/fwhm_${tag}.bin"

        if [ ! -f "$fwhm_tmp" ]; then
            echo "WARNING: missing fwhm file for tag '$tag' -> $fwhm_tmp" >&2
            continue
        fi

        cat "$fwhm_tmp" >> "$ALL_FWHM_BIN"
    done

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    printf "Total runtime: %ds\n" "$elapsed"

    echo "Done. Aggregated avg_fim_fwhm values in: $ALL_FWHM_BIN"
