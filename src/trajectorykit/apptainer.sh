# ── Storage: redirect Apptainer scratch to /VData to avoid filling / ──
# The root filesystem (/) is nearly full. Apptainer's --writable-tmpfs
# overlay, image cache, and temp files all default to /tmp and ~/.apptainer
# on /. Moving them to /VData (NFS, 1.3T+ free) prevents ENOSPC errors.
export APPTAINER_TMPDIR=/VData/linna4335/apptainer_tmp
export APPTAINER_CACHEDIR=/VData/linna4335/apptainer_cache
SANDBOX_SCRATCH=/VData/linna4335/sandbox_scratch
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$SANDBOX_SCRATCH"

# Pull sandbox-fusion from Docker Tags (only needed once)
# apptainer pull sandbox-fusion_server-20250609.sif \
#   docker://volcengine/sandbox-fusion:server-20250609

# Run the sandbox-fusion container
# --bind mounts a writable scratch dir from /VData into /tmp inside the
# container, replacing the tiny 64MB fuse-overlayfs that fills up during
# code execution. This prevents [Errno 28] No space left on device errors.
apptainer run --containall --cleanenv --writable-tmpfs \
  --bind "${SANDBOX_SCRATCH}:/tmp" \
  src/trajectorykit/sandbox-fusion_server-20250609.sif

#in another container
# curl 'http://localhost:8080/run_code' \
#   -H 'Content-Type: application/json' \
#   --data-raw '{"code": "console.log(1+1)", "language": "typescript"}'
