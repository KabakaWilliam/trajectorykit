# ── Storage: redirect Apptainer scratch to /VData to avoid filling / ──
# The root filesystem (/) is nearly full. Apptainer's --writable-tmpfs
# overlay, image cache, and temp files all default to /tmp and ~/.apptainer
# on /. Moving them to /VData (NFS, 1.3T+ free) prevents ENOSPC errors.
export APPTAINER_TMPDIR=/VData/linna4335/apptainer_tmp
export APPTAINER_CACHEDIR=/VData/linna4335/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# Pull sandbox-fusion from Docker Tags
apptainer pull sandbox-fusion_server-20250609.sif \
  docker://volcengine/sandbox-fusion:server-20250609

# Run the Docker Container
apptainer run --containall --cleanenv --writable-tmpfs \
  sandbox-fusion_server-20250609.sif

#in another container
# curl 'http://localhost:8080/run_code' \
#   -H 'Content-Type: application/json' \
#   --data-raw '{"code": "console.log(1+1)", "language": "typescript"}'
