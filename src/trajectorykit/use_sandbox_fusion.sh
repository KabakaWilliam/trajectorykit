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
