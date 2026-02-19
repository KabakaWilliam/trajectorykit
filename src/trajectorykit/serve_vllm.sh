CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1 --port 3030 --gpu-memory-utilization 0.6
