# CUDA Kernel

mini-llm은 GPU kernel을 사용해 matrix multiplication과 attention 연산을 가속한다.

## Matrix Multiplication

GEMM은 LLM inference에서 가장 중요한 연산 중 하나다.
naive kernel은 global memory access가 비효율적이기 때문에 성능이 낮다.

## Coalescing

Coalesced memory access는 warp의 thread들이 연속된 memory address에 접근하도록 만드는 최적화다.
이를 통해 global memory bandwidth utilization을 높일 수 있다.

## Tensor Core

Tensor Core는 mixed precision matrix multiplication을 빠르게 수행하는 하드웨어 유닛이다.
WMMA API나 mma.sync instruction을 통해 사용할 수 있다.
