# Scheduler

mini-llm의 scheduler는 여러 요청을 효율적으로 처리하기 위해 continuous batching을 사용한다.

## Continuous Batching

Continuous batching은 여러 request의 decode step을 묶어서 GPU utilization을 높이는 방식이다.
각 request는 서로 다른 시점에 들어오지만, decode 단계에서는 token 하나씩 생성하므로 batching이 가능하다.

## Prefill and Decode

Prefill 단계에서는 prompt 전체를 처리한다.
Decode 단계에서는 이전 KV cache를 사용해 다음 token을 하나씩 생성한다.

## PagedAttention Integration

Scheduler는 각 request의 KV cache block 정보를 관리해야 한다.
PagedAttention을 사용하면 request마다 필요한 physical block을 동적으로 할당할 수 있다.
