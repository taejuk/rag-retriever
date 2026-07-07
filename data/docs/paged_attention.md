# PagedAttention

PagedAttention은 LLM serving에서 KV cache를 block 단위로 관리하는 방법이다.

## KV Cache Problem

LLM inference에서는 이전 token들의 key/value tensor를 저장해야 한다.
요청마다 sequence length가 다르기 때문에 KV cache memory fragmentation이 발생할 수 있다.

## Block Table

Block table은 logical block id를 physical KV cache block id로 매핑한다.
이를 통해 요청마다 연속된 physical memory를 요구하지 않고, block 단위로 KV cache를 관리할 수 있다.

## BlockAllocator

BlockAllocator는 free physical blocks를 관리한다.
새로운 sequence가 들어오면 필요한 만큼 block을 할당하고, sequence가 끝나면 block을 반환한다.
