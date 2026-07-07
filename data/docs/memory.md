# Memory Management

시스템 프로그래밍에서는 memory allocation과 address translation이 중요하다.

## Page Table

Page table은 virtual address를 physical address로 변환하는 자료구조다.
운영체제는 page table을 사용해 process마다 독립적인 address space를 제공한다.

## Memory Fragmentation

Memory fragmentation은 free memory가 여러 조각으로 나뉘어 큰 연속 메모리를 할당하기 어려운 현상이다.

## Allocator

Allocator는 heap memory를 관리한다.
malloc과 free는 동적 메모리 할당과 해제를 수행한다.
