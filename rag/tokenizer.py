import re
from typing import List


STOPWORDS = {
    "은", "는", "이", "가", "을", "를", "에", "에서", "으로", "로",
    "왜", "어떻게", "무엇", "뭐야", "한다", "있다", "된다", "위해",
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "for",
}


def normalize(text: str) -> str:
    return text.lower().replace("\u200b", " ")


def split_camel_case(token: str) -> List[str]:
    """
    PagedAttention -> Paged Attention
    cudaHostAlloc -> cuda Host Alloc
    KVCacheManager -> KV Cache Manager
    """
    parts = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        token,
    )
    parts = re.sub(
        r"([A-Z]+)([A-Z][a-z])",
        r"\1 \2",
        parts,
    )
    return parts.split()


def expand_identifier(token: str) -> List[str]:
    """
    기술 identifier를 검색에 유리하게 확장한다.

    Examples:
    - PagedAttention -> pagedattention, paged, attention
    - BlockAllocator -> blockallocator, block, allocator
    - mma.sync -> mma.sync, mma, sync
    - cuda_host_alloc -> cuda_host_alloc, cuda, host, alloc
    """
    expanded = []

    raw = token.strip()
    lower = raw.lower()

    if not lower:
        return []

    expanded.append(lower)

    # dotted identifiers: mma.sync
    if "." in raw:
        for part in raw.split("."):
            if part:
                expanded.append(part.lower())

    # snake_case identifiers
    if "_" in raw:
        for part in raw.split("_"):
            if part:
                expanded.append(part.lower())

    # hyphenated identifiers
    if "-" in raw:
        for part in raw.split("-"):
            if part:
                expanded.append(part.lower())

    # CamelCase identifiers
    camel_parts = split_camel_case(raw)
    if len(camel_parts) > 1:
        for part in camel_parts:
            expanded.append(part.lower())

    # remove duplicates while preserving order
    result = []
    seen = set()

    for item in expanded:
        if item in STOPWORDS:
            continue
        if len(item) <= 1:
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def tokenize(text: str) -> List[str]:
    """
    간단한 technical tokenizer.

    - 영어 identifier: PagedAttention, BlockAllocator, cudaHostAlloc
    - dotted identifier: mma.sync
    - snake_case: block_allocator
    - 숫자
    - 한글 연속 문자열
    """
    # normalize 전에 원래 대소문자가 필요하다.
    pattern = r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*|[0-9]+|[가-힣]+"
    raw_tokens = re.findall(pattern, text)

    result = []

    for tok in raw_tokens:
        # 한글 token
        if re.fullmatch(r"[가-힣]+", tok):
            norm = normalize(tok)
            if norm not in STOPWORDS and len(norm) > 1:
                result.append(norm)
            continue

        # 숫자 token
        if tok.isdigit():
            result.append(tok)
            continue

        # 영어/identifier token
        result.extend(expand_identifier(tok))

    return result