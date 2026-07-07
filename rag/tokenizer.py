import re
from typing import List


STOPWORDS = {
    "은", "는", "이", "가", "을", "를", "에", "에서", "으로", "로",
    "왜", "어떻게", "무엇", "뭐야", "한다", "있다", "된다", "위해",
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "for",
}


def normalize(text: str) -> str:
    return text.lower().replace("\u200b", " ")


def tokenize(text: str) -> List[str]:
    """
    간단한 tokenizer.
    - 영어 identifier: PagedAttention, BlockAllocator, cudaHostAlloc 등
    - 숫자
    - 한글 연속 문자열
    을 token으로 뽑는다.
    """
    text = normalize(text)

    pattern = r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[가-힣]+"
    tokens = re.findall(pattern, text)

    result = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        if len(tok) <= 1:
            continue
        result.append(tok)

    return result