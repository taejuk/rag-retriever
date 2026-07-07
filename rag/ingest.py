import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple


@dataclass
class Chunk:
    chunk_id: int
    source_path: str
    section: str
    start_line: int
    end_line: int
    text: str


def iter_markdown_files(docs_dir: Path) -> List[Path]:
    return sorted(docs_dir.glob("*.md"))


def split_buffer_by_lines(
    buffer: List[Tuple[int, str]],
    max_lines: int = 60,
    overlap: int = 8,
) -> List[List[Tuple[int, str]]]:
    """
    긴 section을 line 단위로 나눈다.
    처음에는 token 단위보다 line 단위가 구현하기 쉽다.
    """
    if len(buffer) <= max_lines:
        return [buffer]

    chunks = []
    step = max_lines - overlap
    start = 0

    while start < len(buffer):
        end = min(start + max_lines, len(buffer))
        chunks.append(buffer[start:end])

        if end == len(buffer):
            break

        start += step

    return chunks


def parse_markdown_file(path: Path, start_chunk_id: int) -> List[Chunk]:
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

    chunks: List[Chunk] = []
    section_stack: List[str] = []
    buffer: List[Tuple[int, str]] = []

    def flush_buffer():
        nonlocal chunks, buffer

        if not buffer:
            return

        section = " > ".join(section_stack) if section_stack else path.stem

        for sub_buffer in split_buffer_by_lines(buffer):
            text = "\n".join(line for _, line in sub_buffer).strip()
            if not text:
                continue

            chunk = Chunk(
                chunk_id=start_chunk_id + len(chunks),
                source_path=str(path),
                section=section,
                start_line=sub_buffer[0][0],
                end_line=sub_buffer[-1][0],
                text=text,
            )
            chunks.append(chunk)

        buffer = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            match = heading_pattern.match(line)

            if match:
                flush_buffer()

                level = len(match.group(1))
                title = match.group(2).strip()

                # heading level에 맞게 stack 갱신
                section_stack = section_stack[: level - 1]
                section_stack.append(title)

                # heading 자체도 chunk text에 포함
                buffer.append((line_no, line))
            else:
                buffer.append((line_no, line))

    flush_buffer()
    return chunks


def ingest(docs_dir: Path) -> List[Chunk]:
    all_chunks: List[Chunk] = []

    for md_file in iter_markdown_files(docs_dir):
        file_chunks = parse_markdown_file(md_file, start_chunk_id=len(all_chunks))
        all_chunks.extend(file_chunks)

    return all_chunks


def save_jsonl(chunks: List[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=str, default="data/docs")
    parser.add_argument("--out", type=str, default="data/chunks.jsonl")
    args = parser.parse_args()

    docs_dir = Path(args.docs)
    output_path = Path(args.out)

    chunks = ingest(docs_dir)
    save_jsonl(chunks, output_path)

    print(f"Created {len(chunks)} chunks")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()