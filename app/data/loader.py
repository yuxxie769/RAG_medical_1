from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List


# 流式读取

def iter_raw_records(path: str, batch_id: str | None = None) -> Iterator[Dict[str, Any]]:
    """Stream raw jsonl records from a file.

    Each yielded item contains source metadata and the original record.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    resolved_batch_id = batch_id or file_path.stem

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object on line {line_no}, got {type(record).__name__}")

            yield {
                "raw_batch_id": resolved_batch_id,
                "source_path": str(file_path),
                "source_line_no": line_no,
                "record": record,
            }


# 批量读取

def load_raw_records(path: str, batch_id: str | None = None) -> List[Dict[str, Any]]:
    """Load all raw jsonl records into memory.

    This is a convenience wrapper around `iter_raw_records`.
    """
    return list(iter_raw_records(path, batch_id=batch_id))
