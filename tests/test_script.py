from pathlib import Path
import sys

# 项目根目录加入模块搜索路径，便于直接运行本脚本
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data import load_raw_records, clean_records

DATA_PATH = Path(__file__).resolve().parent / "sample_data.jsonl"

raw = load_raw_records(str(DATA_PATH))
cleaned = clean_records(raw)

print("原始条数:", len(raw))
for item in raw[:5]:
    print(item)
print("清洗后条数:", len(cleaned))

for item in cleaned[:5]:
    print(item)
