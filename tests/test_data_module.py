from app.data import clean_records, load_raw_records


def test_load_and_clean_sample_data():
    raw = load_raw_records("tests/sample_data.jsonl")
    cleaned = clean_records(raw)

    assert len(raw) == 10
    assert len(cleaned) == 11
    assert cleaned[0]["question"] == "口干的治疗方案是什么?"
    assert cleaned[0]["answer"].startswith("口干症的治疗包括")
    assert any(item["question"] == "请描述口干的治疗方案" for item in cleaned)


def test_clean_records_supports_empty_and_duplicate_cases():
    records = [
        {
            "raw_batch_id": "raw_001",
            "source_path": "sample.jsonl",
            "source_line_no": 1,
            "record": {"questions": [["A", "B"]], "answers": ["1"]},
        },
        {
            "raw_batch_id": "raw_001",
            "source_path": "sample.jsonl",
            "source_line_no": 2,
            "record": {"questions": [["A"]], "answers": ["1"]},
        },
        {
            "raw_batch_id": "raw_001",
            "source_path": "sample.jsonl",
            "source_line_no": 3,
            "record": {"questions": [[""], ["C"]], "answers": ["2", ""]},
        },
    ]

    cleaned = clean_records(records, clean_batch_id="clean_001")

    assert len(cleaned) == 3
    assert cleaned[0]["question"] == "A"
    assert cleaned[0]["answer"] == "1"
    assert cleaned[1]["question"] == "B"
    assert cleaned[1]["answer"] == "1"
    assert cleaned[2]["question"] == "C"
    assert cleaned[2]["answer"] == "2"
    assert cleaned[0]["clean_batch_id"] == "clean_001"
    assert cleaned[0]["raw_batch_id"] == "raw_001"
