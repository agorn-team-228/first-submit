import csv
from pathlib import Path


def validate(questions_path: str = "data/demo_questions.csv", sub_path: str = "output/submission.csv") -> None:
    q_file, s_file = Path(questions_path), Path(sub_path)
    assert s_file.is_file(), f"Файл {s_file} не найден!"

    with q_file.open(encoding="utf-8-sig", newline="") as f:
        expected_ids = [row["id"].strip() for row in csv.DictReader(f)]

    with s_file.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        assert header == ["id", "answer"], f"Неверный заголовок: {header} (нужно ['id', 'answer'])"
        rows = list(reader)

    assert len(rows) == len(expected_ids), f"Несовпадение числа строк: {len(rows)} != {len(expected_ids)}"

    seen_ids = set()
    for line_num, row in enumerate(rows, start=2):
        assert len(row) == 2, f"Строка {line_num}: должно быть ровно 2 колонки, получено {len(row)}"
        qid, ans = row[0], row[1]
        assert qid not in seen_ids, f"Строка {line_num}: дубликат ID '{qid}'"
        seen_ids.add(qid)
        assert ans.strip(), f"Строка {line_num} (ID {qid}): пустой ответ"
        assert len(ans) <= 10000, f"Строка {line_num} (ID {qid}): превышен лимит 10 000 символов ({len(ans)})"

    assert [r[0] for r in rows] == expected_ids, "Порядок или состав ID не совпадает с questions.csv!"
    print(f"УСПЕХ! Файл {sub_path} полностью соответствует регламенту ({len(rows)} ответов).")


if __name__ == "__main__":
    validate()
