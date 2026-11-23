import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jsonl_split_id import load_jsonl, split_records
from jsonl_split_id.core import write_splits


class SplitRecordsTests(unittest.TestCase):
    def test_unstratified_split_is_reproducible_and_has_exact_total(self):
        records = [{"id": index} for index in range(10)]
        first = split_records(records, ratios=("0.8", "0.1", "0.1"), seed=17)
        second = split_records(records, ratios=("0.8", "0.1", "0.1"), seed=17)
        self.assertEqual(first, second)
        self.assertEqual({name: len(rows) for name, rows in first.items()}, {"train": 8, "validation": 1, "test": 1})
        ids = [row["id"] for rows in first.values() for row in rows]
        self.assertCountEqual(ids, range(10))

    def test_stratified_split_keeps_per_class_distribution_as_close_as_integer_counts_allow(self):
        records = ([{"id": f"a{i}", "label": "a"} for i in range(6)]
                   + [{"id": f"b{i}", "label": "b"} for i in range(4)])
        splits = split_records(records, ratios=("0.5", "0.25", "0.25"), seed=9, stratify_by="label")
        sizes = {name: len(rows) for name, rows in splits.items()}
        self.assertEqual(sizes, {"train": 5, "validation": 3, "test": 2})
        for label in ("a", "b"):
            counts = [sum(row["label"] == label for row in splits[name]) for name in ("train", "validation", "test")]
            self.assertEqual(counts, [3, 2, 1] if label == "a" else [2, 1, 1])

    def test_deterministic_allocation_ties_follow_split_order(self):
        splits = split_records([{"x": i} for i in range(2)], ratios=("0.5", "0.25", "0.25"), seed=1)
        self.assertEqual([len(splits[name]) for name in ("train", "validation", "test")], [1, 1, 0])

    def test_invalid_ratios_records_or_stratification_are_rejected(self):
        for ratios in (("0.7", "0.2", "0.2"), ("-0.1", "0.6", "0.5"), ("NaN", "0", "1")):
            with self.subTest(ratios=ratios):
                with self.assertRaises(ValueError):
                    split_records([{"x": 1}], ratios=ratios)
        for records in ([], [1], [{"label": []}]):
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    split_records(records, stratify_by="label" if records and isinstance(records[0], dict) else None)
        with self.assertRaises(ValueError):
            split_records([{"x": 1}], stratify_by="label")

    def test_load_jsonl_ignores_blank_lines_and_does_not_echo_bad_rows(self):
        self.assertEqual(load_jsonl('\n{"x":1}\n\n{"x":2}\n'), [{"x": 1}, {"x": 2}])
        private_value = '{"private":"do-not-repeat-591" broken'
        with self.assertRaisesRegex(ValueError, "baris 1") as error:
            load_jsonl(private_value)
        self.assertNotIn("do-not-repeat-591", str(error.exception))


class CliTests(unittest.TestCase):
    def test_cli_writes_jsonl_outputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "splits"
            dataset = "\n".join(json.dumps({"id": i}) for i in range(10))
            command = [sys.executable, "-m", "jsonl_split_id", "--output-dir", str(output), "--seed", "4"]
            first = subprocess.run(command, input=dataset, text=True, capture_output=True, check=True)
            summary = json.loads(first.stdout)
            self.assertEqual(summary, {"total": 10, "splits": {"train": 8, "validation": 1, "test": 1}})
            self.assertEqual(sum(len(path.read_text().splitlines()) for path in output.glob("*.jsonl")), 10)

            second = subprocess.run(command, input=dataset, text=True, capture_output=True)
            self.assertEqual(second.returncode, 2)
            self.assertIn("--force", second.stderr)
            forced = subprocess.run(command + ["--force"], input=dataset, text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(forced.stdout)["total"], 10)

    def test_cli_stratification_and_invalid_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stratified"
            dataset = "\n".join(json.dumps({"id": i, "label": i % 2}) for i in range(8))
            result = subprocess.run(
                [sys.executable, "-m", "jsonl_split_id", "--output-dir", str(output), "--stratify-by", "label"],
                input=dataset,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(json.loads(result.stdout)["total"], 8)

            invalid = subprocess.run(
                [sys.executable, "-m", "jsonl_split_id", "--output-dir", str(Path(directory) / "bad")],
                input='{"secret":"not-valid-443" broken}',
                text=True,
                capture_output=True,
            )
            self.assertEqual(invalid.returncode, 2)
            self.assertEqual(invalid.stdout, "")
            self.assertNotIn("not-valid-443", invalid.stderr)

    def test_write_refuses_existing_output_files(self):
        with tempfile.TemporaryDirectory() as directory:
            splits = {"train": [{"x": 1}], "validation": [], "test": []}
            write_splits(splits, directory)
            before = (Path(directory) / "train.jsonl").read_text()
            with self.assertRaises(FileExistsError):
                write_splits({"train": [{"x": 2}], "validation": [], "test": []}, directory)
            self.assertEqual((Path(directory) / "train.jsonl").read_text(), before)


if __name__ == "__main__":
    unittest.main()
