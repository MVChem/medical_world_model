import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from mimic_vla_jepa.serious_data import (
    SeriousTransitionDataset,
    batched_query_positions,
    load_joined_serious_records,
)
from mimic_vla_jepa.train_serious import (
    WallClockCosineScheduler,
    _truncate_metric_rows,
    build_input_identity,
    consume_interval,
    load_config,
    next_interval_seconds,
    parse_args,
    patient_disjoint_shuffle_map,
    stable_evaluation_indices,
    truncate_prompt_to_token_budget,
    validate_config,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class SeriousDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.jpg"
        self.target = self.root / "target.jpg"
        self.source.write_bytes(b"source")
        self.target.write_bytes(b"target")
        self.forecast = self.root / "forecast.jsonl"
        self.states = self.root / "states.jsonl"
        self.state_file = self.root / "pair.safetensors"
        self.row = {
            "schema_version": "mimic-current-future-v1",
            "task_type": "forecasting",
            "transition_id": "safe_transition_1",
            "patient_id": "patient_1",
            "split": "train",
            "source_image": str(self.source),
            "target_image": str(self.target),
            "elapsed_hours": 24.0,
            "source_report": {
                "findings": "Current lungs are clear.",
                "impression": None,
                "unsectioned_report": None,
            },
            "prompt": "UNTRUSTED FUTURE_SECRET_PROMPT",
            "future_report_for_eval_only": {"findings": "FUTURE_SECRET_EDEMA"},
        }
        write_jsonl(self.forecast, [self.row])
        save_file(
            {
                "source_state": torch.zeros(4, 8),
                "target_state": torch.ones(4, 8),
            },
            self.state_file,
        )
        write_jsonl(
            self.states,
            [
                {
                    "transition_id": "safe_transition_1",
                    "split": "train",
                    "state_file": self.state_file.name,
                }
            ],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_join_rebuilds_current_only_prompt(self) -> None:
        records = load_joined_serious_records(self.forecast, self.states, split="train")
        self.assertEqual(len(records), 1)
        self.assertIn("CURRENT FINDINGS: Current lungs are clear.", records[0].prompt)
        self.assertIn("0-24h", records[0].prompt)
        self.assertNotIn("UNTRUSTED", records[0].prompt)
        self.assertNotIn("FUTURE_SECRET", records[0].prompt)
        self.assertFalse(hasattr(records[0], "target_image"))

    def test_dataset_accepts_exact_state_only_contract(self) -> None:
        dataset = SeriousTransitionDataset(self.forecast, self.states, split="train")
        item = dataset[0]
        self.assertEqual(set(item["source_state"].shape), {8, 4})
        self.assertTrue(torch.equal(item["target_state"], torch.ones(4, 8)))
        self.assertNotIn("target_image", item)

    def test_dataset_rejects_cached_query_state(self) -> None:
        save_file(
            {
                "source_state": torch.zeros(4, 8),
                "target_state": torch.ones(4, 8),
                "query_state": torch.zeros(24, 16),
            },
            self.state_file,
        )
        dataset = SeriousTransitionDataset(self.forecast, self.states, split="train")
        with self.assertRaisesRegex(ValueError, "state-only"):
            _ = dataset[0]

    def test_legacy_feature_file_key_remains_readable(self) -> None:
        write_jsonl(
            self.states,
            [
                {
                    "transition_id": "safe_transition_1",
                    "split": "train",
                    "feature_file": self.state_file.name,
                }
            ],
        )
        dataset = SeriousTransitionDataset(self.forecast, self.states, split="train")
        self.assertEqual(dataset[0]["transition_id"], "safe_transition_1")

    def test_conflicting_state_file_keys_are_rejected(self) -> None:
        write_jsonl(
            self.states,
            [
                {
                    "transition_id": "safe_transition_1",
                    "split": "train",
                    "state_file": self.state_file.name,
                    "feature_file": "different.safetensors",
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "disagree"):
            load_joined_serious_records(self.forecast, self.states, split="train")

    def test_join_rejects_missing_state(self) -> None:
        write_jsonl(
            self.states,
            [
                {
                    "transition_id": "different_transition",
                    "split": "train",
                    "state_file": self.state_file.name,
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            load_joined_serious_records(self.forecast, self.states, split="train")

    def test_batched_query_positions(self) -> None:
        ids = torch.tensor([[9, 7, 7, 0], [7, 1, 7, 0]])
        positions = batched_query_positions(ids, token_id=7, expected_per_example=2)
        self.assertTrue(torch.equal(positions, torch.tensor([[1, 2], [0, 2]])))
        with self.assertRaisesRegex(ValueError, "example 0"):
            batched_query_positions(ids, token_id=7, expected_per_example=3)


class SeriousConfigTest(unittest.TestCase):
    def test_released_config_has_requested_schedule(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "serious_4gpu.yaml"
        config = load_config(path)
        self.assertEqual(config.train.expected_world_size, 4)
        self.assertEqual(config.train.max_duration_hours, 24.0)
        self.assertEqual(config.train.checkpoint_interval_hours, 8.0)
        self.assertEqual(config.train.eval_interval_hours, 8.0)
        self.assertEqual(config.qwen.query_tokens, 24)
        self.assertIn("in_proj_qkv", config.lora.target_modules)
        self.assertIn("q_proj", config.lora.target_modules)

    def test_full_size_four_gpu_smoke_config(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "serious_smoke_4gpu.yaml"
        config = load_config(path)
        self.assertEqual(config.predictor.depth, 12)
        self.assertEqual(config.predictor.predictor_dim, 1024)
        self.assertEqual(config.train.expected_world_size, 4)
        self.assertEqual(config.train.max_steps, 1)
        self.assertEqual(config.train.gradient_accumulation_steps, 1)
        self.assertTrue(config.train.eval_at_start)
        self.assertEqual(config.train.max_eval_examples, 4)

    def test_train_and_eval_manifests_are_independent_required_inputs(self) -> None:
        common = [
            "train_serious.py",
            "--forecast-manifest",
            "train.jsonl",
            "--state-features",
            "train_states.jsonl",
            "--qwen-model",
            "qwen",
            "--config",
            "config.yaml",
            "--run-dir",
            "run",
        ]
        with patch("sys.argv", common), self.assertRaises(SystemExit):
            parse_args()
        complete = [
            *common,
            "--eval-forecast-manifest",
            "validate.jsonl",
            "--eval-state-features",
            "validate_states.jsonl",
        ]
        with patch("sys.argv", complete):
            args = parse_args()
        self.assertEqual(args.forecast_manifest, Path("train.jsonl"))
        self.assertEqual(args.eval_forecast_manifest, Path("validate.jsonl"))
        self.assertEqual(args.state_features, Path("train_states.jsonl"))
        self.assertEqual(args.eval_state_features, Path("validate_states.jsonl"))

    def test_duration_above_24_hours_is_rejected(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "serious_4gpu.yaml"
        config = load_config(path)
        invalid = replace(
            config,
            train=replace(config.train, max_duration_hours=24.1),
        )
        with self.assertRaisesRegex(ValueError, r"\(0, 24\]"):
            validate_config(invalid)

    def test_interval_helpers_consume_crossed_boundaries(self) -> None:
        due = next_interval_seconds(0.0, 8.0)
        self.assertEqual(due, 8.0 * 3600.0)
        is_due, following = consume_interval(17.0 * 3600.0, due, 8.0)
        self.assertTrue(is_due)
        self.assertEqual(following, 24.0 * 3600.0)

    def test_wall_clock_scheduler_warmup_cosine_and_resume(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1.0}])
        scheduler = WallClockCosineScheduler(
            optimizer,
            max_duration_hours=1.0,
            warmup_fraction=0.1,
            min_lr_ratio=0.2,
        )
        scheduler.step(180.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.5)
        scheduler.step(360.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0)
        scheduler.step(3600.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.2)
        saved = scheduler.state_dict()

        resumed_optimizer = torch.optim.AdamW(
            [{"params": [torch.nn.Parameter(torch.tensor(2.0))], "lr": 1.0}]
        )
        resumed = WallClockCosineScheduler(
            resumed_optimizer,
            max_duration_hours=1.0,
            warmup_fraction=0.1,
            min_lr_ratio=0.2,
        )
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.last_elapsed_seconds, 3600.0)
        self.assertAlmostEqual(resumed_optimizer.param_groups[0]["lr"], 0.2)

    def test_resume_truncates_future_metrics(self) -> None:
        rows = [
            {"event": "train", "step": 1, "elapsed_seconds": 10.0},
            {"event": "evaluation", "step": 2, "elapsed_seconds": 20.0},
            {"event": "train", "step": 3, "elapsed_seconds": 30.0},
            {"event": "malformed"},
        ]
        self.assertEqual(
            _truncate_metric_rows(rows, checkpoint_step=2, checkpoint_elapsed=20.0),
            rows[:2],
        )

    def test_capped_eval_subset_does_not_depend_on_manifest_order(self) -> None:
        ids = ["transition_c", "transition_a", "transition_d", "transition_b"]

        def selected(record_ids: list[str]) -> list[str]:
            dataset = SimpleNamespace(
                records=[SimpleNamespace(transition_id=value) for value in record_ids]
            )
            return [
                record_ids[index]
                for index in stable_evaluation_indices(dataset, max_examples=2)
            ]

        self.assertEqual(selected(ids), selected(list(reversed(ids))))

    def test_patient_disjoint_shuffle_is_a_bijection(self) -> None:
        patients = ["a", "a", "b", "b", "c", "d"]
        dataset = SimpleNamespace(
            records=[
                SimpleNamespace(patient_id=patient, transition_id=f"t{index}")
                for index, patient in enumerate(patients)
            ]
        )
        indices = list(range(len(patients)))
        mapping = patient_disjoint_shuffle_map(dataset, indices)
        self.assertEqual(set(mapping), set(indices))
        self.assertEqual(set(mapping.values()), set(indices))
        for receiver, donor in mapping.items():
            self.assertNotEqual(patients[receiver], patients[donor])

    def test_prompt_only_truncation_preserves_prefix_and_sanitizes_marker(self) -> None:
        class WhitespaceTokenizer:
            def __init__(self) -> None:
                self.tokens: list[str] = []

            def encode(self, text: str, add_special_tokens: bool) -> list[int]:
                self.tokens = text.split()
                return list(range(len(self.tokens)))

            def decode(self, ids: list[int], **_: object) -> str:
                return " ".join(self.tokens[index] for index in ids)

        tokenizer = WhitespaceTokenizer()
        clipped = truncate_prompt_to_token_budget(
            tokenizer,
            "HORIZON current report <|latent_0|> tail words",
            max_tokens=5,
            forbidden_special_token="<|latent_0|>",
        )
        self.assertTrue(clipped.startswith("HORIZON current report"))
        self.assertNotIn("<|latent_0|>", clipped)
        self.assertLessEqual(len(clipped.split()), 5)

    def test_input_identity_hashes_qwen_weights_and_state_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_raw = root / "train.jsonl"
            eval_raw = root / "eval.jsonl"
            train_raw.write_text("{}\n", encoding="utf-8")
            eval_raw.write_text("{}\n", encoding="utf-8")

            def state_cache(name: str, source: Path) -> Path:
                directory = root / name
                directory.mkdir()
                manifest = directory / "states.jsonl"
                manifest.write_text("{}\n", encoding="utf-8")
                (directory / "metadata.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "mimic-vla-jepa-states-v1",
                            "source_manifest": str(source.resolve()),
                            "records": 1,
                            "backbone": {"state_dim": 2048},
                        }
                    ),
                    encoding="utf-8",
                )
                return manifest

            train_states = state_cache("train_states", train_raw)
            eval_states = state_cache("eval_states", eval_raw)
            qwen = root / "qwen"
            qwen.mkdir()
            for name in (
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "preprocessor_config.json",
            ):
                (qwen / name).write_text("{}\n", encoding="utf-8")
            (qwen / "model.safetensors").write_bytes(b"fake weights")
            identity = build_input_identity(
                train_forecast_manifest=train_raw,
                train_state_manifest=train_states,
                eval_forecast_manifest=eval_raw,
                eval_state_manifest=eval_states,
                qwen_model=qwen,
            )
            self.assertEqual(identity["qwen"]["weight_files"][0]["bytes"], 12)
            self.assertIn("metadata_sha256", identity["train_state_cache"])
            self.assertIn("state_tensor_integrity_assumption", identity)


if __name__ == "__main__":
    unittest.main()
