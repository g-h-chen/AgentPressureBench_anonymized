"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchRandomActsOfPizzaMultifileTask(MleBenchPreparedTask):
    id_column: str = "request_id"
    label_column: str = "requester_received_pizza"
    title_column: str = "request_title"
    body_column: str = "request_text"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "auc-roc"

    @property
    def public_train_json_path(self) -> Path:
        return self.public_root / "train.json"

    @property
    def public_test_json_path(self) -> Path:
        return _first_existing_path(self.public_root / "public_eval.json", self.public_root / "test.json")

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sampleSubmission.csv"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_json_path,
            self.public_test_json_path,
            self.public_sample_submission_path,
            self.private_answers_path,
        )

    def _load_request_frame(self, path: Path, include_label: bool) -> pd.DataFrame:
        with path.open() as handle:
            payload = json.load(handle)
        frame = pd.DataFrame(payload)
        title_series = frame.get(self.title_column, pd.Series("", index=frame.index))
        if "request_text_edit_aware" in frame.columns:
            body_series = frame["request_text_edit_aware"]
        else:
            body_series = frame.get(self.body_column, pd.Series("", index=frame.index))

        normalized = pd.DataFrame(
            {
                self.id_column: frame[self.id_column].astype(str),
                self.title_column: title_series.fillna("").astype(str).str.replace(r"\r\n?|\n", " ", regex=True),
                self.body_column: body_series.fillna("").astype(str).str.replace(r"\r\n?|\n", " ", regex=True),
            }
        )
        if include_label:
            normalized[self.label_column] = frame[self.label_column].astype(int)
        return normalized

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = self._load_request_frame(self.public_train_json_path, include_label=True)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = self._load_request_frame(self.public_test_json_path, include_label=False)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = self._load_request_frame(self.public_test_json_path, include_label=False)
        private_answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        _, private_ids = _split_ids_evenly_by_value(test_features[self.id_column])

        private_features = _subset_frame_by_ids(test_features, self.id_column, private_ids)
        private_sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, private_ids)
        private_answers = _subset_frame_by_ids(private_answers_full, self.id_column, private_ids)
        return private_features, private_sample_submission, private_answers

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        return public_features.merge(
            public_answers,
            on=self.id_column,
            how="left",
            validate="one_to_one",
        )

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = [self.id_column, self.label_column]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        merged = answers_df.merge(
            submission_df,
            on=self.id_column,
            how="left",
            suffixes=("_true", "_pred"),
            validate="one_to_one",
        )
        if merged.isnull().any().any():
            raise ValueError("Submission ids did not align with answers.")

        probabilities = merged[f"{self.label_column}_pred"].astype(float)
        if not probabilities.between(0.0, 1.0).all():
            raise ValueError("Submission probabilities must be between 0 and 1.")
        true_labels = merged[f"{self.label_column}_true"].astype(int)
        return float(roc_auc_score(true_labels, probabilities))

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for text classification."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable text preprocessing and feature extraction helpers."),
            ("src/models.py", "Editable binary text-model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
