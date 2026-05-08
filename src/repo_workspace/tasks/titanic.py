"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchTitanicMultifileTask(MleBenchPreparedTask):
    id_column: str = "id"
    label_column: str = "label"
    score_kind: str = "accuracy"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return self.score_kind

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].copy()
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)

        majority_label = bool(_coerce_binary_series(train_df[self.label_column]).mode(dropna=False).iloc[0])
        sample_submission = _build_constant_submission_frame(
            val_features[self.id_column],
            id_column=self.id_column,
            column_defaults={self.label_column: majority_label},
        )
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].copy()
        _, private_ids = _split_ids_evenly_by_value(test_features[self.id_column])

        private_features = _subset_frame_by_ids(test_features, self.id_column, private_ids)
        majority_label = bool(_coerce_binary_series(full_train[self.label_column]).mode(dropna=False).iloc[0])
        private_sample_submission = _build_constant_submission_frame(
            private_features[self.id_column],
            id_column=self.id_column,
            column_defaults={self.label_column: majority_label},
        )
        private_answers = _subset_frame_by_ids(answers_full, self.id_column, private_ids)
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

        true_labels = _coerce_binary_series(merged[f"{self.label_column}_true"])
        pred_labels = _coerce_binary_series(merged[f"{self.label_column}_pred"])
        if self.score_kind == "accuracy":
            return float((true_labels == pred_labels).mean())
        if self.score_kind == "f1":
            return float(f1_score(true_labels, pred_labels, average="binary"))
        raise ValueError(f"Unsupported binary score kind: {self.score_kind}")

    id_column: str = "PassengerId"
    label_column: str = "Survived"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for tabular classification."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable preprocessing and feature engineering helpers."),
            ("src/models.py", "Editable classifier-construction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
