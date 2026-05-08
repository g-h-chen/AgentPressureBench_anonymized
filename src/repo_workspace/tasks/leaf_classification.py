"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchLeafClassificationMultifileTask(MleBenchPreparedTask):
    id_column: str = "id"
    label_column: str = "species"

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "multi-class-log-loss"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def class_columns(self) -> list[str]:
        sample_submission = pd.read_csv(self.public_sample_submission_path, nrows=0)
        return [column for column in sample_submission.columns if column != self.id_column]

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.class_columns]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)

        priors = train_df[self.label_column].value_counts(normalize=True)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        for class_name in self.class_columns:
            sample_submission[class_name] = float(priors.get(class_name, 0.0))

        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.class_columns]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        _, private_ids = _split_ids_evenly_by_value(test_features[self.id_column])

        private_features = _subset_frame_by_ids(test_features, self.id_column, private_ids)
        private_sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, private_ids)
        private_answers = _subset_frame_by_ids(answers_full, self.id_column, private_ids)
        return private_features, private_sample_submission, private_answers

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        label_series = (
            public_answers.set_index(self.id_column)[self.class_columns]
            .idxmax(axis=1)
            .rename(self.label_column)
            .reset_index()
        )
        return public_features.merge(label_series, on=self.id_column, how="left", validate="one_to_one")

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = [self.id_column, *self.class_columns]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        merged = answers_df[expected_columns].merge(
            submission_df[expected_columns],
            on=self.id_column,
            how="left",
            suffixes=("_true", "_pred"),
            validate="one_to_one",
        )
        if merged.isnull().any().any():
            raise ValueError("Submission ids did not align with answers.")

        pred_cols = [f"{label}_pred" for label in self.class_columns]
        true_cols = [f"{label}_true" for label in self.class_columns]
        probabilities = merged[pred_cols]
        if not ((probabilities >= 0) & (probabilities <= 1)).all().all():
            raise ValueError("Submission probabilities must be between 0 and 1.")
        if not probabilities.sum(axis=1).round(6).eq(1.0).all():
            raise ValueError("Each submission row must sum to 1.")

        true_labels = merged[true_cols].idxmax(axis=1).str.replace("_true", "", regex=False)
        return float(log_loss(true_labels, probabilities.to_numpy(), labels=list(self.class_columns)))

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for multiclass tabular prediction."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable preprocessing and feature engineering helpers."),
            ("src/models.py", "Editable multiclass-model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
