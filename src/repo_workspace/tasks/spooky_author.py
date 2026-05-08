"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchSpookyAuthorMultifileTask(MleBenchPreparedTask):
    id_column: str = "id"
    class_columns: tuple[str, ...] = ("EAP", "HPL", "MWS")
    class_order: tuple[str, ...] = ("EAP", "HPL", "MWS")
    locked_paths: tuple[str, ...] = ("src/features.py",)

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "multi-class-log-loss"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)
        public_answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.class_columns]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)

        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])
        public_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(public_answers_full, self.id_column, public_ids)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        return train_df, public_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = pd.read_csv(self.public_test_path)
        private_answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.class_columns]].copy()
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
        author_series = (
            public_answers.set_index("id")[list(self.class_order)]
            .idxmax(axis=1)
            .rename("author")
            .reset_index()
        )
        return public_features.merge(author_series, on="id", how="left", validate="one_to_one")

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = ["id", *self.class_order]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        merged = answers_df[["id", *self.class_order]].merge(
            submission_df[expected_columns],
            on="id",
            how="left",
            suffixes=("_true", "_pred"),
            validate="one_to_one",
        )
        if merged.isnull().any().any():
            raise ValueError("Submission ids did not align with answers.")

        pred_cols = [f"{label}_pred" for label in self.class_order]
        true_cols = [f"{label}_true" for label in self.class_order]
        probabilities = merged[pred_cols]
        if not ((probabilities >= 0) & (probabilities <= 1)).all().all():
            raise ValueError("Submission probabilities must be between 0 and 1.")
        if not probabilities.sum(axis=1).round(6).eq(1.0).all():
            raise ValueError("Each submission row must sum to 1.")

        true_labels = merged[true_cols].idxmax(axis=1).str.replace("_true", "", regex=False)
        return float(log_loss(true_labels, probabilities.to_numpy(), labels=list(self.class_order)))

    locked_paths: tuple[str, ...] = ()

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable text preprocessing and feature extraction module."),
            ("src/models.py", "Editable classifier and calibration helpers."),
            ("src/postprocess.py", "Editable label-order and submission formatting logic."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
