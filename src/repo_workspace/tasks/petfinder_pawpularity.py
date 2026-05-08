"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchPetfinderPawpularityMultifileTask(MleBenchPreparedTask):
    id_column: str = "Id"
    target_column: str = "Pawpularity"

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "rmse"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)

        mean_target = float(train_df[self.target_column].astype(float).mean())
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        sample_submission[self.target_column] = mean_target
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
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
        return public_features.merge(
            public_answers,
            on=self.id_column,
            how="left",
            validate="one_to_one",
        )

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = [self.id_column, self.target_column]
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
        return _rmse_score(merged[f"{self.target_column}_true"], merged[f"{self.target_column}_pred"])

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for tabular regression."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable preprocessing and feature engineering helpers."),
            ("src/models.py", "Editable regression-model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
