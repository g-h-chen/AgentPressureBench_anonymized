"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchNomadMultifileTask(MleBenchPreparedTask):
    id_column: str = "id"
    target_columns: tuple[str, ...] = ("formation_energy_ev_natom", "bandgap_energy_ev")

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_test_path,
            self.private_answers_path,
        )

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "mean-column-wise-rmsle"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)
        private_answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.target_columns]].copy()
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(private_answers_full, self.id_column, public_ids)

        means = train_df[list(self.target_columns)].mean()
        sample_submission = pd.DataFrame({self.id_column: val_features[self.id_column]})
        for column in self.target_columns:
            sample_submission[column] = float(means[column])

        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(self.public_train_path)
        test_features = pd.read_csv(self.public_test_path)
        private_answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.target_columns]].copy()
        _, private_ids = _split_ids_evenly_by_value(test_features[self.id_column])

        private_features = _subset_frame_by_ids(test_features, self.id_column, private_ids)
        means = train_df[list(self.target_columns)].mean()
        private_sample_submission = pd.DataFrame({self.id_column: private_features[self.id_column]})
        for column in self.target_columns:
            private_sample_submission[column] = float(means[column])
        private_answers = _subset_frame_by_ids(private_answers_full, self.id_column, private_ids)
        return private_features, private_sample_submission, private_answers

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = ["id", *self.target_columns]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        merged = answers_df[expected_columns].merge(
            submission_df[expected_columns],
            on="id",
            how="left",
            suffixes=("_true", "_pred"),
            validate="one_to_one",
        )
        if merged.isnull().any().any():
            raise ValueError("Submission ids did not align with answers.")

        scores = []
        for column in self.target_columns:
            true_values = merged[f"{column}_true"]
            pred_values = merged[f"{column}_pred"]
            scores.append(root_mean_squared_log_error(true_values, pred_values))
        return float(sum(scores) / len(scores))

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable feature selection and numeric preprocessing helpers."),
            ("src/models.py", "Editable regression-model helpers."),
            ("src/postprocess.py", "Editable prediction clipping and submission formatting logic."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
