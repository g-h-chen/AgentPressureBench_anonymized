"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchIcrAgeRelatedConditionsMultifileTask(MleBenchPreparedTask):
    id_column: str = "id"
    label_column: str = "label"
    probability_zero_column: str = "class_0"
    probability_one_column: str = "class_1"

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "balanced-binary-log-loss"

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
        public_features, _, public_answers, _ = _split_aligned_test_pool(
            test_features,
            answers_full,
            id_column=self.id_column,
            seed=self.public_split_seed,
        )

        pos_rate = float(pd.to_numeric(train_df[self.label_column], errors="raise").astype(int).mean())
        sample_submission = _build_constant_submission_frame(
            public_features[self.id_column],
            id_column=self.id_column,
            column_defaults={
                self.probability_zero_column: 1.0 - pos_rate,
                self.probability_one_column: pos_rate,
            },
        )
        return train_df, public_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].copy()
        _, private_features, _, private_answers = _split_aligned_test_pool(
            test_features,
            answers_full,
            id_column=self.id_column,
            seed=self.public_split_seed,
        )
        pos_rate = float(pd.to_numeric(full_train[self.label_column], errors="raise").astype(int).mean())
        private_sample_submission = _build_constant_submission_frame(
            private_features[self.id_column],
            id_column=self.id_column,
            column_defaults={
                self.probability_zero_column: 1.0 - pos_rate,
                self.probability_one_column: pos_rate,
            },
        )
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
        expected_columns = [self.id_column, self.probability_zero_column, self.probability_one_column]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        merged = answers_df.merge(
            submission_df,
            on=self.id_column,
            how="left",
            validate="one_to_one",
        )
        if merged.isnull().any().any():
            raise ValueError("Submission ids did not align with answers.")

        p0 = np.clip(merged[self.probability_zero_column].astype(float).to_numpy(), 1e-15, 1 - 1e-15)
        p1 = np.clip(merged[self.probability_one_column].astype(float).to_numpy(), 1e-15, 1 - 1e-15)
        if not np.allclose(p0 + p1, 1.0, atol=1e-6):
            raise ValueError("Binary probability rows must sum to 1.")

        y_true = pd.to_numeric(merged[self.label_column], errors="raise").astype(int).to_numpy()
        neg_mask = y_true == 0
        pos_mask = y_true == 1
        neg_loss = -np.mean(np.log(p0[neg_mask])) if np.any(neg_mask) else 0.0
        pos_loss = -np.mean(np.log(p1[pos_mask])) if np.any(pos_mask) else 0.0
        return float((neg_loss + pos_loss) / 2.0)

    id_column: str = "Id"
    label_column: str = "Class"
    probability_zero_column: str = "class_0"
    probability_one_column: str = "class_1"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for balanced binary tabular prediction."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable tabular feature engineering helpers."),
            ("src/models.py", "Editable probability-model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
