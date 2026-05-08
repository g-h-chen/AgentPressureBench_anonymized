"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchTransferLearningStackExchangeTagsMultifileTask(MleBenchPreparedTask):
    id_column: str = "id"
    target_column: str = "tags"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "micro-f1"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)

        test_features_full = pd.read_csv(self.public_test_path).sort_values(self.id_column).reset_index(drop=True)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features_full[self.id_column])

        public_features = _subset_frame_by_ids(test_features_full, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        return train_df, public_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features_full = pd.read_csv(self.public_test_path).sort_values(self.id_column).reset_index(drop=True)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        _, private_ids = _split_ids_evenly_by_value(test_features_full[self.id_column])

        private_features = _subset_frame_by_ids(test_features_full, self.id_column, private_ids)
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

        submission_df = submission_df.sort_values(self.id_column).reset_index(drop=True)
        answers_df = answers_df.sort_values(self.id_column).reset_index(drop=True)
        if not submission_df[self.id_column].equals(answers_df[self.id_column]):
            raise ValueError("Submission ids did not align with answers.")

        true_tags = answers_df[self.target_column].fillna("").astype(str).str.split()
        pred_tags = submission_df[self.target_column].fillna("").astype(str).str.split()
        classes = sorted(set(tag for row in true_tags for tag in row))
        mlb = MultiLabelBinarizer(classes=classes, sparse_output=False)
        y_true = mlb.fit_transform(true_tags)
        y_pred = mlb.transform(pred_tags)
        return float(f1_score(y_true, y_pred, average="micro"))

    id_column: str = "id"
    target_column: str = "tags"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for multilabel text tagging."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable text feature extraction helpers."),
            ("src/models.py", "Editable multilabel tagging helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
