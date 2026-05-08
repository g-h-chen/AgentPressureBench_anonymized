"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchCrowdflowerSearchRelevanceMultifileTask(MleBenchPreparedTask):
    id_column: str = "id"
    label_column: str = "label"
    submission_column: str | None = None
    score_kind: str = "accuracy"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "quadratic-weighted-kappa" if self.score_kind == "qwk" else self.score_kind

    @property
    def prediction_column(self) -> str:
        return self.submission_column or self.label_column

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

        baseline_label = train_df[self.label_column].mode(dropna=False).iloc[0]
        sample_submission = _build_constant_submission_frame(
            public_features[self.id_column],
            id_column=self.id_column,
            column_defaults={self.prediction_column: baseline_label},
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
        baseline_label = full_train[self.label_column].mode(dropna=False).iloc[0]
        private_sample_submission = _build_constant_submission_frame(
            private_features[self.id_column],
            id_column=self.id_column,
            column_defaults={self.prediction_column: baseline_label},
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
        expected_columns = [self.id_column, self.prediction_column]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        submission_df = submission_df.sort_values(self.id_column).reset_index(drop=True)
        answers_df = answers_df.sort_values(self.id_column).reset_index(drop=True)
        if not submission_df[self.id_column].equals(answers_df[self.id_column]):
            raise ValueError("Submission ids did not align with answers.")

        true_labels = answers_df[self.label_column]
        pred_labels = submission_df[self.prediction_column]
        if self.score_kind == "accuracy":
            return float((true_labels == pred_labels).mean())
        if self.score_kind == "qwk":
            true_scores = pd.to_numeric(true_labels, errors="raise").astype(int)
            pred_scores = pd.to_numeric(pred_labels, errors="raise").round().astype(int)
            pred_scores = pred_scores.clip(lower=int(true_scores.min()), upper=int(true_scores.max()))
            return float(cohen_kappa_score(true_scores, pred_scores, weights="quadratic"))
        raise ValueError(f"Unsupported multiclass label score kind: {self.score_kind}")

    id_column: str = "id"
    label_column: str = "median_relevance"
    submission_column: str = "prediction"
    score_kind: str = "qwk"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for ordinal text relevance prediction."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable text-pair feature extraction helpers."),
            ("src/models.py", "Editable ordinal prediction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
