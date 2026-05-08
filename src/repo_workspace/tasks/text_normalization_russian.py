"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchTextNormalizationRussianMultifileTask(MleBenchPreparedTask):
    sentence_column: str = "sentence_id"
    token_column: str = "token_id"
    before_column: str = "before"
    target_column: str = "after"
    id_column: str = "id"

    @property
    def public_train_path(self) -> Path:
        return self.public_root / "en_train.csv.zip"

    @property
    def public_test_path(self) -> Path:
        return self.public_root / "en_test_2.csv.zip"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "en_sample_submission_2.csv.zip"

    @property
    def private_answers_path(self) -> Path:
        return self.private_root / "answers.csv"

    @property
    def private_sample_submission_path(self) -> Path:
        return self.private_root / "sample_submission.csv"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_test_path,
            self.public_sample_submission_path,
            self.private_answers_path,
            self.private_sample_submission_path,
        )

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "accuracy"

    def _with_submission_ids(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        enriched[self.id_column] = (
            enriched[self.sentence_column].astype(str) + "_" + enriched[self.token_column].astype(str)
        )
        return enriched

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = _read_csv_auto(self.public_train_path)
        train_df = full_train.reset_index(drop=True)
        test_features = _read_csv_auto(self.public_test_path)[
            [self.sentence_column, self.token_column, self.before_column]
        ].copy()
        sample_submission_full = _read_csv_auto(self.public_sample_submission_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()

        sentence_ids = sorted(test_features[self.sentence_column].unique().tolist())
        public_sentence_ids, private_sentence_ids = np.array_split(np.asarray(sentence_ids, dtype=object), 2)
        public_sentence_ids = list(public_sentence_ids.tolist())

        val_features = test_features[test_features[self.sentence_column].isin(public_sentence_ids)].reset_index(drop=True)
        public_ids = self._with_submission_ids(val_features[[self.sentence_column, self.token_column]])[self.id_column].tolist()
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        sample_submission[self.target_column] = val_features[self.before_column].astype(str).tolist()
        return train_df, val_features, sample_submission[[self.id_column, self.target_column]], public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = _read_csv_auto(self.public_test_path)[
            [self.sentence_column, self.token_column, self.before_column]
        ].copy()
        sample_submission_full = _read_csv_auto(self.public_sample_submission_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()

        sentence_ids = sorted(test_features[self.sentence_column].unique().tolist())
        public_sentence_ids, private_sentence_ids = np.array_split(np.asarray(sentence_ids, dtype=object), 2)
        private_sentence_ids = list(private_sentence_ids.tolist())

        private_features = test_features[test_features[self.sentence_column].isin(private_sentence_ids)].reset_index(drop=True)
        private_ids = self._with_submission_ids(private_features[[self.sentence_column, self.token_column]])[
            self.id_column
        ].tolist()
        private_sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, private_ids)
        private_answers = _subset_frame_by_ids(answers_full, self.id_column, private_ids)
        return private_features, private_sample_submission, private_answers

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        label_frame = public_answers.copy()
        split_ids = label_frame[self.id_column].str.split("_", n=1, expand=True)
        label_frame[self.sentence_column] = split_ids[0].astype(int)
        label_frame[self.token_column] = split_ids[1].astype(int)
        return public_features.merge(
            label_frame[[self.sentence_column, self.token_column, self.target_column]],
            on=[self.sentence_column, self.token_column],
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

        pred = submission_df[self.target_column].astype(str)
        true = answers_df[self.target_column].astype(str)
        return float((pred == true).mean())

    @property
    def public_train_path(self) -> Path:
        return self.public_root / "ru_train.csv.zip"

    @property
    def public_test_path(self) -> Path:
        return self.public_root / "ru_test_2.csv.zip"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "ru_sample_submission_2.csv.zip"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for Russian text normalization."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable token-normalization helpers."),
            ("src/models.py", "Editable token-mapping helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
