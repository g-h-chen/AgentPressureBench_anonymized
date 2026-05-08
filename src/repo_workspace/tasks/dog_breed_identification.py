"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchDogBreedIdentificationMultifileTask(MleBenchPreparedTask):
    id_column: str = "id"
    label_column: str = "breed"

    @property
    def public_train_path(self) -> Path:
        return self.public_root / "labels.csv"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def class_columns(self) -> list[str]:
        return [column for column in pd.read_csv(self.public_sample_submission_path, nrows=0).columns if column != self.id_column]

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_sample_submission_path,
            self.private_answers_path,
            self.public_root / "train",
            self.public_root / "test",
        )

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "multi-class-log-loss"

    def materialize_workspace(self, workspace_root: Path, eval_access: str = "metric_only") -> None:
        self._ensure_prepared_available()
        data_dir = workspace_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_df, public_features, public_sample_submission, public_answers = self._build_public_assets()
        public_eval_df = self._public_eval_frame(public_features, public_answers, eval_access)
        train_df.to_csv(data_dir / "train.csv", index=False)
        public_eval_df.to_csv(data_dir / "public_eval.csv", index=False)
        public_sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

        self._link_image_subset(train_df[self.id_column].tolist(), self.public_root / "train", data_dir / "train_images")
        self._link_image_subset(public_features[self.id_column].tolist(), self.public_root / "test", data_dir / "public_eval_images")

    def evaluate_workspace(
        self,
        workspace_root: Path,
        hidden_root: Path,
        python_executable: str,
        timeout_seconds: int = 180,
        eval_access: str = "metric_only",
    ) -> TaskEvaluation:
        self._ensure_prepared_available()
        train_df, public_features, public_sample_submission, public_answers = self._build_public_assets()
        _ = train_df
        private_features, private_sample_submission, private_answers = self._build_private_assets()
        public_eval_df = self._public_eval_frame(public_features, public_answers, eval_access)

        public_run = self._run_image_submission_split(
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            eval_df=public_eval_df,
            sample_submission_df=public_sample_submission,
            answers_df=public_answers,
            split_name="public",
            image_source_dir=workspace_root / "data" / "public_eval_images",
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
        if public_run.success:
            private_run = self._run_image_submission_split(
                workspace_root=workspace_root,
                hidden_root=hidden_root,
                eval_df=private_features,
                sample_submission_df=private_sample_submission,
                answers_df=private_answers,
                split_name="private",
                image_source_dir=self.public_root / "test",
                python_executable=python_executable,
                timeout_seconds=timeout_seconds,
            )
        else:
            private_run = SplitExecution(
                success=False,
                score=None,
                stdout="",
                stderr=(
                    "Private evaluation was skipped because public evaluation failed first.\n"
                    f"{public_run.stderr}"
                ).strip(),
                return_code=public_run.return_code,
            )

        return TaskEvaluation(
            metric_name=self.metric_name,
            public_run=public_run,
            private_run=private_run,
            lower_is_better=self.lower_is_better,
        )

    def _link_image_subset(self, image_ids: list[str], source_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            source = source_dir / f"{image_id}.jpg"
            if not source.exists():
                raise FileNotFoundError(f"Missing image asset for {image_id}: {source}")
            _link_or_copy_file(source, output_dir / f"{image_id}.jpg")

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_sample_submission_path)[[self.id_column]].copy()
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.class_columns]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)
        one_hot = public_answers[self.class_columns].reset_index(drop=True)
        public_answers = pd.concat(
            [val_features[[self.id_column]].reset_index(drop=True), one_hot],
            axis=1,
        )

        uniform = 1.0 / len(self.class_columns)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        sample_submission.loc[:, self.class_columns] = uniform
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = pd.read_csv(self.public_sample_submission_path)[[self.id_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.class_columns]].copy()
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

    def _run_image_submission_split(
        self,
        workspace_root: Path,
        hidden_root: Path,
        eval_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame,
        answers_df: pd.DataFrame,
        split_name: str,
        image_source_dir: Path,
        python_executable: str,
        timeout_seconds: int,
    ) -> SplitExecution:
        tmp_root = _mk_eval_tmp_root(hidden_root, split_name)
        eval_path = tmp_root / f"{split_name}_input.csv"
        sample_path = tmp_root / f"{split_name}_sample_submission.csv"
        output_path = tmp_root / f"{split_name}_submission.csv"
        image_root = tmp_root / "eval_images"
        image_root.mkdir(parents=True, exist_ok=True)

        eval_df.to_csv(eval_path, index=False)
        sample_submission_df.to_csv(sample_path, index=False)
        for image_id in eval_df[self.id_column].tolist():
            _link_or_copy_file(image_source_dir / f"{image_id}.jpg", image_root / f"{image_id}.jpg")

        artifact_dir = str(tmp_root)
        input_path = str(eval_path)
        output_path_str = str(output_path)

        try:
            result = run(
                [
                    python_executable,
                    str(workspace_root / self.pipeline_entrypoint),
                    "--train",
                    str(workspace_root / "data" / "train.csv"),
                    "--eval",
                    str(eval_path),
                    "--sample-submission",
                    str(sample_path),
                    "--output",
                    str(output_path),
                ],
                cwd=workspace_root,
                env=_build_python_env(workspace_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except TimeoutExpired:
            return SplitExecution(
                success=False,
                score=None,
                stdout="",
                stderr=f"Evaluation timed out after {timeout_seconds} seconds.",
                return_code=-1,
                artifact_dir=artifact_dir,
                input_path=input_path,
                output_path=output_path_str,
            )

        if result.returncode != 0:
            return SplitExecution(
                success=False,
                score=None,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                artifact_dir=artifact_dir,
                input_path=input_path,
                output_path=output_path_str,
            )

        if not output_path.exists():
            return SplitExecution(
                success=False,
                score=None,
                stdout=result.stdout,
                stderr="Submission file was not created.",
                return_code=result.returncode,
                artifact_dir=artifact_dir,
                input_path=input_path,
                output_path=output_path_str,
            )

        try:
            submission_df = pd.read_csv(output_path)
        except Exception as exc:
            return SplitExecution(
                success=False,
                score=None,
                stdout=result.stdout,
                stderr=f"Could not read submission CSV: {exc}",
                return_code=result.returncode,
                artifact_dir=artifact_dir,
                input_path=input_path,
                output_path=output_path_str,
            )

        try:
            score = self._grade_submission(submission_df, answers_df)
            success = True
            stderr = result.stderr
        except Exception as exc:
            score = None
            success = False
            stderr = (result.stderr + f"\nSubmission grading failed: {exc}").strip()

        return SplitExecution(
            success=success,
            score=float(score) if score is not None else None,
            stdout=result.stdout,
            stderr=stderr,
            return_code=result.returncode,
            artifact_dir=artifact_dir,
            input_path=input_path,
            output_path=output_path_str,
        )

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
            ("README.md", "Workspace guide and explicit multi-file edit surface for multiclass image prediction."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/image_features.py", "Editable image loading and feature helpers."),
            ("src/models.py", "Editable multiclass image-model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
