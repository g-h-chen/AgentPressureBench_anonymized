"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchKuzushijiRecognitionMultifileTask(MleBenchPreparedTask):
    id_column: str = "image_id"
    target_column: str = "labels"
    image_suffix: str = ".jpg"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def public_train_images_zip_path(self) -> Path:
        return self.public_root / "train_images.zip"

    @property
    def public_test_images_zip_path(self) -> Path:
        return self.public_root / "test_images.zip"

    @property
    def unicode_translation_path(self) -> Path:
        return self.public_root / "unicode_translation.csv"

    @property
    def extracted_cache_root(self) -> Path:
        return self.competition_root / "prepared_cache"

    @property
    def train_image_cache_dir(self) -> Path:
        return self.extracted_cache_root / "train_images"

    @property
    def test_image_cache_dir(self) -> Path:
        return self.extracted_cache_root / "test_images"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.private_answers_path,
            self.public_sample_submission_path,
            self.public_train_images_zip_path,
            self.public_test_images_zip_path,
            self.unicode_translation_path,
        )

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "f1"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for page-level character localization."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/image_features.py", "Editable page-image loading and feature helpers."),
            ("src/models.py", "Editable localization baseline helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]

    def materialize_workspace(self, workspace_root: Path, eval_access: str = "metric_only") -> None:
        self._ensure_prepared_available()
        self._ensure_image_caches()

        data_dir = workspace_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_df, public_features, public_sample_submission, public_answers = self._build_public_assets()
        public_eval_df = self._public_eval_frame(public_features, public_answers, eval_access)
        train_df.to_csv(data_dir / "train.csv", index=False)
        public_eval_df.to_csv(data_dir / "public_eval.csv", index=False)
        public_sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)
        _link_or_copy_file(self.unicode_translation_path, data_dir / "unicode_translation.csv")

        self._link_image_subset(
            train_df[self.id_column].astype(str).tolist(),
            self.train_image_cache_dir,
            data_dir / "train_images",
        )
        self._link_image_subset(
            public_features[self.id_column].astype(str).tolist(),
            self.test_image_cache_dir,
            data_dir / "public_eval_images",
        )

    def evaluate_workspace(
        self,
        workspace_root: Path,
        hidden_root: Path,
        python_executable: str,
        timeout_seconds: int = 180,
        eval_access: str = "metric_only",
    ) -> TaskEvaluation:
        self._ensure_prepared_available()
        self._ensure_image_caches()

        train_df, public_features, public_sample_submission, public_answers = self._build_public_assets()
        _ = train_df
        private_features, private_sample_submission, private_answers = self._build_private_assets()
        public_eval_df = self._public_eval_frame(public_features, public_answers, eval_access)

        public_run = self._run_kuzushiji_split(
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            eval_df=public_eval_df,
            sample_submission_df=public_sample_submission,
            answers_df=public_answers,
            split_name="public",
            eval_image_dir=workspace_root / "data" / "public_eval_images",
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
        if public_run.success:
            private_run = self._run_kuzushiji_split(
                workspace_root=workspace_root,
                hidden_root=hidden_root,
                eval_df=private_features,
                sample_submission_df=private_sample_submission,
                answers_df=private_answers,
                split_name="private",
                eval_image_dir=self.test_image_cache_dir,
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

    def _ensure_image_caches(self) -> None:
        _extract_zip_to_cache(self.public_train_images_zip_path, self.train_image_cache_dir)
        _extract_zip_to_cache(self.public_test_images_zip_path, self.test_image_cache_dir)

    def _link_image_subset(self, image_ids: list[str], source_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            source = _resolve_image_asset(source_dir, image_id, suffix_candidates=(self.image_suffix,))
            _link_or_copy_file(source, output_dir / source.name)

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)[[self.id_column, self.target_column]].copy()
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)

        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)[[self.id_column, self.target_column]].copy()
        public_ids, _ = _split_ids_evenly_by_value(answers_full[self.id_column])

        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)
        public_features = public_answers[[self.id_column]].copy()
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        return train_df, public_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)[
            [self.id_column, self.target_column]
        ].copy()
        _, private_ids = _split_ids_evenly_by_value(answers_full[self.id_column])

        private_answers = _subset_frame_by_ids(answers_full, self.id_column, private_ids)
        private_features = private_answers[[self.id_column]].copy()
        private_sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, private_ids)
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

    def _run_kuzushiji_split(
        self,
        workspace_root: Path,
        hidden_root: Path,
        eval_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame,
        answers_df: pd.DataFrame,
        split_name: str,
        eval_image_dir: Path,
        python_executable: str,
        timeout_seconds: int,
    ) -> SplitExecution:
        tmp_root = _mk_eval_tmp_root(hidden_root, split_name)
        eval_path = tmp_root / f"{split_name}_input.csv"
        sample_path = tmp_root / f"{split_name}_sample_submission.csv"
        output_path = tmp_root / f"{split_name}_submission.csv"
        eval_df.to_csv(eval_path, index=False)
        sample_submission_df.to_csv(sample_path, index=False)

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
                    "--train-image-dir",
                    str(workspace_root / "data" / "train_images"),
                    "--eval-image-dir",
                    str(eval_image_dir),
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
        expected_columns = [self.id_column, self.target_column]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        submission_df = submission_df.sort_values(self.id_column).reset_index(drop=True)
        answers_df = answers_df.sort_values(self.id_column).reset_index(drop=True)
        if not submission_df[self.id_column].equals(answers_df[self.id_column]):
            raise ValueError("Submission image ids did not align with answers.")

        for prediction_text in submission_df[self.target_column]:
            if pd.isna(prediction_text):
                continue
            parts = str(prediction_text).split()
            if len(parts) % 3 != 0:
                raise ValueError(f"Malformed prediction string: {prediction_text}")
            for index in range(1, len(parts), 3):
                float(parts[index])
                float(parts[index + 1])

        return _kuzushiji_f1(submission_df, answers_df)
