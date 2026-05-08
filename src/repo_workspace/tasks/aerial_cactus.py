"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchAerialCactusMultifileTask(MleBenchPreparedTask):
    label_column: str = "has_cactus"
    id_column: str = "id"

    @property
    def public_eval_archive_path(self) -> Path:
        return _first_existing_path(self.public_root / "public_eval.zip", self.public_root / "test.zip")

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_sample_submission_path,
            self.private_answers_path,
            self.public_root / "train.zip",
            self.public_eval_archive_path,
        )

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "auc-roc"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and image-directory layout."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Editable image-classification pipeline."),
            ("src/image_features.py", "Editable image loading and feature helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]

    def materialize_workspace(self, workspace_root: Path, eval_access: str = "metric_only") -> None:
        self._ensure_prepared_available()
        data_dir = workspace_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_df, public_features, public_sample_submission, public_answers = self._build_public_assets()
        public_eval_df = self._public_eval_frame(public_features, public_answers, eval_access)
        train_df.to_csv(data_dir / "train.csv", index=False)
        public_eval_df.to_csv(data_dir / "public_eval.csv", index=False)
        public_sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

        train_image_ids = train_df[self.id_column].tolist()
        public_eval_ids = public_features[self.id_column].tolist()
        self._extract_zip_subset(
            archive_path=self.public_root / "train.zip",
            image_ids=train_image_ids,
            output_dir=data_dir / "train_images",
        )
        self._extract_zip_subset(
            archive_path=self.public_eval_archive_path,
            image_ids=public_eval_ids,
            output_dir=data_dir / "public_eval_images",
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
                image_source_dir=self._ensure_private_image_cache(
                    hidden_root,
                    private_features[self.id_column].astype(str).tolist(),
                ),
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

    def _extract_zip_subset(self, archive_path: Path, image_ids: list[str], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        wanted = set(image_ids)
        with zipfile.ZipFile(archive_path) as archive:
            name_map = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
            missing = sorted(image_id for image_id in wanted if image_id not in name_map)
            if missing:
                raise FileNotFoundError(
                    f"Archive {archive_path} is missing {len(missing)} expected images, "
                    f"including {missing[:3]}"
                )
            for image_id in image_ids:
                destination = output_dir / image_id
                with archive.open(name_map[image_id]) as source, destination.open("wb") as sink:
                    sink.write(source.read())

    def _ensure_private_image_cache(self, hidden_root: Path, image_ids: list[str]) -> Path:
        cache_dir = hidden_root / "private_image_cache"
        marker = cache_dir / ".ready"
        if marker.exists() and all((cache_dir / image_id).exists() for image_id in image_ids):
            return cache_dir

        cache_dir.mkdir(parents=True, exist_ok=True)
        self._extract_zip_subset(
            archive_path=self.public_eval_archive_path,
            image_ids=image_ids,
            output_dir=cache_dir,
        )
        marker.write_text("ready\n")
        return cache_dir

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(sample_submission_full[self.id_column])

        val_features = _subset_frame_by_ids(sample_submission_full[[self.id_column]], self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)

        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_df = pd.read_csv(self.private_answers_path)
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        _, private_ids = _split_ids_evenly_by_value(sample_submission_full[self.id_column])

        private_features = _subset_frame_by_ids(sample_submission_full[[self.id_column]], self.id_column, private_ids)
        private_sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, private_ids)
        private_answers = _subset_frame_by_ids(private_df[[self.id_column, self.label_column]], self.id_column, private_ids)
        return private_features, private_sample_submission, private_answers

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
            _link_or_copy_file(image_source_dir / image_id, image_root / image_id)

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
        expected_columns = [self.id_column, self.label_column]
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

        probabilities = merged[f"{self.label_column}_pred"].astype(float)
        if not probabilities.between(0.0, 1.0).all():
            raise ValueError("Submission probabilities must be between 0 and 1.")

        true_labels = merged[f"{self.label_column}_true"].astype(int)
        return float(roc_auc_score(true_labels, probabilities))

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for image classification."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/image_features.py", "Editable image loading and feature helpers."),
            ("src/models.py", "Editable image-model construction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
