"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchTgsSaltIdentificationMultifileTask(MleBenchPreparedTask):
    image_id_column: str = "id"
    target_column: str = "rle_mask"
    image_height: int = 101
    image_width: int = 101
    protected_paths: tuple[str, ...] = (
        "data/train.csv",
        "data/public_eval.csv",
        "data/sample_submission.csv",
        "data/train_images",
        "data/train_masks",
        "data/public_eval_images",
        "data/public_eval_masks",
    )

    @property
    def public_depths_path(self) -> Path:
        return self.public_root / "depths.csv"

    @property
    def public_train_images_root(self) -> Path:
        return self.public_root / "train" / "images"

    @property
    def public_train_masks_root(self) -> Path:
        return self.public_root / "train" / "masks"

    @property
    def public_test_images_root(self) -> Path:
        return self.public_root / "test" / "images"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_sample_submission_path,
            self.public_depths_path,
            self.private_answers_path,
            self.public_train_images_root,
            self.public_train_masks_root,
            self.public_test_images_root,
        )

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "mean-precision-at-iou-thresholds"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for salt-mask segmentation."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and CPU-only library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/image_features.py", "Editable seismic-image and mask loading helpers."),
            ("src/models.py", "Editable lightweight segmentation helpers."),
            ("src/postprocess.py", "Editable RLE submission formatting helpers."),
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

        train_ids = train_df[self.image_id_column].astype(str).tolist()
        public_ids = public_features[self.image_id_column].astype(str).tolist()
        self._link_asset_subset(train_ids, self.public_train_images_root, data_dir / "train_images")
        self._link_asset_subset(train_ids, self.public_train_masks_root, data_dir / "train_masks")
        self._link_asset_subset(public_ids, self.public_test_images_root, data_dir / "public_eval_images")
        if eval_access == "full":
            self._materialize_public_mask_subset(public_answers, data_dir / "public_eval_masks")

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

        public_run = self._run_segmentation_submission_split(
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            eval_df=public_eval_df,
            sample_submission_df=public_sample_submission,
            answers_df=public_answers,
            split_name="public",
            train_image_dir=workspace_root / "data" / "train_images",
            train_mask_dir=workspace_root / "data" / "train_masks",
            eval_image_dir=workspace_root / "data" / "public_eval_images",
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
        if public_run.success:
            private_run = self._run_segmentation_submission_split(
                workspace_root=workspace_root,
                hidden_root=hidden_root,
                eval_df=private_features,
                sample_submission_df=private_sample_submission,
                answers_df=private_answers,
                split_name="private",
                train_image_dir=workspace_root / "data" / "train_images",
                train_mask_dir=workspace_root / "data" / "train_masks",
                eval_image_dir=self.public_test_images_root,
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

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cached = getattr(self, "_cached_public_assets", None)
        if cached is not None:
            return tuple(frame.copy() for frame in cached)

        labels_df = pd.read_csv(self.public_train_path)[[self.image_id_column, self.target_column]].copy()
        depths_df = pd.read_csv(self.public_depths_path)[[self.image_id_column, "z"]].copy()
        train_df = labels_df.merge(depths_df, on=self.image_id_column, how="left")
        train_df = train_df.sort_values(self.image_id_column).reset_index(drop=True)

        sample_submission_full = pd.read_csv(self.public_sample_submission_path)[
            [self.image_id_column, self.target_column]
        ].copy()
        sample_submission_full = sample_submission_full.sort_values(self.image_id_column).reset_index(drop=True)
        answers_full = pd.read_csv(self.private_answers_path)[[self.image_id_column, self.target_column]].copy()
        answers_full = self._build_answers_frame(answers_full)
        public_ids, _ = _split_ids_evenly_by_value(sample_submission_full[self.image_id_column])

        public_features = pd.DataFrame({self.image_id_column: public_ids, "z": np.nan})
        public_sample_submission = _subset_frame_by_ids(sample_submission_full, self.image_id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.image_id_column, public_ids)
        cached = (
            train_df.copy(),
            public_features.copy(),
            public_sample_submission.copy(),
            public_answers.copy(),
        )
        setattr(self, "_cached_public_assets", cached)
        return train_df, public_features, public_sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cached = getattr(self, "_cached_private_assets", None)
        if cached is not None:
            return tuple(frame.copy() for frame in cached)

        sample_submission_full = pd.read_csv(self.public_sample_submission_path)[
            [self.image_id_column, self.target_column]
        ].copy()
        sample_submission_full = sample_submission_full.sort_values(self.image_id_column).reset_index(drop=True)
        answers_full = pd.read_csv(self.private_answers_path)[[self.image_id_column, self.target_column]].copy()
        answers_full = self._build_answers_frame(answers_full)
        _, private_ids = _split_ids_evenly_by_value(sample_submission_full[self.image_id_column])

        private_features = pd.DataFrame({self.image_id_column: private_ids, "z": np.nan})
        private_sample_submission = _subset_frame_by_ids(sample_submission_full, self.image_id_column, private_ids)
        private_answers = _subset_frame_by_ids(answers_full, self.image_id_column, private_ids)
        cached = (
            private_features.copy(),
            private_sample_submission.copy(),
            private_answers.copy(),
        )
        setattr(self, "_cached_private_assets", cached)
        return private_features, private_sample_submission, private_answers

    def _build_answers_frame(self, labels_df: pd.DataFrame) -> pd.DataFrame:
        answers = labels_df[[self.image_id_column, self.target_column]].copy()
        answers["width"] = self.image_width
        answers["height"] = self.image_height
        return answers.sort_values(self.image_id_column).reset_index(drop=True)

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        return public_features.merge(
            public_answers,
            on=self.image_id_column,
            how="left",
            validate="one_to_one",
        )

    def _link_asset_subset(self, image_ids: list[str], source_dir: Path, output_dir: Path) -> None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            source = _resolve_image_asset(source_dir, image_id)
            _link_or_copy_file(source, output_dir / source.name)

    def _materialize_public_mask_subset(self, answers_df: pd.DataFrame, output_dir: Path) -> None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in answers_df.itertuples(index=False):
            image_id = str(getattr(row, self.image_id_column))
            image_path = _resolve_image_asset(self.public_test_images_root, image_id)
            mask = _decode_binary_mask_rle(
                str(getattr(row, self.target_column)),
                height=int(getattr(row, "height")),
                width=int(getattr(row, "width")),
            )
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(output_dir / image_path.name)

    def _run_segmentation_submission_split(
        self,
        workspace_root: Path,
        hidden_root: Path,
        eval_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame,
        answers_df: pd.DataFrame,
        split_name: str,
        train_image_dir: Path,
        train_mask_dir: Path,
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
                    str(train_image_dir),
                    "--train-mask-dir",
                    str(train_mask_dir),
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
        return _tgs_mean_average_precision(
            submission_df,
            answers_df,
            image_id_column=self.image_id_column,
            target_column=self.target_column,
            default_height=self.image_height,
            default_width=self.image_width,
        )
