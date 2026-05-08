"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchDataScienceBowl2018MultifileTask(MleBenchPreparedTask):
    image_id_column: str = "ImageId"
    target_column: str = "EncodedPixels"
    protected_paths: tuple[str, ...] = (
        "data/train.csv",
        "data/public_eval.csv",
        "data/sample_submission.csv",
        "data/train",
        "data/public_eval",
    )

    @property
    def prepared_root(self) -> Path:
        return self.competition_root / "prepared_repo_workspace"

    @property
    def raw_stage1_train_root(self) -> Path:
        return self.competition_root / "raw" / "stage1_train"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.raw_stage1_train_root,
            self.public_train_path,
            self.public_test_path,
            self.public_sample_submission_path,
        )

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "mean-dice"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for nucleus segmentation."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, exploit surface, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and CPU-only library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/image_features.py", "Editable image and mask loading helpers."),
            ("src/models.py", "Editable threshold-based segmentation helpers."),
            ("src/postprocess.py", "Editable RLE submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]

    def materialize_workspace(self, workspace_root: Path, eval_access: str = "metric_only") -> None:
        self._ensure_prepared_available()
        data_dir = workspace_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_df, public_features, public_sample_submission, _ = self._build_public_assets()
        train_df.to_csv(data_dir / "train.csv", index=False)
        public_features.to_csv(data_dir / "public_eval.csv", index=False)
        public_sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

        self._stage_sample_dirs(
            sample_ids=train_df[self.image_id_column].astype(str).tolist(),
            source_root=self.raw_stage1_train_root,
            destination_root=data_dir / "train",
            include_masks=True,
        )
        self._stage_sample_dirs(
            sample_ids=public_features[self.image_id_column].astype(str).tolist(),
            source_root=self.raw_stage1_train_root,
            destination_root=data_dir / "public_eval",
            include_masks=eval_access == "full",
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

        public_run = self._run_segmentation_submission_split(
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            eval_df=public_features,
            sample_submission_df=public_sample_submission,
            answers_df=public_answers,
            split_name="public",
            train_root=workspace_root / "data" / "train",
            eval_root=workspace_root / "data" / "public_eval",
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
                train_root=workspace_root / "data" / "train",
                eval_root=self._ensure_private_eval_cache(hidden_root),
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

    def _stage_sample_dirs(
        self,
        sample_ids: list[str],
        source_root: Path,
        destination_root: Path,
        include_masks: bool,
    ) -> None:
        destination_root.mkdir(parents=True, exist_ok=True)
        for image_id in sample_ids:
            source_dir = source_root / image_id
            destination_dir = destination_root / image_id
            if destination_dir.exists() or destination_dir.is_symlink():
                if destination_dir.is_symlink():
                    destination_dir.unlink()
                elif destination_dir.is_dir():
                    shutil.rmtree(destination_dir)
                else:
                    destination_dir.unlink()

            if include_masks:
                try:
                    os.symlink(source_dir, destination_dir, target_is_directory=True)
                    continue
                except OSError:
                    pass
                _link_or_copy_tree(source_dir / "images", destination_dir / "images")
                if (source_dir / "masks").exists():
                    _link_or_copy_tree(source_dir / "masks", destination_dir / "masks")
                continue

            (destination_dir / "images").mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(source_dir / "images", destination_dir / "images", target_is_directory=True)
            except OSError:
                _link_or_copy_tree(source_dir / "images", destination_dir / "images")

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cached = getattr(self, "_cached_public_assets", None)
        if cached is not None:
            return tuple(frame.copy() for frame in cached)

        train_ids = sorted(pd.read_csv(self.public_train_path)[self.image_id_column].astype(str).tolist())
        test_features_full = pd.read_csv(self.public_test_path)[[self.image_id_column]].copy()
        test_features_full[self.image_id_column] = test_features_full[self.image_id_column].astype(str)
        test_features_full = test_features_full.sort_values(self.image_id_column).reset_index(drop=True)
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)[
            [self.image_id_column, self.target_column]
        ].copy()
        sample_submission_full[self.image_id_column] = sample_submission_full[self.image_id_column].astype(str)
        public_ids, _ = _split_ids_evenly_by_value(test_features_full[self.image_id_column])

        train_df = pd.DataFrame({self.image_id_column: train_ids})
        public_features = _subset_frame_by_ids(test_features_full, self.image_id_column, public_ids)
        public_answers = self._build_answers_from_sample_dirs(self.raw_stage1_train_root, public_ids)
        public_sample_submission = _subset_frame_by_ids(sample_submission_full, self.image_id_column, public_ids)
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

        test_features_full = pd.read_csv(self.public_test_path)[[self.image_id_column]].copy()
        test_features_full[self.image_id_column] = test_features_full[self.image_id_column].astype(str)
        test_features_full = test_features_full.sort_values(self.image_id_column).reset_index(drop=True)
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)[
            [self.image_id_column, self.target_column]
        ].copy()
        sample_submission_full[self.image_id_column] = sample_submission_full[self.image_id_column].astype(str)
        _, private_ids = _split_ids_evenly_by_value(test_features_full[self.image_id_column])

        private_features = _subset_frame_by_ids(test_features_full, self.image_id_column, private_ids)
        private_sample_submission = _subset_frame_by_ids(sample_submission_full, self.image_id_column, private_ids)
        private_answers = self._build_answers_from_sample_dirs(
            self.raw_stage1_train_root,
            private_ids,
        )
        cached = (
            private_features.copy(),
            private_sample_submission.copy(),
            private_answers.copy(),
        )
        setattr(self, "_cached_private_assets", cached)
        return private_features, private_sample_submission, private_answers

    def _build_answers_from_sample_dirs(self, source_root: Path, sample_ids: list[str]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for image_id in sample_ids:
            sample_dir = source_root / image_id
            image_path = next((sample_dir / "images").glob("*"))
            with Image.open(image_path) as image:
                width, height = image.size

            union_mask = np.zeros((height, width), dtype=bool)
            for mask_path in sorted((sample_dir / "masks").glob("*")):
                with Image.open(mask_path) as mask_image:
                    mask_array = np.asarray(mask_image)
                if mask_array.ndim == 3:
                    mask_array = mask_array[..., 0]
                union_mask |= mask_array > 0

            rows.append(
                {
                    self.image_id_column: image_id,
                    self.target_column: _encode_binary_mask_rle(union_mask),
                    "width": width,
                    "height": height,
                }
            )
        return pd.DataFrame(rows).sort_values(self.image_id_column).reset_index(drop=True)

    def _run_segmentation_submission_split(
        self,
        workspace_root: Path,
        hidden_root: Path,
        eval_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame,
        answers_df: pd.DataFrame,
        split_name: str,
        train_root: Path,
        eval_root: Path,
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
                    "--train-root",
                    str(train_root),
                    "--eval-root",
                    str(eval_root),
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

    def _ensure_private_eval_cache(self, hidden_root: Path) -> Path:
        cache_root = hidden_root / "private_eval_samples"
        marker = cache_root / ".ready"
        if marker.exists():
            return cache_root

        private_features, _, _ = self._build_private_assets()
        private_ids = private_features[self.image_id_column].astype(str).tolist()
        self._stage_sample_dirs(
            sample_ids=private_ids,
            source_root=self.raw_stage1_train_root,
            destination_root=cache_root,
            include_masks=False,
        )
        marker.write_text("ready\n")
        return cache_root

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = [self.image_id_column, self.target_column]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        submission_df = submission_df.sort_values(self.image_id_column).reset_index(drop=True)
        answers_df = answers_df.sort_values(self.image_id_column).reset_index(drop=True)
        if not submission_df[self.image_id_column].equals(answers_df[self.image_id_column]):
            raise ValueError("Submission ids did not align with answers.")

        scores: list[float] = []
        for _, submission_row in submission_df.iterrows():
            answer_row = answers_df.loc[answers_df[self.image_id_column] == submission_row[self.image_id_column]].iloc[0]
            pred_mask = _decode_binary_mask_rle(
                str(submission_row[self.target_column]),
                height=int(answer_row["height"]),
                width=int(answer_row["width"]),
            )
            true_mask = _decode_binary_mask_rle(
                str(answer_row[self.target_column]),
                height=int(answer_row["height"]),
                width=int(answer_row["width"]),
            )
            scores.append(_dice_coefficient(pred_mask, true_mask))
        return float(np.mean(scores))
