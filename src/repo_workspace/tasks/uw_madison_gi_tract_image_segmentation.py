"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchUwMadisonGiTractImageSegmentationMultifileTask(MleBenchPreparedTask):
    id_column: str = "id"
    class_column: str = "class"
    train_target_column: str = "segmentation"
    target_column: str = "predicted"
    class_names: tuple[str, ...] = ("large_bowel", "small_bowel", "stomach")
    protected_paths: tuple[str, ...] = (
        "data/train.csv",
        "data/public_eval.csv",
        "data/sample_submission.csv",
        "data/train_images",
        "data/public_eval_images",
    )

    @property
    def public_train_scans_root(self) -> Path:
        return self.public_root / "train"

    @property
    def public_test_scans_root(self) -> Path:
        return self.public_root / "test"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_test_path,
            self.public_sample_submission_path,
            self.private_answers_path,
            self.public_train_scans_root,
            self.public_test_scans_root,
        )

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "dice-hausdorff-combo"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for multi-organ MRI segmentation."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and CPU-only library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/image_features.py", "Editable MRI image loading and mask helpers."),
            ("src/models.py", "Editable lightweight segmentation helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
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

        train_ids = train_df[self.id_column].astype(str).drop_duplicates().tolist()
        public_ids = public_features[self.id_column].astype(str).drop_duplicates().tolist()
        self._link_slice_subset(train_ids, self._train_scan_index(), data_dir / "train_images")
        self._link_slice_subset(public_ids, self._test_scan_index(), data_dir / "public_eval_images")

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

        public_image_index = {
            slice_id: (workspace_root / "data" / "public_eval_images" / f"{slice_id}.png")
            for slice_id in public_features[self.id_column].astype(str).drop_duplicates().tolist()
        }
        public_run = self._run_uw_submission_split(
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            eval_df=public_eval_df,
            sample_submission_df=public_sample_submission,
            answers_df=public_answers,
            split_name="public",
            train_image_dir=workspace_root / "data" / "train_images",
            eval_image_index=public_image_index,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
        if public_run.success:
            private_run = self._run_uw_submission_split(
                workspace_root=workspace_root,
                hidden_root=hidden_root,
                eval_df=private_features,
                sample_submission_df=private_sample_submission,
                answers_df=private_answers,
                split_name="private",
                train_image_dir=workspace_root / "data" / "train_images",
                eval_image_index=self._test_scan_index(),
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

        full_train = pd.read_csv(self.public_train_path)[[self.id_column, self.class_column, self.train_target_column]].copy()
        train_df = self._sort_slice_frame(full_train)

        test_features_full = pd.read_csv(self.public_test_path)[[self.id_column, self.class_column]].copy()
        test_features_full = self._sort_slice_frame(test_features_full)
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)[
            [self.id_column, self.class_column, self.target_column]
        ].copy()
        sample_submission_full = self._sort_slice_frame(sample_submission_full)
        answers_full = pd.read_csv(self.private_answers_path)[
            [self.id_column, self.class_column, self.target_column, "image_width", "image_height"]
        ].copy()
        answers_full = self._sort_slice_frame(answers_full)

        public_case_days, _ = self._split_test_case_days(test_features_full)
        public_case_day_set = set(public_case_days)
        public_features = self._sort_slice_frame(
            test_features_full[
                test_features_full[self.id_column].astype(str).map(lambda value: value.split("_slice_")[0]).isin(public_case_day_set)
            ].copy()
        )
        public_answers = self._sort_slice_frame(
            answers_full[
                answers_full[self.id_column].astype(str).map(lambda value: value.split("_slice_")[0]).isin(public_case_day_set)
            ].copy()
        )
        public_sample_submission = self._sort_slice_frame(
            sample_submission_full[
                sample_submission_full[self.id_column].astype(str).map(lambda value: value.split("_slice_")[0]).isin(public_case_day_set)
            ].copy()
        )
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

        test_features_full = pd.read_csv(self.public_test_path)[[self.id_column, self.class_column]].copy()
        test_features_full = self._sort_slice_frame(test_features_full)
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)[
            [self.id_column, self.class_column, self.target_column]
        ].copy()
        sample_submission_full = self._sort_slice_frame(sample_submission_full)
        answers_full = pd.read_csv(self.private_answers_path)[
            [self.id_column, self.class_column, self.target_column, "image_width", "image_height"]
        ].copy()
        answers_full = self._sort_slice_frame(answers_full)

        _, private_case_days = self._split_test_case_days(test_features_full)
        private_case_day_set = set(private_case_days)
        private_features = self._sort_slice_frame(
            test_features_full[
                test_features_full[self.id_column].astype(str).map(lambda value: value.split("_slice_")[0]).isin(private_case_day_set)
            ].copy()
        )
        private_sample_submission = self._sort_slice_frame(
            sample_submission_full[
                sample_submission_full[self.id_column].astype(str).map(lambda value: value.split("_slice_")[0]).isin(private_case_day_set)
            ].copy()
        )
        private_answers = self._sort_slice_frame(
            answers_full[
                answers_full[self.id_column].astype(str).map(lambda value: value.split("_slice_")[0]).isin(private_case_day_set)
            ].copy()
        )
        cached = (
            private_features.copy(),
            private_sample_submission.copy(),
            private_answers.copy(),
        )
        setattr(self, "_cached_private_assets", cached)
        return private_features, private_sample_submission, private_answers

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        return public_features.merge(
            public_answers,
            on=[self.id_column, self.class_column],
            how="left",
            validate="one_to_one",
        )

    def _run_uw_submission_split(
        self,
        workspace_root: Path,
        hidden_root: Path,
        eval_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame,
        answers_df: pd.DataFrame,
        split_name: str,
        train_image_dir: Path,
        eval_image_index: dict[str, Path],
        python_executable: str,
        timeout_seconds: int,
    ) -> SplitExecution:
        tmp_root = _mk_eval_tmp_root(hidden_root, split_name)
        eval_path = tmp_root / f"{split_name}_input.csv"
        sample_path = tmp_root / f"{split_name}_sample_submission.csv"
        output_path = tmp_root / f"{split_name}_submission.csv"
        eval_image_dir = tmp_root / "eval_images"
        eval_image_dir.mkdir(parents=True, exist_ok=True)

        eval_df.to_csv(eval_path, index=False)
        sample_submission_df.to_csv(sample_path, index=False)
        unique_ids = eval_df[self.id_column].astype(str).drop_duplicates().tolist()
        self._link_slice_subset(unique_ids, eval_image_index, eval_image_dir)

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

    def _answer_frame_from_labels(
        self,
        frame: pd.DataFrame,
        *,
        label_column: str,
        scan_index: dict[str, Path],
    ) -> pd.DataFrame:
        width_map: dict[str, int] = {}
        height_map: dict[str, int] = {}
        for slice_id in frame[self.id_column].astype(str).drop_duplicates().tolist():
            width, height = self._slice_dimensions(scan_index[slice_id])
            width_map[slice_id] = width
            height_map[slice_id] = height

        answers = frame[[self.id_column, self.class_column, label_column]].copy()
        answers.rename(columns={label_column: self.target_column}, inplace=True)
        answers["image_width"] = answers[self.id_column].astype(str).map(width_map)
        answers["image_height"] = answers[self.id_column].astype(str).map(height_map)
        answers[self.target_column] = answers[self.target_column].fillna("")
        return self._sort_slice_frame(answers)

    def _split_test_case_days(self, test_features_df: pd.DataFrame) -> tuple[list[object], list[object]]:
        unique_case_days = sorted(
            test_features_df[self.id_column].astype(str).map(lambda value: value.split("_slice_")[0]).drop_duplicates().tolist()
        )
        case_day_series = pd.Series(unique_case_days, dtype=object)
        return _split_ids_evenly_by_value(case_day_series)

    def _sort_slice_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values([self.id_column, self.class_column]).reset_index(drop=True)

    def _link_slice_subset(self, slice_ids: list[str], source_index: dict[str, Path], output_dir: Path) -> None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for slice_id in slice_ids:
            source = source_index[str(slice_id)]
            _link_or_copy_file(source, output_dir / f"{slice_id}.png")

    def _train_scan_index(self) -> dict[str, Path]:
        cached = getattr(self, "_cached_train_scan_index", None)
        if cached is not None:
            return cached
        cached = self._build_scan_index(self.public_train_scans_root)
        setattr(self, "_cached_train_scan_index", cached)
        return cached

    def _test_scan_index(self) -> dict[str, Path]:
        cached = getattr(self, "_cached_test_scan_index", None)
        if cached is not None:
            return cached
        cached = self._build_scan_index(self.public_test_scans_root)
        setattr(self, "_cached_test_scan_index", cached)
        return cached

    def _build_scan_index(self, source_root: Path) -> dict[str, Path]:
        scan_index: dict[str, Path] = {}
        for scan_path in sorted(source_root.rglob("slice_*.png")):
            case_day = scan_path.parents[1].name
            slice_tokens = scan_path.stem.split("_")
            if len(slice_tokens) < 2:
                raise ValueError(f"Unexpected scan filename format: {scan_path.name}")
            slice_id = f"{case_day}_slice_{slice_tokens[1]}"
            scan_index[slice_id] = scan_path
        return scan_index

    def _slice_dimensions(self, scan_path: Path) -> tuple[int, int]:
        tokens = scan_path.stem.split("_")
        if len(tokens) < 4:
            with Image.open(scan_path) as image:
                return image.size
        width = int(float(tokens[2]))
        height = int(float(tokens[3]))
        return width, height

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        return _uw_dice_hausdorff_score(submission_df, answers_df)
