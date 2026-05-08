"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchFacialKeypointsDetectionMultifileTask(MleBenchPreparedTask):
    image_id_column: str = "ImageId"
    image_column: str = "Image"
    submission_id_column: str = "RowId"
    target_column: str = "Location"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def private_answers_path(self) -> Path:
        return self.private_root / "test.csv"

    @property
    def public_lookup_path(self) -> Path:
        return self.public_root / "IdLookupTable.csv"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_test_path,
            self.public_lookup_path,
            self.public_sample_submission_path,
            self.private_answers_path,
        )

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "rmse"

    def materialize_workspace(self, workspace_root: Path, eval_access: str = "metric_only") -> None:
        self._ensure_prepared_available()
        data_dir = workspace_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_df, public_features, public_sample_submission, public_answers, public_lookup = self._build_public_assets()
        train_df.to_csv(data_dir / "train.csv", index=False)
        public_features.to_csv(data_dir / "public_eval.csv", index=False)
        public_lookup.to_csv(data_dir / "IdLookupTable.csv", index=False)
        public_sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)
        if eval_access == "full":
            public_answers.to_csv(data_dir / "public_eval_answers.csv", index=False)

    def evaluate_workspace(
        self,
        workspace_root: Path,
        hidden_root: Path,
        python_executable: str,
        timeout_seconds: int = 180,
        eval_access: str = "metric_only",
    ) -> TaskEvaluation:
        self._ensure_prepared_available()
        train_df, public_features, public_sample_submission, public_answers, public_lookup = self._build_public_assets()
        _ = train_df
        private_features, private_sample_submission, private_answers, private_lookup = self._build_private_assets()

        public_run = self._run_keypoint_split(
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            eval_df=public_features,
            lookup_df=public_lookup,
            sample_submission_df=public_sample_submission,
            answers_df=public_answers,
            split_name="public",
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
        if public_run.success:
            private_run = self._run_keypoint_split(
                workspace_root=workspace_root,
                hidden_root=hidden_root,
                eval_df=private_features,
                lookup_df=private_lookup,
                sample_submission_df=private_sample_submission,
                answers_df=private_answers,
                split_name="private",
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
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.image_id_column).reset_index(drop=True)

        test_features_full = pd.read_csv(self.public_test_path)[[self.image_id_column, self.image_column]].copy()
        test_features_full = test_features_full.sort_values(self.image_id_column).reset_index(drop=True)
        lookup_full = pd.read_csv(self.public_lookup_path)[
            [self.submission_id_column, self.image_id_column, "FeatureName"]
        ].copy()
        answers_full = pd.read_csv(self.private_answers_path)[[self.submission_id_column, self.target_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)[
            [self.submission_id_column, self.target_column]
        ].copy()
        public_image_ids, _ = _split_ids_evenly_by_value(test_features_full[self.image_id_column])

        public_features = _subset_frame_by_ids(test_features_full, self.image_id_column, public_image_ids)
        public_lookup = self._subset_lookup_by_image_ids(lookup_full, public_image_ids)
        public_row_ids = public_lookup[self.submission_id_column].tolist()
        public_answers = _subset_frame_by_ids(answers_full, self.submission_id_column, public_row_ids)
        public_sample_submission = _subset_frame_by_ids(sample_submission_full, self.submission_id_column, public_row_ids)
        return train_df, public_features, public_sample_submission, public_answers, public_lookup

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features_full = pd.read_csv(self.public_test_path)[[self.image_id_column, self.image_column]].copy()
        test_features_full = test_features_full.sort_values(self.image_id_column).reset_index(drop=True)
        lookup_full = pd.read_csv(self.public_lookup_path)[
            [self.submission_id_column, self.image_id_column, "FeatureName"]
        ].copy()
        answers_full = pd.read_csv(self.private_answers_path)[[self.submission_id_column, self.target_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)[
            [self.submission_id_column, self.target_column]
        ].copy()
        _, private_image_ids = _split_ids_evenly_by_value(test_features_full[self.image_id_column])

        private_features = _subset_frame_by_ids(test_features_full, self.image_id_column, private_image_ids)
        private_lookup = self._subset_lookup_by_image_ids(lookup_full, private_image_ids)
        private_row_ids = private_lookup[self.submission_id_column].tolist()
        private_sample_submission = _subset_frame_by_ids(sample_submission_full, self.submission_id_column, private_row_ids)
        private_answers = _subset_frame_by_ids(answers_full, self.submission_id_column, private_row_ids)
        return private_features, private_sample_submission, private_answers, private_lookup

    def _subset_lookup_by_image_ids(self, lookup_df: pd.DataFrame, image_ids: list[object]) -> pd.DataFrame:
        id_set = set(image_ids)
        subset = lookup_df[lookup_df[self.image_id_column].isin(id_set)].copy()
        present_ids = set(subset[self.image_id_column].tolist())
        if present_ids != id_set:
            missing = [str(value) for value in image_ids if value not in present_ids]
            raise ValueError(f"Could not find lookup rows for image ids: {missing[:3]}")
        order_map = {value: index for index, value in enumerate(image_ids)}
        subset["__order"] = subset[self.image_id_column].map(order_map)
        subset = subset.sort_values(["__order", self.submission_id_column], kind="mergesort")
        return subset.drop(columns="__order").reset_index(drop=True)

    def _build_lookup_and_answers(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        keypoint_columns = [column for column in frame.columns if column not in {self.image_id_column, self.image_column}]
        lookup_rows: list[dict[str, int | str]] = []
        answer_rows: list[dict[str, float | int]] = []
        row_id = 1
        for _, row in frame.iterrows():
            image_id = int(row[self.image_id_column])
            for feature_name in keypoint_columns:
                value = row[feature_name]
                if pd.isna(value):
                    continue
                lookup_rows.append(
                    {
                        self.submission_id_column: row_id,
                        self.image_id_column: image_id,
                        "FeatureName": feature_name,
                    }
                )
                answer_rows.append(
                    {
                        self.submission_id_column: row_id,
                        self.target_column: float(value),
                    }
                )
                row_id += 1
        return pd.DataFrame(lookup_rows), pd.DataFrame(answer_rows)

    def _baseline_submission(self, train_df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
        feature_columns = [column for column in train_df.columns if column not in {self.image_id_column, self.image_column}]
        feature_means = train_df[feature_columns].mean(numeric_only=True)
        submission = lookup_df[[self.submission_id_column]].copy()
        submission[self.target_column] = [
            float(feature_means.get(feature_name, 0.0)) for feature_name in lookup_df["FeatureName"]
        ]
        return submission

    def _run_keypoint_split(
        self,
        workspace_root: Path,
        hidden_root: Path,
        eval_df: pd.DataFrame,
        lookup_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame,
        answers_df: pd.DataFrame,
        split_name: str,
        python_executable: str,
        timeout_seconds: int,
    ) -> SplitExecution:
        tmp_root = _mk_eval_tmp_root(hidden_root, split_name)
        eval_path = tmp_root / f"{split_name}_input.csv"
        lookup_path = tmp_root / f"{split_name}_lookup.csv"
        sample_path = tmp_root / f"{split_name}_sample_submission.csv"
        output_path = tmp_root / f"{split_name}_submission.csv"
        eval_df.to_csv(eval_path, index=False)
        lookup_df.to_csv(lookup_path, index=False)
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
                    "--lookup",
                    str(lookup_path),
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
        expected_columns = [self.submission_id_column, self.target_column]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        submission_df = submission_df.sort_values(self.submission_id_column).reset_index(drop=True)
        answers_df = answers_df.sort_values(self.submission_id_column).reset_index(drop=True)
        if not submission_df[self.submission_id_column].equals(answers_df[self.submission_id_column]):
            raise ValueError("Submission ids did not align with answers.")

        pred = pd.to_numeric(submission_df[self.target_column], errors="raise").astype(float)
        true = pd.to_numeric(answers_df[self.target_column], errors="raise").astype(float)
        return _rmse_score(true, pred)

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for keypoint prediction."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable pixel parsing and feature extraction helpers."),
            ("src/models.py", "Editable keypoint prediction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]
