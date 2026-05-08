"""Active repo_workspace task definition.

Flattened from tasks/mle_bench.py so task preparation logic is local to this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..task_common import *  # noqa: F401,F403


@dataclass
class MleBenchDenoisingDirtyDocumentsMultifileTask(MleBenchPreparedTask):
    id_column: str = "id"
    target_column: str = "value"
    image_column: str = "image_id"
    pixel_stride: int = 8

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sampleSubmission.csv"

    @property
    def private_answers_path(self) -> Path:
        return self.private_root / "answers.csv"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_root / "train",
            self.public_root / "train_cleaned",
            self.public_root / "test",
            self.public_sample_submission_path,
            self.private_answers_path,
        )

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "rmse"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for grayscale document denoising."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/image_features.py", "Editable image loading helpers."),
            ("src/models.py", "Editable denoising-model helpers."),
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

        train_ids = train_df[self.image_column].astype(str).tolist()
        public_ids = public_features[self.image_column].astype(str).tolist()
        self._link_image_subset(train_ids, self.public_root / "train", data_dir / "train_dirty")
        self._link_image_subset(train_ids, self.public_root / "train_cleaned", data_dir / "train_clean")
        self._link_image_subset(public_ids, self.public_root / "test", data_dir / "public_eval_dirty")
        self._write_sparse_clean_subset(public_answers, self.public_root / "test", data_dir / "public_eval_clean")

    def _link_image_subset(self, image_ids: list[str], source_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            _link_or_copy_file(source_dir / f"{image_id}.png", output_dir / f"{image_id}.png")

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_ids = sorted(path.stem for path in (self.public_root / "train").glob("*.png"))
        public_sample_submission_full = self._filter_sampled_pixels(
            pd.read_csv(self.public_sample_submission_path)[[self.id_column, self.target_column]]
        )
        private_answers_full = self._filter_sampled_pixels(
            pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]]
        )
        all_test_image_ids = sorted(self._extract_image_ids(public_sample_submission_full[self.id_column]))
        val_ids, _ = _split_ids_evenly_by_value(pd.Series(all_test_image_ids, dtype=object))

        train_df = pd.DataFrame(
            {
                self.image_column: train_ids,
                "dirty_path": [f"data/train_dirty/{image_id}.png" for image_id in train_ids],
                "clean_path": [f"data/train_clean/{image_id}.png" for image_id in train_ids],
            }
        )
        val_features = pd.DataFrame(
            {
                self.image_column: val_ids,
                "dirty_path": [f"data/public_eval_dirty/{image_id}.png" for image_id in val_ids],
            }
        )
        public_answers = self._subset_pixel_frame_by_image_ids(private_answers_full, val_ids)
        public_sample_submission = self._subset_pixel_frame_by_image_ids(public_sample_submission_full, val_ids)
        return train_df, val_features, public_sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_answers_full = self._filter_sampled_pixels(pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]])
        private_sample_submission_full = self._filter_sampled_pixels(
            pd.read_csv(self.public_sample_submission_path)[[self.id_column, self.target_column]]
        )
        all_test_image_ids = sorted(self._extract_image_ids(private_answers_full[self.id_column]))
        _, private_image_ids = _split_ids_evenly_by_value(pd.Series(all_test_image_ids, dtype=object))
        private_answers = self._subset_pixel_frame_by_image_ids(private_answers_full, private_image_ids)
        private_sample_submission = self._subset_pixel_frame_by_image_ids(private_sample_submission_full, private_image_ids)
        private_features = pd.DataFrame(
            {
                self.image_column: private_image_ids,
                "dirty_path": [str((self.public_root / "test" / f"{image_id}.png").resolve()) for image_id in private_image_ids],
            }
        )
        return private_features, private_sample_submission, private_answers

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = public_features.copy()
        frame["clean_path"] = frame[self.image_column].astype(str).map(
            lambda image_id: f"data/public_eval_clean/{image_id}.png"
        )
        return frame

    def _build_answers_from_clean_images(self, clean_dir: Path, image_ids: list[str]) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for image_id in image_ids:
            image = np.asarray(Image.open(clean_dir / f"{image_id}.png").convert("L"), dtype=np.float32) / 255.0
            height, width = image.shape
            for row in range(0, height, self.pixel_stride):
                for col in range(0, width, self.pixel_stride):
                    rows.append(
                        {
                            self.id_column: f"{image_id}_{row + 1}_{col + 1}",
                            self.target_column: float(image[row, col]),
                        }
                    )
        return pd.DataFrame(rows)

    def _filter_sampled_pixels(self, frame: pd.DataFrame) -> pd.DataFrame:
        split_ids = frame[self.id_column].astype(str).str.rsplit("_", n=2, expand=True)
        row_idx = pd.to_numeric(split_ids[1], errors="raise").astype(int)
        col_idx = pd.to_numeric(split_ids[2], errors="raise").astype(int)
        mask = ((row_idx - 1) % self.pixel_stride == 0) & ((col_idx - 1) % self.pixel_stride == 0)
        return frame.loc[mask].reset_index(drop=True)

    def _extract_image_ids(self, pixel_ids: pd.Series) -> list[str]:
        split_ids = pixel_ids.astype(str).str.rsplit("_", n=2, expand=True)
        return split_ids[0].astype(str).drop_duplicates().tolist()

    def _subset_pixel_frame_by_image_ids(self, frame: pd.DataFrame, image_ids: list[str]) -> pd.DataFrame:
        image_set = set(str(image_id) for image_id in image_ids)
        split_ids = frame[self.id_column].astype(str).str.rsplit("_", n=2, expand=True)
        mask = split_ids[0].isin(image_set)
        return frame.loc[mask].reset_index(drop=True)

    def _write_sparse_clean_subset(self, answers_df: pd.DataFrame, source_dir: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        grouped = answers_df.groupby(answers_df[self.id_column].astype(str).str.rsplit("_", n=2).str[0], sort=False)
        for image_id, group in grouped:
            source_image = np.asarray(Image.open(source_dir / f"{image_id}.png").convert("L"), dtype=np.uint8)
            canvas = np.zeros_like(source_image, dtype=np.uint8)
            split_ids = group[self.id_column].astype(str).str.rsplit("_", n=2, expand=True)
            row_idx = pd.to_numeric(split_ids[1], errors="raise").astype(int) - 1
            col_idx = pd.to_numeric(split_ids[2], errors="raise").astype(int) - 1
            values = (pd.to_numeric(group[self.target_column], errors="raise").astype(float).clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
            canvas[row_idx.to_numpy(), col_idx.to_numpy()] = values.to_numpy()
            Image.fromarray(canvas, mode="L").save(output_dir / f"{image_id}.png")

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

        pred = pd.to_numeric(submission_df[self.target_column], errors="raise").astype(float)
        true = pd.to_numeric(answers_df[self.target_column], errors="raise").astype(float)
        return _rmse_score(true, pred)
