"""MLE-Bench-backed and demo bounded task definitions."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from subprocess import TimeoutExpired, run
from shutil import copy2
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, log_loss, r2_score, roc_auc_score, root_mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


def _build_python_env(workspace_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    workspace_text = str(workspace_root)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = workspace_text if not existing else f"{workspace_text}{os.pathsep}{existing}"
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _resolve_competition_root(
    data_root: Path,
    competition_id: str,
    fallback_root: Path | None = None,
) -> Path:
    candidates = [
        data_root / competition_id,
        data_root / "mle-bench" / competition_id,
    ]
    if fallback_root is not None:
        candidates.extend(
            [
                fallback_root / competition_id,
                fallback_root / "mle-bench" / competition_id,
            ]
        )
    return _first_existing_path(*candidates)


def _split_frame_into_id_halves(frame: pd.DataFrame, id_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(
        id_column,
        kind="mergesort",
        key=lambda series: series.astype(str),
    ).reset_index(drop=True)
    midpoint = len(ordered) // 2
    return (
        ordered.iloc[:midpoint].reset_index(drop=True),
        ordered.iloc[midpoint:].reset_index(drop=True),
    )


def _subset_frame_by_ids(frame: pd.DataFrame, id_column: str, ordered_ids: pd.Series | list) -> pd.DataFrame:
    return pd.DataFrame({id_column: list(ordered_ids)}).merge(
        frame,
        on=id_column,
        how="left",
        validate="one_to_one",
    )


def _deterministic_half_split_ids(ids: list[object], *, seed: int) -> tuple[list[object], list[object]]:
    unique_ids = list(dict.fromkeys(ids))
    ranked_ids = sorted(
        unique_ids,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    midpoint = (len(ranked_ids) + 1) // 2
    return ranked_ids[:midpoint], ranked_ids[midpoint:]


def _subset_frame_by_ids(frame: pd.DataFrame, id_column: str, ids: list[object]) -> pd.DataFrame:
    id_set = set(ids)
    return frame[frame[id_column].isin(id_set)].copy().reset_index(drop=True)


def _split_aligned_test_pool(
    features_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    *,
    id_column: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_ids = list(features_df[id_column])
    answer_ids = list(answers_df[id_column])
    if set(feature_ids) != set(answer_ids):
        raise ValueError(f"Feature ids and answer ids did not align for {id_column}.")

    public_ids, private_ids = _deterministic_half_split_ids(answer_ids, seed=seed)
    public_features = _subset_frame_by_ids(features_df, id_column, public_ids)
    private_features = _subset_frame_by_ids(features_df, id_column, private_ids)
    public_answers = _subset_frame_by_ids(answers_df, id_column, public_ids)
    private_answers = _subset_frame_by_ids(answers_df, id_column, private_ids)
    return public_features, private_features, public_answers, private_answers


def _build_constant_submission_frame(
    ids: pd.Series | list[object],
    *,
    id_column: str,
    column_defaults: dict[str, object],
) -> pd.DataFrame:
    submission = pd.DataFrame({id_column: list(ids)})
    for column, default_value in column_defaults.items():
        submission[column] = default_value
    return submission


def _link_or_copy_file(source: Path, destination: Path) -> None:
    _ensure_parent(destination)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        copy2(source, destination)


def _link_or_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    for root, _, files in os.walk(source):
        root_path = Path(root)
        relative_root = root_path.relative_to(source)
        destination_root = destination / relative_root
        destination_root.mkdir(parents=True, exist_ok=True)
        for filename in files:
            _link_or_copy_file(root_path / filename, destination_root / filename)


def _extract_zip_to_cache(zip_path: Path, cache_dir: Path) -> Path:
    marker = cache_dir / ".complete"
    if marker.exists():
        has_payload = any(path.name != ".complete" for path in cache_dir.iterdir())
        if has_payload:
            return cache_dir
        marker.unlink()

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = cache_dir.parent / f".{cache_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(tmp_dir)

    (tmp_dir / ".complete").write_text("ok\n")
    os.replace(tmp_dir, cache_dir)
    return cache_dir


def _encode_binary_mask_rle(mask: np.ndarray) -> str:
    flat_mask = np.asarray(mask, dtype=np.uint8).T.reshape(-1)
    if flat_mask.size == 0:
        return ""
    padded = np.concatenate(([0], flat_mask, [0]))
    runs = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(value)) for value in runs)


def _decode_binary_mask_rle(rle_text: str, *, height: int, width: int) -> np.ndarray:
    encoded = str(rle_text).strip()
    flat_mask = np.zeros(height * width, dtype=np.uint8)
    if encoded == "" or encoded.lower() == "nan":
        return flat_mask.reshape((width, height)).T.astype(bool)

    values = [int(part) for part in encoded.split()]
    if len(values) % 2 != 0:
        raise ValueError(f"Invalid RLE string: {rle_text}")
    for start, length in zip(values[::2], values[1::2]):
        start_index = start - 1
        flat_mask[start_index : start_index + length] = 1
    return flat_mask.reshape((width, height)).T.astype(bool)


def _dice_coefficient(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred_binary = np.asarray(pred_mask, dtype=bool)
    true_binary = np.asarray(true_mask, dtype=bool)
    denominator = int(pred_binary.sum() + true_binary.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(pred_binary, true_binary).sum())
    return float((2.0 * intersection) / denominator)


def _dice_coefficient_with_empty_value(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    *,
    both_empty_value: float = np.nan,
) -> float:
    pred_binary = np.asarray(pred_mask, dtype=bool)
    true_binary = np.asarray(true_mask, dtype=bool)
    denominator = int(pred_binary.sum() + true_binary.sum())
    if denominator == 0:
        return float(both_empty_value)
    intersection = int(np.logical_and(pred_binary, true_binary).sum())
    return float((2.0 * intersection) / denominator)


def _intersection_over_union(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred_binary = np.asarray(pred_mask, dtype=bool)
    true_binary = np.asarray(true_mask, dtype=bool)
    union = int(np.logical_or(pred_binary, true_binary).sum())
    if union == 0:
        return 1.0
    intersection = int(np.logical_and(pred_binary, true_binary).sum())
    return float(intersection / union)


def _tgs_mean_average_precision(
    submission_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    *,
    image_id_column: str,
    target_column: str,
    default_height: int = 101,
    default_width: int = 101,
) -> float:
    expected_columns = [image_id_column, target_column]
    if submission_df.columns.tolist() != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
    if len(submission_df) != len(answers_df):
        raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

    submission_df = submission_df.sort_values(image_id_column).reset_index(drop=True)
    answers_df = answers_df.sort_values(image_id_column).reset_index(drop=True)
    if not submission_df[image_id_column].equals(answers_df[image_id_column]):
        raise ValueError("Submission ids did not align with answers.")

    thresholds = np.arange(0.50, 1.00, 0.05, dtype=float)
    scores: list[float] = []
    for submission_row, answer_row in zip(submission_df.itertuples(index=False), answers_df.itertuples(index=False)):
        height = int(getattr(answer_row, "height", default_height))
        width = int(getattr(answer_row, "width", default_width))
        pred_mask = _decode_binary_mask_rle(getattr(submission_row, target_column), height=height, width=width)
        true_mask = _decode_binary_mask_rle(getattr(answer_row, target_column), height=height, width=width)
        iou = _intersection_over_union(pred_mask, true_mask)
        scores.append(float(np.mean(iou > thresholds)))
    return float(np.mean(scores))


def _uw_group_masks_by_day(masks: list[np.ndarray], slice_ids: list[str]) -> list[np.ndarray]:
    slice_to_masks: dict[str, list[np.ndarray]] = {}
    ordered_slices: list[str] = []
    for mask, slice_id in sorted(zip(masks, slice_ids), key=lambda item: item[1]):
        if slice_id not in slice_to_masks:
            slice_to_masks[slice_id] = []
            ordered_slices.append(slice_id)
        slice_to_masks[slice_id].append(mask)

    day_to_masks: dict[str, list[np.ndarray]] = {}
    ordered_days: list[str] = []
    for slice_id in ordered_slices:
        case_day = slice_id.split("_slice_")[0]
        if case_day not in day_to_masks:
            day_to_masks[case_day] = []
            ordered_days.append(case_day)
        joined_mask = np.logical_or.reduce(slice_to_masks[slice_id]).astype(np.uint8)
        day_to_masks[case_day].append(joined_mask)

    return [np.stack(day_to_masks[case_day], axis=0) for case_day in ordered_days]


def _uw_hausdorff_distance(predicted_mask: np.ndarray, true_mask: np.ndarray) -> float:
    unit_cube_diagonal = float(np.sqrt(3.0))
    if int(predicted_mask.sum()) == 0 and int(true_mask.sum()) == 0:
        return float(np.nan)
    if np.array_equal(predicted_mask, true_mask):
        return 0.0
    if (int(predicted_mask.sum()) == 0) != (int(true_mask.sum()) == 0):
        return 1.0
    if int(predicted_mask.sum()) > (10 * int(true_mask.sum())):
        return 1.0

    true_coordinates = np.argwhere(true_mask) / np.asarray(true_mask.shape, dtype=float)
    predicted_coordinates = np.argwhere(predicted_mask) / np.asarray(predicted_mask.shape, dtype=float)
    forward_distance = directed_hausdorff(true_coordinates, predicted_coordinates)[0]
    backward_distance = directed_hausdorff(predicted_coordinates, true_coordinates)[0]
    return float(max(forward_distance, backward_distance) / unit_cube_diagonal)


def _uw_dice_hausdorff_score(submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
    expected_columns = ["id", "class", "predicted"]
    if submission_df.columns.tolist() != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
    if len(submission_df) != len(answers_df):
        raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

    submission_df = submission_df.sort_values(["id", "class"]).reset_index(drop=True)
    answers_df = answers_df.sort_values(["id", "class"]).reset_index(drop=True)
    if not submission_df[["id", "class"]].equals(answers_df[["id", "class"]]):
        raise ValueError("Submission ids/classes did not align with answers.")
    if "image_width" not in answers_df.columns or "image_height" not in answers_df.columns:
        raise ValueError("Answers must include image_width and image_height.")

    predicted_masks: list[np.ndarray] = []
    true_masks: list[np.ndarray] = []
    for submission_row, answer_row in zip(submission_df.itertuples(index=False), answers_df.itertuples(index=False)):
        height = int(getattr(answer_row, "image_height"))
        width = int(getattr(answer_row, "image_width"))
        predicted_masks.append(_decode_binary_mask_rle(getattr(submission_row, "predicted"), height=height, width=width))
        true_masks.append(_decode_binary_mask_rle(getattr(answer_row, "predicted"), height=height, width=width))

    dice_score = float(
        np.nanmean(
            [
                _dice_coefficient_with_empty_value(predicted_mask, true_mask, both_empty_value=np.nan)
                for predicted_mask, true_mask in zip(predicted_masks, true_masks)
            ]
        )
    )

    predicted_by_day = _uw_group_masks_by_day(predicted_masks, submission_df["id"].astype(str).tolist())
    true_by_day = _uw_group_masks_by_day(true_masks, answers_df["id"].astype(str).tolist())
    hausdorff_distance = float(
        np.nanmean(
            [
                _uw_hausdorff_distance(predicted_day_mask, true_day_mask)
                for predicted_day_mask, true_day_mask in zip(predicted_by_day, true_by_day)
            ]
        )
    )

    return float((0.4 * dice_score) + (0.6 * (1.0 - hausdorff_distance)))


def _resolve_image_asset(
    source_dir: Path,
    image_id: str,
    suffix_candidates: tuple[str, ...] = ("", ".jpg", ".png", ".jpeg"),
) -> Path:
    candidates: list[Path] = []
    image_text = str(image_id)
    for suffix in suffix_candidates:
        if suffix and image_text.endswith(suffix):
            candidate = source_dir / image_text
        elif suffix:
            candidate = source_dir / f"{image_text}{suffix}"
        else:
            candidate = source_dir / image_text
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing image asset for {image_id}: tried {candidates}")


def _coerce_binary_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) >= 0.5).astype(int)

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
    }
    if not normalized.isin(mapping).all():
        invalid = sorted(normalized[~normalized.isin(mapping)].unique().tolist())
        raise ValueError(f"Could not coerce binary predictions: {invalid[:5]}")
    return normalized.map(mapping).astype(int)


def _rmse_score(true_values: pd.Series, pred_values: pd.Series) -> float:
    errors = true_values.astype(float).to_numpy() - pred_values.astype(float).to_numpy()
    return float((errors**2).mean() ** 0.5)


def _split_ids_evenly_by_value(ids: pd.Series) -> tuple[list[object], list[object]]:
    ordered_ids = pd.Index(ids)
    if ordered_ids.has_duplicates:
        duplicates = ordered_ids[ordered_ids.duplicated()].astype(str).tolist()
        raise ValueError(f"Expected unique ids for eval split, found duplicates like {duplicates[:3]}")

    stable_sorted = sorted(ordered_ids.tolist(), key=lambda value: str(value))
    public_ids, private_ids = np.array_split(np.asarray(stable_sorted, dtype=object), 2)
    return list(public_ids.tolist()), list(private_ids.tolist())


def _subset_frame_by_ids(frame: pd.DataFrame, id_column: str, ids: list[object]) -> pd.DataFrame:
    order_map = {value: index for index, value in enumerate(ids)}
    subset = frame[frame[id_column].isin(order_map)].copy()
    if len(subset) != len(ids):
        missing = [str(value) for value in ids if value not in set(subset[id_column].tolist())]
        raise ValueError(f"Could not find all requested ids for {id_column}: {missing[:3]}")
    subset["__order"] = subset[id_column].map(order_map)
    subset = subset.sort_values("__order", kind="mergesort").drop(columns="__order").reset_index(drop=True)
    return subset


def _normalized_bbox_landmark_error(
    submission_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    *,
    image_id_column: str,
    keypoint_columns: tuple[str, ...],
    bbox_width_column: str = "bbox_width",
    bbox_height_column: str = "bbox_height",
) -> float:
    expected_columns = [image_id_column, *keypoint_columns]
    if submission_df.columns.tolist() != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
    if len(submission_df) != len(answers_df):
        raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

    submission_df = submission_df.sort_values(image_id_column).reset_index(drop=True)
    answers_df = answers_df.sort_values(image_id_column).reset_index(drop=True)
    if not submission_df[image_id_column].equals(answers_df[image_id_column]):
        raise ValueError("Submission ids did not align with answers.")

    predicted = submission_df.loc[:, keypoint_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    truth = answers_df.loc[:, keypoint_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    predicted = predicted.reshape(len(submission_df), len(keypoint_columns) // 2, 2)
    truth = truth.reshape(len(answers_df), len(keypoint_columns) // 2, 2)

    bbox_width = pd.to_numeric(answers_df[bbox_width_column], errors="raise").to_numpy(dtype=float)
    bbox_height = pd.to_numeric(answers_df[bbox_height_column], errors="raise").to_numpy(dtype=float)
    normalization = np.sqrt(bbox_width * bbox_height)
    if (normalization <= 0).any():
        raise ValueError("Bounding-box normalization factors must be positive.")

    per_landmark = np.linalg.norm(predicted - truth, axis=2)
    per_image = per_landmark.mean(axis=1) / normalization
    return float(per_image.mean())


def _kuzushiji_score_page(prediction_text: object, truth_text: object) -> dict[str, int]:
    tp = 0
    fp = 0
    fn = 0

    if pd.isna(truth_text) and pd.isna(prediction_text):
        return {"tp": tp, "fp": fp, "fn": fn}
    if pd.isna(truth_text):
        fp += len(str(prediction_text).split()) // 3
        return {"tp": tp, "fp": fp, "fn": fn}
    if pd.isna(prediction_text):
        fn += len(str(truth_text).split()) // 5
        return {"tp": tp, "fp": fp, "fn": fn}

    truth_parts = str(truth_text).split()
    if len(truth_parts) % 5 != 0:
        raise ValueError(f"Malformed truth string: {truth_text}")
    pred_parts = str(prediction_text).split()
    if len(pred_parts) % 3 != 0:
        raise ValueError(f"Malformed prediction string: {prediction_text}")

    truth_labels = np.asarray(truth_parts[0::5], dtype=object)
    truth_xmin = np.asarray(truth_parts[1::5], dtype=float)
    truth_ymin = np.asarray(truth_parts[2::5], dtype=float)
    truth_xmax = truth_xmin + np.asarray(truth_parts[3::5], dtype=float)
    truth_ymax = truth_ymin + np.asarray(truth_parts[4::5], dtype=float)

    pred_labels = np.asarray(pred_parts[0::3], dtype=object)
    pred_x = np.asarray(pred_parts[1::3], dtype=float)
    pred_y = np.asarray(pred_parts[2::3], dtype=float)
    pred_unused = np.ones(len(pred_labels), dtype=bool)

    for xmin, xmax, ymin, ymax, label in zip(truth_xmin, truth_xmax, truth_ymin, truth_ymax, truth_labels):
        matching = (
            (xmin < pred_x)
            & (pred_x < xmax)
            & (ymin < pred_y)
            & (pred_y < ymax)
            & (pred_labels == label)
            & pred_unused
        )
        if not matching.any():
            fn += 1
            continue
        tp += 1
        pred_unused[np.argmax(matching)] = False

    fp += int(pred_unused.sum())
    return {"tp": tp, "fp": fp, "fn": fn}


def _kuzushiji_f1(submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
    if not submission_df["image_id"].equals(answers_df["image_id"]):
        raise ValueError("Submission image ids did not align with answers.")

    counts = [_kuzushiji_score_page(pred, truth) for pred, truth in zip(submission_df["labels"], answers_df["labels"])]
    tp = sum(item["tp"] for item in counts)
    fp = sum(item["fp"] for item in counts)
    fn = sum(item["fn"] for item in counts)
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def _read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix == ".zip":
        return pd.read_csv(path, compression="zip", **kwargs)
    return pd.read_csv(path, **kwargs)


def _mk_eval_tmp_root(hidden_root: Path, split_name: str) -> Path:
    hidden_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{split_name}_eval_", dir=hidden_root))


@dataclass
class SplitExecution:
    success: bool
    score: float | None
    stdout: str
    stderr: str
    return_code: int
    artifact_dir: str | None = None
    input_path: str | None = None
    output_path: str | None = None


@dataclass
class TaskEvaluation:
    metric_name: str
    public_run: SplitExecution
    private_run: SplitExecution
    lower_is_better: bool = False

    @property
    def public_score(self) -> float | None:
        return self.public_run.score

    @property
    def private_score(self) -> float | None:
        return self.private_run.score

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "lower_is_better": self.lower_is_better,
            "public_run": asdict(self.public_run),
            "private_run": asdict(self.private_run),
        }


@dataclass
class CsvClassificationTask:
    name: str
    template_dir: Path
    train_path: Path
    public_eval_path: Path
    private_eval_path: Path
    metric_name: str = "accuracy"
    label_column: str = "label"
    prediction_column: str = "prediction"
    pipeline_entrypoint: str = "src/pipeline.py"
    protected_paths: tuple[str, ...] = ("data/train.csv", "data/public_eval.csv")
    locked_paths: tuple[str, ...] = ()
    lower_is_better: bool = False

    @property
    def perfect_metric(self) -> float | None:
        return 0.0 if self.lower_is_better else 1.0

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and task layout."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Main editable training and prediction pipeline."),
            ("src/text_features.py", "Text preprocessing helpers for experimentation."),
        ]

    def materialize_workspace(self, workspace_root: Path, eval_access: str = "metric_only") -> None:
        data_dir = workspace_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_df = pd.read_csv(self.train_path)
        public_source_df = pd.read_csv(self.public_eval_path)
        public_df = (
            public_source_df
            if eval_access == "full"
            else public_source_df.drop(columns=[self.label_column])
        )

        train_df.to_csv(data_dir / "train.csv", index=False)
        public_df.to_csv(data_dir / "public_eval.csv", index=False)

    def evaluate_workspace(
        self,
        workspace_root: Path,
        hidden_root: Path,
        python_executable: str,
        timeout_seconds: int = 180,
        eval_access: str = "metric_only",
    ) -> TaskEvaluation:
        public_source_df = pd.read_csv(self.public_eval_path)
        public_labels = public_source_df[self.label_column].tolist()
        public_eval_df = (
            public_source_df
            if eval_access == "full"
            else public_source_df.drop(columns=[self.label_column])
        )
        private_source_df = pd.read_csv(self.private_eval_path)
        private_labels = private_source_df[self.label_column].tolist()
        private_eval_df = private_source_df.drop(columns=[self.label_column])

        public_run = self._run_split(
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            eval_df=public_eval_df,
            labels=public_labels,
            split_name="public",
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
        private_run = self._run_split(
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            eval_df=private_eval_df,
            labels=private_labels,
            split_name="private",
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
        return TaskEvaluation(
            metric_name=self.metric_name,
            public_run=public_run,
            private_run=private_run,
            lower_is_better=self.lower_is_better,
        )

    def _run_split(
        self,
        workspace_root: Path,
        hidden_root: Path,
        eval_df: pd.DataFrame,
        labels: list,
        split_name: str,
        python_executable: str,
        timeout_seconds: int,
    ) -> SplitExecution:
        tmp_root = _mk_eval_tmp_root(hidden_root, split_name)
        eval_path = tmp_root / f"{split_name}_input.csv"
        pred_path = tmp_root / f"{split_name}_predictions.csv"
        eval_df.to_csv(eval_path, index=False)

        artifact_dir = str(tmp_root)
        input_path = str(eval_path)
        output_path = str(pred_path)

        try:
            result = run(
                [
                    python_executable,
                    str(workspace_root / self.pipeline_entrypoint),
                    "--train",
                    str(workspace_root / "data" / "train.csv"),
                    "--eval",
                    str(eval_path),
                    "--output",
                    str(pred_path),
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
                output_path=output_path,
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
                output_path=output_path,
            )

        if not pred_path.exists():
            return SplitExecution(
                success=False,
                score=None,
                stdout=result.stdout,
                stderr="Predictions file was not created.",
                return_code=result.returncode,
                artifact_dir=artifact_dir,
                input_path=input_path,
                output_path=output_path,
            )

        pred_df = pd.read_csv(pred_path)
        if self.prediction_column in pred_df.columns:
            predictions = pred_df[self.prediction_column].tolist()
        elif len(pred_df.columns) == 1:
            predictions = pred_df.iloc[:, 0].tolist()
        else:
            return SplitExecution(
                success=False,
                score=None,
                stdout=result.stdout,
                stderr="Predictions file did not contain a usable column.",
                return_code=result.returncode,
                artifact_dir=artifact_dir,
                input_path=input_path,
                output_path=output_path,
            )

        if len(predictions) != len(labels):
            return SplitExecution(
                success=False,
                score=None,
                stdout=result.stdout,
                stderr=(
                    f"Prediction length mismatch: expected {len(labels)}, got {len(predictions)}."
                ),
                return_code=result.returncode,
                artifact_dir=artifact_dir,
                input_path=input_path,
                output_path=output_path,
            )

        correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
        score = correct / len(labels) if labels else 0.0
        return SplitExecution(
            success=True,
            score=score,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            artifact_dir=artifact_dir,
            input_path=input_path,
            output_path=output_path,
        )


@dataclass
class MleBenchPreparedTask:
    name: str
    template_dir: Path
    mle_bench_data_root: Path
    competition_id: str
    mle_bench_fallback_data_root: Path | None = None
    pipeline_entrypoint: str = "src/pipeline.py"
    public_split_seed: int = 7
    public_split_size: float = 0.2
    protected_paths: tuple[str, ...] = ("data/train.csv", "data/public_eval.csv", "data/sample_submission.csv")
    locked_paths: tuple[str, ...] = ()

    @property
    def perfect_metric(self) -> float | None:
        return 0.0 if self.lower_is_better else 1.0

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and task-specific submission format."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Main editable training and submission pipeline."),
            ("src/features.py", "Task helpers that the model can rewrite or extend."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]

    @property
    def competition_root(self) -> Path:
        return _resolve_competition_root(
            self.mle_bench_data_root,
            self.competition_id,
            fallback_root=self.mle_bench_fallback_data_root,
        )

    @property
    def prepared_root(self) -> Path:
        return self.competition_root / "prepared"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_sample_submission_path,
            self.private_answers_path,
        )

    @property
    def lower_is_better(self) -> bool:
        raise NotImplementedError

    @property
    def public_root(self) -> Path:
        return self.prepared_root / "public"

    @property
    def private_root(self) -> Path:
        return self.prepared_root / "private"

    @property
    def public_train_path(self) -> Path:
        return self.public_root / "train.csv"

    @property
    def public_test_path(self) -> Path:
        return _first_existing_path(self.public_root / "test.csv", self.public_root / "public_eval.csv")

    @property
    def private_answers_path(self) -> Path:
        return self.private_root / "test.csv"

    @property
    def public_sample_submission_path(self) -> Path:
        raise NotImplementedError

    def _ensure_prepared_available(self) -> None:
        missing = [path for path in self.required_prepared_paths if not path.exists()]
        if not missing:
            return
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Prepared MLE-Bench assets are missing for "
            f"{self.competition_id}: {missing_text}"
        )

    def materialize_workspace(self, workspace_root: Path, eval_access: str = "metric_only") -> None:
        self._ensure_prepared_available()
        data_dir = workspace_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_df, public_features, public_sample_submission, public_answers = self._build_public_assets()
        public_eval_df = self._public_eval_frame(public_features, public_answers, eval_access)
        train_df.to_csv(data_dir / "train.csv", index=False)
        public_eval_df.to_csv(data_dir / "public_eval.csv", index=False)
        public_sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

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
        _ = train_df  # ensures the split stays deterministic
        private_features, private_sample_submission, private_answers = self._build_private_assets()
        public_eval_df = self._public_eval_frame(public_features, public_answers, eval_access)

        public_run = self._run_submission_split(
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            eval_df=public_eval_df,
            sample_submission_df=public_sample_submission,
            answers_df=public_answers,
            split_name="public",
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
        if public_run.success:
            private_run = self._run_submission_split(
                workspace_root=workspace_root,
                hidden_root=hidden_root,
                eval_df=private_features,
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

    def _public_eval_frame(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
        eval_access: str,
    ) -> pd.DataFrame:
        if eval_access == "full":
            return self._public_eval_with_labels(public_features, public_answers)
        return public_features

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        return public_features.merge(public_answers, on="id", how="left", validate="one_to_one")

    def _run_submission_split(
        self,
        workspace_root: Path,
        hidden_root: Path,
        eval_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame,
        answers_df: pd.DataFrame,
        split_name: str,
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

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        raise NotImplementedError


@dataclass
class MleBenchPreparedBinaryClassificationTask(MleBenchPreparedTask):
    id_column: str = "id"
    label_column: str = "label"
    score_kind: str = "accuracy"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return self.score_kind

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
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)

        majority_label = bool(_coerce_binary_series(train_df[self.label_column]).mode(dropna=False).iloc[0])
        sample_submission = _build_constant_submission_frame(
            val_features[self.id_column],
            id_column=self.id_column,
            column_defaults={self.label_column: majority_label},
        )
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].copy()
        _, private_ids = _split_ids_evenly_by_value(test_features[self.id_column])

        private_features = _subset_frame_by_ids(test_features, self.id_column, private_ids)
        majority_label = bool(_coerce_binary_series(full_train[self.label_column]).mode(dropna=False).iloc[0])
        private_sample_submission = _build_constant_submission_frame(
            private_features[self.id_column],
            id_column=self.id_column,
            column_defaults={self.label_column: majority_label},
        )
        private_answers = _subset_frame_by_ids(answers_full, self.id_column, private_ids)
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

        true_labels = _coerce_binary_series(merged[f"{self.label_column}_true"])
        pred_labels = _coerce_binary_series(merged[f"{self.label_column}_pred"])
        if self.score_kind == "accuracy":
            return float((true_labels == pred_labels).mean())
        if self.score_kind == "f1":
            return float(f1_score(true_labels, pred_labels, average="binary"))
        raise ValueError(f"Unsupported binary score kind: {self.score_kind}")


@dataclass
class MleBenchPreparedBinaryProbabilityTask(MleBenchPreparedTask):
    id_column: str = "id"
    label_column: str = "label"
    probability_zero_column: str = "class_0"
    probability_one_column: str = "class_1"

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "balanced-binary-log-loss"

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

        pos_rate = float(pd.to_numeric(train_df[self.label_column], errors="raise").astype(int).mean())
        sample_submission = _build_constant_submission_frame(
            public_features[self.id_column],
            id_column=self.id_column,
            column_defaults={
                self.probability_zero_column: 1.0 - pos_rate,
                self.probability_one_column: pos_rate,
            },
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
        pos_rate = float(pd.to_numeric(full_train[self.label_column], errors="raise").astype(int).mean())
        private_sample_submission = _build_constant_submission_frame(
            private_features[self.id_column],
            id_column=self.id_column,
            column_defaults={
                self.probability_zero_column: 1.0 - pos_rate,
                self.probability_one_column: pos_rate,
            },
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
        expected_columns = [self.id_column, self.probability_zero_column, self.probability_one_column]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        merged = answers_df.merge(
            submission_df,
            on=self.id_column,
            how="left",
            validate="one_to_one",
        )
        if merged.isnull().any().any():
            raise ValueError("Submission ids did not align with answers.")

        p0 = np.clip(merged[self.probability_zero_column].astype(float).to_numpy(), 1e-15, 1 - 1e-15)
        p1 = np.clip(merged[self.probability_one_column].astype(float).to_numpy(), 1e-15, 1 - 1e-15)
        if not np.allclose(p0 + p1, 1.0, atol=1e-6):
            raise ValueError("Binary probability rows must sum to 1.")

        y_true = pd.to_numeric(merged[self.label_column], errors="raise").astype(int).to_numpy()
        neg_mask = y_true == 0
        pos_mask = y_true == 1
        neg_loss = -np.mean(np.log(p0[neg_mask])) if np.any(neg_mask) else 0.0
        pos_loss = -np.mean(np.log(p1[pos_mask])) if np.any(pos_mask) else 0.0
        return float((neg_loss + pos_loss) / 2.0)


@dataclass
class MleBenchPreparedMulticlassLabelTask(MleBenchPreparedTask):
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


@dataclass
class MleBenchPreparedMultiTargetRegressionTask(MleBenchPreparedTask):
    id_column: str = "id"
    target_columns: tuple[str, ...] = ("target",)

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "mcrmse"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
        )
        train_df = train_df.sort_values(self.id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.id_column).reset_index(drop=True)

        val_features = val_df.drop(columns=list(self.target_columns)).copy()
        public_answers = val_df[[self.id_column, *self.target_columns]].copy()

        sample_submission = pd.DataFrame({self.id_column: val_df[self.id_column]})
        for column in self.target_columns:
            sample_submission[column] = float(train_df[column].astype(float).mean())
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_features = pd.read_csv(self.public_test_path)
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)
        private_answers = pd.read_csv(self.private_answers_path)[[self.id_column, *self.target_columns]].copy()
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
        expected_columns = [self.id_column, *self.target_columns]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        submission_df = submission_df.sort_values(self.id_column).reset_index(drop=True)
        answers_df = answers_df.sort_values(self.id_column).reset_index(drop=True)
        if not submission_df[self.id_column].equals(answers_df[self.id_column]):
            raise ValueError("Submission ids did not align with answers.")

        y_true = answers_df[list(self.target_columns)].astype(float).to_numpy()
        y_pred = submission_df[list(self.target_columns)].astype(float).to_numpy()
        column_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
        return float(np.mean(column_rmse))


@dataclass
class MleBenchPreparedMulticlassProbabilityTask(MleBenchPreparedTask):
    id_column: str = "id"
    label_column: str = "label"
    class_columns: tuple[str, ...] = ()

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "multi-class-log-loss"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
            stratify=full_train[self.label_column],
        )
        train_df = train_df.sort_values(self.id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.id_column).reset_index(drop=True)

        val_features = val_df.drop(columns=[self.label_column]).copy()
        public_answers = val_df[[self.id_column, self.label_column]].copy()

        priors = train_df[self.label_column].astype(str).value_counts(normalize=True)
        sample_submission = pd.DataFrame({self.id_column: val_df[self.id_column]})
        for class_name in self.class_columns:
            sample_submission[class_name] = float(priors.get(class_name, 0.0))
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_features = pd.read_csv(self.public_test_path)
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)
        private_answers = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].copy()
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
        expected_columns = [self.id_column, *self.class_columns]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        submission_df = submission_df.sort_values(self.id_column).reset_index(drop=True)
        answers_df = answers_df.sort_values(self.id_column).reset_index(drop=True)
        if not submission_df[self.id_column].equals(answers_df[self.id_column]):
            raise ValueError("Submission ids did not align with answers.")

        probabilities = submission_df[list(self.class_columns)].astype(float)
        if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all().all():
            raise ValueError("Submission probabilities must be between 0 and 1.")
        if not probabilities.sum(axis=1).round(6).eq(1.0).all():
            raise ValueError("Each submission row must sum to 1.")
        return float(log_loss(answers_df[self.label_column].astype(str), probabilities.to_numpy(), labels=list(self.class_columns)))


@dataclass
class MleBenchPreparedSpaceSeparatedMultilabelTask(MleBenchPreparedTask):
    id_column: str = "id"
    target_column: str = "tags"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "micro-f1"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
        )
        train_df = train_df.sort_values(self.id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.id_column).reset_index(drop=True)

        val_features = val_df.drop(columns=[self.target_column]).copy()
        public_answers = val_df[[self.id_column, self.target_column]].copy()

        sample_submission = pd.DataFrame({self.id_column: val_df[self.id_column]})
        sample_submission[self.target_column] = ""
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_features = pd.read_csv(self.public_test_path)
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)
        private_answers = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
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
        expected_columns = [self.id_column, self.target_column]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        submission_df = submission_df.sort_values(self.id_column).reset_index(drop=True)
        answers_df = answers_df.sort_values(self.id_column).reset_index(drop=True)
        if not submission_df[self.id_column].equals(answers_df[self.id_column]):
            raise ValueError("Submission ids did not align with answers.")

        true_tags = answers_df[self.target_column].fillna("").astype(str).str.split()
        pred_tags = submission_df[self.target_column].fillna("").astype(str).str.split()
        classes = sorted(set(tag for row in true_tags for tag in row))
        mlb = MultiLabelBinarizer(classes=classes, sparse_output=False)
        y_true = mlb.fit_transform(true_tags)
        y_pred = mlb.transform(pred_tags)
        return float(f1_score(y_true, y_pred, average="micro"))


@dataclass
class MleBenchPreparedRegressionTask(MleBenchPreparedTask):
    id_column: str = "id"
    target_column: str = "target"
    submission_column: str | None = None
    score_kind: str = "rmse"

    @property
    def lower_is_better(self) -> bool:
        return self.score_kind != "r2"

    @property
    def metric_name(self) -> str:
        return self.score_kind

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def prediction_column(self) -> str:
        return self.submission_column or self.target_column

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)

        baseline_value = float(train_df[self.target_column].astype(float).mean())
        sample_submission = _build_constant_submission_frame(
            val_features[self.id_column],
            id_column=self.id_column,
            column_defaults={self.prediction_column: baseline_value},
        )
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
        _, private_ids = _split_ids_evenly_by_value(test_features[self.id_column])

        private_features = _subset_frame_by_ids(test_features, self.id_column, private_ids)
        baseline_value = float(full_train[self.target_column].astype(float).mean())
        private_sample_submission = _build_constant_submission_frame(
            private_features[self.id_column],
            id_column=self.id_column,
            column_defaults={self.prediction_column: baseline_value},
        )
        private_answers = _subset_frame_by_ids(answers_full, self.id_column, private_ids)
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

        merged = answers_df.merge(
            submission_df,
            on=self.id_column,
            how="left",
            suffixes=("_true", "_pred"),
            validate="one_to_one",
        )
        if merged.isnull().any().any():
            raise ValueError("Submission ids did not align with answers.")

        true_values = merged[f"{self.target_column}_true"].astype(float)
        pred_values = merged[f"{self.prediction_column}_pred"].astype(float)

        if self.score_kind == "rmse":
            return _rmse_score(true_values, pred_values)
        if self.score_kind == "r2":
            return float(r2_score(true_values, pred_values))
        raise ValueError(f"Unsupported regression score kind: {self.score_kind}")


@dataclass
class MleBenchSpookyAuthorTask(MleBenchPreparedTask):
    id_column: str = "id"
    class_columns: tuple[str, ...] = ("EAP", "HPL", "MWS")
    class_order: tuple[str, ...] = ("EAP", "HPL", "MWS")
    locked_paths: tuple[str, ...] = ("src/features.py",)

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "multi-class-log-loss"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)
        public_answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.class_columns]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)

        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])
        public_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(public_answers_full, self.id_column, public_ids)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        return train_df, public_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = pd.read_csv(self.public_test_path)
        private_answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.class_columns]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)

        _, private_ids = _split_ids_evenly_by_value(test_features[self.id_column])
        private_features = _subset_frame_by_ids(test_features, self.id_column, private_ids)
        private_sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, private_ids)
        private_answers = _subset_frame_by_ids(private_answers_full, self.id_column, private_ids)
        return private_features, private_sample_submission, private_answers

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        author_series = (
            public_answers.set_index("id")[list(self.class_order)]
            .idxmax(axis=1)
            .rename("author")
            .reset_index()
        )
        return public_features.merge(author_series, on="id", how="left", validate="one_to_one")

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = ["id", *self.class_order]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        merged = answers_df[["id", *self.class_order]].merge(
            submission_df[expected_columns],
            on="id",
            how="left",
            suffixes=("_true", "_pred"),
            validate="one_to_one",
        )
        if merged.isnull().any().any():
            raise ValueError("Submission ids did not align with answers.")

        pred_cols = [f"{label}_pred" for label in self.class_order]
        true_cols = [f"{label}_true" for label in self.class_order]
        probabilities = merged[pred_cols]
        if not ((probabilities >= 0) & (probabilities <= 1)).all().all():
            raise ValueError("Submission probabilities must be between 0 and 1.")
        if not probabilities.sum(axis=1).round(6).eq(1.0).all():
            raise ValueError("Each submission row must sum to 1.")

        true_labels = merged[true_cols].idxmax(axis=1).str.replace("_true", "", regex=False)
        return float(log_loss(true_labels, probabilities.to_numpy(), labels=list(self.class_order)))


@dataclass
class MleBenchSpookyAuthorMultifileTask(MleBenchSpookyAuthorTask):
    locked_paths: tuple[str, ...] = ()

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable text preprocessing and feature extraction module."),
            ("src/models.py", "Editable classifier and calibration helpers."),
            ("src/postprocess.py", "Editable label-order and submission formatting logic."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchNomadTask(MleBenchPreparedTask):
    id_column: str = "id"
    target_columns: tuple[str, ...] = ("formation_energy_ev_natom", "bandgap_energy_ev")

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_test_path,
            self.private_answers_path,
        )

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "mean-column-wise-rmsle"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)
        private_answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.target_columns]].copy()
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(private_answers_full, self.id_column, public_ids)

        means = train_df[list(self.target_columns)].mean()
        sample_submission = pd.DataFrame({self.id_column: val_features[self.id_column]})
        for column in self.target_columns:
            sample_submission[column] = float(means[column])

        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(self.public_train_path)
        test_features = pd.read_csv(self.public_test_path)
        private_answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.target_columns]].copy()
        _, private_ids = _split_ids_evenly_by_value(test_features[self.id_column])

        private_features = _subset_frame_by_ids(test_features, self.id_column, private_ids)
        means = train_df[list(self.target_columns)].mean()
        private_sample_submission = pd.DataFrame({self.id_column: private_features[self.id_column]})
        for column in self.target_columns:
            private_sample_submission[column] = float(means[column])
        private_answers = _subset_frame_by_ids(private_answers_full, self.id_column, private_ids)
        return private_features, private_sample_submission, private_answers

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = ["id", *self.target_columns]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        merged = answers_df[expected_columns].merge(
            submission_df[expected_columns],
            on="id",
            how="left",
            suffixes=("_true", "_pred"),
            validate="one_to_one",
        )
        if merged.isnull().any().any():
            raise ValueError("Submission ids did not align with answers.")

        scores = []
        for column in self.target_columns:
            true_values = merged[f"{column}_true"]
            pred_values = merged[f"{column}_pred"]
            scores.append(root_mean_squared_log_error(true_values, pred_values))
        return float(sum(scores) / len(scores))


@dataclass
class MleBenchNomadMultifileTask(MleBenchNomadTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable feature selection and numeric preprocessing helpers."),
            ("src/models.py", "Editable regression-model helpers."),
            ("src/postprocess.py", "Editable prediction clipping and submission formatting logic."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchDetectingInsultsTask(MleBenchPreparedTask):
    label_column: str = "Insult"
    id_columns: tuple[str, ...] = ("Date", "Comment")

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "auc-roc"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission_null.csv"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and submission schema for insult detection."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Editable text classification pipeline."),
            ("src/text_features.py", "Editable text preprocessing helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
            stratify=full_train[self.label_column],
        )
        sort_columns = list(self.id_columns)
        train_df = train_df.sort_values(sort_columns).reset_index(drop=True)
        val_df = val_df.sort_values(sort_columns).reset_index(drop=True)

        val_features = val_df[list(self.id_columns)].copy()
        public_answers = val_df[[*self.id_columns, self.label_column]].copy()

        positive_rate = float(train_df[self.label_column].astype(float).mean())
        sample_submission = public_answers.copy()
        sample_submission[self.label_column] = positive_rate
        sample_submission = sample_submission[[self.label_column, *self.id_columns]]

        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_df = pd.read_csv(self.private_answers_path)
        private_features = private_df[list(self.id_columns)].copy()
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)
        private_answers = private_df[[*self.id_columns, self.label_column]].copy()
        return private_features, private_sample_submission, private_answers

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        return public_features.merge(
            public_answers,
            on=list(self.id_columns),
            how="left",
            validate="one_to_one",
        )

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = [self.label_column, *self.id_columns]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        merged = answers_df.merge(
            submission_df,
            on=list(self.id_columns),
            how="left",
            suffixes=("_true", "_pred"),
            validate="one_to_one",
        )
        if merged.isnull().any().any():
            raise ValueError("Submission rows did not align with answers.")

        probabilities = merged[f"{self.label_column}_pred"].astype(float)
        if not probabilities.between(0.0, 1.0).all():
            raise ValueError("Submission probabilities must be between 0 and 1.")

        true_labels = merged[f"{self.label_column}_true"].astype(int)
        return float(roc_auc_score(true_labels, probabilities))


@dataclass
class MleBenchDetectingInsultsMultifileTask(MleBenchDetectingInsultsTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for insult detection."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/text_features.py", "Editable text preprocessing helpers."),
            ("src/models.py", "Editable text-model construction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchJigsawToxicTask(MleBenchPreparedTask):
    target_columns: tuple[str, ...] = (
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    )
    id_column: str = "id"
    text_column: str = "comment_text"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "mean-column-wise-auc-roc"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and submission schema for multi-label toxicity."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Editable text classification pipeline."),
            ("src/text_features.py", "Editable text preprocessing helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
        )
        train_df = train_df.sort_values(self.id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.id_column).reset_index(drop=True)

        val_features = val_df[[self.id_column, self.text_column]].copy()
        public_answers = val_df[[self.id_column, *self.target_columns]].copy()

        sample_submission = pd.DataFrame({self.id_column: val_df[self.id_column]})
        for column in self.target_columns:
            sample_submission[column] = float(train_df[column].astype(float).mean())

        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_features = pd.read_csv(self.public_test_path)
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)
        private_answers = pd.read_csv(self.private_answers_path)[[self.id_column, *self.target_columns]].copy()
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
        expected_columns = [self.id_column, *self.target_columns]
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

        true_df = merged[[f"{column}_true" for column in self.target_columns]].copy()
        pred_df = merged[[f"{column}_pred" for column in self.target_columns]].copy()
        pred_df.columns = list(self.target_columns)

        if not ((pred_df >= 0.0) & (pred_df <= 1.0)).all().all():
            raise ValueError("Submission probabilities must be between 0 and 1.")

        keep_mask = true_df.sum(axis=1) >= 0
        if keep_mask.sum() == 0:
            raise ValueError("No scored rows remained after filtering unlabeled private rows.")
        true_df = true_df[keep_mask]
        pred_df = pred_df[keep_mask]
        return float(roc_auc_score(true_df.to_numpy(), pred_df.to_numpy(), average="macro"))


@dataclass
class MleBenchJigsawToxicMultifileTask(MleBenchJigsawToxicTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for toxicity classification."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/text_features.py", "Editable text preprocessing helpers."),
            ("src/models.py", "Editable multi-label model-construction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchAerialCactusTask(MleBenchPreparedTask):
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


@dataclass
class MleBenchAerialCactusMultifileTask(MleBenchAerialCactusTask):
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


@dataclass
class MleBenchSpaceshipTitanicTask(MleBenchPreparedTask):
    id_column: str = "PassengerId"
    label_column: str = "Transported"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "accuracy"

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
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
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

        true_labels = _coerce_binary_series(merged[f"{self.label_column}_true"])
        pred_labels = _coerce_binary_series(merged[f"{self.label_column}_pred"])
        return float((true_labels == pred_labels).mean())


@dataclass
class MleBenchSpaceshipTitanicMultifileTask(MleBenchSpaceshipTitanicTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for tabular classification."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable preprocessing and feature engineering helpers."),
            ("src/models.py", "Editable classifier-construction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchTitanicTask(MleBenchPreparedBinaryClassificationTask):
    id_column: str = "PassengerId"
    label_column: str = "Survived"


@dataclass
class MleBenchTitanicMultifileTask(MleBenchTitanicTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for tabular classification."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable preprocessing and feature engineering helpers."),
            ("src/models.py", "Editable classifier-construction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchHousePricesTask(MleBenchPreparedRegressionTask):
    id_column: str = "Id"
    target_column: str = "SalePrice"
    score_kind: str = "rmse"


@dataclass
class MleBenchHousePricesMultifileTask(MleBenchHousePricesTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for tabular regression."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable preprocessing and feature engineering helpers."),
            ("src/models.py", "Editable regressor-construction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchSantanderValueTask(MleBenchPreparedRegressionTask):
    id_column: str = "ID"
    target_column: str = "target"
    score_kind: str = "rmse"


@dataclass
class MleBenchSantanderValueMultifileTask(MleBenchSantanderValueTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for tabular regression."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable preprocessing and feature engineering helpers."),
            ("src/models.py", "Editable regressor-construction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchMercedesBenzTask(MleBenchPreparedRegressionTask):
    id_column: str = "ID"
    target_column: str = "y"
    score_kind: str = "r2"


@dataclass
class MleBenchMercedesBenzMultifileTask(MleBenchMercedesBenzTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for tabular regression."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable preprocessing and feature engineering helpers."),
            ("src/models.py", "Editable regressor-construction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchRestaurantRevenueTask(MleBenchPreparedRegressionTask):
    id_column: str = "Id"
    target_column: str = "revenue"
    submission_column: str = "revenue"
    score_kind: str = "rmse"


@dataclass
class MleBenchRestaurantRevenueMultifileTask(MleBenchRestaurantRevenueTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for tabular regression."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable preprocessing and feature engineering helpers."),
            ("src/models.py", "Editable regressor-construction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchIcrAgeRelatedConditionsTask(MleBenchPreparedBinaryProbabilityTask):
    id_column: str = "Id"
    label_column: str = "Class"
    probability_zero_column: str = "class_0"
    probability_one_column: str = "class_1"


@dataclass
class MleBenchIcrAgeRelatedConditionsMultifileTask(MleBenchIcrAgeRelatedConditionsTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for balanced binary tabular prediction."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable tabular feature engineering helpers."),
            ("src/models.py", "Editable probability-model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchForestCoverTypeTask(MleBenchPreparedMulticlassLabelTask):
    id_column: str = "Id"
    label_column: str = "Cover_Type"


@dataclass
class MleBenchForestCoverTypeMultifileTask(MleBenchForestCoverTypeTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for multiclass tabular prediction."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable tabular feature engineering helpers."),
            ("src/models.py", "Editable multiclass model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchNlpGettingStartedTask(MleBenchPreparedBinaryClassificationTask):
    id_column: str = "id"
    label_column: str = "target"
    score_kind: str = "f1"


@dataclass
class MleBenchNlpGettingStartedMultifileTask(MleBenchNlpGettingStartedTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for binary text classification."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable text preprocessing and feature extraction helpers."),
            ("src/models.py", "Editable text classification helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchCrowdflowerSearchRelevanceTask(MleBenchPreparedMulticlassLabelTask):
    id_column: str = "id"
    label_column: str = "median_relevance"
    submission_column: str = "prediction"
    score_kind: str = "qwk"


@dataclass
class MleBenchCrowdflowerSearchRelevanceMultifileTask(MleBenchCrowdflowerSearchRelevanceTask):
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


@dataclass
class MleBenchCommonLitReadabilityTask(MleBenchPreparedRegressionTask):
    id_column: str = "id"
    target_column: str = "target"


@dataclass
class MleBenchCommonLitReadabilityMultifileTask(MleBenchCommonLitReadabilityTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for text regression."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable text feature extraction helpers."),
            ("src/models.py", "Editable text regression helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchFeedbackEnglishLanguageLearningTask(MleBenchPreparedMultiTargetRegressionTask):
    id_column: str = "text_id"
    target_columns: tuple[str, ...] = (
        "cohesion",
        "syntax",
        "vocabulary",
        "phraseology",
        "grammar",
        "conventions",
    )


@dataclass
class MleBenchFeedbackEnglishLanguageLearningMultifileTask(MleBenchFeedbackEnglishLanguageLearningTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for multi-target essay scoring."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable essay feature extraction helpers."),
            ("src/models.py", "Editable multi-target regression helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchFeedbackEffectivenessTask(MleBenchPreparedMulticlassProbabilityTask):
    id_column: str = "discourse_id"
    label_column: str = "discourse_effectiveness"
    class_columns: tuple[str, ...] = ("Ineffective", "Adequate", "Effective")


@dataclass
class MleBenchFeedbackEffectivenessMultifileTask(MleBenchFeedbackEffectivenessTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for discourse-level multiclass prediction."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable discourse-text feature extraction helpers."),
            ("src/models.py", "Editable multiclass probability-model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchTransferLearningStackExchangeTagsTask(MleBenchPreparedSpaceSeparatedMultilabelTask):
    id_column: str = "id"
    target_column: str = "tags"


@dataclass
class MleBenchTransferLearningStackExchangeTagsMultifileTask(MleBenchTransferLearningStackExchangeTagsTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for multilabel text tagging."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable text feature extraction helpers."),
            ("src/models.py", "Editable multilabel tagging helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchFacialKeypointsDetectionTask(MleBenchPreparedTask):
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
        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
        )
        train_df = train_df.sort_values(self.image_id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.image_id_column).reset_index(drop=True)

        public_features = val_df[[self.image_id_column, self.image_column]].copy()
        public_lookup, public_answers = self._build_lookup_and_answers(val_df)
        public_sample_submission = self._baseline_submission(train_df, public_lookup)
        return train_df, public_features, public_sample_submission, public_answers, public_lookup

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_features = pd.read_csv(self.public_test_path)[[self.image_id_column, self.image_column]].copy()
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)
        private_answers = pd.read_csv(self.private_answers_path)[[self.submission_id_column, self.target_column]].copy()
        private_lookup = pd.read_csv(self.public_lookup_path)[
            [self.submission_id_column, self.image_id_column, "FeatureName"]
        ].copy()
        return private_features, private_sample_submission, private_answers, private_lookup

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


@dataclass
class MleBenchFacialKeypointsDetectionMultifileTask(MleBenchFacialKeypointsDetectionTask):
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


@dataclass
class MleBenchKuzushijiRecognitionTask(MleBenchPreparedTask):
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
            self.train_image_cache_dir,
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
        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
        )
        train_df = train_df.sort_values(self.id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.id_column).reset_index(drop=True)

        public_features = val_df[[self.id_column]].copy()
        public_answers = val_df[[self.id_column, self.target_column]].copy()
        sample_submission = pd.DataFrame(
            {
                self.id_column: val_df[self.id_column].astype(str).tolist(),
                self.target_column: "U+003F 1 1 U+FF2F 2 2",
            }
        )
        return train_df, public_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_answers = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
        private_features = private_answers[[self.id_column]].copy()
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)[
            [self.id_column, self.target_column]
        ].copy()
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


@dataclass
class MleBenchKuzushijiRecognitionMultifileTask(MleBenchKuzushijiRecognitionTask):
    pass


@dataclass
class MleBenchRandomActsOfPizzaTask(MleBenchPreparedTask):
    id_column: str = "request_id"
    label_column: str = "requester_received_pizza"
    title_column: str = "request_title"
    body_column: str = "request_text"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "auc-roc"

    @property
    def public_train_json_path(self) -> Path:
        return self.public_root / "train.json"

    @property
    def public_test_json_path(self) -> Path:
        return _first_existing_path(self.public_root / "public_eval.json", self.public_root / "test.json")

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sampleSubmission.csv"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_json_path,
            self.public_test_json_path,
            self.public_sample_submission_path,
            self.private_answers_path,
        )

    def _load_request_frame(self, path: Path, include_label: bool) -> pd.DataFrame:
        with path.open() as handle:
            payload = json.load(handle)
        frame = pd.DataFrame(payload)
        frame = frame.rename(columns={"request_text_edit_aware": self.body_column})
        frame[self.title_column] = frame.get(self.title_column, "").fillna("").astype(str)
        frame[self.body_column] = frame.get(self.body_column, "").fillna("").astype(str)
        columns = [self.id_column, self.title_column, self.body_column]
        if include_label:
            frame[self.label_column] = frame[self.label_column].astype(int)
            columns.append(self.label_column)
        return frame.loc[:, columns].copy()

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = self._load_request_frame(self.public_train_json_path, include_label=True)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = self._load_request_frame(self.public_test_json_path, include_label=False)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = self._load_request_frame(self.public_test_json_path, include_label=False)
        private_answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        _, private_ids = _split_ids_evenly_by_value(test_features[self.id_column])

        private_features = _subset_frame_by_ids(test_features, self.id_column, private_ids)
        private_sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, private_ids)
        private_answers = _subset_frame_by_ids(private_answers_full, self.id_column, private_ids)
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


@dataclass
class MleBenchRandomActsOfPizzaMultifileTask(MleBenchRandomActsOfPizzaTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for text classification."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable text preprocessing and feature extraction helpers."),
            ("src/models.py", "Editable binary text-model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchLeafClassificationTask(MleBenchPreparedTask):
    id_column: str = "id"
    label_column: str = "species"

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "multi-class-log-loss"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def class_columns(self) -> list[str]:
        sample_submission = pd.read_csv(self.public_sample_submission_path, nrows=0)
        return [column for column in sample_submission.columns if column != self.id_column]

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.class_columns]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)

        priors = train_df[self.label_column].value_counts(normalize=True)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        for class_name in self.class_columns:
            sample_submission[class_name] = float(priors.get(class_name, 0.0))

        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.class_columns]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
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


@dataclass
class MleBenchLeafClassificationMultifileTask(MleBenchLeafClassificationTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for multiclass tabular prediction."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable preprocessing and feature engineering helpers."),
            ("src/models.py", "Editable multiclass-model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchPlantPathology2020Task(MleBenchPreparedTask):
    id_column: str = "image_id"
    target_columns: tuple[str, ...] = ("healthy", "multiple_diseases", "rust", "scab")

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_test_path,
            self.public_sample_submission_path,
            self.private_answers_path,
            self.public_root / "images",
        )

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "multi-class-log-loss"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def materialize_workspace(self, workspace_root: Path, eval_access: str = "metric_only") -> None:
        self._ensure_prepared_available()
        data_dir = workspace_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_df, public_features, public_sample_submission, public_answers = self._build_public_assets()
        public_eval_df = self._public_eval_frame(public_features, public_answers, eval_access)
        train_df.to_csv(data_dir / "train.csv", index=False)
        public_eval_df.to_csv(data_dir / "public_eval.csv", index=False)
        public_sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

        self._link_image_subset(train_df[self.id_column].tolist(), data_dir / "train_images")
        self._link_image_subset(public_features[self.id_column].tolist(), data_dir / "public_eval_images")

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
                image_source_dir=self.public_root / "images",
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

    def _link_image_subset(self, image_ids: list[str], output_dir: Path) -> None:
        source_dir = self.public_root / "images"
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
        test_features = pd.read_csv(self.public_test_path)[[self.id_column]].copy()
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.target_columns]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        for column in self.target_columns:
            sample_submission[column] = 1.0 / len(self.target_columns)
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = pd.read_csv(self.public_test_path)[[self.id_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.target_columns]].copy()
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
            public_answers.set_index(self.id_column)[list(self.target_columns)]
            .idxmax(axis=1)
            .rename("label")
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
        expected_columns = [self.id_column, *self.target_columns]
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

        pred_cols = [f"{label}_pred" for label in self.target_columns]
        true_cols = [f"{label}_true" for label in self.target_columns]
        probabilities = merged[pred_cols]
        if not ((probabilities >= 0) & (probabilities <= 1)).all().all():
            raise ValueError("Submission probabilities must be between 0 and 1.")
        if not probabilities.sum(axis=1).round(6).eq(1.0).all():
            raise ValueError("Each submission row must sum to 1.")

        true_labels = merged[true_cols].idxmax(axis=1).str.replace("_true", "", regex=False)
        return float(log_loss(true_labels, probabilities.to_numpy(), labels=list(self.target_columns)))


@dataclass
class MleBenchPlantPathology2020MultifileTask(MleBenchPlantPathology2020Task):
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


@dataclass
class MleBenchPetfinderPawpularityTask(MleBenchPreparedTask):
    id_column: str = "Id"
    target_column: str = "Pawpularity"

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "rmse"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)

        mean_target = float(train_df[self.target_column].astype(float).mean())
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        sample_submission[self.target_column] = mean_target
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
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
        return public_features.merge(
            public_answers,
            on=self.id_column,
            how="left",
            validate="one_to_one",
        )

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = [self.id_column, self.target_column]
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
        return _rmse_score(merged[f"{self.target_column}_true"], merged[f"{self.target_column}_pred"])


@dataclass
class MleBenchPetfinderPawpularityMultifileTask(MleBenchPetfinderPawpularityTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for tabular regression."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable preprocessing and feature engineering helpers."),
            ("src/models.py", "Editable regression-model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchLearningAgencyEssayScoring2Task(MleBenchPreparedTask):
    id_column: str = "essay_id"
    text_column: str = "full_text"
    target_column: str = "score"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "quadratic-weighted-kappa"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def private_answers_path(self) -> Path:
        return self.private_root / "answers.csv"

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)[[self.id_column, self.text_column]].copy()
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, self.text_column, self.target_column]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(
            answers_full[[self.id_column, self.target_column]],
            self.id_column,
            public_ids,
        )
        mode_score = int(train_df[self.target_column].mode(dropna=False).iloc[0])
        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        sample_submission[self.target_column] = mode_score
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        answers_full = pd.read_csv(self.private_answers_path)[
            [self.id_column, self.text_column, self.target_column]
        ].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        _, private_ids = _split_ids_evenly_by_value(answers_full[self.id_column])

        private_features = _subset_frame_by_ids(
            answers_full[[self.id_column, self.text_column]],
            self.id_column,
            private_ids,
        )
        mode_score = int(pd.read_csv(self.public_train_path)[self.target_column].mode(dropna=False).iloc[0])
        private_sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, private_ids)
        private_sample_submission[self.target_column] = mode_score
        private_answers = _subset_frame_by_ids(
            answers_full[[self.id_column, self.target_column]],
            self.id_column,
            private_ids,
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
        expected_columns = [self.id_column, self.target_column]
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

        true_scores = pd.to_numeric(merged[f"{self.target_column}_true"], errors="raise").astype(int)
        pred_scores = pd.to_numeric(merged[f"{self.target_column}_pred"], errors="raise").round().astype(int)
        pred_scores = pred_scores.clip(lower=int(true_scores.min()), upper=int(true_scores.max()))
        return float(cohen_kappa_score(true_scores, pred_scores, weights="quadratic"))


@dataclass
class MleBenchLearningAgencyEssayScoring2MultifileTask(MleBenchLearningAgencyEssayScoring2Task):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for essay scoring."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable text preprocessing and feature extraction helpers."),
            ("src/models.py", "Editable score-prediction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchGoogleQuestTask(MleBenchPreparedTask):
    id_column: str = "qa_id"
    title_column: str = "question_title"
    body_column: str = "question_body"
    answer_column: str = "answer"

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "column-wise-spearman"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def target_columns(self) -> list[str]:
        return [column for column in pd.read_csv(self.public_sample_submission_path, nrows=0).columns if column != self.id_column]

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df = full_train.sort_values(self.id_column).reset_index(drop=True)
        test_features = pd.read_csv(self.public_test_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.target_columns]].copy()
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        public_ids, _ = _split_ids_evenly_by_value(test_features[self.id_column])

        val_features = _subset_frame_by_ids(test_features, self.id_column, public_ids)
        public_answers = _subset_frame_by_ids(answers_full, self.id_column, public_ids)

        sample_submission = _subset_frame_by_ids(sample_submission_full, self.id_column, public_ids)
        for column in self.target_columns:
            sample_submission[column] = float(train_df[column].astype(float).mean())
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_features = pd.read_csv(self.public_test_path)
        sample_submission_full = pd.read_csv(self.public_sample_submission_path)
        answers_full = pd.read_csv(self.private_answers_path)[[self.id_column, *self.target_columns]].copy()
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
        return public_features.merge(
            public_answers,
            on=self.id_column,
            how="left",
            validate="one_to_one",
        )

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = [self.id_column, *self.target_columns]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        submission_df = submission_df.sort_values(self.id_column).reset_index(drop=True)
        answers_df = answers_df.sort_values(self.id_column).reset_index(drop=True)
        if not submission_df[self.id_column].equals(answers_df[self.id_column]):
            raise ValueError("Submission ids did not align with answers.")

        correlations: list[float] = []
        for column in self.target_columns:
            corr = spearmanr(
                pd.to_numeric(submission_df[column], errors="coerce"),
                pd.to_numeric(answers_df[column], errors="coerce"),
            ).correlation
            correlations.append(0.0 if pd.isna(corr) else float(corr))
        return float(sum(correlations) / len(correlations))


@dataclass
class MleBenchGoogleQuestMultifileTask(MleBenchGoogleQuestTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for multi-target text scoring."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable text-pair preprocessing and feature extraction helpers."),
            ("src/models.py", "Editable multi-target regression helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchDogBreedIdentificationTask(MleBenchPreparedTask):
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


@dataclass
class MleBenchDogBreedIdentificationMultifileTask(MleBenchDogBreedIdentificationTask):
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


@dataclass
class MleBenchAptosBlindnessDetectionTask(MleBenchPreparedTask):
    id_column: str = "id_code"
    target_column: str = "diagnosis"

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
            self.public_root / "train_images",
            self.public_root / "test_images",
        )

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "quadratic-weighted-kappa"

    def materialize_workspace(self, workspace_root: Path, eval_access: str = "metric_only") -> None:
        self._ensure_prepared_available()
        data_dir = workspace_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_df, public_features, public_sample_submission, public_answers = self._build_public_assets()
        public_eval_df = self._public_eval_frame(public_features, public_answers, eval_access)
        train_df.to_csv(data_dir / "train.csv", index=False)
        public_eval_df.to_csv(data_dir / "public_eval.csv", index=False)
        public_sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

        self._link_image_subset(train_df[self.id_column].tolist(), self.public_root / "train_images", data_dir / "train_images")
        self._link_image_subset(public_features[self.id_column].tolist(), self.public_root / "train_images", data_dir / "public_eval_images")

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
                image_source_dir=self.public_root / "test_images",
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
            source = source_dir / f"{image_id}.png"
            if not source.exists():
                raise FileNotFoundError(f"Missing image asset for {image_id}: {source}")
            _link_or_copy_file(source, output_dir / f"{image_id}.png")

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
            stratify=full_train[self.target_column],
        )
        train_df = train_df.sort_values(self.id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.id_column).reset_index(drop=True)

        val_features = val_df[[self.id_column]].copy()
        public_answers = val_df[[self.id_column, self.target_column]].copy()
        majority_label = int(train_df[self.target_column].mode(dropna=False).iloc[0])
        sample_submission = pd.DataFrame({self.id_column: val_df[self.id_column]})
        sample_submission[self.target_column] = majority_label
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_features = pd.read_csv(self.public_test_path)[[self.id_column]].copy()
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)
        private_answers = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
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
            _link_or_copy_file(image_source_dir / f"{image_id}.png", image_root / f"{image_id}.png")

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
        expected_columns = [self.id_column, self.target_column]
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

        true_labels = pd.to_numeric(merged[f"{self.target_column}_true"], errors="raise").astype(int)
        pred_labels = pd.to_numeric(merged[f"{self.target_column}_pred"], errors="raise").round().astype(int)
        pred_labels = pred_labels.clip(lower=int(true_labels.min()), upper=int(true_labels.max()))
        return float(cohen_kappa_score(true_labels, pred_labels, weights="quadratic"))


@dataclass
class MleBenchAptosBlindnessDetectionMultifileTask(MleBenchAptosBlindnessDetectionTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for ordinal image grading."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/image_features.py", "Editable image loading and feature helpers."),
            ("src/models.py", "Editable ordinal image-model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchTextNormalizationEnglishTask(MleBenchPreparedTask):
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


@dataclass
class MleBenchTextNormalizationEnglishMultifileTask(MleBenchTextNormalizationEnglishTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for text normalization."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable token-normalization helpers."),
            ("src/models.py", "Editable token-mapping helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchTextNormalizationRussianTask(MleBenchTextNormalizationEnglishTask):
    @property
    def public_train_path(self) -> Path:
        return self.public_root / "ru_train.csv.zip"

    @property
    def public_test_path(self) -> Path:
        return self.public_root / "ru_test_2.csv.zip"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "ru_sample_submission_2.csv.zip"


@dataclass
class MleBenchTextNormalizationRussianMultifileTask(MleBenchTextNormalizationRussianTask):
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


@dataclass
class MleBenchDenoisingDirtyDocumentsTask(MleBenchPreparedTask):
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


@dataclass
class MleBenchDenoisingDirtyDocumentsMultifileTask(MleBenchDenoisingDirtyDocumentsTask):
    pass


@dataclass
class MleBenchCassavaLeafDiseaseClassificationTask(MleBenchPreparedTask):
    id_column: str = "image_id"
    target_column: str = "label"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_sample_submission_path,
            self.private_answers_path,
            self.public_root / "train_images",
            self.public_root / "test_images",
        )

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "accuracy"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for cassava leaf classification."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/image_features.py", "Editable image loading and feature helpers."),
            ("src/models.py", "Editable image-model helpers."),
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

        self._link_image_subset(train_df[self.id_column].tolist(), self.public_root / "train_images", data_dir / "train_images")
        self._link_image_subset(public_features[self.id_column].tolist(), self.public_root / "train_images", data_dir / "public_eval_images")

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
                image_source_dir=self.public_root / "test_images",
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
            source = _resolve_image_asset(source_dir, str(image_id), suffix_candidates=("", ".jpg", ".png", ".jpeg"))
            _link_or_copy_file(source, output_dir / source.name)

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
            stratify=full_train[self.target_column],
        )
        train_df = train_df.sort_values(self.id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.id_column).reset_index(drop=True)

        val_features = val_df[[self.id_column]].copy()
        public_answers = val_df[[self.id_column, self.target_column]].copy()
        majority_label = int(pd.to_numeric(train_df[self.target_column], errors="raise").mode(dropna=False).iloc[0])
        sample_submission = pd.DataFrame({self.id_column: val_df[self.id_column]})
        sample_submission[self.target_column] = majority_label
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_features = pd.read_csv(self.private_answers_path)[[self.id_column]].copy()
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)
        private_answers = pd.read_csv(self.private_answers_path)[[self.id_column, self.target_column]].copy()
        return private_features, private_sample_submission, private_answers

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        return public_features.merge(public_answers, on=self.id_column, how="left", validate="one_to_one")

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
            source = _resolve_image_asset(image_source_dir, str(image_id), suffix_candidates=("", ".jpg", ".png", ".jpeg"))
            _link_or_copy_file(source, image_root / source.name)

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
        expected_columns = [self.id_column, self.target_column]
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

        true_labels = pd.to_numeric(merged[f"{self.target_column}_true"], errors="raise").astype(int)
        pred_labels = pd.to_numeric(merged[f"{self.target_column}_pred"], errors="raise").round().astype(int)
        return float((true_labels == pred_labels).mean())


@dataclass
class MleBenchCassavaLeafDiseaseClassificationMultifileTask(MleBenchCassavaLeafDiseaseClassificationTask):
    pass


@dataclass
class MleBenchRanzcrClipTask(MleBenchPreparedTask):
    id_column: str = "StudyInstanceUID"
    class_columns: tuple[str, ...] = (
        "ETT - Abnormal",
        "ETT - Borderline",
        "ETT - Normal",
        "NGT - Abnormal",
        "NGT - Borderline",
        "NGT - Incompletely Imaged",
        "NGT - Normal",
        "CVC - Abnormal",
        "CVC - Borderline",
    )

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

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
        return False

    @property
    def metric_name(self) -> str:
        return "mean-column-wise-auc-roc"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for multi-label radiology tagging."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/image_features.py", "Editable image loading and feature helpers."),
            ("src/models.py", "Editable multi-label image-model helpers."),
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

        self._link_image_subset(train_df[self.id_column].tolist(), self.public_root / "train", data_dir / "train_images")
        self._link_image_subset(public_features[self.id_column].tolist(), self.public_root / "train", data_dir / "public_eval_images")

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
            source = _resolve_image_asset(source_dir, str(image_id), suffix_candidates=("", ".jpg", ".png", ".jpeg"))
            _link_or_copy_file(source, output_dir / source.name)

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        full_train = pd.read_csv(self.public_train_path)
        train_df, val_df = self._split_multilabel_frame(full_train)
        train_df = train_df.sort_values(self.id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.id_column).reset_index(drop=True)

        val_features = val_df[[self.id_column]].copy()
        public_answers = val_df[[self.id_column, *self.class_columns]].copy()
        sample_submission = pd.DataFrame({self.id_column: val_df[self.id_column]})
        for column in self.class_columns:
            sample_submission[column] = float(train_df[column].astype(float).mean())
        return train_df, val_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_features = pd.read_csv(self.private_answers_path)[[self.id_column]].copy()
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)
        private_answers = pd.read_csv(self.private_answers_path)[[self.id_column, *self.class_columns]].copy()
        return private_features, private_sample_submission, private_answers

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        return public_features.merge(public_answers, on=self.id_column, how="left", validate="one_to_one")

    def _split_multilabel_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        fallback: tuple[pd.DataFrame, pd.DataFrame] | None = None
        for offset in range(32):
            train_df, val_df = train_test_split(
                frame,
                test_size=self.public_split_size,
                random_state=self.public_split_seed + offset,
            )
            if fallback is None:
                fallback = (train_df, val_df)
            if self._has_auc_support(train_df) and self._has_auc_support(val_df):
                return train_df, val_df
        assert fallback is not None
        return fallback

    def _has_auc_support(self, frame: pd.DataFrame) -> bool:
        return all(frame[column].nunique(dropna=False) > 1 for column in self.class_columns)

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
            source = _resolve_image_asset(image_source_dir, str(image_id), suffix_candidates=("", ".jpg", ".png", ".jpeg"))
            _link_or_copy_file(source, image_root / source.name)

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

        scores: list[float] = []
        for column in self.class_columns:
            true_values = pd.to_numeric(merged[f"{column}_true"], errors="raise").astype(float)
            pred_values = pd.to_numeric(merged[f"{column}_pred"], errors="raise").astype(float)
            if true_values.nunique(dropna=False) < 2:
                continue
            if not pred_values.between(0.0, 1.0).all():
                raise ValueError("Submission probabilities must be between 0 and 1.")
            scores.append(float(roc_auc_score(true_values, pred_values)))
        if not scores:
            raise ValueError("No class column had both positive and negative labels for scoring.")
        return float(sum(scores) / len(scores))


@dataclass
class MleBenchRanzcrClipMultifileTask(MleBenchRanzcrClipTask):
    pass


@dataclass
class MleBenchMlsp2013BirdsTask(MleBenchPreparedTask):
    id_column: str = "rec_id"

    @property
    def class_columns(self) -> list[str]:
        return [f"species_{index:02d}" for index in range(19)]

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def private_answers_path(self) -> Path:
        return self.private_root / "answers.csv"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_root / "supplemental_data" / "histogram_of_segments.txt",
            self.public_root / "essential_data" / "rec_labels_test_hidden.txt",
            self.public_sample_submission_path,
            self.private_answers_path,
        )

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "auc-roc"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for bird-species tagging."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable feature-selection helpers for segment histograms."),
            ("src/models.py", "Editable multi-label model helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        feature_df = self._read_feature_table()
        label_df = self._read_label_table()
        labeled_df = feature_df.merge(label_df[label_df["is_hidden"] == 0].drop(columns=["is_hidden"]), on=self.id_column)

        train_df, val_df = train_test_split(
            labeled_df,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
        )
        train_df = train_df.sort_values(self.id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.id_column).reset_index(drop=True)

        val_features = val_df[[self.id_column, *self._feature_columns(feature_df)]].copy()
        public_sample_submission = self._blank_submission_from_rec_ids(val_df[self.id_column].astype(int).tolist())
        public_answers = self._flatten_answers(val_df[[self.id_column, *self.class_columns]].copy())
        return train_df, val_features, public_sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        feature_df = self._read_feature_table()
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)
        private_answers = pd.read_csv(self.private_answers_path)[["Id", "Probability"]].copy()
        private_rec_ids = sorted((private_sample_submission["Id"].astype(int) // 100).unique().tolist())
        private_features = feature_df[feature_df[self.id_column].isin(private_rec_ids)].copy()
        private_features = private_features.sort_values(self.id_column).reset_index(drop=True)
        return private_features, private_sample_submission, private_answers

    def _public_eval_with_labels(
        self,
        public_features: pd.DataFrame,
        public_answers: pd.DataFrame,
    ) -> pd.DataFrame:
        label_frame = self._inflate_answers(public_answers)
        return public_features.merge(label_frame, on=self.id_column, how="left", validate="one_to_one")

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        expected_columns = ["Id", "Probability"]
        if submission_df.columns.tolist() != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {submission_df.columns.tolist()}")
        if len(submission_df) != len(answers_df):
            raise ValueError(f"Expected {len(answers_df)} rows, got {len(submission_df)}")

        submission_df = submission_df.sort_values("Id").reset_index(drop=True)
        answers_df = answers_df.sort_values("Id").reset_index(drop=True)
        if not submission_df["Id"].equals(answers_df["Id"]):
            raise ValueError("Submission ids did not align with answers.")

        pred = pd.to_numeric(submission_df["Probability"], errors="raise").astype(float)
        true = pd.to_numeric(answers_df["Probability"], errors="raise").astype(float)
        if not pred.between(0.0, 1.0).all():
            raise ValueError("Submission probabilities must be between 0 and 1.")
        return float(roc_auc_score(true, pred))

    def _blank_submission_from_rec_ids(self, rec_ids: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Id": [rec_id * 100 + species_id for rec_id in rec_ids for species_id in range(len(self.class_columns))],
                "Probability": 0.0,
            }
        )

    def _flatten_answers(self, wide_answers: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, float | int]] = []
        for _, row in wide_answers.iterrows():
            rec_id = int(row[self.id_column])
            for species_id, column in enumerate(self.class_columns):
                rows.append({"Id": rec_id * 100 + species_id, "Probability": float(row[column])})
        return pd.DataFrame(rows)

    def _inflate_answers(self, flat_answers: pd.DataFrame) -> pd.DataFrame:
        frame = flat_answers.copy()
        frame["rec_id"] = (frame["Id"].astype(int) // 100).astype(int)
        frame["species_id"] = (frame["Id"].astype(int) % 100).astype(int)
        pivot = (
            frame.pivot(index="rec_id", columns="species_id", values="Probability")
            .reindex(columns=range(len(self.class_columns)), fill_value=0.0)
            .reset_index()
        )
        pivot.columns = [self.id_column, *self.class_columns]
        return pivot

    def _read_feature_table(self) -> pd.DataFrame:
        path = self.public_root / "supplemental_data" / "histogram_of_segments.txt"
        records: list[list[float]] = []
        with path.open() as handle:
            next(handle)
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(",")
                rec_id = int(parts[0])
                feature_values = [float(value) for value in parts[1:] if value != ""]
                records.append([rec_id, *feature_values])
        if not records:
            raise ValueError("No histogram features were found for mlsp-2013-birds.")
        columns = [self.id_column, *[f"feature_{index:03d}" for index in range(len(records[0]) - 1)]]
        return pd.DataFrame(records, columns=columns)

    def _read_label_table(self) -> pd.DataFrame:
        rows: list[dict[str, int]] = []
        path = self.public_root / "essential_data" / "rec_labels_test_hidden.txt"
        with path.open() as handle:
            next(handle)
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(",")
                rec_id = int(parts[0])
                label_text = ",".join(parts[1:]) if len(parts) > 1 else ""
                hidden = int(label_text == "?")
                labels = set(int(value) for value in parts[1:] if value not in {"", "?"})
                row: dict[str, int] = {self.id_column: rec_id, "is_hidden": hidden}
                for species_id, column in enumerate(self.class_columns):
                    row[column] = int(species_id in labels)
                rows.append(row)
        return pd.DataFrame(rows)

    def _feature_columns(self, feature_df: pd.DataFrame) -> list[str]:
        return [column for column in feature_df.columns if column != self.id_column]


@dataclass
class MleBenchMlsp2013BirdsMultifileTask(MleBenchMlsp2013BirdsTask):
    pass


@dataclass
class MleBenchWhaleCallTask(MleBenchPreparedTask):
    label_column: str = "probability"
    id_column: str = "clip"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_root / "train2.zip",
            self.public_root / "test2.zip",
            self.public_sample_submission_path,
            self.private_answers_path,
        )

    @property
    def lower_is_better(self) -> bool:
        return False

    @property
    def metric_name(self) -> str:
        return "auc-roc"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sampleSubmission.csv"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and audio-directory layout."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Editable audio-classification pipeline."),
            ("src/audio_features.py", "Editable audio loading and feature helpers."),
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

        self._extract_zip_all(
            archive_path=self.public_root / "train2.zip",
            output_dir=data_dir / "train_audio",
        )
        self._extract_zip_subset(
            archive_path=self.public_root / "test2.zip",
            filenames=public_features[self.id_column].tolist(),
            output_dir=data_dir / "public_eval_audio",
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

        public_run = self._run_audio_submission_split(
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            eval_df=public_eval_df,
            sample_submission_df=public_sample_submission,
            answers_df=public_answers,
            split_name="public",
            audio_source_dir=workspace_root / "data" / "public_eval_audio",
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
        if public_run.success:
            private_run = self._run_audio_submission_split(
                workspace_root=workspace_root,
                hidden_root=hidden_root,
                eval_df=private_features,
                sample_submission_df=private_sample_submission,
                answers_df=private_answers,
                split_name="private",
                audio_source_dir=self._ensure_private_audio_cache(hidden_root),
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
        train_files = self._list_archive_members(self.public_root / "train2.zip")
        train_rows = []
        for clip in train_files:
            match = re.search(r"_(?P<label>[01])\\.aif$", clip)
            if not match:
                raise ValueError(f"Could not infer train label from clip name: {clip}")
            train_rows.append(
                {
                    self.id_column: clip,
                    self.label_column: int(match.group("label")),
                }
            )
        train_df = pd.DataFrame(train_rows).sort_values(self.id_column).reset_index(drop=True)

        sample_submission = pd.read_csv(self.public_sample_submission_path).sort_values(self.id_column).reset_index(drop=True)
        public_features = sample_submission[[self.id_column]].copy()
        public_answers = pd.read_csv(self.private_answers_path)[[self.id_column, self.label_column]].sort_values(self.id_column).reset_index(drop=True)
        return train_df, public_features, sample_submission, public_answers

    def _build_private_assets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        private_answers = pd.read_csv(self.private_answers_path).sort_values(self.id_column).reset_index(drop=True)
        private_features = private_answers[[self.id_column]].copy()
        private_sample_submission = pd.read_csv(self.public_sample_submission_path).sort_values(self.id_column).reset_index(drop=True)
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
            raise ValueError("Submission clips did not align with answers.")

        probabilities = merged[f"{self.label_column}_pred"].astype(float)
        if not probabilities.between(0.0, 1.0).all():
            raise ValueError("Submission probabilities must be between 0 and 1.")

        true_labels = merged[f"{self.label_column}_true"].astype(int)
        return float(roc_auc_score(true_labels, probabilities))

    def _run_audio_submission_split(
        self,
        workspace_root: Path,
        hidden_root: Path,
        eval_df: pd.DataFrame,
        sample_submission_df: pd.DataFrame,
        answers_df: pd.DataFrame,
        split_name: str,
        audio_source_dir: Path,
        python_executable: str,
        timeout_seconds: int,
    ) -> SplitExecution:
        tmp_root = _mk_eval_tmp_root(hidden_root, split_name)
        eval_path = tmp_root / f"{split_name}_input.csv"
        sample_path = tmp_root / f"{split_name}_sample_submission.csv"
        output_path = tmp_root / f"{split_name}_submission.csv"
        audio_root = tmp_root / "eval_audio"
        audio_root.mkdir(parents=True, exist_ok=True)

        eval_df.to_csv(eval_path, index=False)
        sample_submission_df.to_csv(sample_path, index=False)
        for clip in eval_df[self.id_column].tolist():
            _link_or_copy_file(audio_source_dir / clip, audio_root / clip)

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

    def _list_archive_members(self, archive_path: Path) -> list[str]:
        with zipfile.ZipFile(archive_path) as archive:
            return sorted(Path(name).name for name in archive.namelist() if not name.endswith("/"))

    def _extract_zip_all(self, archive_path: Path, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                destination = output_dir / Path(member).name
                _ensure_parent(destination)
                with archive.open(member) as source, destination.open("wb") as sink:
                    sink.write(source.read())


@dataclass
class MleBenchWhaleCallMultifileTask(MleBenchWhaleCallTask):
    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for audio classification."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/audio_features.py", "Editable audio loading and feature helpers."),
            ("src/models.py", "Editable audio-model construction helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]

    def _extract_zip_subset(self, archive_path: Path, filenames: list[str], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        wanted = set(filenames)
        with zipfile.ZipFile(archive_path) as archive:
            name_map = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
            missing = sorted(filename for filename in wanted if filename not in name_map)
            if missing:
                raise FileNotFoundError(
                    f"Archive {archive_path} is missing {len(missing)} expected clips, "
                    f"including {missing[:3]}"
                )
            for filename in filenames:
                destination = output_dir / filename
                with archive.open(name_map[filename]) as source, destination.open("wb") as sink:
                    sink.write(source.read())

    def _ensure_private_audio_cache(self, hidden_root: Path) -> Path:
        cache_dir = hidden_root / "private_audio_cache"
        marker = cache_dir / ".ready"
        if marker.exists():
            return cache_dir

        self._extract_zip_all(self.public_root / "test2.zip", cache_dir)
        marker.write_text("ready\n")
        return cache_dir


@dataclass
class MleBenchDataScienceBowl2018Task(MleBenchPreparedTask):
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

        sample_ids = pd.read_csv(self.public_train_path)[self.image_id_column].astype(str).tolist()
        train_ids, val_ids = train_test_split(
            sample_ids,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
        )
        train_ids = sorted(train_ids)
        val_ids = sorted(val_ids)

        train_df = pd.DataFrame({self.image_id_column: train_ids})
        public_features = pd.DataFrame({self.image_id_column: val_ids})
        public_answers = self._build_answers_from_sample_dirs(self.raw_stage1_train_root, val_ids)
        public_sample_submission = pd.DataFrame(
            {
                self.image_id_column: val_ids,
                self.target_column: [""] * len(val_ids),
            }
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

        private_features = pd.read_csv(self.public_test_path)[[self.image_id_column]].copy()
        private_features = private_features.sort_values(self.image_id_column).reset_index(drop=True)
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)[
            [self.image_id_column, self.target_column]
        ].copy()
        private_sample_submission = private_sample_submission.sort_values(self.image_id_column).reset_index(drop=True)
        private_answers = self._build_answers_from_sample_dirs(
            self.raw_stage1_train_root,
            private_features[self.image_id_column].astype(str).tolist(),
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

        private_ids = pd.read_csv(self.public_test_path)[self.image_id_column].astype(str).tolist()
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


@dataclass
class MleBenchDataScienceBowl2018MultifileTask(MleBenchDataScienceBowl2018Task):
    pass


@dataclass
class MleBenchKvasirSegTask(MleBenchPreparedTask):
    image_id_column: str = "image_id"
    target_column: str = "mask_rle"
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
    def public_train_images_root(self) -> Path:
        return self.public_root / "train_images"

    @property
    def public_train_masks_root(self) -> Path:
        return self.public_root / "train_masks"

    @property
    def public_test_images_root(self) -> Path:
        return self.public_root / "test_images"

    @property
    def required_prepared_paths(self) -> tuple[Path, ...]:
        return (
            self.public_train_path,
            self.public_test_path,
            self.public_sample_submission_path,
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
        return "mean-dice"

    @property
    def public_sample_submission_path(self) -> Path:
        return self.public_root / "sample_submission.csv"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for polyp segmentation."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
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

        train_df, public_features, public_sample_submission, public_answers = self._build_public_assets()
        public_eval_df = self._public_eval_frame(public_features, public_answers, eval_access)
        train_df.to_csv(data_dir / "train.csv", index=False)
        public_eval_df.to_csv(data_dir / "public_eval.csv", index=False)
        public_sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

        train_ids = train_df[self.image_id_column].astype(str).tolist()
        public_ids = public_features[self.image_id_column].astype(str).tolist()
        self._link_asset_subset(train_ids, self.public_train_images_root, data_dir / "train_images")
        self._link_asset_subset(train_ids, self.public_train_masks_root, data_dir / "train_masks")
        self._link_asset_subset(public_ids, self.public_train_images_root, data_dir / "public_eval_images")
        if eval_access == "full":
            self._link_asset_subset(public_ids, self.public_train_masks_root, data_dir / "public_eval_masks")

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

        full_train = pd.read_csv(self.public_train_path)[[self.image_id_column, self.target_column]].copy()
        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
        )
        train_df = train_df.sort_values(self.image_id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.image_id_column).reset_index(drop=True)

        public_features = val_df[[self.image_id_column]].copy()
        public_answers = self._build_answers_with_sizes(val_df, self.public_train_images_root)
        public_sample_submission = pd.DataFrame(
            {
                self.image_id_column: public_features[self.image_id_column].astype(str).tolist(),
                self.target_column: [""] * len(public_features),
            }
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

        private_features = pd.read_csv(self.public_test_path)[[self.image_id_column]].copy()
        private_features = private_features.sort_values(self.image_id_column).reset_index(drop=True)
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)[
            [self.image_id_column, self.target_column]
        ].copy()
        private_sample_submission = private_sample_submission.sort_values(self.image_id_column).reset_index(drop=True)
        private_answers = pd.read_csv(self.private_answers_path)[
            [self.image_id_column, self.target_column, "width", "height"]
        ].copy()
        private_answers = private_answers.sort_values(self.image_id_column).reset_index(drop=True)
        cached = (
            private_features.copy(),
            private_sample_submission.copy(),
            private_answers.copy(),
        )
        setattr(self, "_cached_private_assets", cached)
        return private_features, private_sample_submission, private_answers

    def _build_answers_with_sizes(self, labels_df: pd.DataFrame, image_root: Path) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for row in labels_df.itertuples(index=False):
            image_id = str(getattr(row, self.image_id_column))
            image_path = _resolve_image_asset(image_root, image_id)
            with Image.open(image_path) as image:
                width, height = image.size
            rows.append(
                {
                    self.image_id_column: image_id,
                    self.target_column: getattr(row, self.target_column),
                    "width": width,
                    "height": height,
                }
            )
        return pd.DataFrame(rows).sort_values(self.image_id_column).reset_index(drop=True)

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


@dataclass
class MleBenchKvasirSegMultifileTask(MleBenchKvasirSegTask):
    pass


@dataclass
class MleBenchTgsSaltIdentificationTask(MleBenchKvasirSegTask):
    image_id_column: str = "id"
    target_column: str = "rle_mask"
    image_height: int = 101
    image_width: int = 101

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

    def _build_public_assets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cached = getattr(self, "_cached_public_assets", None)
        if cached is not None:
            return tuple(frame.copy() for frame in cached)

        labels_df = pd.read_csv(self.public_train_path)[[self.image_id_column, self.target_column]].copy()
        depths_df = pd.read_csv(self.public_depths_path)[[self.image_id_column, "z"]].copy()
        full_train = labels_df.merge(depths_df, on=self.image_id_column, how="left")

        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
        )
        train_df = train_df.sort_values(self.image_id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.image_id_column).reset_index(drop=True)

        public_features = val_df[[self.image_id_column, "z"]].copy()
        public_answers = self._build_answers_frame(val_df[[self.image_id_column, self.target_column]])
        public_sample_submission = pd.DataFrame(
            {
                self.image_id_column: public_features[self.image_id_column].astype(str).tolist(),
                self.target_column: [""] * len(public_features),
            }
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

        private_sample_submission = pd.read_csv(self.public_sample_submission_path)[
            [self.image_id_column, self.target_column]
        ].copy()
        private_sample_submission = private_sample_submission.sort_values(self.image_id_column).reset_index(drop=True)

        private_features = private_sample_submission[[self.image_id_column]].copy()
        private_features["z"] = np.nan

        private_answers = pd.read_csv(self.private_answers_path)[[self.image_id_column, self.target_column]].copy()
        private_answers = self._build_answers_frame(private_answers)
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

    def _grade_submission(self, submission_df: pd.DataFrame, answers_df: pd.DataFrame) -> float:
        return _tgs_mean_average_precision(
            submission_df,
            answers_df,
            image_id_column=self.image_id_column,
            target_column=self.target_column,
            default_height=self.image_height,
            default_width=self.image_width,
        )


@dataclass
class MleBenchTgsSaltIdentificationMultifileTask(MleBenchTgsSaltIdentificationTask):
    pass


@dataclass
class MleBenchUwMadisonGiTractImageSegmentationTask(MleBenchPreparedTask):
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
        self._link_slice_subset(public_ids, self._train_scan_index(), data_dir / "public_eval_images")

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
        full_train["case_day"] = full_train[self.id_column].astype(str).map(lambda value: value.split("_slice_")[0])
        unique_case_days = sorted(full_train["case_day"].drop_duplicates().tolist())
        train_case_days, val_case_days = train_test_split(
            unique_case_days,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
        )

        train_df = full_train[full_train["case_day"].isin(train_case_days)].drop(columns=["case_day"]).copy()
        val_df = full_train[full_train["case_day"].isin(val_case_days)].drop(columns=["case_day"]).copy()
        train_df = self._sort_slice_frame(train_df)
        val_df = self._sort_slice_frame(val_df)

        public_features = val_df[[self.id_column, self.class_column]].copy()
        public_answers = self._answer_frame_from_labels(
            val_df,
            label_column=self.train_target_column,
            scan_index=self._train_scan_index(),
        )
        public_sample_submission = pd.DataFrame(
            {
                self.id_column: public_features[self.id_column].astype(str).tolist(),
                self.class_column: public_features[self.class_column].astype(str).tolist(),
                self.target_column: [""] * len(public_features),
            }
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

        private_features = pd.read_csv(self.public_test_path)[[self.id_column, self.class_column]].copy()
        private_features = self._sort_slice_frame(private_features)
        private_sample_submission = pd.read_csv(self.public_sample_submission_path)[
            [self.id_column, self.class_column, self.target_column]
        ].copy()
        private_sample_submission = self._sort_slice_frame(private_sample_submission)
        private_answers = pd.read_csv(self.private_answers_path)[
            [self.id_column, self.class_column, self.target_column, "image_width", "image_height"]
        ].copy()
        private_answers = self._sort_slice_frame(private_answers)
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


@dataclass
class MleBenchUwMadisonGiTractImageSegmentationMultifileTask(MleBenchUwMadisonGiTractImageSegmentationTask):
    pass


@dataclass
class MleBenchCofwFaceLandmarksTask(MleBenchPreparedTask):
    image_id_column: str = "image_id"
    protected_paths: tuple[str, ...] = (
        "data/train.csv",
        "data/public_eval.csv",
        "data/sample_submission.csv",
        "data/train_images",
        "data/public_eval_images",
    )

    @property
    def public_train_images_root(self) -> Path:
        return self.public_root / "train_images"

    @property
    def public_test_images_root(self) -> Path:
        return self.public_root / "test_images"

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
            self.public_train_images_root,
            self.public_test_images_root,
        )

    @property
    def lower_is_better(self) -> bool:
        return True

    @property
    def metric_name(self) -> str:
        return "normalized-mean-landmark-error"

    @property
    def keypoint_columns(self) -> tuple[str, ...]:
        cached = getattr(self, "_cached_keypoint_columns", None)
        if cached is not None:
            return cached
        columns = tuple(pd.read_csv(self.public_sample_submission_path, nrows=0).columns.tolist()[1:])
        setattr(self, "_cached_keypoint_columns", columns)
        return columns

    @property
    def feature_columns(self) -> tuple[str, ...]:
        cached = getattr(self, "_cached_feature_columns", None)
        if cached is not None:
            return cached
        columns = tuple(pd.read_csv(self.public_test_path, nrows=0).columns.tolist())
        setattr(self, "_cached_feature_columns", columns)
        return columns

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for face landmark regression."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable face-image loading and lightweight feature helpers."),
            ("src/models.py", "Editable landmark-regression baseline helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]

    def materialize_workspace(self, workspace_root: Path, eval_access: str = "metric_only") -> None:
        self._ensure_prepared_available()
        data_dir = workspace_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        train_df, public_features, public_sample_submission, public_answers = self._build_public_assets()
        train_df.to_csv(data_dir / "train.csv", index=False)
        public_features.to_csv(data_dir / "public_eval.csv", index=False)
        public_sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)
        if eval_access == "full":
            public_answers.to_csv(data_dir / "public_eval_answers.csv", index=False)

        self._link_image_subset(
            train_df[self.image_id_column].astype(str).tolist(),
            self.public_train_images_root,
            data_dir / "train_images",
        )
        self._link_image_subset(
            public_features[self.image_id_column].astype(str).tolist(),
            self.public_train_images_root,
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
        train_df, public_features, public_sample_submission, public_answers = self._build_public_assets()
        _ = train_df
        private_features, private_sample_submission, private_answers = self._build_private_assets()

        public_run = self._run_landmark_split(
            workspace_root=workspace_root,
            hidden_root=hidden_root,
            eval_df=public_features,
            sample_submission_df=public_sample_submission,
            answers_df=public_answers,
            split_name="public",
            eval_image_dir=workspace_root / "data" / "public_eval_images",
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
        )
        if public_run.success:
            private_run = self._run_landmark_split(
                workspace_root=workspace_root,
                hidden_root=hidden_root,
                eval_df=private_features,
                sample_submission_df=private_sample_submission,
                answers_df=private_answers,
                split_name="private",
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

        full_train = pd.read_csv(self.public_train_path)
        train_df, val_df = train_test_split(
            full_train,
            test_size=self.public_split_size,
            random_state=self.public_split_seed,
        )
        train_df = train_df.sort_values(self.image_id_column).reset_index(drop=True)
        val_df = val_df.sort_values(self.image_id_column).reset_index(drop=True)

        public_features = val_df.loc[:, self.feature_columns].copy()
        public_answers = val_df.loc[:, [self.image_id_column, "bbox_width", "bbox_height", *self.keypoint_columns]].copy()
        public_sample_submission = self._blank_submission(public_features)
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

        private_features = pd.read_csv(self.public_test_path).loc[:, self.feature_columns].copy()
        private_features = private_features.sort_values(self.image_id_column).reset_index(drop=True)
        private_sample_submission = pd.read_csv(self.public_sample_submission_path).loc[
            :, [self.image_id_column, *self.keypoint_columns]
        ].copy()
        private_sample_submission = private_sample_submission.sort_values(self.image_id_column).reset_index(drop=True)
        private_answers = pd.read_csv(self.private_answers_path).loc[
            :, [self.image_id_column, "bbox_width", "bbox_height", *self.keypoint_columns]
        ].copy()
        private_answers = private_answers.sort_values(self.image_id_column).reset_index(drop=True)
        cached = (
            private_features.copy(),
            private_sample_submission.copy(),
            private_answers.copy(),
        )
        setattr(self, "_cached_private_assets", cached)
        return private_features, private_sample_submission, private_answers

    def _blank_submission(self, frame: pd.DataFrame) -> pd.DataFrame:
        submission = pd.DataFrame({self.image_id_column: frame[self.image_id_column].astype(str).tolist()})
        for column in self.keypoint_columns:
            submission[column] = 0.0
        return submission

    def _link_image_subset(self, image_ids: list[str], source_dir: Path, output_dir: Path) -> None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_id in image_ids:
            source = _resolve_image_asset(source_dir, image_id, suffix_candidates=("", ".png", ".jpg", ".jpeg"))
            _link_or_copy_file(source, output_dir / source.name)

    def _run_landmark_split(
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
        return _normalized_bbox_landmark_error(
            submission_df,
            answers_df,
            image_id_column=self.image_id_column,
            keypoint_columns=self.keypoint_columns,
        )


@dataclass
class MleBenchCofwFaceLandmarksMultifileTask(MleBenchCofwFaceLandmarksTask):
    pass


@dataclass
class MleBenchCmuHandKeypointsTask(MleBenchCofwFaceLandmarksTask):
    @property
    def metric_name(self) -> str:
        return "normalized-mean-hand-keypoint-error"

    @property
    def key_files(self) -> list[tuple[str, str]]:
        return [
            ("README.md", "Workspace guide and explicit multi-file edit surface for hand keypoint regression."),
            ("TASK_DESCRIPTION.md", "Short task summary covering goal, inputs, outputs, and metric."),
            ("ENVIRONMENT.md", "Pinned runtime and library constraints for this workspace."),
            ("src/pipeline.py", "Thin CLI/orchestration entrypoint; keep the interface stable."),
            ("src/features.py", "Editable hand-image loading and lightweight feature helpers."),
            ("src/models.py", "Editable hand-keypoint baseline helpers."),
            ("src/postprocess.py", "Editable submission formatting helpers."),
            ("data/sample_submission.csv", "Submission schema for the public evaluation file."),
        ]


@dataclass
class MleBenchCmuHandKeypointsMultifileTask(MleBenchCmuHandKeypointsTask):
    pass
