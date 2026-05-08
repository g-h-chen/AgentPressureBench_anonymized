"""Shared repo_workspace task infrastructure copied from tasks/mle_bench.py.

The active task modules import from here so tasks/mle_bench.py can remain unchanged
as reference-only code.
"""

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


__all__ = [name for name in globals() if not name.startswith("__")]
