"""
Sandboxed code execution for model-generated Python scripts.

Executes code in a temporary directory with only the necessary data files,
with a timeout to prevent runaway processes.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import pandas as pd


@dataclass
class ExecResult:
    success: bool
    predictions: list | None
    stdout: str
    stderr: str
    return_code: int


def extract_code(response: str) -> str | None:
    """Extract Python code block from model response."""
    # Try to find ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # Return the last (most complete) code block
        return matches[-1].strip()

    # Try generic ``` blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return None


def execute_code(
    code: str,
    data_files: dict[str, str],
    timeout: int = 120,
    python_path: str = "python",
) -> ExecResult:
    """
    Execute model-generated code in a sandboxed temp directory.

    Args:
        code: Python code string to execute.
        data_files: Mapping of {filename: source_path} for data files to copy in.
        timeout: Maximum execution time in seconds.
        python_path: Path to Python interpreter.

    Returns:
        ExecResult with success status, predictions, and output.
    """
    tmpdir = tempfile.mkdtemp(prefix="sandbox_")

    try:
        # Copy data files into sandbox
        for filename, src_path in data_files.items():
            dst_path = os.path.join(tmpdir, filename)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

        # Write code to file
        script_path = os.path.join(tmpdir, "solution.py")
        with open(script_path, "w") as f:
            f.write(code)

        # Execute
        result = subprocess.run(
            [python_path, script_path],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Read predictions if they exist
        pred_path = os.path.join(tmpdir, "predictions.csv")
        predictions = None
        if os.path.exists(pred_path):
            try:
                pred_df = pd.read_csv(pred_path)
                if "prediction" in pred_df.columns:
                    predictions = pred_df["prediction"].tolist()
                elif len(pred_df.columns) == 1:
                    predictions = pred_df.iloc[:, 0].tolist()
            except Exception:
                pass

        return ExecResult(
            success=(result.returncode == 0 and predictions is not None),
            predictions=predictions,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return ExecResult(
            success=False,
            predictions=None,
            stdout="",
            stderr=f"Execution timed out after {timeout} seconds.",
            return_code=-1,
        )
    except Exception as e:
        return ExecResult(
            success=False,
            predictions=None,
            stdout="",
            stderr=str(e),
            return_code=-1,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
