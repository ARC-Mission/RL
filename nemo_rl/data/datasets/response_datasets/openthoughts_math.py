# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


def _extract_reflection_blocks(text: str) -> str:
    """Extract all <reflection>...</reflection> blocks from a reasoning trace.

    Returns the concatenated content of every <reflection> block separated by
    double newlines, or the empty string when no blocks are found.
    """
    blocks = re.findall(r"<reflection>(.*?)</reflection>", text, re.DOTALL)
    return "\n\n".join(b.strip() for b in blocks)

# Columns from yllkryeziu/openthoughts114k-math-qwen3 to keep alongside messages/task_name.
# These are available as {column_name} placeholders in teacher prompt templates
# (use data.default.column_mapping to alias them, e.g. qwen3_1b7_answer → trace).
_KEEP_COLUMNS = {
    "qwen3_1b7_answer",
    "qwen3_4b_answer",
    "qwen3_8b_answer",
    "qwen3_14b_answer",
    "qwen3_8b_reflectionsv1",
    "qwen3_8b_reflectionsv2",
    "qwen3_8b_reflectionsv1_extracted",
    "qwen3_8b_reflectionsv2_extracted",
}


class OpenThoughtsMathDataset(RawDataset):
    """Dataset wrapping yllkryeziu/openthoughts114k-math-qwen3.

    Builds a messages column from `problem` (user) and `expected_answer` (assistant),
    and preserves the qwen3_*_answer columns so they can be injected into teacher
    prompt templates via data.default.column_mapping.
    """

    def __init__(self, **kwargs) -> None:
        self.task_name = "OpenThoughtsMath"

        self.dataset = load_dataset(
            "parquet",
            data_files="hf://datasets/yllkryeziu/openthoughts114k-math-qwen3/data/*.parquet",
            split="train",
        )

        # Add messages + task_name without stripping extra columns first,
        # then drop everything we don't need to keep memory use reasonable.
        self.dataset = self.dataset.map(self.format_data)

        # Derive extracted-reflection columns (only the <reflection> blocks).
        self.dataset = self.dataset.map(self._extract_reflections)

        cols_to_drop = [
            c for c in self.dataset.column_names
            if c not in {"messages", "task_name"} | _KEEP_COLUMNS
        ]
        if cols_to_drop:
            self.dataset = self.dataset.remove_columns(cols_to_drop)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": data["expected_answer"]},
            ],
            "task_name": self.task_name,
        }

    @staticmethod
    def _extract_reflections(data: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for col in ("qwen3_8b_reflectionsv1", "qwen3_8b_reflectionsv2"):
            raw = data.get(col)
            out[f"{col}_extracted"] = _extract_reflection_blocks(raw) if raw else ""
        return out
