# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream and Fast-dLLM repos: https://github.com/HKUNLP/Dream
# and https://github.com/NVlabs/Fast-dLLM

import os
import sys

from humaneval_execution import check_correctness
from sanitize import sanitize

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
EVALUATOR_VERSION = "humaneval_spawn_official_checker_v3"

def pass_at_1(references, predictions):
    if len(references) != 1 or len(predictions) != 1 or len(predictions[0]) != 1:
        raise ValueError("sequential pass@1 expects exactly one task and one candidate")
    test_program = predictions[0][0] + "\n" + references[0]
    result = check_correctness(test_program, 3.0, 0, 0)
    return float(result["passed"])

import json

        
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def main(file_path):
    data = read_jsonl(file_path)
    references = [sample['target'] for sample in data]
    predictions = [[sanitize(
        sample['doc']['prompt'] + "\n"
        + sample['resps'][0][0].split('```python\n', 1)[-1].split('```')[0],
        sample['doc']["entry_point"],
    )] for sample in data]
    pass_at_1s = [
        pass_at_1([reference], [prediction])
        for reference, prediction in zip(references, predictions)
    ]
    print(sum(pass_at_1s) / len(pass_at_1s))
    records = [{
        "task_id": sample['doc']['task_id'],
        "completion": prediction,
        "pass_at_1": result,
        "evaluator_version": EVALUATOR_VERSION,
    } for sample, prediction, result in zip(data, predictions, pass_at_1s)]
    write_jsonl(records, file_path + '.cleaned')


if __name__ == "__main__":
    main(sys.argv[1])
