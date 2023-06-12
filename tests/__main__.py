# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import subprocess
import sys


here = pathlib.Path(__file__).resolve().parent


# Each file is ran separately to avoid out-of-memorying.
running_out = 0
for file in here.iterdir():
    if file.is_file() and file.name.startswith("test"):
        out = subprocess.run(f"pytest {file}", shell=True).returncode
        running_out = max(running_out, out)
sys.exit(running_out)
