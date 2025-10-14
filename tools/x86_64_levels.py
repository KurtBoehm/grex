# This file is part of https://github.com/KurtBoehm/grex.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from subprocess import PIPE, run
from typing import Final


def parse_line(line: str):
    line = line.strip()
    assert len(line) > 0
    lhs, rhs = line.split(":")
    return lhs.strip(), rhs.strip()


arch: Final = run(["uname", "--machine"], stdout=PIPE).stdout.decode().strip()
assert arch == "x86_64"


x86_64_levels: Final = {
    "x86-64": ["cmov", "cx8", "fpu", "fxsr", "mmx", "syscall", "sse", "sse2"],
    "x86-64-v2": ["cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"],
    "x86-64-v3": [
        "avx",
        "avx2",
        "bmi1",
        "bmi2",
        "f16c",
        "fma",
        "abm",
        "movbe",
        "xsave",
    ],
    "x86-64-v4": ["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"],
}


with open("/proc/cpuinfo", "r") as f:
    raw_info: Final = f.read()
info: Final = {
    k: v
    for k, v in (
        parse_line(line) for line in raw_info.splitlines() if len(line.strip()) != 0
    )
}
flags: Final = {f.strip() for f in info["flags"].split()}
supported: Final = [k for k, v in x86_64_levels.items() if all(f in flags for f in v)]
print(" ".join(supported))
