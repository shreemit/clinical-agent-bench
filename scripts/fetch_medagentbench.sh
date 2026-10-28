#!/usr/bin/env bash
# Pull the pinned MedAgentBench HAPI image and verify vendored task files.
source "$(dirname "$0")/lib.sh"

echo "Pulling $HAPI_IMAGE"
"${DOCKER[@]}" pull "$HAPI_IMAGE"

python3 - << 'PY'
import hashlib
import pathlib
import sys
import urllib.request

root = pathlib.Path("vendor/medagentbench")
expected = {
    "test_data_v2.json": "b6e89b2ef82f1bef27c5778a644a8a66c7b25738bbd6a48d03ab3e5182df6ca8",
    "funcs_v1.json": "c977266db5eca75182c0d12cac1962b33ae6b5a54b9029ceb104e1a93b4c38c1",
}
urls = {
    "test_data_v2.json": "https://raw.githubusercontent.com/stanfordmlgroup/MedAgentBench/main/data/medagentbench/test_data_v2.json",
    "funcs_v1.json": "https://raw.githubusercontent.com/stanfordmlgroup/MedAgentBench/main/data/medagentbench/funcs_v1.json",
}

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

for name, digest in expected.items():
    path = root / name
    if not path.exists():
        print(f"downloading {name}")
        urllib.request.urlretrieve(urls[name], path)
    got = sha256(path)
    if got != digest:
        print(f"checksum mismatch for {name}: {got} != {digest}", file=sys.stderr)
        sys.exit(1)
    print(f"ok {name} {got}")
PY

echo "Fetch finished. Next: make load"
