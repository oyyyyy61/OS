"""Clean-room dependency and imported evidence contract tests."""

from __future__ import annotations

import ast
import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"offload", "Agentrix", "vllm"}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_runtime_has_no_historical_project_imports() -> None:
    """The new runtime crosses legacy boundaries only through future adapters."""

    violations: list[str] = []
    for path in sorted((ROOT / "src" / "dagkv").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if module.split(".", maxsplit=1)[0] in FORBIDDEN:
                    violations.append(f"{path.relative_to(ROOT)} imports {module}")
    assert violations == []


def test_imported_evidence_hashes_match_manifest() -> None:
    """Every copied M1 contract artifact remains byte-identical."""

    evidence_root = ROOT / "evidence"
    manifest = json.loads((evidence_root / "IMPORT_MANIFEST.json").read_text())
    artifacts = [manifest["m1_authority"]["copied_manifest"]]
    artifacts.extend(manifest["copied_artifacts"])
    mismatches: list[str] = []
    for artifact in artifacts:
        path = evidence_root / artifact["path"]
        actual = _file_sha256(path)
        if actual != artifact["sha256"]:
            mismatches.append(f"{artifact['path']}: {actual}")
    assert mismatches == []

    for inventory in manifest["external_root_inventories"]:
        for field in ("sha256sums", "manifest_sha256_file"):
            artifact = inventory[field]
            actual = _file_sha256(evidence_root / artifact["path"])
            if actual != artifact["sha256"]:
                mismatches.append(f"{artifact['path']}: {actual}")
    assert mismatches == []


def test_complete_acceptance_copy_is_self_verifying() -> None:
    """The small M1 decision package can be verified without sibling paths."""

    acceptance = ROOT / "evidence" / "m1" / "acceptance"
    checksum_file = acceptance / "SHA256SUMS"
    mismatches: list[str] = []
    for line in checksum_file.read_text().splitlines():
        expected, relative_path = line.split(maxsplit=1)
        path = acceptance / relative_path
        if _file_sha256(path) != expected:
            mismatches.append(relative_path)
    assert mismatches == []
    manifest_expected, manifest_name = (
        (acceptance / "MANIFEST.sha256").read_text().strip().split(maxsplit=1)
    )
    assert _file_sha256(acceptance / manifest_name) == manifest_expected
