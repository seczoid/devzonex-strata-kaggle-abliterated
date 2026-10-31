"""SC117-specific model preparation and launch helpers for the Kaggle notebook."""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from scripts import strata_kaggle as sk

MODEL_REPO = "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF"
SUPPORTED_TIERS = {"Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"}
STRATA_SOURCE = Path("/kaggle/working/Strata")
SERVER_CONFIG = Path("/root/strata-local/.secrets/strata-sc117-server.json")
API_KEY_FILE = Path("/root/strata-local/.secrets/strata-api-key.txt")
SERVER_LOG = Path("/kaggle/working/strata-sc117-server.log")
ENGINE_LOG = Path("/kaggle/working/strata-sc117-engine.log")


def normalize_tier(tier: str) -> str:
    value = str(tier).strip().upper()
    if value not in SUPPORTED_TIERS:
        raise ValueError("MODEL_TIER must be one of: " + ", ".join(sorted(SUPPORTED_TIERS)))
    return value


def ensure_strata_source() -> Path:
    """Get Strata's source tree for its packer and the OpenAI-compatible server."""
    if not (STRATA_SOURCE / "tools" / "iq_pack.py").is_file() or not (STRATA_SOURCE / "serve" / "server.py").is_file():
        if STRATA_SOURCE.exists():
            shutil.rmtree(STRATA_SOURCE)
        sk.run_command(
            ["git", "clone", "--depth", "1", "https://github.com/Niko1221/Strata.git", str(STRATA_SOURCE)],
            timeout=600,
        )
    return STRATA_SOURCE


def _merge_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for src in source.rglob("*"):
        if not src.is_file():
            continue
        dst = destination / src.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            src.unlink()
            continue
        os.replace(src, dst)
    shutil.rmtree(source, ignore_errors=True)


def _merge_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        source.unlink()
        return
    os.replace(source, destination)


def sc117_model_paths(data_dir: Path, tier: str) -> dict[str, Path]:
    tier = normalize_tier(tier)
    model_dir = Path(data_dir) / "models" / "sc117" / tier
    pack_dir = Path(data_dir) / "packs" / ("sc117_" + tier.lower())
    mtp_dir = Path(data_dir) / "mtp" / "rt"
    shards = sorted(model_dir.glob("*-0000[12]-of-00002.gguf"))
    first = next((p for p in shards if "-00001-of-00002.gguf" in p.name), None)
    second = next((p for p in shards if "-00002-of-00002.gguf" in p.name), None)
    return {
        "model_dir": model_dir,
        "shard1": first or model_dir / f"Qwen3.8-Flash-Next-GSQ-RCO-abliterated-{tier}-00001-of-00002.gguf",
        "shard2": second or model_dir / f"Qwen3.8-Flash-Next-GSQ-RCO-abliterated-{tier}-00002-of-00002.gguf",
        "pack_dir": pack_dir,
        "mtp_dir": mtp_dir,
        "mtp_gguf": Path(data_dir) / "mtp" / "mtp-q2_0.gguf",
        "tokenizer_dir": pack_dir / "tokenizer",
    }


def ensure_gguf_py(source: Path) -> Path:
    """Use Strata's pinned llama.cpp gguf-py, which understands its GGUF tensor types."""
    candidates = [
        sk.STRATA_ROOT / "third_party" / "llama.cpp" / "gguf-py",
        sk.STRATA_ROOT / "terminal" / "current" / "third_party" / "llama.cpp" / "gguf-py",
    ]
    for candidate in candidates:
        if (candidate / "gguf").is_dir():
            os.environ["STRATA_GGUF_PY"] = str(candidate)
            return candidate

    setup_text = (source / "setup.py").read_text(encoding="utf-8", errors="replace")
    match = re.search(r'^LLAMA_CPP_COMMIT\s*=\s*"([0-9a-f]{40})"', setup_text, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find Strata's pinned LLAMA_CPP_COMMIT in setup.py.")
    commit = match.group(1)
    archive = Path("/kaggle/working") / f"llama.cpp-{commit[:7]}.zip"
    gguf_py = Path("/kaggle/working/strata-gguf-py")
    if not (gguf_py / "gguf").is_dir():
        if not archive.is_file():
            sk.log("Fetching Strata's pinned llama.cpp gguf-py dependency for native-pack construction.")
            sk.run_command([
                "curl", "-fL", "--retry", "3",
                f"https://github.com/ggml-org/llama.cpp/archive/{commit}.zip",
                "-o", str(archive),
            ], timeout=900)
        gguf_py.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            init_member = next(
                (name for name in names if name.endswith("/gguf-py/gguf/__init__.py")),
                None,
            )
            if not init_member:
                raise RuntimeError("Pinned llama.cpp archive does not contain gguf-py/gguf/__init__.py.")
            prefix = init_member.split("/gguf-py/gguf/__init__.py")[0] + "/gguf-py/"
            for name in names:
                if not name.startswith(prefix):
                    continue
                relative = name[len(prefix):]
                rel_path = PurePosixPath(relative)
                if not relative or rel_path.is_absolute() or ".." in rel_path.parts:
                    continue
                target = gguf_py.joinpath(*rel_path.parts)
                if name.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)
    if not (gguf_py / "gguf").is_dir():
        raise RuntimeError("Could not extract Strata's pinned gguf-py dependency.")
    os.environ["STRATA_GGUF_PY"] = str(gguf_py)
    return gguf_py


def _has_runtime(paths: dict[str, Path]) -> bool:
    required = ("dense.bin", "experts.bin", "draft_vocab.bin", "dense.txt")
    return paths["mtp_dir"].is_dir() and all((paths["mtp_dir"] / name).is_file() for name in required)


def prepare_sc117_model(repo_id: str, tier: str, data_dir: Path) -> dict[str, Path]:
    """Download only one tier plus SC117's Strata MTP files, then build its own native pack."""
    tier = normalize_tier(tier)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = sc117_model_paths(data_dir, tier)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["mtp_dir"].mkdir(parents=True, exist_ok=True)

    shards_ready = paths["shard1"].is_file() and paths["shard2"].is_file()
    mtp_ready = _has_runtime(paths)
    if not shards_ready or not mtp_ready:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            sk.run_command([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"], timeout=600)
            from huggingface_hub import snapshot_download

        download_root = data_dir / ".sc117-hf-download"
        download_root.mkdir(parents=True, exist_ok=True)
        sk.log(f"Downloading missing SC117 files for {tier}; existing complete files are reused.")
        snapshot_download(
            repo_id=repo_id or MODEL_REPO,
            local_dir=str(download_root),
            allow_patterns=[f"{tier}/*", "mtp-q2_0.gguf", "strata/rt/*"],
            max_workers=8,
        )
        _merge_tree(download_root / tier, paths["model_dir"])
        _merge_file(download_root / "mtp-q2_0.gguf", paths["mtp_gguf"])
        _merge_tree(download_root / "strata" / "rt", paths["mtp_dir"])

    paths = sc117_model_paths(data_dir, tier)
    if not paths["shard1"].is_file() or not paths["shard2"].is_file():
        raise RuntimeError(
            f"SC117 {tier} download is incomplete. Expected both GGUF shards in {paths['model_dir']}."
        )
    if paths["shard1"].stat().st_size < 1024 * 1024 or paths["shard2"].stat().st_size < 1024 * 1024:
        raise RuntimeError("One of the SC117 GGUF shards is unexpectedly small; refusing to build a pack.")
    if not _has_runtime(paths):
        raise RuntimeError(
            "SC117's Strata MTP runtime is incomplete. Expected dense.bin, experts.bin, draft_vocab.bin and dense.txt in "
            + str(paths["mtp_dir"])
        )

    source = ensure_strata_source()
    marker = paths["pack_dir"] / "sc117-pack-source.json"
    marker_data = {
        "repo_id": repo_id or MODEL_REPO,
        "tier": tier,
        "shard1": paths["shard1"].name,
        "shard1_size": paths["shard1"].stat().st_size,
        "shard2": paths["shard2"].name,
        "shard2_size": paths["shard2"].stat().st_size,
    }
    pack_ready = (
        marker.is_file()
        and (paths["pack_dir"] / "index.txt").is_file()
        and (paths["pack_dir"] / "native_experts.txt").is_file()
        and (paths["tokenizer_dir"] / "tokenizer.json").is_file()
    )
    if pack_ready:
        try:
            pack_ready = json.loads(marker.read_text(encoding="utf-8")) == marker_data
        except Exception:
            pack_ready = False

    if not pack_ready:
        ensure_gguf_py(source)
        if paths["pack_dir"].exists():
            shutil.rmtree(paths["pack_dir"])
        paths["pack_dir"].parent.mkdir(parents=True, exist_ok=True)
        sk.log("Building a dedicated SC117 native pack from shard 1. This is not the upstream Gyro-S pack.")
        result = sk.run_streaming(
            [sys.executable, str(source / "tools" / "iq_pack.py"),
             "--gguf", str(paths["shard1"]), "--out", str(paths["pack_dir"])],
            timeout=7200,
            check=False,
            heartbeat_interval=20,
            heartbeat_label="SC117 native-pack build",
            progress_paths=[paths["model_dir"], paths["pack_dir"]],
        )
        if result.returncode:
            raise RuntimeError("SC117 native-pack build failed. Review the packer output above.")
        required_pack_files = [
            paths["pack_dir"] / "index.txt",
            paths["pack_dir"] / "native_experts.txt",
            paths["tokenizer_dir"] / "tokenizer.json",
        ]
        missing = [str(p) for p in required_pack_files if not p.is_file()]
        if missing:
            raise RuntimeError("SC117 packer exited without required outputs: " + ", ".join(missing))
        marker.write_text(json.dumps(marker_data, indent=2) + "\n", encoding="utf-8")

    sk.log("SC117 model shards, model-specific native pack, tokenizer and MTP runtime are ready.")
    return paths


def _context_tokens(context: str) -> int:
    value = str(context).strip().lower()
    if value.endswith("k"):
        return int(float(value[:-1]) * 1024)
    return int(value)


def _engine_component_present() -> bool:
    root = sk.STRATA_ROOT / "components" / "engine"
    return root.is_dir() and any(p.is_dir() for p in root.iterdir())


def ensure_engine_component(paths: dict[str, Path], context: str, gpu_list: str) -> None:
    """Refresh/check Strata's GPU engine, using only the already-local SC117 model files."""
    if _engine_component_present():
        # Kaggle may assign a different GPU generation on a later session. Let Strata
        # update/check the cached engine for the currently visible hardware first.
        update = sk.run_streaming(
            ["strata", "launch", "--update", "--yes"],
            timeout=1800, check=False,
            heartbeat_interval=20, heartbeat_label="Strata engine compatibility check",
            progress_paths=[sk.STRATA_ROOT / "components"],
        )
        if update.returncode:
            sk.log("Strata engine update/check returned a non-zero exit code; continuing with the cached engine and validating startup.")
        if _engine_component_present():
            sk.log("Strata engine component is present; using the component Strata selected for this session.")
            return

    sk.log("No Strata engine component is cached. Asking Strata to prepare from the local SC117 GGUF files.")
    bootstrap_data = Path("/root/strata-sc117-engine-bootstrap")
    # Strata Terminal 0.2.x uses --model-path for local GGUF files and has no
    # setup-only --no-start flag. The engine is fetched/test-started as part of
    # launch, so start against the already-downloaded SC117 files, then stop it
    # before the notebook starts the explicit custom server config.
    base = [
        "strata", "launch", "--model-path", str(paths["model_dir"]),
        "--data-dir", str(bootstrap_data),
        "--context", context, "--vision", "off", "--mtp", "off",
        "--access", "local", "--port", "8081", "--gpu", gpu_list,
        "--yes", "--no-tui",
    ]
    if not _engine_component_present():
        sk.log("Engine not installed yet; bootstrapping it using the local SC117 files (no stock model download).")
        started = sk.run_streaming(
            base, timeout=7200, check=False,
            heartbeat_interval=20, heartbeat_label="Local SC117 engine bootstrap",
            progress_paths=[bootstrap_data, sk.STRATA_ROOT / "components"],
        )
        if _engine_component_present():
            sk.run_command(["strata", "launch", "--stop"], check=False, timeout=120)
        elif started.returncode:
            raise RuntimeError(
                "Strata did not install a compatible engine component. Inspect the bootstrap output above; "
                "no upstream model weights were intentionally downloaded."
            )

    if not _engine_component_present():
        raise RuntimeError("Strata's engine component is still missing after local-model preparation.")

    # The bootstrap config/pack is disposable; the actual server uses packs/sc117_<tier>.
    shutil.rmtree(bootstrap_data, ignore_errors=True)
    sk.log("Strata engine component is ready.")


def run_sc117_plan(paths: dict[str, Path], context: str, gpu_list: str, data_dir: Path) -> str:
    """Preflight the custom config and ensure the engine exists; do not start the final server yet."""
    required = [
        paths["shard1"], paths["shard2"], paths["pack_dir"] / "index.txt",
        paths["pack_dir"] / "native_experts.txt", paths["tokenizer_dir"] / "tokenizer.json",
        paths["mtp_dir"] / "dense.bin", paths["mtp_dir"] / "experts.bin",
        paths["mtp_dir"] / "draft_vocab.bin", paths["mtp_dir"] / "dense.txt",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("SC117 preflight failed; missing required files:\n" + "\n".join(missing))
    source = ensure_strata_source()
    profile = source / "data" / "expert-profile.bin"
    if not profile.is_file():
        raise RuntimeError("Strata source checkout is missing data/expert-profile.bin.")
    ensure_engine_component(paths, context, gpu_list)
    plan = {
        "model": f"qwen3.8-flash-next-sc117-{paths['model_dir'].name.lower()}",
        "tier": paths["model_dir"].name,
        "shard1": str(paths["shard1"]),
        "shard2": str(paths["shard2"]),
        "native_pack": str(paths["pack_dir"]),
        "tokenizer": str(paths["tokenizer_dir"]),
        "mtp_runtime": str(paths["mtp_dir"]),
        "context_tokens": _context_tokens(context),
        "gpu_ids": gpu_list,
        "engine": "component:strata-engine",
        "api": "OpenAI-compatible API on 127.0.0.1:8080, protected by a generated API key",
    }
    print(json.dumps(plan, indent=2))
    return json.dumps(plan, indent=2)


def launch_sc117(paths: dict[str, Path], context: str, gpu_list: str, data_dir: Path) -> str:
    """Start the SC117 config with a generated API key; no Strata CLI stock preset is used."""
    source = ensure_strata_source()
    if not _engine_component_present():
        ensure_engine_component(paths, context, gpu_list)

    SERVER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    API_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if API_KEY_FILE.is_file():
        api_key = API_KEY_FILE.read_text(encoding="utf-8").strip()
        if not api_key.startswith("sk-strata-") or len(api_key) < 32:
            api_key = "sk-strata-" + secrets.token_urlsafe(36)
    else:
        api_key = "sk-strata-" + secrets.token_urlsafe(36)
    API_KEY_FILE.write_text(api_key + "\n", encoding="utf-8")
    API_KEY_FILE.chmod(0o600)
    os.environ["STRATA_API_KEY_FILE"] = str(API_KEY_FILE)

    config = {
        "engine": "strata",
        "exe": "component:strata-engine",
        "args": [
            "--pack", str(paths["pack_dir"]),
            "--native", str(paths["shard1"]),
            "--ple-gguf", str(paths["shard2"]),
            "--expert-profile", str(source / "data" / "expert-profile.bin"),
            "--expert-cache", "auto",
            "--prefill", "auto",
            "--spec", "4",
            "--spec-min-p", "0.5",
            "--mtp", str(paths["mtp_dir"]),
            "--max-context", str(_context_tokens(context)),
            "--kv", "int8",
            "--vram-reserve-mib", "1024",
            "--gpu", gpu_list,
        ],
        "cwd": str(source),
        "tokenizer": str(paths["tokenizer_dir"]),
        "model_name": f"qwen3.8-flash-next-sc117-{paths['model_dir'].name.lower()}",
        "log": str(ENGINE_LOG),
        "host": "127.0.0.1",
        "port": 8080,
        "api_key": api_key,
    }
    SERVER_CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    SERVER_CONFIG.chmod(0o600)

    try:
        import requests
        response = requests.get("http://127.0.0.1:8080/health", timeout=3)
        if response.ok and response.json().get("loaded"):
            current = str(response.json().get("model") or "")
            if "sc117" in current.lower():
                sk.log("The SC117 model is already serving; reusing it.")
                return "SC117 server already running."
            raise RuntimeError(
                f"Port 8080 is already serving a different model ({current}). Stop that server before launching SC117."
            )
    except RuntimeError:
        raise
    except Exception:
        pass

    env = os.environ.copy()
    env["PYTHONPATH"] = str(source) + os.pathsep + str(source / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    with SERVER_LOG.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "serve.server", "--engine", "strata",
             "--config", str(SERVER_CONFIG), "--port", "8080"],
            cwd=str(source), env=env, stdout=output, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, text=True,
        )
    time.sleep(2)
    if process.poll() is not None:
        tail = SERVER_LOG.read_text(encoding="utf-8", errors="replace")[-6000:] if SERVER_LOG.exists() else ""
        raise RuntimeError("SC117 server exited during startup. Server log tail:\n" + tail)
    sk.log("Started the custom SC117 server process. Waiting for health check in the next cell.")
    return "SC117 custom server process started."
