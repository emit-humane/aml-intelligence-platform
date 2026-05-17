"""
S3 — Artifact Store.

ArtifactStore wraps a filesystem directory and provides:
  - save(name, obj)          — serialise obj to <root>/<name>.joblib
  - load(name)               — deserialise and return
  - save_torch(name, sd)     — torch.save state_dict → <root>/<name>.pt
  - load_torch(name)         — torch.load → state_dict
  - save_npy(name, arr)      — np.save → <root>/<name>.npy
  - load_npy(name)           — np.load → ndarray
  - save_pkl(name, obj)      — pickle.dump → <root>/<name>.pkl
  - load_pkl(name)           — pickle.load
  - save_json(name, obj)     — json.dump → <root>/<name>.json
  - load_json(name)          — json.load
  - save_all(artifacts)      — batch-save a dict of name→obj
  - save_manifest()          — scan root and write manifest.json
  - load_manifest()          — parse and return manifest.json
  - exists(name)             — quick existence check (any extension)
  - list_artifacts()         — names of .joblib artifacts
  - metadata(name)           — size + mtime

ALL offline outputs held by this store:
  multigraph            nx.MultiDiGraph from S2
  node_index            account→int mapping
  edge_index            list of (src,dst) int pairs
  behavioral_features   52-dim feature matrix
  feature_scaler        scaler_behavioral → StandardScaler
  graph_features        per-account graph feature DataFrame
  community_profiles    per-community DataFrame
  suspicious_paths      layering/circular path DataFrame
  behavioral_profiles   per-account behavioral profile DataFrame
  iso_forest            isolation_forest → IsolationForest
  lof_model             LocalOutlierFactor (novelty=True)
  autoencoder           autoencoder → .pt state_dict
  tgn_model             TGN+MegaGNN → .pt state_dicts
  node_embeddings       node_embeddings → .npy (N, EMBEDDING_DIM)
  node_embedding_index  account→row mapping → .json
  tgn_memory_state      {node_int: Tensor} → .pkl
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import joblib


class ArtifactStore:
    """
    Filesystem-backed artifact store.

    Parameters
    ----------
    root : str | Path
        Directory where artifacts are persisted.  Created on first use.
    """

    # Names that are stored by L4A as raw files (not via save())
    _RAW_NPY   = {"node_embeddings"}
    _RAW_JSON  = {"node_embedding_index"}
    _RAW_PKL   = {"tgn_memory_state"}
    _RAW_PT    = {"tgn_model", "autoencoder"}

    def __init__(self, root: str | Path = "data/artifacts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # joblib (primary serialisation for sklearn / NetworkX / DataFrames / numpy)
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, name: str, obj: Any, compress: int = 3) -> Path:
        """Serialise obj with joblib and return the path written."""
        path = self._jl_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path, compress=compress)
        self._write_meta(name, {"saved_at": time.time(), "type": type(obj).__name__})
        return path

    def load(self, name: str) -> Any:
        """Load and return a previously saved joblib artifact."""
        path = self._jl_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Artifact '{name}' not found at {path}")
        return joblib.load(path)

    # ──────────────────────────────────────────────────────────────────────────
    # PyTorch
    # ──────────────────────────────────────────────────────────────────────────

    def save_torch(self, name: str, state_dict: Any) -> Path:
        """Save a torch state_dict (or any torch-serialisable object)."""
        import torch
        path = self.root / f"{name}.pt"
        torch.save(state_dict, path)
        self._write_meta(name + "_torch", {"saved_at": time.time(), "type": "torch"})
        return path

    def load_torch(self, name: str, map_location: str = "cpu") -> Any:
        """Load a torch artifact from <root>/<name>.pt."""
        import torch
        path = self.root / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Torch artifact '{name}' not found at {path}")
        return torch.load(path, map_location=map_location, weights_only=False)

    # ──────────────────────────────────────────────────────────────────────────
    # NumPy
    # ──────────────────────────────────────────────────────────────────────────

    def save_npy(self, name: str, arr: Any) -> Path:
        """Save a numpy array to <root>/<name>.npy."""
        import numpy as np
        path = self.root / f"{name}.npy"
        np.save(str(path), arr)
        self._write_meta(name + "_npy", {"saved_at": time.time(), "type": "numpy"})
        return path

    def load_npy(self, name: str) -> Any:
        """Load a numpy array from <root>/<name>.npy."""
        import numpy as np
        path = self.root / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"NumPy artifact '{name}' not found at {path}")
        return np.load(str(path))

    # ──────────────────────────────────────────────────────────────────────────
    # Pickle  (used for tgn_memory_state which contains Tensor objects)
    # ──────────────────────────────────────────────────────────────────────────

    def save_pkl(self, name: str, obj: Any) -> Path:
        """Pickle-dump obj to <root>/<name>.pkl."""
        path = self.root / f"{name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        self._write_meta(name + "_pkl", {"saved_at": time.time(), "type": "pickle"})
        return path

    def load_pkl(self, name: str) -> Any:
        """Pickle-load from <root>/<name>.pkl."""
        path = self.root / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Pickle artifact '{name}' not found at {path}")
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ──────────────────────────────────────────────────────────────────────────
    # JSON
    # ──────────────────────────────────────────────────────────────────────────

    def save_json(self, name: str, obj: Any) -> Path:
        """JSON-dump obj to <root>/<name>.json."""
        path = self.root / f"{name}.json"
        with open(path, "w") as fh:
            json.dump(obj, fh)
        return path

    def load_json(self, name: str) -> Any:
        """JSON-load from <root>/<name>.json."""
        path = self.root / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"JSON artifact '{name}' not found at {path}")
        with open(path) as fh:
            return json.load(fh)

    # ──────────────────────────────────────────────────────────────────────────
    # Batch save / manifest
    # ──────────────────────────────────────────────────────────────────────────

    def save_all(self, artifacts: dict[str, Any]) -> dict[str, Path]:
        """
        Batch-save a dict of {name: obj}.

        Dispatches to the right serialiser based on the name:
          - <name> in _RAW_NPY  → save_npy
          - <name> in _RAW_PKL  → save_pkl
          - <name> in _RAW_JSON → save_json
          - <name> in _RAW_PT   → save_torch
          - everything else     → save (joblib)

        Returns a dict of {name: path} for every artifact saved.
        """
        paths: dict[str, Path] = {}
        for name, obj in artifacts.items():
            if name in self._RAW_NPY:
                paths[name] = self.save_npy(name, obj)
            elif name in self._RAW_PKL:
                paths[name] = self.save_pkl(name, obj)
            elif name in self._RAW_JSON:
                paths[name] = self.save_json(name, obj)
            elif name in self._RAW_PT:
                paths[name] = self.save_torch(name, obj)
            else:
                paths[name] = self.save(name, obj)
            print(f"[S3] saved {name} -> {paths[name]}")
        return paths

    def save_manifest(self) -> Path:
        """
        Scan artifact root for all recognised files and write manifest.json.

        manifest schema:
          {
            "generated_at": <float unix ts>,
            "root": "<path>",
            "artifacts": {
              "<name>": {"path": "...", "size_bytes": N, "type": "..."}
            }
          }
        """
        artifacts: dict[str, dict] = {}

        # joblib
        for p in sorted(self.root.glob("*.joblib")):
            name = p.stem
            artifacts[name] = {
                "path":       str(p),
                "size_bytes": p.stat().st_size,
                "type":       "joblib",
            }

        # torch .pt
        for p in sorted(self.root.glob("*.pt")):
            name = p.stem
            artifacts[name] = {
                "path":       str(p),
                "size_bytes": p.stat().st_size,
                "type":       "torch",
            }

        # numpy .npy
        for p in sorted(self.root.glob("*.npy")):
            name = p.stem
            artifacts[name] = {
                "path":       str(p),
                "size_bytes": p.stat().st_size,
                "type":       "numpy",
            }

        # pickle .pkl
        for p in sorted(self.root.glob("*.pkl")):
            name = p.stem
            artifacts[name] = {
                "path":       str(p),
                "size_bytes": p.stat().st_size,
                "type":       "pickle",
            }

        # json (skip manifest.json itself, skip hidden .meta.json)
        for p in sorted(self.root.glob("*.json")):
            if p.name in ("manifest.json",) or p.name.startswith("."):
                continue
            name = p.stem
            artifacts[name] = {
                "path":       str(p),
                "size_bytes": p.stat().st_size,
                "type":       "json",
            }

        manifest = {
            "generated_at": time.time(),
            "root":         str(self.root),
            "artifacts":    artifacts,
        }
        manifest_path = self.root / "manifest.json"
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        total_mb = sum(a["size_bytes"] for a in artifacts.values()) / 1e6
        print(
            f"[S3] manifest.json written: {len(artifacts)} artifacts, "
            f"{total_mb:.1f} MB total -> {manifest_path}"
        )
        return manifest_path

    def load_manifest(self) -> dict:
        """Load and return manifest.json, or empty dict if not found."""
        path = self.root / "manifest.json"
        if not path.exists():
            return {}
        with open(path) as fh:
            return json.load(fh)

    # ──────────────────────────────────────────────────────────────────────────
    # Existence / listing / metadata
    # ──────────────────────────────────────────────────────────────────────────

    def exists(self, name: str) -> bool:
        """True if any recognised artifact file for <name> exists."""
        for ext in (".joblib", ".pt", ".npy", ".pkl", ".json"):
            if (self.root / f"{name}{ext}").exists():
                return True
        return False

    def list_artifacts(self) -> list[str]:
        """Names of all .joblib artifacts (primary store)."""
        return sorted(p.stem for p in self.root.glob("*.joblib"))

    def list_all(self) -> dict[str, str]:
        """
        Return {name: type} for every known artifact file in root.
        """
        result: dict[str, str] = {}
        for ext, typ in [
            (".joblib", "joblib"),
            (".pt",     "torch"),
            (".npy",    "numpy"),
            (".pkl",    "pickle"),
        ]:
            for p in self.root.glob(f"*{ext}"):
                result[p.stem] = typ
        for p in self.root.glob("*.json"):
            if not p.name.startswith(".") and p.name != "manifest.json":
                result[p.stem] = "json"
        return result

    def metadata(self, name: str) -> dict:
        """Size + mtime for the primary joblib artifact, if it exists."""
        path = self._jl_path(name)
        if not path.exists():
            # Try other extensions
            for ext in (".pt", ".npy", ".pkl", ".json"):
                candidate = self.root / f"{name}{ext}"
                if candidate.exists():
                    path = candidate
                    break
            else:
                return {}
        stat = path.stat()
        meta = self._read_meta(name)
        return {
            "name":       name,
            "path":       str(path),
            "size_bytes": stat.st_size,
            "mtime":      stat.st_mtime,
            **meta,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _jl_path(self, name: str) -> Path:
        return self.root / f"{name}.joblib"

    def _meta_path(self, name: str) -> Path:
        return self.root / f".{name}.meta.json"

    def _write_meta(self, name: str, meta: dict) -> None:
        try:
            with open(self._meta_path(name), "w") as fh:
                json.dump(meta, fh)
        except OSError:
            pass

    def _read_meta(self, name: str) -> dict:
        try:
            with open(self._meta_path(name)) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
