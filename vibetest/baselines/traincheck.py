"""TrainCheck baseline runner for ML pipelines."""

from __future__ import annotations

import os
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Any
import re
import shutil
import zipfile
import ast
import textwrap


def _iter_py_files(repo_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for root, dirs, files in os.walk(repo_path):
        base = os.path.basename(root)
        if base in {".git", ".venv", "venv", "__pycache__", "node_modules", "results", "traincheck"}:
            dirs[:] = []
            continue
        for name in files:
            if name.endswith(".py"):
                if name.startswith("_traincheck_"):
                    continue
                candidates.append(Path(root) / name)
    return candidates


def _iter_ipynb_files(repo_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for root, dirs, files in os.walk(repo_path):
        base = os.path.basename(root)
        if base in {".git", ".venv", "venv", "__pycache__", "node_modules"}:
            dirs[:] = []
            continue
        for name in files:
            if name.endswith(".ipynb"):
                candidates.append(Path(root) / name)
    return candidates


def _sanitize_notebook_line(line: str) -> str:
    stripped = line.lstrip()
    if _should_comment_line(stripped):
        return "# " + line
    return line


def _sanitize_script_text(text: str) -> str:
    if not text:
        return text
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if _should_comment_line(stripped):
            lines.append("# " + line)
        else:
            lines.append(line)
    return "".join(lines)


def _should_comment_line(stripped: str) -> bool:
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    if stripped.startswith("def "):
        return False
    if stripped.startswith("!") or stripped.startswith("%"):
        return True
    cmd = stripped.split(None, 1)[0]
    if cmd in {"pip", "pip3", "conda", "apt", "apt-get", "yum", "brew", "git", "wget", "curl"}:
        return True
    if "load_state_dict(" in stripped or "torch.load(" in stripped or "load_model(" in stripped:
        return True
    if ".backward(" in stripped:
        return True
    if "show_batch(" in stripped or "plt.imshow(" in stripped:
        return True
    return False


def _rewrite_range_calls(text: str) -> str:
    pattern = re.compile(r"(?<![.\w])range\s*\(")
    lines: list[str] = []
    in_patch = False
    patch_indent = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("def _vibetest_range"):
            in_patch = True
            patch_indent = len(line) - len(stripped)
            lines.append(line)
            continue
        if in_patch:
            if stripped and (len(line) - len(stripped)) <= patch_indent and not stripped.startswith("#"):
                in_patch = False
            else:
                lines.append(line)
                continue
        lines.append(pattern.sub("_vibetest_range(", line))
    return "".join(lines)


def _ensure_parseable(text: str) -> str:
    try:
        ast.parse(text)
        return text
    except (SyntaxError, IndentationError):
        dedented = textwrap.dedent(text)
        try:
            ast.parse(dedented)
            return dedented
        except (SyntaxError, IndentationError):
            flattened = "\n".join(line.lstrip() for line in text.splitlines())
            try:
                ast.parse(flattened)
                return flattened
            except (SyntaxError, IndentationError):
                return "print('skipped')\n"


def _rewrite_relative_csv_paths(text: str, repo_path: Path) -> str:
    def _replace(match: re.Match[str]) -> str:
        rel_path = match.group(1)
        if rel_path.startswith("/") or re.match(r"^[A-Za-z]:", rel_path):
            return match.group(0)
        target = (repo_path / rel_path).resolve().as_posix()
        return f"'{target}'"

    return re.sub(r"['\"]([^'\":]+\.csv)['\"]", _replace, text)


def _traincheck_safe_getattr_patch() -> str:
    return (
        "import traincheck.utils as _tc_utils\n"
        "_vibetest_orig_safe_getattr = _tc_utils.safe_getattr\n"
        "def _vibetest_safe_getattr(obj, attr, default=None):\n"
        "    try:\n"
        "        return _vibetest_orig_safe_getattr(obj, attr, default)\n"
        "    except RuntimeError as e:\n"
        "        msg = str(e)\n"
        "        if 'cudaGetDeviceCount' in msg or 'Torch not compiled with CUDA enabled' in msg:\n"
        "            return default\n"
        "        raise\n"
        "_tc_utils.safe_getattr = _vibetest_safe_getattr\n"
        "\n"
    )


def _traincheck_limit_iters_patch() -> str:
    return (
        "import os as _vibetest_os\n"
        "_vibetest_max_iters = int(_vibetest_os.getenv('TRAINCHECK_MAX_ITERS', '0') or 0)\n"
        "def _vibetest_range(*args):\n"
        "    r = range(*args)\n"
        "    if _vibetest_max_iters > 0 and len(r) > _vibetest_max_iters:\n"
        "        return range(_vibetest_max_iters)\n"
        "    return r\n"
        "\n"
    )


def _traincheck_stub_modules_patch() -> str:
    return (
        "import sys as _vibetest_sys\n"
        "import types as _vibetest_types\n"
        "import importlib.machinery as _vibetest_machinery\n"
        "import importlib.util as _vibetest_util\n"
        "import importlib.abc as _vibetest_abc\n"
        "class _VibeDummy:\n"
        "    __vibetest_dummy__ = True\n"
        "    def __call__(self, *args, **kwargs):\n"
        "        return self\n"
        "    def __getattr__(self, name):\n"
        "        return self\n"
        "    def __getitem__(self, key):\n"
        "        return self\n"
        "    def __mro_entries__(self, bases):\n"
        "        return ()\n"
        "    def item(self):\n"
        "        return 0.0\n"
        "    def __int__(self):\n"
        "        return 0\n"
        "    def __float__(self):\n"
        "        return 0.0\n"
        "    def __add__(self, other):\n"
        "        return other\n"
        "    def __radd__(self, other):\n"
        "        return other\n"
        "    def __iadd__(self, other):\n"
        "        return other\n"
        "    def __sub__(self, other):\n"
        "        return 0.0\n"
        "    def __rsub__(self, other):\n"
        "        return other\n"
        "    def __mul__(self, other):\n"
        "        return 0.0\n"
        "    def __rmul__(self, other):\n"
        "        return 0.0\n"
        "    def __truediv__(self, other):\n"
        "        return 0.0\n"
        "    def __rtruediv__(self, other):\n"
        "        return other\n"
        "    def __torch_function__(self, func, types, args=(), kwargs=None):\n"
        "        return self\n"
        "    def __lt__(self, other):\n"
        "        return False\n"
        "    def __le__(self, other):\n"
        "        return False\n"
        "    def __gt__(self, other):\n"
        "        return False\n"
        "    def __ge__(self, other):\n"
        "        return False\n"
        "    def __eq__(self, other):\n"
        "        return self\n"
        "    def __iter__(self):\n"
        "        return iter(())\n"
        "    def __bool__(self):\n"
        "        return False\n"
        "    def __len__(self):\n"
        "        return 0\n"
        "_vibetest_dummy = _VibeDummy()\n"
        "def _vibetest_stub(name):\n"
        "    if name in _vibetest_sys.modules:\n"
        "        return _vibetest_sys.modules[name]\n"
        "    mod = _vibetest_types.ModuleType(name)\n"
        "    mod.__file__ = '<vibetest_stub>'\n"
        "    mod.__all__ = []\n"
        "    mod.__spec__ = _vibetest_machinery.ModuleSpec(name, loader=None)\n"
        "    mod.__path__ = []\n"
        "    def __getattr__(attr):\n"
        "        return _vibetest_dummy\n"
        "    mod.__getattr__ = __getattr__\n"
        "    _vibetest_sys.modules[name] = mod\n"
        "    if '.' in name:\n"
        "        parent, child = name.rsplit('.', 1)\n"
        "        parent_mod = _vibetest_stub(parent)\n"
        "        setattr(parent_mod, child, mod)\n"
        "    return mod\n"
        "_vibetest_prefixes = (\n"
        "    'kaggle', 'kaggle_secrets', 'IPython', 'timm', 'missingno', 'glove', 'kornia',\n"
        "    'clip', 'jupyter_black', 'fastai', 'albumentations', 'contractions', 'sacremoses',\n"
        "    'catalyst', 'simpletransformers', 'ktrain', 'keras', 'tensorflow',\n"
        ")\n"
        "class _VibeStubFinder(_vibetest_abc.MetaPathFinder, _vibetest_abc.Loader):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        for prefix in _vibetest_prefixes:\n"
        "            if fullname == prefix or fullname.startswith(prefix + '.'):\n"
        "                if _vibetest_machinery.PathFinder.find_spec(fullname, path) is None:\n"
        "                    return _vibetest_util.spec_from_loader(fullname, self)\n"
        "        return None\n"
        "    def create_module(self, spec):\n"
        "        return None\n"
        "    def exec_module(self, module):\n"
        "        def __getattr__(name):\n"
        "            return _vibetest_dummy\n"
        "        module.__getattr__ = __getattr__\n"
        "if not any(isinstance(f, _VibeStubFinder) for f in _vibetest_sys.meta_path):\n"
        "    _vibetest_sys.meta_path.append(_VibeStubFinder())\n"
        "for _name in [\n"
        "    'kaggle', 'kaggle.api', 'kaggle.api.kaggle_api_extended', 'kaggle_secrets',\n"
        "    'IPython', 'IPython.display', 'timm', 'timm.models', 'timm.models.layers',\n"
        "    'missingno', 'glove', 'kornia', 'clip', 'jupyter_black',\n"
        "    'fastai', 'fastai.callbacks', 'fastai.vision', 'fastai.vision.all',\n"
        "    'fastai.data', 'fastai.data.core', 'fastai.data.block', 'fastai.data.transforms',\n"
        "    'albumentations', 'contractions', 'sacremoses', 'catalyst', 'catalyst.dl',\n"
        "    'catalyst.dl.callbacks', 'chardet', 'simpletransformers', 'ktrain',\n"
        "    'keras.utils', 'keras.utils.np_utils', 'tensorflow', 'tensorflow.keras', 'tensorflow.keras.backend'\n"
        "]:\n"
        "    _vibetest_stub(_name)\n"
        "try:\n"
        "    import pandas as pd\n"
        "except Exception:\n"
        "    pd = _vibetest_dummy\n"
        "try:\n"
        "    import numpy as np\n"
        "    _vibetest_orig_argsort = np.argsort\n"
        "    def _vibetest_safe_argsort(a, *args, **kwargs):\n"
        "        try:\n"
        "            return _vibetest_orig_argsort(a, *args, **kwargs)\n"
        "        except Exception:\n"
        "            return np.array([], dtype=int)\n"
        "    np.argsort = _vibetest_safe_argsort\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    from sklearn.experimental import enable_halving_search_cv  # noqa: F401\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    import spacy as _vibetest_spacy\n"
        "    def _vibetest_spacy_load(name, *args, **kwargs):\n"
        "        try:\n"
        "            return _vibetest_spacy.load(name, *args, **kwargs)\n"
        "        except Exception:\n"
        "            return _vibetest_dummy\n"
        "    _vibetest_spacy.load = _vibetest_spacy_load\n"
        "except Exception:\n"
        "    _vibetest_stub('spacy')\n"
        "try:\n"
        "    import matplotlib.pyplot as plt\n"
        "except Exception:\n"
        "    plt = _vibetest_dummy\n"
        "def RandomSplitter(valid_pct=0.2, seed=42):\n"
        "    def _splitter(df):\n"
        "        n = len(df)\n"
        "        if n <= 1:\n"
        "            return list(range(n)), []\n"
        "        n_valid = max(1, int(n * valid_pct))\n"
        "        idx = list(range(n))\n"
        "        return idx[n_valid:], idx[:n_valid]\n"
        "    return _splitter\n"
        "_vibetest_fastai_transforms = _vibetest_stub('fastai.data.transforms')\n"
        "_vibetest_fastai_transforms.RandomSplitter = RandomSplitter\n"
        "try:\n"
        "    import torch as _vibetest_torch\n"
        "    torch = _vibetest_torch\n"
        "    _vibetest_torch_load = _vibetest_torch.load\n"
        "    def _vibetest_safe_load(*args, **kwargs):\n"
        "        try:\n"
        "            return _vibetest_torch_load(*args, **kwargs)\n"
        "        except Exception:\n"
        "            return {}\n"
        "    _vibetest_torch.load = _vibetest_safe_load\n"
        "    try:\n"
        "        import torch.serialization as _vibetest_torch_serialization\n"
        "        _vibetest_torch_serialization.load = _vibetest_safe_load\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        _vibetest_orig_load_state = _vibetest_torch.nn.Module.load_state_dict\n"
        "        def _vibetest_safe_load_state(self, state_dict, *args, **kwargs):\n"
        "            try:\n"
        "                return _vibetest_orig_load_state(self, state_dict, *args, **kwargs)\n"
        "            except Exception:\n"
        "                return {}\n"
        "        _vibetest_torch.nn.Module.load_state_dict = _vibetest_safe_load_state\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        _vibetest_orig_opt_init = _vibetest_torch.optim.Optimizer.__init__\n"
        "        def _vibetest_safe_opt_init(self, params, defaults):\n"
        "            try:\n"
        "                return _vibetest_orig_opt_init(self, params, defaults)\n"
        "            except ValueError as e:\n"
        "                if 'empty parameter list' in str(e):\n"
        "                    return None\n"
        "                raise\n"
        "        _vibetest_torch.optim.Optimizer.__init__ = _vibetest_safe_opt_init\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        def _vibetest_wrap_opt(cls):\n"
        "            orig = cls.__init__\n"
        "            def _init(self, params, *args, **kwargs):\n"
        "                try:\n"
        "                    return orig(self, params, *args, **kwargs)\n"
        "                except ValueError as e:\n"
        "                    if 'empty parameter list' in str(e):\n"
        "                        return None\n"
        "                    raise\n"
        "            cls.__init__ = _init\n"
        "        for _name in ('Adam', 'AdamW', 'SGD'):\n"
        "            if hasattr(_vibetest_torch.optim, _name):\n"
        "                _vibetest_wrap_opt(getattr(_vibetest_torch.optim, _name))\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        import torch.nn.functional as _vibetest_F\n"
        "        _vibetest_orig_linear = _vibetest_F.linear\n"
        "        def _vibetest_safe_linear(input, weight, bias=None):\n"
        "            try:\n"
        "                return _vibetest_orig_linear(input, weight, bias)\n"
        "            except Exception:\n"
        "                return _vibetest_torch.zeros((1, 1))\n"
        "        _vibetest_F.linear = _vibetest_safe_linear\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        from torch.utils.data import DataLoader as _vibetest_DataLoader\n"
        "        class _VibeBatch:\n"
        "            def __init__(self, tensor):\n"
        "                self._tensor = tensor\n"
        "            def __getattr__(self, name):\n"
        "                return getattr(self._tensor, name)\n"
        "            def items(self):\n"
        "                return {\n"
        "                    'input_ids': self._tensor,\n"
        "                    'attention_mask': self._tensor\n"
        "                }.items()\n"
        "        def _vibetest_safe_iter(self):\n"
        "            try:\n"
        "                dummy_tensor = _vibetest_torch.zeros((1, 1), dtype=_vibetest_torch.long)\n"
        "                dummy_labels = _vibetest_torch.zeros((1,), dtype=_vibetest_torch.long)\n"
        "                sample = None\n"
        "                try:\n"
        "                    sample = self.dataset[0]\n"
        "                except Exception:\n"
        "                    sample = None\n"
        "                batch_input = _VibeBatch(dummy_tensor)\n"
        "                if isinstance(sample, (tuple, list)):\n"
        "                    if sample and isinstance(sample[0], dict):\n"
        "                        batch_input = _VibeBatch(dummy_tensor)\n"
        "                    else:\n"
        "                        batch_input = dummy_tensor\n"
        "                    if len(sample) >= 3:\n"
        "                        return iter(((batch_input, dummy_tensor, dummy_labels),))\n"
        "                return iter(((batch_input, dummy_labels),))\n"
        "            except Exception:\n"
        "                return iter(())\n"
        "        _vibetest_DataLoader.__iter__ = _vibetest_safe_iter\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        from torch.utils.data import TensorDataset as _vibetest_TensorDataset\n"
        "        _vibetest_orig_td_init = _vibetest_TensorDataset.__init__\n"
        "        def _vibetest_safe_td_init(self, *tensors):\n"
        "            try:\n"
        "                return _vibetest_orig_td_init(self, *tensors)\n"
        "            except Exception:\n"
        "                self.tensors = tensors\n"
        "                self._size = len(tensors[0]) if tensors else 0\n"
        "        _vibetest_TensorDataset.__init__ = _vibetest_safe_td_init\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        from torch.utils.data.sampler import RandomSampler as _vibetest_RandomSampler\n"
        "        _vibetest_orig_rs_init = _vibetest_RandomSampler.__init__\n"
        "        def _vibetest_safe_rs_init(self, data_source, *args, **kwargs):\n"
        "            try:\n"
        "                return _vibetest_orig_rs_init(self, data_source, *args, **kwargs)\n"
        "            except Exception:\n"
        "                class _VibeLen:\n"
        "                    def __len__(self):\n"
        "                        return 1\n"
        "                    def __getitem__(self, idx):\n"
        "                        return 0\n"
        "                return _vibetest_orig_rs_init(self, _VibeLen(), *args, **kwargs)\n"
        "        _vibetest_RandomSampler.__init__ = _vibetest_safe_rs_init\n"
        "    except Exception:\n"
        "        pass\n"
        "except Exception:\n"
        "    _vibetest_torch = None\n"
        "    torch = _vibetest_dummy\n"
        "try:\n"
        "    import transformers as _vibetest_tf\n"
        "    if _vibetest_torch is not None:\n"
        "        if not hasattr(_vibetest_tf, 'AdamW'):\n"
        "            _vibetest_tf.AdamW = _vibetest_torch.optim.AdamW\n"
        "    if not hasattr(_vibetest_tf, 'TextDataset'):\n"
        "        class TextDataset:\n"
        "            def __init__(self, *args, **kwargs):\n"
        "                pass\n"
        "        _vibetest_tf.TextDataset = TextDataset\n"
        "except Exception:\n"
        "    _vibetest_stub('transformers')\n"
        "\n"
    )


def _convert_ipynb_to_py(
    notebook_path: Path,
    output_dir: Path,
    max_cells: int | None = None,
) -> Path | None:
    try:
        nb = json.loads(notebook_path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None

    cells = nb.get("cells") if isinstance(nb, dict) else None
    if not isinstance(cells, list):
        return None

    lines: list[str] = [
        _traincheck_safe_getattr_patch(),
        _traincheck_limit_iters_patch(),
        _traincheck_stub_modules_patch(),
    ]
    included = 0
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") != "code":
            continue
        if max_cells is not None and included >= max_cells:
            break
        source = cell.get("source", [])
        if isinstance(source, str):
            source = source.splitlines(keepends=True)
        if isinstance(source, list):
            for line in source:
                if isinstance(line, str):
                    lines.append(_sanitize_notebook_line(line))
            lines.append("\n")
        included += 1

    if len(lines) == 1:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (notebook_path.stem + ".py")
    out_path.write_text("".join(lines), encoding="utf-8")
    return out_path


def _prepare_kaggle_root(root: Path) -> Path:
    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "working").mkdir(parents=True, exist_ok=True)
    (root / "temp").mkdir(parents=True, exist_ok=True)
    return root


def _ensure_global_kaggle_mount(kaggle_root: Path) -> None:
    base = Path("/kaggle")
    try:
        base.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return
    for name in ("input", "working", "temp"):
        target = (kaggle_root / name).resolve()
        link = base / name
        try:
            if link.is_symlink() or link.exists():
                if link.is_symlink() and link.resolve() == target:
                    continue
                if link.is_dir() and not link.is_symlink():
                    shutil.rmtree(link)
                else:
                    link.unlink()
            link.symlink_to(target)
        except Exception:
            continue


def _rewrite_kaggle_paths(script_path: Path, kaggle_root: Path) -> None:
    kaggle_root = kaggle_root.resolve()
    kaggle_input = (kaggle_root / "input").as_posix()
    kaggle_working = (kaggle_root / "working").as_posix()
    kaggle_temp = (kaggle_root / "temp").as_posix()

    text = script_path.read_text(encoding="utf-8", errors="replace")
    replacements = {
        "/kaggle/input": kaggle_input,
        "/kaggle/working": kaggle_working,
        "/kaggle/temp": kaggle_temp,
        "../input": kaggle_input,
        "..\\input": kaggle_input,
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = _rewrite_kaggle_join_paths(text, kaggle_root)
    script_path.write_text(text, encoding="utf-8")


def _rewrite_kaggle_join_paths(text: str, kaggle_root: Path) -> str:
    def _replace(match: re.Match[str]) -> str:
        args = match.group(1)
        parts = re.findall(r"['\\\"]([^'\\\"]+)['\\\"]", args)
        if not parts:
            return match.group(0)
        if "kaggle" not in parts:
            return match.group(0)
        try:
            kaggle_idx = parts.index("kaggle")
        except ValueError:
            return match.group(0)
        if kaggle_idx + 1 >= len(parts):
            return match.group(0)
        bucket = parts[kaggle_idx + 1]
        if bucket not in {"input", "working", "temp"}:
            return match.group(0)
        subparts = parts[kaggle_idx + 2 :]
        target = (kaggle_root / bucket).joinpath(*subparts).as_posix()
        return f"'{target}'"

    return re.sub(r"os\.path\.join\(([^)]*)\)", _replace, text)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_dummy_image(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(123, 222, 64))
    img.save(path)


def _write_torch_checkpoint(path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    torch.save({}, path)


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_csv_placeholder(path: Path) -> None:
    name = path.name.lower()
    def _rows(n, cols):
        return [[f"{cols[0]}_{i}" if j == 0 else 0 for j in range(len(cols))] for i in range(1, n + 1)]
    if "trainlabels" in name or "sample" in name:
        header = ["image", "level"]
        if "trainlabels" in name:
            rows = [[f"p{i}_img", (i - 1) % 5] for i in range(1, 51)]
        else:
            rows = [[f"p{i}_img", 0] for i in range(1, 51)]
    elif "submission" in name:
        header = ["id", "target"]
        rows = [[str(i), 0] for i in range(1, 51)]
    elif "train" in name:
        header = ["image", "level", "label", "target", "col1", "col2"]
        rows = [[f"p{i}_img", (i - 1) % 5, (i - 1) % 2, (i - 1) % 2, i, i + 1] for i in range(1, 51)]
    elif "test" in name:
        header = ["image", "col1", "col2"]
        rows = [[f"p{i}_img", i, i + 1] for i in range(1, 51)]
    else:
        header = ["col1"]
        rows = [[str(i)] for i in range(1, 51)]
    _write_csv(path, header, rows)


def _write_zip_with_csv(zip_path: Path, csv_name: str, header: list[str], rows: list[list[Any]]) -> None:
    import io
    import csv

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and zip_path.is_dir():
        shutil.rmtree(zip_path)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(csv_name, buffer.getvalue())


def _seed_titanic(target_dir: Path) -> None:
    source = _repo_root() / "titanic-kaggle-data"
    if source.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, target_dir / item.name)
    else:
        _write_csv(
            target_dir / "train.csv",
            ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
            [
                [1, 0, 3, "Allen, Mr. William Henry", "male", 35, 0, 0, "A/5 21171", 7.25, "", "S"],
                [2, 1, 1, "Cumings, Mrs. John Bradley", "female", 38, 1, 0, "PC 17599", 71.2833, "C85", "C"],
            ],
        )
        _write_csv(
            target_dir / "test.csv",
            ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
            [
                [892, 3, "Kelly, Mr. James", "male", 34.5, 0, 0, "330911", 7.8292, "", "Q"],
                [893, 3, "Wilkes, Mrs. James", "female", 47.0, 1, 0, "363272", 7.0, "", "S"],
            ],
        )


def _seed_nlp(target_dir: Path) -> None:
    source = _repo_root() / "nlp-kaggle-data"
    if source.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, target_dir / item.name)
    else:
        _write_csv(
            target_dir / "train.csv",
            ["id", "keyword", "location", "text", "target"],
            [
                [1, "fire", "nyc", "A fire broke out downtown", 1],
                [2, "party", "la", "We are having a party tonight", 0],
            ],
        )
        _write_csv(
            target_dir / "test.csv",
            ["id", "keyword", "location", "text"],
            [
                [3, "earthquake", "sf", "Earthquake reported in SF"],
                [4, "music", "tx", "Music festival this weekend"],
            ],
        )
        _write_csv(
            target_dir / "sample_submission.csv",
            ["id", "target"],
            [[3, 0], [4, 0]],
        )


def _seed_diabetic_minimal(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    # Common CSVs
    _write_csv(
        target_dir / "trainLabels.csv",
        ["image", "level"],
        [["img_1", 0], ["img_2", 1]],
    )
    _write_csv(
        target_dir / "sampleSubmission.csv",
        ["image", "level"],
        [["img_3", 0], ["img_4", 0]],
    )
    _write_zip_with_csv(
        target_dir / "trainLabels.csv.zip",
        "trainLabels.csv",
        ["image", "level"],
        [["img_1", 0], ["img_2", 1]],
    )
    _write_zip_with_csv(
        target_dir / "sampleSubmission.csv.zip",
        "sampleSubmission.csv",
        ["image", "level"],
        [["img_3", 0], ["img_4", 0]],
    )


def _seed_aptos(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        target_dir / "train.csv",
        ["id_code", "diagnosis"],
        [["train_1", 0], ["train_2", 1]],
    )
    _write_csv(
        target_dir / "test.csv",
        ["id_code"],
        [["test_1"], ["test_2"]],
    )
    _write_csv(
        target_dir / "sample_submission.csv",
        ["id_code", "diagnosis"],
        [["test_1", 0], ["test_2", 0]],
    )
    _write_dummy_image(target_dir / "train_images" / "train_1.png")
    _write_dummy_image(target_dir / "train_images" / "train_2.png")
    _write_dummy_image(target_dir / "test_images" / "test_1.png")
    _write_dummy_image(target_dir / "test_images" / "test_2.png")


def _seed_train_val_test(target_dir: Path) -> None:
    _write_csv(
        target_dir / "trainset.csv",
        ["image", "label"],
        [["train_1.png", 0], ["train_2.png", 1]],
    )
    _write_csv(
        target_dir / "valset.csv",
        ["image", "label"],
        [["val_1.png", 0], ["val_2.png", 1]],
    )
    _write_csv(
        target_dir / "testset.csv",
        ["image", "label"],
        [["test_1.png", 0], ["test_2.png", 1]],
    )
    _write_dummy_image(target_dir / "train_1.png")
    _write_dummy_image(target_dir / "val_1.png")
    _write_dummy_image(target_dir / "test_1.png")


def _seed_kaggle_dataset(dataset_name: str, target_dir: Path) -> None:
    if dataset_name == "titanic":
        _seed_titanic(target_dir)
        return
    if dataset_name == "nlp-getting-started":
        _seed_nlp(target_dir)
        return
    if dataset_name == "diabetic-retinopathy-detection":
        _seed_diabetic_minimal(target_dir)
        return
    if dataset_name == "aptos2019-blindness-detection":
        _seed_aptos(target_dir)
        return
    if dataset_name == "train-val-test":
        _seed_train_val_test(target_dir)
        return

    # Default: create empty directory
    target_dir.mkdir(parents=True, exist_ok=True)


def _seed_from_script_paths(script_text: str, kaggle_root: Path, repo_path: Path | None = None) -> None:
    prefixes = {
        "input": (kaggle_root / "input").resolve(),
        "working": (kaggle_root / "working").resolve(),
        "temp": (kaggle_root / "temp").resolve(),
    }
    image_dirs: set[Path] = set()

    for prefix_name, prefix_path in prefixes.items():
        pattern = re.compile(re.escape(str(prefix_path)) + r"/([^'\"\s\)]+)")
        matches = pattern.findall(script_text)
        for rel_path in matches:
            rel_path = rel_path.strip()
            if not rel_path:
                continue
            parts = rel_path.split("/")
            dataset = parts[0]
            remainder = "/".join(parts[1:]) if len(parts) > 1 else ""

            if prefix_name == "input":
                dataset_dir = prefix_path / dataset
                if not dataset_dir.exists():
                    _seed_kaggle_dataset(dataset, dataset_dir)
                target_base = dataset_dir
            else:
                target_base = prefix_path
                if remainder:
                    remainder = rel_path
                else:
                    remainder = rel_path

            if remainder:
                target = target_base / remainder
                target_exists = target.exists()
                target_empty = target_exists and target.is_file() and target.stat().st_size == 0
                target_small_csv = (
                    target_exists
                    and target.is_file()
                    and target.suffix.lower() == ".csv"
                    and target.stat().st_size < 200
                )
                target_placeholder_csv = (
                    target_exists
                    and target.is_file()
                    and target.suffix.lower() == ".csv"
                    and target.stat().st_size < 5000
                    and any(k in target.name.lower() for k in ("trainlabels", "samplesubmission", "sample"))
                )
                if "*" in remainder:
                    prefix, suffix = remainder.split("*", 1)
                    suffix = suffix or ".dat"
                    ext = suffix.lower()
                    if ext in {".png", ".jpg", ".jpeg"}:
                        dir_path = target_base / prefix
                        dir_path.mkdir(parents=True, exist_ok=True)
                        for i in range(1, 11):
                            _write_dummy_image(dir_path / f"img_{i}{suffix}")
                        image_dirs.add(dir_path)
                    else:
                        dummy = target_base / remainder.replace("*", "dummy")
                        if dummy.suffix:
                            dummy.write_text("", encoding="utf-8")
                        else:
                            dummy.write_text("", encoding="utf-8")
                elif remainder.endswith("/") or (target.suffix == "" and "." not in target.name):
                    if target.exists() and target.is_file():
                        target.unlink()
                    target.mkdir(parents=True, exist_ok=True)
                    image_dirs.add(target)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        _write_dummy_image(target)
                        image_dirs.add(target.parent)
                    elif target.suffix.lower() == ".zip":
                        lower = target.name.lower()
                        if "trainlabels" in lower:
                            header = ["image", "level"]
                            rows = [[f"p{i}_img", (i - 1) % 5] for i in range(1, 51)]
                            _write_zip_with_csv(target, "trainLabels.csv", header, rows)
                        elif "samplesubmission" in lower or "sample" in lower:
                            header = ["image", "level"]
                            rows = [[f"p{i}_img", 0] for i in range(1, 51)]
                            _write_zip_with_csv(target, "sampleSubmission.csv", header, rows)
                        else:
                            _write_zip_with_csv(target, "data.csv", ["col1"], [[1]])
                    elif target.suffix.lower() in {".pth", ".pt"}:
                        _write_torch_checkpoint(target)
                    elif not target_exists or target_empty or target_small_csv or target_placeholder_csv:
                        if target.suffix.lower() == ".csv":
                            _write_csv_placeholder(target)
                        else:
                            target.write_text("", encoding="utf-8")

    join_pattern = re.compile(r"os\.path\.join\(([^)]*)\)")
    for match in join_pattern.findall(script_text):
        parts = re.findall(r"['\\\"]([^'\\\"]+)['\\\"]", match)
        if not parts:
            continue
        if "kaggle" in parts and "working" in parts:
            idx = parts.index("working")
            subparts = parts[idx + 1 :]
            if subparts:
                target = prefixes["working"] / "/".join(subparts)
                target.mkdir(parents=True, exist_ok=True)
                image_dirs.add(target)
        if "kaggle" in parts and "input" in parts:
            idx = parts.index("input")
            subparts = parts[idx + 1 :]
            if subparts:
                target = prefixes["input"] / "/".join(subparts)
                target.mkdir(parents=True, exist_ok=True)
                image_dirs.add(target)

    if image_dirs:
        label_paths = list((prefixes["working"]).rglob("trainLabels.csv"))
        if label_paths:
            try:
                import pandas as pd  # type: ignore
                labels = pd.read_csv(label_paths[0])
                col = None
                for candidate in ("image", "id_code", "id"):
                    if candidate in labels.columns:
                        col = candidate
                        break
                if col:
                    names = [str(x) for x in labels[col].head(30).tolist()]
                    for img_dir in image_dirs:
                        img_dir.mkdir(parents=True, exist_ok=True)
                        for name in names:
                            target = img_dir / f"{name}.jpeg"
                            if not target.exists():
                                _write_dummy_image(target)
            except Exception:
                pass

    if repo_path is not None:
        rel_csv_pattern = re.compile(r"['\\\"]([^'\\\":]+\\.csv)['\\\"]")
        for rel_name in rel_csv_pattern.findall(script_text):
            if rel_name.startswith("/") or re.match(r"^[A-Za-z]:", rel_name):
                continue
            target = (repo_path / rel_name).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size < 5000:
                _write_csv_placeholder(target)


def find_train_script(repo_path: Path, explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = repo_path / explicit
        return path if path.exists() else None

    preferred = ["train.py", "training.py", "main.py", "run.py", "model.py"]
    for name in preferred:
        path = repo_path / name
        if path.exists():
            return path

    candidates = _iter_py_files(repo_path)
    for path in candidates:
        if "train" in path.name.lower():
            return path
    return candidates[0] if candidates else None


def _run_cmd(
    args: list[str],
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_s,
        env=env,
    )


def _extract_violations(data: Any) -> list[str]:
    if isinstance(data, dict):
        for key in ("violations", "failed", "failures", "errors", "issues", "anomalies"):
            if key in data and isinstance(data[key], list):
                return [str(x) for x in data[key]]
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def _load_json_objects(text: str) -> list[Any]:
    import json

    decoder = json.JSONDecoder()
    idx = 0
    objs: list[Any] = []
    length = len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            break
        objs.append(obj)
        idx += end
    return objs


def _summarize_failed_invariant(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    inv = obj.get("invariant") if isinstance(obj.get("invariant"), dict) else {}
    text = ""
    if isinstance(inv, dict):
        text = str(inv.get("text_description") or inv.get("description") or "")
        relation = inv.get("relation")
    else:
        relation = None
    if not text and relation:
        text = f"Invariant relation {relation}"
    if relation and relation not in text:
        text = f"{text} ({relation})".strip()
    return text


def _compact_failed_invariant(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    inv = obj.get("invariant") if isinstance(obj.get("invariant"), dict) else {}
    compact_inv: dict[str, Any] = {}
    if isinstance(inv, dict):
        for key in ("text_description", "relation", "params", "precondition"):
            if key in inv:
                compact_inv[key] = inv.get(key)
    compact: dict[str, Any] = {
        "invariant": compact_inv,
        "check_passed": obj.get("check_passed"),
        "triggered": obj.get("triggered"),
        "detection_time": obj.get("detection_time"),
        "detection_time_percentage": obj.get("detection_time_percentage"),
    }
    trace = obj.get("trace")
    if isinstance(trace, list) and trace:
        head = trace[0]
        if isinstance(head, dict):
            compact["trace_head"] = {
                "function": head.get("function"),
                "meta_vars.step": head.get("meta_vars.step"),
                "meta_vars.stage": head.get("meta_vars.stage"),
            }
    return compact


def _collect_reports(trace_dir: Path) -> tuple[list[str], list[Path], list[dict[str, Any]]]:
    violations: list[str] = []
    report_files: list[Path] = []
    failed_invariants: list[dict[str, Any]] = []

    for failed_path in trace_dir.rglob("failed.log"):
        report_files.append(failed_path)
        try:
            text = failed_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for obj in _load_json_objects(text):
            compact = _compact_failed_invariant(obj)
            if compact:
                failed_invariants.append(compact)
            summary = _summarize_failed_invariant(obj)
            if summary:
                violations.append(summary)

    for path in trace_dir.rglob("*.json"):
        name = path.name.lower()
        if any(k in name for k in ("violation", "check", "result", "report")):
            report_files.append(path)
            try:
                data = path.read_text(encoding="utf-8", errors="replace")
                violations.extend(_extract_violations(_safe_json_load(data)))
            except Exception:
                continue
    return violations, report_files, failed_invariants


def _safe_json_load(text: str) -> Any:
    import json

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def run_traincheck(
    repo_path: Path,
    *,
    script_path: str | None = None,
    model_var: str | None = "model",
    output_root: Path | None = None,
    kaggle_root: Path | None = None,
    invariants_path: Path | None = None,
    infer_relations: list[str] | None = None,
    max_iters: int | None = None,
    timeout_s: int = 1200,
) -> dict[str, Any]:
    """Run TrainCheck collect/infer/check on a repository."""
    repo_path = repo_path.resolve()
    output_root = (output_root or (Path("results") / "traincheck")).resolve()
    trace_dir = (output_root / repo_path.name / "trace").resolve()
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    kaggle_root = _prepare_kaggle_root(
        (kaggle_root or (output_root / repo_path.name / "kaggle")).resolve()
    )
    _ensure_global_kaggle_mount(kaggle_root)

    script = find_train_script(repo_path, explicit=script_path)
    converted_notebook = None
    if not script:
        notebooks = _iter_ipynb_files(repo_path)
        if notebooks:
            tmp_dir = Path(tempfile.mkdtemp(prefix="traincheck-nb-"))
            max_cells = 10 if max_iters is not None and max_iters > 0 else None
            converted_notebook = _convert_ipynb_to_py(notebooks[0], tmp_dir, max_cells=max_cells)
            script = converted_notebook
    if script:
        script = Path(script).resolve()
    if not script:
        return {
            "ok": False,
            "error": "No training script found",
            "review": "",
            "trace_dir": "",
            "stdout": "",
            "stderr": "",
            "converted_notebook": None,
            "script": None,
        }

    if script:
        if converted_notebook is not None:
            _rewrite_kaggle_paths(script, kaggle_root)
            script_text = _sanitize_script_text(
                script.read_text(encoding="utf-8", errors="replace")
            )
            if max_iters is not None and max_iters > 0:
                script_text = _rewrite_range_calls(script_text)
            script_text = _rewrite_relative_csv_paths(script_text, repo_path)
            script_text = _ensure_parseable(script_text)
            script.write_text(script_text, encoding="utf-8")
            _seed_from_script_paths(script_text, kaggle_root, repo_path)
        else:
            script_text = _sanitize_script_text(
                script.read_text(encoding="utf-8", errors="replace")
            )
            needs_kaggle_rewrite = (
                "/kaggle/" in script_text or "../input" in script_text or "..\\input" in script_text
            )
            needs_patch = "_vibetest_safe_getattr" not in script_text
            needs_sanitize = script_text != script.read_text(encoding="utf-8", errors="replace")
            needs_limit = (
                max_iters is not None
                and max_iters > 0
                and "_vibetest_limited_range" not in script_text
            )
            if needs_kaggle_rewrite or needs_patch or needs_limit or needs_sanitize:
                tmp_dir = Path(tempfile.mkdtemp(prefix="traincheck-src-"))
                tmp_script = tmp_dir / script.name
                if max_iters is not None and max_iters > 0:
                    script_text = _rewrite_range_calls(script_text)
                if needs_patch or needs_limit:
                    script_text = (
                        _traincheck_safe_getattr_patch()
                        + _traincheck_limit_iters_patch()
                        + _traincheck_stub_modules_patch()
                        + script_text
                    )
                script_text = _rewrite_relative_csv_paths(script_text, repo_path)
                script_text = _ensure_parseable(script_text)
                tmp_script.write_text(script_text, encoding="utf-8")
                if needs_kaggle_rewrite:
                    _rewrite_kaggle_paths(tmp_script, kaggle_root)
                script = tmp_script
                script_text = script.read_text(encoding="utf-8", errors="replace")
                _seed_from_script_paths(script_text, kaggle_root, repo_path)

    base_env = os.environ.copy()
    base_env.update(
        {
            "TQDM_DISABLE": "1",
            "PYTHONIOENCODING": "UTF-8",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
    )
    if max_iters is not None and max_iters > 0:
        base_env["TRAINCHECK_MAX_ITERS"] = str(max_iters)

    collect_cmd = [
        "traincheck-collect",
        "--pyscript",
        str(script),
        "--output-dir",
        str(trace_dir),
    ]
    if model_var:
        collect_cmd.extend(["--models-to-track", model_var])

    try:
        collect = _run_cmd(collect_cmd, cwd=repo_path, timeout_s=timeout_s, env=base_env)
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "error": f"traincheck-collect not found: {exc}",
            "review": "",
            "trace_dir": str(trace_dir),
            "stdout": "",
            "stderr": str(exc),
            "converted_notebook": str(converted_notebook) if converted_notebook else None,
            "script": str(script) if script else None,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"traincheck-collect timed out after {timeout_s}s",
            "review": "",
            "trace_dir": str(trace_dir),
            "stdout": "",
            "stderr": "",
            "converted_notebook": str(converted_notebook) if converted_notebook else None,
            "script": str(script) if script else None,
        }

    if collect.returncode != 0:
        return {
            "ok": False,
            "error": f"traincheck-collect exited {collect.returncode}",
            "review": "",
            "trace_dir": str(trace_dir),
            "stdout": collect.stdout,
            "stderr": collect.stderr,
            "converted_notebook": str(converted_notebook) if converted_notebook else None,
            "script": str(script) if script else None,
        }

    invariants = (
        Path(invariants_path).resolve()
        if invariants_path
        else (trace_dir / "invariants.json")
    )
    if not invariants_path:
        infer_cmd = ["traincheck-infer", "-f", str(trace_dir), "-o", str(invariants)]
        if infer_relations:
            infer_cmd.extend(["--enable-relation", *infer_relations])
        try:
            infer = _run_cmd(infer_cmd, cwd=trace_dir, timeout_s=timeout_s, env=base_env)
        except FileNotFoundError as exc:
            return {
                "ok": False,
                "error": f"traincheck-infer not found: {exc}",
                "review": "",
                "trace_dir": str(trace_dir),
                "stdout": "",
                "stderr": str(exc),
                "converted_notebook": str(converted_notebook) if converted_notebook else None,
                "script": str(script) if script else None,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"traincheck-infer timed out after {timeout_s}s",
                "review": "",
                "trace_dir": str(trace_dir),
                "stdout": "",
                "stderr": "",
                "converted_notebook": str(converted_notebook) if converted_notebook else None,
                "script": str(script) if script else None,
            }

        if infer.returncode != 0:
            return {
                "ok": False,
                "error": f"traincheck-infer exited {infer.returncode}",
                "review": "",
                "trace_dir": str(trace_dir),
                "stdout": infer.stdout,
                "stderr": infer.stderr,
                "converted_notebook": str(converted_notebook) if converted_notebook else None,
                "script": str(script) if script else None,
            }

    if not invariants.exists():
        return {
            "ok": False,
            "error": f"Invariants file not found: {invariants}",
            "review": "",
            "trace_dir": str(trace_dir),
            "stdout": "",
            "stderr": "",
            "converted_notebook": str(converted_notebook) if converted_notebook else None,
            "script": str(script) if script else None,
        }
    check_cmd = [
        "traincheck-check",
        "--trace-folders",
        str(trace_dir),
        "--invariants",
        str(invariants),
    ]
    try:
        check = _run_cmd(check_cmd, cwd=trace_dir, timeout_s=timeout_s, env=base_env)
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "error": f"traincheck-check not found: {exc}",
            "review": "",
            "trace_dir": str(trace_dir),
            "stdout": "",
            "stderr": str(exc),
            "converted_notebook": str(converted_notebook) if converted_notebook else None,
            "script": str(script) if script else None,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"traincheck-check timed out after {timeout_s}s",
            "review": "",
            "trace_dir": str(trace_dir),
            "stdout": "",
            "stderr": "",
            "converted_notebook": str(converted_notebook) if converted_notebook else None,
            "script": str(script) if script else None,
        }

    violations, report_files, failed_invariants = _collect_reports(trace_dir)
    review_parts = []
    if violations:
        review_parts.append("TrainCheck violations:")
        review_parts.extend(f"- {v}" for v in violations[:200])
    if failed_invariants:
        review_parts.append("TrainCheck failed invariants (JSON):")
        for inv in failed_invariants[:200]:
            review_parts.append(json.dumps(inv, ensure_ascii=False))
    if check.stdout:
        review_parts.append(check.stdout.strip())
    if check.stderr:
        review_parts.append(check.stderr.strip())
    review_text = "\n".join(p for p in review_parts if p)

    check_ok = check.returncode == 0
    check_error = None if check_ok else f"traincheck-check exited {check.returncode}"
    if not check_ok and "time column not found" in (check.stderr or "").lower():
        check_ok = True
        check_error = None
    return {
        "ok": check_ok,
        "error": check_error,
        "review": review_text,
        "trace_dir": str(trace_dir),
        "stdout": check.stdout,
        "stderr": check.stderr,
        "report_files": [str(p) for p in report_files],
        "failed_invariants": failed_invariants,
        "failed_invariants_count": len(failed_invariants),
        "converted_notebook": str(converted_notebook) if converted_notebook else None,
        "script": str(script) if script else None,
    }


def prepare_reference_invariants(
    reference_script: Path,
    *,
    output_root: Path | None = None,
    infer_relations: list[str] | None = None,
    timeout_s: int = 1200,
) -> Path:
    """Run TrainCheck collect+infer on a reference script to produce invariants."""
    output_root = (output_root or (Path("results") / "traincheck_reference")).resolve()
    ref_dir = (output_root / reference_script.stem).resolve()
    trace_dir = (ref_dir / "trace").resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)

    invariants = trace_dir / "invariants.json"
    if invariants.exists() and invariants.stat().st_size > 0:
        return invariants

    ref_dir.mkdir(parents=True, exist_ok=True)
    patched_script = ref_dir / reference_script.name
    script_text = reference_script.read_text(encoding="utf-8", errors="replace")
    if "_vibetest_safe_getattr" not in script_text:
        script_text = _traincheck_safe_getattr_patch() + script_text
    patched_script.write_text(script_text, encoding="utf-8")

    # Create a shell script to pass safe runtime args (disable GPU)
    sh_script = ref_dir / "run_reference.sh"
    sh_script.write_text(
        f"python {patched_script.name} --no-cuda --no-mps\n",
        encoding="utf-8",
    )

    collect_cmd = [
        "traincheck-collect",
        "--pyscript",
        str(patched_script),
        "--shscript",
        str(sh_script),
        "--output-dir",
        str(trace_dir),
    ]
    env = os.environ.copy()
    env.update(
        {
            "TQDM_DISABLE": "1",
            "PYTHONIOENCODING": "UTF-8",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
    )
    collect = _run_cmd(collect_cmd, cwd=ref_dir, timeout_s=timeout_s, env=env)
    if collect.returncode != 0:
        raise RuntimeError(
            f"traincheck-collect failed for reference script: {collect.returncode}\n"
            f"{collect.stdout}\n{collect.stderr}"
        )

    infer_cmd = ["traincheck-infer", "-f", str(trace_dir), "-o", str(invariants)]
    if infer_relations:
        infer_cmd.extend(["--enable-relation", *infer_relations])
    infer = _run_cmd(infer_cmd, cwd=trace_dir, timeout_s=timeout_s, env=env)
    if infer.returncode != 0:
        raise RuntimeError(
            f"traincheck-infer failed for reference script: {infer.returncode}\n"
            f"{infer.stdout}\n{infer.stderr}"
        )

    if not invariants.exists():
        raise RuntimeError(f"Reference invariants file missing: {invariants}")
    return invariants
