#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小化的 PhantomDet 配置文件（仅服务于运行 test_single_image.py）。

保留：
- 常量：PROJECT_ROOT、TEST_RESULTS_DIR、DEFAULT_TEST_IMAGE、BEST_MODEL_PATH
- 函数：get_config、ensure_directories、validate_paths

去除与测试无关的其余配置与函数。
"""

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


# 项目根目录（优先环境变量，否则回退到默认路径）
PROJECT_ROOT: Path = Path(
    os.environ.get("PROJECT_ROOT")
    or os.environ.get("RM_PROJECT_ROOT")
    or "/home/user/Project/RealManArmControl"
).resolve()


# 关键路径与文件
DATA_DIR: Path = PROJECT_ROOT / "src" / "realmanarmcontrol" / "algorithms" / "yolo"
RAW_DATA_DIR: Path = DATA_DIR / "test"
PROCESSED_DATA_DIR: Path = DATA_DIR / "test" / "output"

MODEL_DIR: Path = DATA_DIR / "model"
BEST_MODEL_PATH: Path = MODEL_DIR / "best.pt"
LAST_MODEL_PATH: Path = MODEL_DIR / "last.pt"

TEST_RESULTS_DIR: Path = DATA_DIR / "test" / "output"
LOGS_DIR: Path = DATA_DIR / "logs"
TMP_DIR: Path = PROJECT_ROOT / "tmp"

DEFAULT_TEST_IMAGE: Path = DATA_DIR / "test" / "RBG_1280x720_1754913593976.png"


@dataclass(frozen=True)
class PhantomConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    model_dir: Path = MODEL_DIR
    best_model_path: Path = BEST_MODEL_PATH
    last_model_path: Path = LAST_MODEL_PATH
    test_results_dir: Path = TEST_RESULTS_DIR
    logs_dir: Path = LOGS_DIR
    tmp_dir: Path = TMP_DIR
    default_test_image: Path = DEFAULT_TEST_IMAGE


_GLOBAL_CONFIG = PhantomConfig()


def get_config() -> PhantomConfig:
    return _GLOBAL_CONFIG


def ensure_directories() -> Dict[str, Path]:
    dirs = {
        'logs_dir': LOGS_DIR,
        'tmp_dir': TMP_DIR,
        'test_results_dir': TEST_RESULTS_DIR,
        'model_dir': MODEL_DIR,
        'processed_data_dir': PROCESSED_DATA_DIR,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def validate_paths() -> List[Path]:
    must_exist: List[Path] = [
        PROJECT_ROOT,
        DATA_DIR,
        RAW_DATA_DIR,
        LOGS_DIR,
        TMP_DIR,
    ]
    important_files: List[Path] = [
        BEST_MODEL_PATH,
    ]
    missing: List[Path] = []
    for path in must_exist + important_files:
        if not path.exists():
            missing.append(path)
    return missing


__all__ = [
    'get_config', 'ensure_directories', 'validate_paths',
    'PhantomConfig',
    'PROJECT_ROOT', 'DATA_DIR', 'RAW_DATA_DIR', 'PROCESSED_DATA_DIR',
    'MODEL_DIR', 'BEST_MODEL_PATH', 'LAST_MODEL_PATH',
    'TEST_RESULTS_DIR', 'LOGS_DIR', 'TMP_DIR', 'DEFAULT_TEST_IMAGE',
]
