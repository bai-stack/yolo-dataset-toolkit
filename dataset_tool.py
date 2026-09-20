# -*- coding: utf-8 -*-
"""YOLO 数据集工具。

两个标签页：
  转换 —— 把 X-AnyLabeling 的 XLABEL 标注（图片 + 同名 .json）整理成
          Ultralytics 目录结构并写出 data.yaml。
  查看 —— 左侧图像（可叠加标注框），右侧展示 labels 下 txt 的解析结果，
          images / labels 两个来源可独立切换。

本工具不修改标注源目录；只在指定的输出目录里创建文件。
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import shutil
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional, Sequence, Tuple

import customtkinter as ctk
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageTk

APP_TITLE: str = "YOLO 数据集工具"
DEFAULT_SOURCE: str = r"C:\Users\99259\Desktop\sucai\images"
DEFAULT_OUTPUT: str = r"C:\Users\99259\Desktop\sucai\test"
DEFAULT_YAML: str = "data.yaml"
IMAGE_EXTS: Tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
)
RECT_SHAPES: frozenset = frozenset({"rectangle"})
POLY_SHAPES: frozenset = frozenset({"polygon"})
CLASS_COLORS: Tuple[str, ...] = (
    "#e5484d",
    "#0d9488",
    "#7c3aed",
    "#d97706",
    "#2563eb",
    "#db2777",
    "#059669",
    "#dc2626",
)
FONT_CANDIDATES: Tuple[str, ...] = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
)
try:
    RESAMPLE: int = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    RESAMPLE = Image.LANCZOS

# 画布底色（浅色 / 深色），用 Tk 内置颜色名，与 CTk 默认主题观感一致
CANVAS_BG: Tuple[str, str] = ("gray95", "gray17")


# ====================================================================== 数据结构
@dataclass(frozen=True)
class Box:
    """一个归一化的 YOLO 检测框。"""

    class_id: int
    cx: float
    cy: float
    width: float
    height: float
    raw_line: str


@dataclass
class Frame:
    """查看器当前展示的一页。"""

    stem: str
    image_path: str
    label_path: Optional[str]
    boxes: List[Box]
    errors: List[str]
    image_size: Tuple[int, int]


@dataclass
class Sample:
    """转换过程中的一条样本。"""

    stem: str
    image_path: str
    width: int
    height: int
    items: List[Tuple[str, Tuple[float, float, float, float]]]


@dataclass
class ConvertOptions:
    """转换参数。"""

    source_dir: str
    output_dir: str
    val_ratio: float = 0.2
    seed: int = 42
    rename: Dict[str, str] = field(default_factory=dict)
    overwrite: bool = False


@dataclass
class ConvertReport:
    """转换结果报告。"""

    ok: bool
    message: str
    log: List[str] = field(default_factory=list)
    output_dir: str = ""
    yaml_path: str = ""
    total_images: int = 0
    pairs: int = 0
    unlabeled: int = 0
    train: int = 0
    val: int = 0
    boxes: int = 0
    poly_boxes: int = 0
    empty_labels: int = 0
    skipped_shapes: List[str] = field(default_factory=list)
    class_counts: Dict[str, int] = field(default_factory=dict)
    display_names: Dict[str, str] = field(default_factory=dict)


def format_class_line(report: ConvertReport, label: str, count: int) -> str:
    """格式化「类别: 框数」；若该类别被重命名过，同时显示原名。"""
    final = report.display_names.get(label, label)
    if final == label:
        return f"  {label}: {count}"
    return f"  {label} -> {final}: {count}"


# ==================================================================== 通用工具
_font_cache: Dict[int, ImageFont.FreeTypeFont] = {}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """按字号加载一个支持中文的字体，带缓存。"""
    if size in _font_cache:
        return _font_cache[size]
    font: ImageFont.FreeTypeFont
    for candidate in FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(candidate, size)
            _font_cache[size] = font
            return font
        except OSError:
            continue
    font = ImageFont.load_default()  # type: ignore[assignment]
    _font_cache[size] = font
    return font


def load_class_names(yaml_path: str) -> Dict[int, str]:
    """从 data.yaml 的 names 字段读取类别映射，兼容 dict 与 list 两种写法。"""
    if not os.path.isfile(yaml_path):
        return {}
    try:
        with open(yaml_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    names = data.get("names")
    mapping: Dict[int, str] = {}
    if isinstance(names, dict):
        for key, value in names.items():
            try:
                mapping[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(names, list):
        for index, value in enumerate(names):
            mapping[index] = str(value)
    return mapping


def scan_split_dirs(root: str) -> Tuple[List[str], List[str]]:
    """扫描 <root>/images 与 <root>/labels 下的子目录名。"""

    def _subdirs(parent: str) -> List[str]:
        if not os.path.isdir(parent):
            return []
        return sorted(
            name
            for name in os.listdir(parent)
            if os.path.isdir(os.path.join(parent, name))
        )

    return _subdirs(os.path.join(root, "images")), _subdirs(
        os.path.join(root, "labels")
    )


def list_image_stems(directory: str) -> List[str]:
    """列出目录下所有图片的文件名主干（不含扩展名），按名称排序。"""
    if not os.path.isdir(directory):
        return []
    stems: List[str] = []
    for name in sorted(os.listdir(directory)):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTS:
            stems.append(stem)
    return stems


def find_image_path(directory: str, stem: str) -> Optional[str]:
    """在目录中查找指定主干对应的图片文件。"""
    for ext in IMAGE_EXTS:
        candidate = os.path.join(directory, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def parse_yolo_label(label_path: Optional[str]) -> Tuple[List[Box], List[str]]:
    """解析 YOLO txt：每行 `cls cx cy w h`，返回 (框列表, 错误说明列表)。"""
    if not label_path or not os.path.isfile(label_path):
        return [], []
    boxes: List[Box] = []
    errors: List[str] = []
    try:
        with open(label_path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return [], [f"读取失败: {exc}"]

    for line_no, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 5:
            errors.append(f"第 {line_no} 行字段数 = {len(parts)}，应为 5")
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, box_w, box_h = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f"第 {line_no} 行无法解析为数字: {text}")
            continue
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, box_w, box_h)):
            errors.append(f"第 {line_no} 行坐标超出 [0,1]: {text}")
        if box_w <= 0 or box_h <= 0:
            errors.append(f"第 {line_no} 行宽高非正数: {text}")
            continue
        boxes.append(Box(class_id, cx, cy, box_w, box_h, text))
    return boxes, errors


def parse_rename_rules(text: str) -> Dict[str, str]:
    """解析类别重命名规则，每条 `原名=新名`，可用换行或分号分隔，忽略注释。"""
    rules: Dict[str, str] = {}
    for raw in text.replace(";", "\n").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        old, new = line.split("=", 1)
        old, new = old.strip(), new.strip()
        if old and new:
            rules[old] = new
    return rules


# ====================================================================== 转换
def _shape_bbox(shape: Dict) -> Optional[Tuple[float, float, float, float]]:
    """把 XLABEL 的 rectangle / polygon 形状转成像素级外接框。"""
    shape_type = str(shape.get("shape_type", ""))
    if shape_type not in RECT_SHAPES and shape_type not in POLY_SHAPES:
        return None
    points = shape.get("points") or []
    xs: List[float] = []
    ys: List[float] = []
    for point in points:
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except (TypeError, ValueError, IndexError):
            continue
    if len(xs) < 2 or len(ys) < 2:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def convert_dataset(options: ConvertOptions) -> ConvertReport:
    """把 XLABEL 标注转换成 Ultralytics 的 YOLO 检测数据集。"""
    report = ConvertReport(ok=False, message="")
    log = report.log

    source = os.path.abspath(options.source_dir)
    output = os.path.abspath(options.output_dir)
    if not os.path.isdir(source):
        report.message = f"标注源目录不存在: {source}"
        return report
    if os.path.abspath(source) == os.path.abspath(output):
        report.message = "输出目录不能和标注源目录相同"
        return report

    # ---- 1. 扫描并读取标注 ----
    samples: List[Sample] = []
    unlabeled = 0
    all_images = list_image_stems(source)
    for stem in all_images:
        json_path = os.path.join(source, stem + ".json")
        if not os.path.isfile(json_path):
            unlabeled += 1
            continue
        image_path = find_image_path(source, stem)
        if image_path is None:
            log.append(f"跳过 {stem}: 只有 json，找不到图片")
            continue
        try:
            with open(json_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            log.append(f"跳过 {stem}: json 解析失败 ({exc})")
            continue
        try:
            width = int(data["imageWidth"])
            height = int(data["imageHeight"])
        except (KeyError, TypeError, ValueError):
            with Image.open(image_path) as handle:
                width, height = handle.size
        items: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for shape in data.get("shapes") or []:
            bbox = _shape_bbox(shape)
            label = str(shape.get("label", "")).strip()
            if bbox is None or not label:
                shape_type = shape.get("shape_type")
                report.skipped_shapes.append(f"{stem}: {label or '?'} ({shape_type})")
                continue
            if str(shape.get("shape_type")) in POLY_SHAPES:
                report.poly_boxes += 1
            items.append((label, bbox))
        samples.append(Sample(stem, image_path, width, height, items))

    report.total_images = len(all_images)
    report.unlabeled = unlabeled
    report.pairs = len(samples)
    log.append(f"源目录图片 {len(all_images)} 张，其中 {len(samples)} 张有标注，{unlabeled} 张未标注（不参与）")
    if not samples:
        report.message = "没有找到任何「图片 + 同名 json」的配对"
        return report

    # ---- 2. 类别表（排序保证多次运行 id 稳定）----
    raw_labels = sorted({label for sample in samples for label, _ in sample.items})
    class_names = [options.rename.get(name, name) for name in raw_labels]
    if len(set(class_names)) != len(class_names):
        report.message = "类别重命名后出现重复名称，请检查映射规则"
        return report
    id_of: Dict[str, int] = {name: index for index, name in enumerate(raw_labels)}
    report.display_names = dict(zip(raw_labels, class_names))
    log.append(f"共 {len(raw_labels)} 个类别: " + "、".join(class_names))

    # ---- 3. 划分 ----
    shuffled = samples[:]
    random.Random(options.seed).shuffle(shuffled)
    n_train = int(len(shuffled) * (1.0 - options.val_ratio))
    n_train = max(1, min(n_train, len(shuffled))) if len(shuffled) > 1 else len(shuffled)
    splits = {"train": shuffled[:n_train], "val": shuffled[n_train:]}
    if not splits["val"]:
        log.append("提示: 验证集为空（样本太少），建议增加数据量或调大验证集比例")

    # ---- 4. 准备输出目录 ----
    if os.path.isdir(output) and os.listdir(output) and not options.overwrite:
        report.message = f"输出目录已存在且非空: {output}\n如需覆盖，请勾选「覆盖已存在的输出目录」"
        report.output_dir = output
        return report
    try:
        if os.path.isdir(output):
            shutil.rmtree(output)
            log.append(f"已清空输出目录: {output}")
        for split in ("train", "val"):
            os.makedirs(os.path.join(output, "images", split), exist_ok=True)
            os.makedirs(os.path.join(output, "labels", split), exist_ok=True)
    except OSError as exc:
        report.message = f"创建输出目录失败: {exc}"
        return report

    # ---- 5. 写图片与标签 ----
    for split, group in splits.items():
        for sample in group:
            rows: List[str] = []
            for label, (x1, y1, x2, y2) in sample.items:
                x1 = max(0.0, min(x1, float(sample.width)))
                x2 = max(0.0, min(x2, float(sample.width)))
                y1 = max(0.0, min(y1, float(sample.height)))
                y2 = max(0.0, min(y2, float(sample.height)))
                box_w, box_h = x2 - x1, y2 - y1
                if box_w <= 0 or box_h <= 0:
                    report.skipped_shapes.append(f"{sample.stem}: {label} (零面积)")
                    continue
                cx = (x1 + x2) / 2.0 / sample.width
                cy = (y1 + y2) / 2.0 / sample.height
                rows.append(
                    f"{id_of[label]} {cx:.6f} {cy:.6f} "
                    f"{box_w / sample.width:.6f} {box_h / sample.height:.6f}"
                )
                report.class_counts[label] = report.class_counts.get(label, 0) + 1
            if not rows:
                report.empty_labels += 1

            ext = os.path.splitext(sample.image_path)[1]
            try:
                shutil.copy2(
                    sample.image_path,
                    os.path.join(output, "images", split, sample.stem + ext),
                )
                with open(
                    os.path.join(output, "labels", split, sample.stem + ".txt"),
                    "w",
                    encoding="utf-8",
                    newline="\n",
                ) as handle:
                    if rows:
                        handle.write("\n".join(rows) + "\n")
            except OSError as exc:
                log.append(f"写入失败 {sample.stem}: {exc}")
                continue
            report.boxes += len(rows)

    report.train = len(splits["train"])
    report.val = len(splits["val"])

    # ---- 6. 写 data.yaml ----
    lines = [
        "# 由 YOLO 数据集工具从 X-AnyLabeling 标注生成",
        f"path: {output.replace(os.sep, '/')}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    lines.append("")
    yaml_path = os.path.join(output, DEFAULT_YAML)
    try:
        with open(yaml_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines))
    except OSError as exc:
        report.message = f"写 data.yaml 失败: {exc}"
        return report

    report.ok = True
    report.output_dir = output
    report.yaml_path = yaml_path
    report.message = (
        f"完成：{report.pairs} 组样本，训练 {report.train} / 验证 {report.val}，"
        f"共 {report.boxes} 个标注框"
    )
    log.append(report.message)
    log.append(f"data.yaml: {yaml_path}")
    return report


# ====================================================================== 查看页
class ViewerTab(ctk.CTkFrame):
    """数据集查看器：左图右标注。"""

    def __init__(self, master: tk.Misc, root_dir: str = DEFAULT_OUTPUT) -> None:
        super().__init__(master, fg_color="transparent")
        self._root_dir: str = root_dir
        self._class_names: Dict[int, str] = {}
        self._image_splits: List[str] = []
        self._label_splits: List[str] = []
        self._stems: List[str] = []
        self._label_index: Dict[str, set] = {}
        self._index: int = 0
        self._frame: Optional[Frame] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._last_canvas_size: Tuple[int, int] = (0, 0)
        self._rendering: bool = False
        self._resize_job: Optional[str] = None

        self._build_layout()
        self._bind_keys()
        self.reload(initial=True)

    # ------------------------------------------------------------------ 布局
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        top = ctk.CTkFrame(self, corner_radius=10)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        top.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(top, text="数据集根目录").grid(
            row=0, column=0, padx=(12, 8), pady=12, sticky="w"
        )
        self._root_entry = ctk.CTkEntry(top)
        self._root_entry.insert(0, self._root_dir)
        self._root_entry.grid(row=0, column=1, sticky="ew", pady=12)
        ctk.CTkButton(top, text="浏览", width=70, command=self._on_browse_root).grid(
            row=0, column=2, padx=8, pady=12
        )
        ctk.CTkButton(top, text="重新扫描", width=90, command=self.reload).grid(
            row=0, column=3, padx=(0, 12), pady=12
        )

        bar = ctk.CTkFrame(self, corner_radius=10)
        bar.grid(row=1, column=0, sticky="ew", pady=6)

        ctk.CTkLabel(bar, text="图像来源").grid(
            row=0, column=0, padx=(12, 8), pady=10
        )
        self._image_menu = ctk.CTkOptionMenu(
            bar, values=["-"], width=170, command=self._on_image_split
        )
        self._image_menu.grid(row=0, column=1, pady=10)

        ctk.CTkLabel(bar, text="标签来源").grid(
            row=0, column=2, padx=(20, 8), pady=10
        )
        self._label_menu = ctk.CTkOptionMenu(
            bar, values=["-"], width=170, command=self._on_label_split
        )
        self._label_menu.grid(row=0, column=3, pady=10)

        self._draw_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            bar,
            text="在图上绘制标注框",
            variable=self._draw_var,
            command=self._render,
        ).grid(row=0, column=4, padx=(24, 8), pady=10)

        self._theme_btn = ctk.CTkButton(
            bar, text="切换深色", width=100, command=self._toggle_theme
        )
        self._theme_btn.grid(row=0, column=5, padx=8, pady=10)

        self._yaml_label = ctk.CTkLabel(
            bar, text="", text_color="#888888", anchor="w"
        )
        self._yaml_label.grid(row=0, column=6, padx=(16, 12), pady=10, sticky="w")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew", pady=6)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        self._canvas_frame = ctk.CTkFrame(
            body, corner_radius=10, fg_color=CANVAS_BG
        )
        self._canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._canvas_frame.grid_rowconfigure(0, weight=1)
        self._canvas_frame.grid_columnconfigure(0, weight=1)
        # 用原生 tk.Label 承载图像：CTkLabel 会自动重算缩放并缓存 PhotoImage，
        # 频繁重绘时容易把正在使用的图像对象回收掉（TclError: image ... doesn't exist）。
        self._image_label = tk.Label(
            self._canvas_frame,
            text="",
            bg=CANVAS_BG[0],
            bd=0,
            highlightthickness=0,
        )
        self._image_label.grid(row=0, column=0, sticky="nsew")
        self._canvas_frame.bind("<Configure>", self._on_canvas_resize)

        right = ctk.CTkFrame(body, corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(right, text="标注内容（解析自 labels 下的 txt）").grid(
            row=0, column=0, padx=12, pady=(10, 4), sticky="w"
        )
        self._text = ctk.CTkTextbox(
            right, wrap="none", font=ctk.CTkFont(family="Consolas", size=12)
        )
        self._text.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        bottom = ctk.CTkFrame(self, corner_radius=10)
        bottom.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        bottom.grid_columnconfigure(2, weight=1)

        self._prev_btn = ctk.CTkButton(
            bottom, text="◀ 上一页", width=110, command=self.prev_page
        )
        self._prev_btn.grid(row=0, column=0, padx=12, pady=12)
        self._page_label = ctk.CTkLabel(bottom, text="- / -", width=140)
        self._page_label.grid(row=0, column=1, padx=8, pady=12)
        self._next_btn = ctk.CTkButton(
            bottom, text="下一页 ▶", width=110, command=self.next_page
        )
        self._next_btn.grid(row=0, column=2, padx=0, pady=12, sticky="w")
        self._status = ctk.CTkLabel(
            bottom, text="", text_color="#888888", anchor="e"
        )
        self._status.grid(row=0, column=3, padx=12, pady=12, sticky="e")

    def _bind_keys(self) -> None:
        top = self.winfo_toplevel()
        top.bind_all("<Left>", self._on_key_prev)
        top.bind_all("<Right>", self._on_key_next)
        top.bind_all("<Prior>", self._on_key_prev)
        top.bind_all("<Next>", self._on_key_next)

    def _active(self) -> bool:
        """只有本标签页可见时才响应全局快捷键。"""
        try:
            return bool(self.winfo_ismapped())
        except tk.TclError:
            return False

    def _typing(self) -> bool:
        focused = self.focus_get()
        return bool(focused) and focused.winfo_class() in ("Entry", "Text")

    def _on_key_prev(self, _event: tk.Event) -> None:
        if self._active() and not self._typing():
            self.prev_page()

    def _on_key_next(self, _event: tk.Event) -> None:
        if self._active() and not self._typing():
            self.next_page()

    # -------------------------------------------------------------- 数据加载
    def set_root_dir(self, root_dir: str) -> None:
        """外部设置根目录并立即刷新（转换完成后调用）。"""
        self._root_entry.delete(0, "end")
        self._root_entry.insert(0, root_dir)
        self.reload(initial=True)

    def reload(self, initial: bool = False) -> None:
        """重新扫描根目录、读取 data.yaml 并刷新下拉框。"""
        self._root_dir = self._root_entry.get().strip().strip('"')
        yaml_path = os.path.join(self._root_dir, DEFAULT_YAML)
        self._class_names = load_class_names(yaml_path)
        self._image_splits, self._label_splits = scan_split_dirs(self._root_dir)

        if not self._image_splits or not self._label_splits:
            self._image_menu.configure(values=["-"])
            self._image_menu.set("-")
            self._label_menu.configure(values=["-"])
            self._label_menu.set("-")
            self._stems = []
            self._yaml_label.configure(text="未找到 images/ 或 labels/ 子目录")
            self._render()
            return

        self._image_menu.configure(values=self._image_splits)
        self._label_menu.configure(values=self._label_splits)
        keep_image = self._image_menu.get()
        keep_label = self._label_menu.get()
        self._image_menu.set(
            keep_image if keep_image in self._image_splits else self._image_splits[0]
        )
        self._label_menu.set(
            keep_label if keep_label in self._label_splits else self._label_splits[0]
        )
        names_text = "、".join(
            f"{key}:{value}" for key, value in sorted(self._class_names.items())
        )
        self._yaml_label.configure(
            text=f"data.yaml 类别 → {names_text or '（未读取到 names）'}"
        )
        if initial:
            self._index = 0
        self._refresh_stems(reset_index=initial)

    def _refresh_stems(self, reset_index: bool = False) -> None:
        image_dir = os.path.join(self._root_dir, "images", self._image_menu.get())
        label_root = os.path.join(self._root_dir, "labels")
        self._label_index = {
            name: set(
                os.path.splitext(f)[0]
                for f in os.listdir(os.path.join(label_root, name))
                if f.lower().endswith(".txt")
            )
            for name in self._label_splits
        }
        self._stems = list_image_stems(image_dir)
        if reset_index:
            self._index = 0
        self._index = (
            max(0, min(self._index, len(self._stems) - 1)) if self._stems else 0
        )
        self._render()

    def _build_frame(self) -> Optional[Frame]:
        if not self._stems:
            return None
        stem = self._stems[self._index]
        image_dir = os.path.join(self._root_dir, "images", self._image_menu.get())
        label_dir = os.path.join(self._root_dir, "labels", self._label_menu.get())
        image_path = find_image_path(image_dir, stem)
        if image_path is None:
            return None
        label_path: Optional[str] = os.path.join(label_dir, stem + ".txt")
        if not os.path.isfile(label_path):
            label_path = None
        boxes, errors = parse_yolo_label(label_path)
        try:
            with Image.open(image_path) as handle:
                image_size = handle.size
        except OSError:
            image_size = (0, 0)
        return Frame(stem, image_path, label_path, boxes, errors, image_size)

    # ------------------------------------------------------------------ 渲染
    def _render(self) -> None:
        self._frame = self._build_frame()
        self._render_image()
        self._render_text()
        total = len(self._stems)
        current = self._index + 1 if total else 0
        self._page_label.configure(text=f"{current} / {total}")
        state = "normal" if total > 1 else "disabled"
        self._prev_btn.configure(state=state)
        self._next_btn.configure(state=state)

    def _render_image(self) -> None:
        """把图像缩放并居中放到左侧画布上。

        给标签设置图片会改变布局，进而再次触发 <Configure> 回调；
        这里用 _rendering 标志阻止重入，避免图像对象在配置过程中被回收。
        """
        if self._rendering:
            return
        self._rendering = True
        try:
            self._render_image_inner()
        finally:
            self._rendering = False

    def _render_image_inner(self) -> None:
        frame = self._frame
        if frame is None:
            self._safe_set_image(None, "没有可显示的图像")
            return
        try:
            with Image.open(frame.image_path) as handle:
                image = handle.convert("RGB")
        except OSError as exc:
            self._safe_set_image(None, f"图像读取失败\n{exc}")
            return

        panel_w = min(max(120, self._canvas_frame.winfo_width() - 16), 4000)
        panel_h = min(max(120, self._canvas_frame.winfo_height() - 16), 4000)
        scale = min(panel_w / image.width, panel_h / image.height)
        target = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )
        resized = image.resize(target, RESAMPLE)

        if self._draw_var.get() and frame.boxes:
            self._draw_boxes(resized, frame.boxes)

        photo = ImageTk.PhotoImage(resized)
        self._photo = photo
        self._safe_set_image(photo, "")

    def _safe_set_image(
        self, image: Optional[ImageTk.PhotoImage], text: str
    ) -> None:
        """设置图像内容；Tk 图像句柄失效时降级为纯文字，避免整个界面崩掉。"""
        try:
            self._image_label.configure(
                image=image if image is not None else "", text=text
            )
        except tk.TclError:
            self._photo = None
            try:
                self._image_label.configure(
                    image="",
                    text="图像渲染失败，已降级为文字提示\n（翻到上一页再回来可重试）",
                )
            except tk.TclError:
                pass

    def _draw_boxes(self, image: Image.Image, boxes: Sequence[Box]) -> None:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        line_width = max(2, int(min(width, height) / 320))
        font = load_font(max(12, int(min(width, height) / 42)))
        for box in boxes:
            color = CLASS_COLORS[box.class_id % len(CLASS_COLORS)]
            x1 = (box.cx - box.width / 2.0) * width
            y1 = (box.cy - box.height / 2.0) * height
            x2 = (box.cx + box.width / 2.0) * width
            y2 = (box.cy + box.height / 2.0) * height
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            name = self._class_names.get(box.class_id, str(box.class_id))
            caption = f"{name} ({box.class_id})"
            try:
                text_box = draw.textbbox((0, 0), caption, font=font)
            except (AttributeError, OSError):
                continue
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            pad = 4
            top = max(0, y1 - text_h - 2 * pad)
            draw.rectangle(
                [x1, top, x1 + text_w + 2 * pad, top + text_h + 2 * pad],
                fill=color,
            )
            draw.text(
                (x1 + pad, top + pad - text_box[1]),
                caption,
                fill="#ffffff",
                font=font,
            )

    def _render_text(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", self._compose_text())
        self._text.configure(state="disabled")

    def _compose_text(self) -> str:
        frame = self._frame
        if frame is None:
            return "没有图像。请检查数据集根目录，以及 images/ 与 labels/ 子目录。"

        label_dir = os.path.join(self._root_dir, "labels", self._label_menu.get())
        lines: List[str] = []
        lines.append(f"图像     : {frame.stem}")
        lines.append(
            f"相对路径 : images/{self._image_menu.get()}/"
            f"{os.path.basename(frame.image_path)}"
        )
        lines.append(f"尺寸     : {frame.image_size[0]} x {frame.image_size[1]}")
        lines.append("")

        if frame.label_path:
            lines.append(
                f"标签文件 : {os.path.relpath(frame.label_path, self._root_dir)}"
            )
        else:
            expected = os.path.relpath(
                os.path.join(label_dir, frame.stem + ".txt"), self._root_dir
            )
            lines.append(f"标签文件 : 未找到 {expected}")
            others = [
                name
                for name, stems in self._label_index.items()
                if frame.stem in stems and name != self._label_menu.get()
            ]
            if others:
                lines.append(
                    "          该图片在 labels/{} 中存在".format(
                        "、labels/".join(others)
                    )
                )
        lines.append("")

        if not frame.label_path:
            lines.append("状态     : 无标签文件（该图无标注）")
        elif not frame.boxes:
            lines.append("状态     : 空标签文件 —— 背景图，无目标")
        else:
            lines.append(f"状态     : {len(frame.boxes)} 个目标")
        lines.append("")

        if frame.boxes:
            lines.append("=" * 58)
            img_w, img_h = frame.image_size
            for order, box in enumerate(frame.boxes, start=1):
                name = self._class_names.get(box.class_id, "未知类别")
                x1 = (box.cx - box.width / 2.0) * img_w
                y1 = (box.cy - box.height / 2.0) * img_h
                x2 = (box.cx + box.width / 2.0) * img_w
                y2 = (box.cy + box.height / 2.0) * img_h
                lines.append(f"[{order}] {name}   class_id = {box.class_id}")
                lines.append(
                    f"    YOLO  : {box.cx:.6f} {box.cy:.6f} "
                    f"{box.width:.6f} {box.height:.6f}"
                )
                lines.append(
                    f"    中心  : ({box.cx * img_w:.1f}, {box.cy * img_h:.1f}) px"
                )
                lines.append(
                    f"    范围  : x {x1:.1f} -> {x2:.1f}   y {y1:.1f} -> {y2:.1f}"
                )
                lines.append(f"    像素框: {(x2 - x1):.1f} x {(y2 - y1):.1f}")
                lines.append(f"    原文  : {box.raw_line}")
                lines.append("")
        elif frame.label_path:
            lines.append("（标签文件里没有任何有效行）")
            lines.append("")

        if frame.errors:
            lines.append("!" * 58)
            lines.append("解析告警:")
            lines.extend(f"  - {message}" for message in frame.errors)
        return "\n".join(lines)

    def current_report(self) -> str:
        """返回当前页的文本报告，供自检与外部调用。"""
        return self._compose_text()

    # ------------------------------------------------------------------ 交互
    def _on_browse_root(self) -> None:
        chosen = filedialog.askdirectory(
            title="选择数据集根目录（应包含 images/ 与 labels/）"
        )
        if chosen:
            self.set_root_dir(chosen)

    def _on_image_split(self, _value: str) -> None:
        self._index = 0
        self._refresh_stems(reset_index=True)

    def _on_label_split(self, _value: str) -> None:
        self._render()

    def _on_canvas_resize(self, event: tk.Event) -> None:
        size = (event.width, event.height)
        if abs(size[0] - self._last_canvas_size[0]) < 4 and abs(
            size[1] - self._last_canvas_size[1]
        ) < 4:
            return
        self._last_canvas_size = size
        # 防抖：拖动窗口时尺寸连续变化，等停下来再重绘
        if self._resize_job is not None:
            try:
                self.after_cancel(self._resize_job)
            except (tk.TclError, ValueError):
                pass
        self._resize_job = self.after(90, self._on_resize_settled)

    def _on_resize_settled(self) -> None:
        self._resize_job = None
        self._render_image()

    def _toggle_theme(self) -> None:
        new_mode = "Dark" if ctk.get_appearance_mode() == "Light" else "Light"
        ctk.set_appearance_mode(new_mode)
        self._theme_btn.configure(
            text="切换浅色" if new_mode == "Dark" else "切换深色"
        )
        self._image_label.configure(
            bg=CANVAS_BG[1] if new_mode == "Dark" else CANVAS_BG[0]
        )
        self.after(60, self._render_image)

    def prev_page(self) -> None:
        if len(self._stems) < 2:
            return
        self._index = (self._index - 1) % len(self._stems)
        self._render()

    def next_page(self) -> None:
        if len(self._stems) < 2:
            return
        self._index = (self._index + 1) % len(self._stems)
        self._render()


# ====================================================================== 转换页
class ConvertTab(ctk.CTkFrame):
    """XLABEL → YOLO 数据集转换面板。"""

    def __init__(
        self,
        master: tk.Misc,
        on_converted: Optional[object] = None,
        source_dir: str = DEFAULT_SOURCE,
        output_dir: str = DEFAULT_OUTPUT,
    ) -> None:
        super().__init__(master, fg_color="transparent")
        self._on_converted = on_converted
        self._queue: "queue.Queue[ConvertReport]" = queue.Queue()
        self._running: bool = False
        self._last_output: str = ""

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        form = ctk.CTkFrame(self, corner_radius=10)
        form.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text="标注源目录").grid(
            row=0, column=0, padx=(12, 8), pady=(14, 6), sticky="w"
        )
        self._source_entry = ctk.CTkEntry(form)
        self._source_entry.insert(0, source_dir)
        self._source_entry.grid(row=0, column=1, sticky="ew", pady=(14, 6))
        ctk.CTkButton(
            form, text="浏览", width=70, command=self._browse_source
        ).grid(row=0, column=2, padx=8, pady=(14, 6))

        ctk.CTkLabel(form, text="输出目录").grid(
            row=1, column=0, padx=(12, 8), pady=6, sticky="w"
        )
        self._output_entry = ctk.CTkEntry(form)
        self._output_entry.insert(0, output_dir)
        self._output_entry.grid(row=1, column=1, sticky="ew", pady=6)
        ctk.CTkButton(
            form, text="浏览", width=70, command=self._browse_output
        ).grid(row=1, column=2, padx=8, pady=6)

        opts = ctk.CTkFrame(form, fg_color="transparent")
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=6)

        ctk.CTkLabel(opts, text="验证集比例").grid(row=0, column=0, padx=(0, 8))
        self._ratio_value = 20
        self._ratio_label = ctk.CTkLabel(opts, text="0.20", width=44)
        self._ratio_slider = ctk.CTkSlider(
            opts,
            from_=5,
            to=50,
            number_of_steps=45,
            width=180,
            command=self._on_ratio,
        )
        self._ratio_slider.set(self._ratio_value)
        self._ratio_slider.grid(row=0, column=1, padx=(0, 6))
        self._ratio_label.grid(row=0, column=2, padx=(0, 24))

        ctk.CTkLabel(opts, text="随机种子").grid(row=0, column=3, padx=(0, 8))
        self._seed_entry = ctk.CTkEntry(opts, width=80)
        self._seed_entry.insert(0, "42")
        self._seed_entry.grid(row=0, column=4, padx=(0, 24))

        self._overwrite_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opts,
            text="覆盖已存在的输出目录",
            variable=self._overwrite_var,
        ).grid(row=0, column=5, padx=(0, 8))

        rename_box = ctk.CTkFrame(self, corner_radius=10)
        rename_box.grid(row=2, column=0, sticky="ew", pady=6)
        rename_box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            rename_box,
            text="类别重命名（可选）—— 每行一条「原名=新名」，例如 物资箱=supply_box",
            anchor="w",
        ).grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")
        self._rename_text = ctk.CTkTextbox(
            rename_box,
            height=76,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self._rename_text.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))

        actions = ctk.CTkFrame(self, corner_radius=10)
        actions.grid(row=3, column=0, sticky="ew", pady=6)
        actions.grid_columnconfigure(2, weight=1)

        self._run_btn = ctk.CTkButton(
            actions, text="开始转换", width=120, command=self.start_convert
        )
        self._run_btn.grid(row=0, column=0, padx=12, pady=12)
        self._open_btn = ctk.CTkButton(
            actions,
            text="打开输出目录",
            width=120,
            command=self._open_output,
            state="disabled",
        )
        self._open_btn.grid(row=0, column=1, padx=0, pady=12)
        self._status = ctk.CTkLabel(
            actions, text="等待开始", text_color="#888888", anchor="e"
        )
        self._status.grid(row=0, column=2, padx=12, pady=12, sticky="e")

        log_box = ctk.CTkFrame(self, corner_radius=10)
        log_box.grid(row=5, column=0, sticky="nsew", pady=(6, 0))
        log_box.grid_columnconfigure(0, weight=1)
        log_box.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(log_box, text="转换日志", anchor="w").grid(
            row=0, column=0, padx=12, pady=(10, 4), sticky="w"
        )
        self._log_text = ctk.CTkTextbox(
            log_box, wrap="word", font=ctk.CTkFont(family="Consolas", size=12)
        )
        self._log_text.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        self.after(200, self._poll_queue)

    # ------------------------------------------------------------------ 交互
    def _on_ratio(self, value: float) -> None:
        self._ratio_value = int(round(value))
        self._ratio_label.configure(text=f"{self._ratio_value / 100.0:.2f}")

    def _browse_source(self) -> None:
        chosen = filedialog.askdirectory(title="选择存放图片与同名 json 的目录")
        if chosen:
            self._source_entry.delete(0, "end")
            self._source_entry.insert(0, chosen)

    def _browse_output(self) -> None:
        chosen = filedialog.askdirectory(title="选择输出目录")
        if chosen:
            self._output_entry.delete(0, "end")
            self._output_entry.insert(0, chosen)

    def _open_output(self) -> None:
        if self._last_output and os.path.isdir(self._last_output):
            try:
                os.startfile(self._last_output)  # type: ignore[attr-defined]
            except OSError as exc:
                messagebox.showerror(APP_TITLE, f"无法打开目录: {exc}")

    def _log(self, message: str) -> None:
        self._log_text.configure(state="normal")
        self._log_text.insert("end", message + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def start_convert(self) -> None:
        if self._running:
            messagebox.showinfo(APP_TITLE, "转换正在进行中")
            return
        source = self._source_entry.get().strip().strip('"')
        output = self._output_entry.get().strip().strip('"')
        if not os.path.isdir(source):
            messagebox.showerror(APP_TITLE, f"标注源目录不存在:\n{source}")
            return
        if not output:
            messagebox.showerror(APP_TITLE, "请填写输出目录")
            return
        if os.path.abspath(source) == os.path.abspath(output):
            messagebox.showerror(APP_TITLE, "输出目录不能和标注源目录相同")
            return
        if (
            os.path.isdir(output)
            and os.listdir(output)
            and not self._overwrite_var.get()
        ):
            messagebox.showerror(
                APP_TITLE,
                f"输出目录已存在且非空:\n{output}\n\n"
                "如确认要清空重建，请勾选「覆盖已存在的输出目录」。",
            )
            return

        try:
            seed = int(self._seed_entry.get().strip())
        except ValueError:
            messagebox.showerror(APP_TITLE, "随机种子必须是整数")
            return
        rename = parse_rename_rules(self._rename_text.get("1.0", "end"))

        options = ConvertOptions(
            source_dir=source,
            output_dir=output,
            val_ratio=self._ratio_value / 100.0,
            seed=seed,
            rename=rename,
            overwrite=bool(self._overwrite_var.get()),
        )
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")
        self._log("开始转换…")
        self._running = True
        self._run_btn.configure(state="disabled", text="转换中…")
        self._status.configure(text="转换中…")
        threading.Thread(
            target=self._worker, args=(options,), daemon=True
        ).start()

    def _worker(self, options: ConvertOptions) -> None:
        """后台线程里执行转换，结果放队列，由主线程取出。"""
        try:
            report = convert_dataset(options)
        except Exception as exc:  # noqa: BLE001 - 兜底，避免线程静默死掉
            report = ConvertReport(ok=False, message=f"转换异常: {exc!r}")
        self._queue.put(report)

    def _poll_queue(self) -> None:
        try:
            while True:
                report = self._queue.get_nowait()
                self._finish(report)
        except queue.Empty:
            pass
        self.after(200, self._poll_queue)

    def _finish(self, report: ConvertReport) -> None:
        self._running = False
        self._run_btn.configure(state="normal", text="开始转换")
        for line in report.log:
            self._log(line)
        if report.skipped_shapes:
            self._log(f"跳过 {len(report.skipped_shapes)} 个形状:")
            for item in report.skipped_shapes[:20]:
                self._log(f"  - {item}")
            if len(report.skipped_shapes) > 20:
                self._log(f"  …（共 {len(report.skipped_shapes)} 条）")

        if not report.ok:
            self._status.configure(text="失败")
            self._log("")
            self._log(f"失败: {report.message}")
            messagebox.showerror(APP_TITLE, report.message)
            return

        self._last_output = report.output_dir
        self._open_btn.configure(state="normal")
        self._status.configure(text="完成")
        self._log("")
        self._log(f"训练 {report.train} 张 / 验证 {report.val} 张")
        self._log(f"空标签(背景图) {report.empty_labels} 张")
        if report.class_counts:
            self._log("类别框数:")
            for name, count in sorted(report.class_counts.items()):
                self._log(format_class_line(report, name, count))
        if report.unlabeled:
            self._log(f"源目录中还有 {report.unlabeled} 张图未标注，未纳入")
        self._log("")
        self._log(f"data.yaml 已写入: {report.yaml_path}")

        if callable(self._on_converted):
            self._on_converted(report.output_dir)


# ====================================================================== 主窗口
class DatasetToolApp(ctk.CTk):
    """转换 + 查看 一体化的主窗口。"""

    def __init__(self, source_dir: str, output_dir: str) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1400x900")
        self.minsize(1080, 680)
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self._tabs = ctk.CTkTabview(self)
        self._tabs.pack(fill="both", expand=True, padx=12, pady=12)
        convert_frame = self._tabs.add("转换")
        viewer_frame = self._tabs.add("查看")

        self.convert_tab = ConvertTab(
            convert_frame,
            on_converted=self._on_converted,
            source_dir=source_dir,
            output_dir=output_dir,
        )
        self.convert_tab.pack(fill="both", expand=True)

        self.viewer_tab = ViewerTab(viewer_frame, root_dir=output_dir)
        self.viewer_tab.pack(fill="both", expand=True)

    def _on_converted(self, output_dir: str) -> None:
        """转换完成后，把查看页指到新数据集并切过去。"""
        self.viewer_tab.set_root_dir(output_dir)
        self._tabs.set("查看")


# ====================================================================== 命令行
def run_selftest(root_dir: str) -> int:
    """不开窗口，只检查数据集能否被正确解析。"""
    print(f"根目录: {root_dir}")
    yaml_path = os.path.join(root_dir, DEFAULT_YAML)
    names = load_class_names(yaml_path)
    print(f"data.yaml: {os.path.basename(yaml_path)} -> {names}")

    image_splits, label_splits = scan_split_dirs(root_dir)
    print(f"images 子目录: {image_splits}")
    print(f"labels 子目录: {label_splits}")
    if not image_splits or not label_splits:
        print("错误: 没有找到 images/ 或 labels/ 子目录")
        return 1

    for split in image_splits:
        stems = list_image_stems(os.path.join(root_dir, "images", split))
        print(f"\n[images/{split}] {len(stems)} 张图")
        for label_split in label_splits:
            total_boxes = 0
            missing = 0
            empty = 0
            bad = 0
            for stem in stems:
                label_path = os.path.join(
                    root_dir, "labels", label_split, stem + ".txt"
                )
                if not os.path.isfile(label_path):
                    missing += 1
                    continue
                boxes, errors = parse_yolo_label(label_path)
                total_boxes += len(boxes)
                if not boxes:
                    empty += 1
                bad += len(errors)
            print(
                f"  对 labels/{label_split}: 框 {total_boxes} 个, "
                f"缺标签 {missing} 张, 空标签 {empty} 张, 解析告警 {bad} 条"
            )
    return 0


def run_convert_cli(args: argparse.Namespace) -> int:
    """命令行转换，便于脚本化与验证。"""
    options = ConvertOptions(
        source_dir=args.source,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        rename=parse_rename_rules(args.rename or ""),
        overwrite=args.force,
    )
    report = convert_dataset(options)
    for line in report.log:
        print(line)
    if report.skipped_shapes:
        print(f"跳过 {len(report.skipped_shapes)} 个形状: {report.skipped_shapes[:10]}")
    if not report.ok:
        print(f"失败: {report.message}", file=sys.stderr)
        return 1
    print()
    print(f"训练 {report.train} / 验证 {report.val} / 框 {report.boxes} / 空标签 {report.empty_labels}")
    print(f"未标注图片 {report.unlabeled} 张")
    print("类别框数:")
    for name, count in sorted(report.class_counts.items()):
        print(format_class_line(report, name, count))
    print(f"data.yaml: {report.yaml_path}")
    return 0


def run_smoke(root_dir: str) -> int:
    """构造窗口、反复重绘并翻页后销毁，用于验证界面代码路径。"""
    app = DatasetToolApp(DEFAULT_SOURCE, root_dir)
    app.update_idletasks()
    app.update()
    app.update()
    viewer = app.viewer_tab
    print("=== 查看页 第 1 页 ===")
    print(viewer.current_report())

    viewer.next_page()
    app.update_idletasks()
    print("\n=== 翻页后 ===")
    print(viewer.current_report())

    for _ in range(30):
        viewer._render_image()
        app.update_idletasks()
    for _ in range(12):
        viewer.next_page()
        app.update_idletasks()
        app.update()
    for _ in range(6):
        viewer.prev_page()
        app.update_idletasks()
        app.update()
    print(f"\nSTRESS OK, 当前第 {viewer._index + 1} / {len(viewer._stems)} 页")

    app._tabs.set("转换")
    app.update_idletasks()
    app.update()
    print("TAB SWITCH OK")

    app.destroy()
    print("SMOKE OK")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("root", nargs="?", default=DEFAULT_OUTPUT,
                        help=f"查看页的数据集根目录（默认 {DEFAULT_OUTPUT}）")
    parser.add_argument("--check", action="store_true",
                        help="只做查看页解析自检，不打开窗口")
    parser.add_argument("--smoke", action="store_true",
                        help="构造窗口渲染后销毁，用于验证界面代码（会短暂闪窗）")
    parser.add_argument("--convert", action="store_true",
                        help="命令行执行转换后退出")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help=f"转换的标注源目录（默认 {DEFAULT_SOURCE}）")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"转换的输出目录（默认 {DEFAULT_OUTPUT}）")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="验证集比例，0~1（默认 0.2）")
    parser.add_argument("--seed", type=int, default=42,
                        help="划分随机种子（默认 42）")
    parser.add_argument("--rename", default="",
                        help="类别重命名，多条用分号隔开，如 物资箱=supply_box")
    parser.add_argument("--force", action="store_true",
                        help="输出目录非空时也覆盖（会先删除原目录）")
    args = parser.parse_args(argv)

    if args.check:
        return run_selftest(os.path.abspath(os.path.expanduser(args.root)))
    if args.smoke:
        return run_smoke(os.path.abspath(os.path.expanduser(args.root)))
    if args.convert:
        return run_convert_cli(args)

    root_dir = os.path.abspath(os.path.expanduser(args.root))
    app = DatasetToolApp(
        os.path.abspath(os.path.expanduser(args.source)), root_dir
    )
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
