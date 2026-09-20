
# yolo-dataset-toolkit

> 把 X-AnyLabeling 的标注一键变成能直接训练的 YOLO 数据集，并且能亲眼看见转换结果对不对。

**English blurb:** A desktop toolkit that converts X-AnyLabeling (XLABEL JSON) annotations into an
Ultralytics-ready YOLO dataset, and provides a built-in viewer so you can actually verify that
every label file matches its image before you spend GPU hours on training.
[→ English docs](README.en.md)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![GUI](https://img.shields.io/badge/GUI-CustomTkinter-informational)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

***

## 目录

- [它解决什么问题](#它解决什么问题)
- [演示](#演示)
- [功能一览](#功能一览)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细用法](#详细用法)
- [架构与数据流](#架构与数据流)
- [关键设计取舍](#关键设计取舍)
- [常见问题](#常见问题)
- [已知限制](#已知限制)
- [项目结构](#项目结构)
- [路线图](#路线图)
- [贡献](#贡献)
- [许可](#许可)
- [致谢](#致谢)

***

## 它解决什么问题

用 X-AnyLabeling 标注完数据，到真正能跑 `yolo train` 之间，有一段没人管的「最后一公里」。这个工具就是来填这段路的。

### 痛点 1：标注完了，却不知道训练时该填什么

X-AnyLabeling 自带一键训练面板，但它的 `Data` 字段**必须指向一个真实存在的 `data.yaml `**&#x20;
也就是整个训练数据集要符合yolo官方所设定的结构，而程序里**没有任何「根据当前标注生成数据集结构」的按钮**。
结果是：图也标了、`Train` 菜单也打开了，卡在 `Data` 不知道填什么。

官方 FAQ 里「预编译 EXE 不支持 Ultralytics 训练」的说法，和更新日志里「已增强外部训练」的说法
互相矛盾，用户很难判断自己该走哪条路。

**本工具的转换页就干这一件事**：指定源目录和输出目录，点一下，`images/`、`labels/`、`data.yaml`
全部就位，`Data` 字段直接 Browse 选它生成好的data.yaml就能开始训练。

### 痛点 2：转换结果是个黑盒

YOLO 的标签是纯数字：

```
0 0.356186 0.352402 0.021702 0.023102
```

`images/` 和 `labels/` 又是两个分开的目录。于是产生一个很难回答的问题：

> 这张图对应的标签文件，到底是不是它的？train/val 划分之后有没有错位？

划分错位这种事，训练**不会报错**，只会悄悄让指标变差，排起来极其痛苦。

**本工具的查看页**把两侧并排放在一起：左边是图（可以直接把框画在图上），右边是解析后的
标注内容。而且 **images 和 labels 两个来源可以各自独立切换**——把 `images/train` 配上
`labels/val`，缺哪个标签一眼就露馅。

### 痛点 3：类别索引不可读

txt 里的 `0`、`1`、`2` 是什么类别？查看页会自动读 `data.yaml` 的 `names` 字段做映射，
显示成 `物资箱 (class_id = 0)`，不需要你手写 `classes.txt` 或对着数字猜。

### 痛点 4：中文类别名的编码风险

类别名带中文时，在 Windows + YAML + 训练框架这条链路上容易出编码问题。
转换页提供了**类别重命名**通道（`物资箱=supply_box`），一次改完，不用手动编辑几十个文件。

### 痛点 5：换一次数据集就要重写一次脚本

同一个转换脚本，每个人都在自己的项目里写一遍，而且**大多是无条件** **`rmtree`** **输出目录**——
手滑一次就没了。

本工具把它固化下来：默认**拒绝覆盖非空目录**，必须显式勾选；命令行模式则要求显式 `--force`。
转换逻辑本身是**不依赖 GUI 的纯函数**，可以直接 `import` 进你自己的流水线或 CI。

***

## 演示

<img width="1444" height="752" alt="演示" src="https://github.com/user-attachments/assets/ae3f34f5-78af-4c6d-913e-ad6becdd03ea" />



**演示覆盖的流程：**

1. 在【转换】页填入标注源目录与输出目录
2. 设置验证集比例、类别重命名（可选）
3. 点「开始转换」，在下方日志区查看统计与警告
4. 自动跳转到【查看】页，逐页核对图片与标注是否对应
5. 用「图像来源 / 标签来源」两个下拉独立切换，故意错配以暴露问题



***

## 功能一览

### 转换页：XLABEL → YOLO

- 识别 `图片 + 同名 .json` 的配对，自动跳过未标注的图片并单独报数
- 支持 `rectangle`（矩形）与 `polygon`（多边形）标注，多边形自动取外接框
- 按比例随机划分 train / val，**随机种子可固定**，划分结果可复现
- 类别索引按名称排序生成，**多次运行 id 稳定**
- 可选**类别重命名**（`原名=新名`，一行一条）
- 无标注的图片生成**空标签文件**（背景图），而不是被丢掉
- 自动写出 `data.yaml`，路径统一用正斜杠（避开 YAML 里反斜杠的坑）
- 后台线程执行，界面不卡死；日志实时输出
- **默认拒绝覆盖非空输出目录**，需要显式勾选

### 查看页：数据集校验

- 左侧图像，**可直接把标注框画在图上**
- 右侧完整解析 txt：类别名、YOLO 归一化值、像素中心、像素范围、原始行
- **图像来源 / 标签来源两个下拉框独立切换**，专治 train/val 错配
- 标签文件缺失时提示「该图片在其他 labels 分支中存在」
- 自动读 `data.yaml` 的 `names` 做类别映射，兼容 dict 与 list 两种写法
- 方向键 `←` `→` 翻页
- **纯只读**，不往数据集写任何文件

### 命令行

三种无界面模式，方便脚本化和 CI：

| 命令          | 用途                  |
| ----------- | ------------------- |
| `--convert` | 无界面执行转换             |
| `--check`   | 只解析数据集并打印配对统计，用于断言  |
| `--smoke`   | 构造界面、压力重绘后销毁，用于回归测试 |

***

## 安装

### 依赖

- Python **3.10+**（开发环境为 3.10.16）
- 仅需三个第三方库：

```
customtkinter
pillow
pyyaml
```

### 安装命令

```bash
git clone https://github.com/bai-stack/yolo-dataset-toolkit.git
cd yolo-dataset-toolkit
pip install customtkinter pillow pyyaml
```

<details>
<summary>Conda 用户</summary>

```bash
conda create -n yolo-toolkit python=3.11 -y
conda activate yolo-toolkit
pip install customtkinter pillow pyyaml
```

</details>

***

## 快速开始

### 图形界面

```bash
python dataset_tool.py
```

程序启动后有两个标签页：

1. **转换** —— 填「标注源目录」（存图片和同名 `.json` 的目录）和「输出目录」，点「开始转换」
2. **查看** —— 转换完成后会**自动跳转**到这里并指向新数据集

> **⚠️ 关于默认路径（第一次用必读）**
>
> 两个输入框的初值来自文件顶部的 `DEFAULT_SOURCE` 和 `DEFAULT_OUTPUT` 两个常量。
> 本仓库中这两个值**指向开发者本人的本地目录**（`C:\Users\99259\...`）——
> 也就是说，在你自己的机器上这两个目录**并不存在**，输入框里显示的是一个无效路径。
>
> **这不会导致程序出错**，但要正常使用，你需要：
>
> - 点旁边的「浏览」按钮，选你自己的目录；或者
> - 直接编辑 `dataset_tool.py` 顶部那两行，把它们改成你的常用目录（推荐，省得每次都要选）
>
> 之所以没有把它们改成相对路径，是为了保持代码直白——路径就写在那儿，一眼能看到、能改。

### 命令行

```bash
# 转换（不覆盖已存在的非空目录）
python dataset_tool.py --convert \
    --source ./dataset/images \
    --output ./dataset/yolo

# 转换并重命名类别
python dataset_tool.py --convert \
    --source ./dataset/images \
    --output ./dataset/yolo \
    --rename "物资箱=supply_box"

# 校验生成结果
python dataset_tool.py --check ./dataset/yolo
```

***

## 详细用法

### 转换参数

| 参数            | 默认    | 说明                                       |
| ------------- | ----- | ---------------------------------------- |
| `--source`    | —     | 标注源目录，需含图片与其同名 `.json`                   |
| `--output`    | —     | 输出目录，会创建 `images/`、`labels/`、`data.yaml` |
| `--val-ratio` | `0.2` | 验证集比例，0\~1                               |
| `--seed`      | `42`  | 划分随机种子，固定它可复现划分                          |
| `--rename`    | 空     | 类别重命名，分号分隔多条：`旧名=新名;旧名2=新名2`             |
| `--force`     | 关     | 输出目录非空时也覆盖。**会先删除整个输出目录**，谨慎使用           |

### 输出结构

```
<output_dir>/
├── data.yaml                 # 训练配置，填进 X-AnyLabeling 的 Data 字段
├── images/
│   ├── train/                # 训练图片
│   └── val/                  # 验证图片
└── labels/
    ├── train/                # 与 images/train 同名的 .txt
    └── val/
```

`data.yaml` 形如：

```yaml
# 由 yolo-dataset-toolkit 从 X-AnyLabeling 标注生成
path: D:/dataset/yolo
train: images/train
val: images/val
names:
  0: supply_box
```

这个结构同时满足两个下游要求：

- **Ultralytics** 的自定义数据集约定
- **X-AnyLabeling** 的 `resolve_prepared_dataset()` —— 它要求 `path`/`train`/`val`
  三者对应的目录都真实存在，才认定为「已准备数据集」并直接使用

### 标注格式要求

源目录里每一对需要满足：

```
<source_dir>/
├── 0001.jpg          # 图片
└── 0001.json         # 同名标注（X-AnyLabeling 的 XLABEL 格式）
```

支持的 `shape_type`：

| 类型          | 处理方式                     |
| ----------- | ------------------------ |
| `rectangle` | 由四个角点取外接框                |
| `polygon`   | 由全部顶点取外接框（会损失轮廓信息，属预期行为） |
| 其他          | 跳过，并在日志中列出               |

***

## 架构与数据流

### 分层

```
dataset_tool.py
│
├─ 数据层（不依赖 GUI，可单独 import）
│   ├─ 数据结构   Box / Frame / Sample / ConvertOptions / ConvertReport
│   ├─ 通用工具   load_class_names() / list_image_stems() / find_image_path()
│   │              parse_yolo_label() / scan_split_dirs() / parse_rename_rules()
│   └─ 转换核心   convert_dataset(ConvertOptions) -> ConvertReport
│
├─ 界面层（CustomTkinter）
│   ├─ ViewerTab    左图右标注 + 双下拉独立切换
│   ├─ ConvertTab   参数表单 + 日志 + 后台线程调度
│   └─ DatasetToolApp   CTkTabview 容器，负责两页联动
│
└─ 命令行层
    ├─ run_convert_cli()   无界面转换
    ├─ run_selftest()      纯解析自检（--check）
    └─ run_smoke()         界面冒烟 + 压力测试（--smoke）
```

**关键点：数据层不 import 任何 GUI 库。** 也就是说你可以这样用：

```python
from dataset_tool import ConvertOptions, convert_dataset

report = convert_dataset(ConvertOptions(
    source_dir=r"D:\labels\images",
    output_dir=r"D:\labels\yolo",
    val_ratio=0.2,
    seed=42,
))

if report.ok:
    print(f"{report.train} 训练 / {report.val} 验证 / {report.boxes} 个框")
else:
    print("失败:", report.message)
```

### 转换流程

```
  源目录扫描
      │
      ├─ 列出所有图片 ──────────────► 未标注的图片计数（不参与，单独上报）
      │
      ├─ 对每张图找同名 .json
      │      │
      │      ├─ 缺 json ────────────► 计入「未标注」
      │      └─ 有 json ────────────► 解析 imageWidth / imageHeight / shapes
      │                                    │
      │                                    ├─ rectangle / polygon ─► 像素外接框
      │                                    └─ 其他类型 ─────────────► 跳过并记录
      │
      ├─ 汇总所有类别名 ─► 排序（保证 id 稳定）─► 应用重命名规则
      │
      ├─ 固定种子随机划分 ─► train / val
      │
      ├─ 安全检查：输出目录非空？
      │      ├─ 非空且未勾选覆盖 ──► 直接中止，返回失败报告
      │      └─ 通过 ──────────────► 清空并重建 images/ labels/ 目录
      │
      ├─ 逐样本写盘
      │      ├─ 复制图片 ──────────────────► images/<split>/<stem>.<ext>
      │      └─ 像素框转归一化 YOLO ────────► labels/<split>/<stem>.txt
      │                                          （无目标则写空文件，保留背景图）
      │
      └─ 写 data.yaml（正斜杠路径）
             │
             └─► ConvertReport { ok, log[], train, val, boxes, class_counts, ... }
```

### 界面数据流

```
        用户点击「开始转换」
                │
                ▼
      ConvertTab.start_convert()
        ├─ 前置校验（目录存在 / 不与源目录相同 / 非空且未勾选覆盖）
        ├─ 禁用按钮，清空日志
        └─ threading.Thread(target=self._worker).start()
                │
                ▼ （子线程）
          convert_dataset(options)
                │
                └─► self._queue.put(report)
                              │
                              ▼
          主线程 self.after(200, self._poll_queue)   ← 每 200ms 轮询
                │
                ├─ _finish(report)  打印日志、统计、失败弹窗
                └─ on_converted(output_dir)
                        │
                        ▼
              DatasetToolApp._on_converted()
                ├─ viewer_tab.set_root_dir(输出目录)
                └─ tabs.set("查看")          ← 自动跳到查看页
```

**为什么用队列轮询而不是让子线程直接操作界面？**
Tkinter 不是线程安全的，从子线程调用 `widget.configure()` 会随机崩溃。
「子线程只放队列、主线程轮询取出」是跨线程更新界面的标准安全做法。

***

## 关键设计取舍

这一节记录的都是踩过坑之后的决定，不是凭空的偏好。

### 1. 图像用原生 `tk.Label`，而不是 `CTkLabel`

CTkLabel 内部会把 `ImageTk.PhotoImage` 缓存在 `CTkImage` 对象里，
而 `CTkImage` 的生命周期被绑在那个 Python 对象上。频繁重绘时（切图、拖窗口），
旧对象一旦被回收，它缓存的 PhotoImage 也被销毁，Tk 侧就报：

```
_tkinter.TclError: image "pyimage7" doesn't exist
```

改用原生 `tk.Label` + 自己持有 `ImageTk.PhotoImage` 的强引用后，这个问题消失。
代价是要手动同步深浅色背景，但换来了可预测的生命周期。

### 2. 重绘要防重入

给标签设置图片会改变它的尺寸 → 父容器重新布局 → 触发 `<Configure>` 事件 →
又调回重绘函数。**这个重入会把正在使用的图像对象回收掉。**

解决方式是两道防线：

```python
def _render_image(self) -> None:
    if self._rendering:      # 防重入
        return
    self._rendering = True
    try:
        self._render_image_inner()
    finally:
        self._rendering = False
```

外加对 `<Configure>` 做 90ms 防抖，拖动窗口时只在停下来之后重绘一次。

### 3. 图像设置要有降级兜底

即使有上面两道防线，`configure(image=...)` 理论上仍可能因为 Tk 句柄失效而抛异常。
一旦抛出，整个界面就崩了。所以再包一层：

```python
try:
    self._image_label.configure(image=photo, text="")
except tk.TclError:
    self._photo = None
    self._image_label.configure(image="", text="图像渲染失败，已降级为文字提示")
```

**宁可显示不出图，也不能让程序直接退出。**

### 4. 拒绝静默覆盖

早期版本无条件 `shutil.rmtree(输出目录)`。这在交互式使用里非常危险——
手滑点错目录，别人几天的标注就没了。

现在：非空目录必须显式勾选「覆盖」，命令行必须显式传 `--force`，
并且报告里会写明「已清空输出目录」。

### 5. 类别 id 排序而非按出现顺序

按首次出现顺序分配 id 也能跑，但同一批数据只要样本顺序变了（换机器、换文件系统），
id 就可能变，导致两次转换的标签对不上。

改成**按类别名排序**，只要类别集合不变，id 就永远一致。

### 6. 保留空标签文件

没有标注的图片有两种可能：**「还没标」** 和 **「确认里面没目标」**。
前者不该进数据集，后者是很有价值的背景样本。

本工具的判定依据是**有没有同名** **`.json`**：有 json 但 shapes 为空 → 视为背景图，
生成 0 字节的 `.txt`（这正是 YOLO 约定）；根本没有 json → 视为未标注，不参与。

### 7. `data.yaml` 里路径用正斜杠

YAML 的普通标量里反斜杠虽然按字面处理，但在不同解析器和字符串插值路径下容易出歧义。
统一写成 `D:/dataset/yolo` 这种形式，Ultralytics 和 Python 的 `pathlib` 都能正确处理。

### 8. 转换核心不依赖 GUI

`convert_dataset()` 是纯函数，输入 `ConvertOptions`、输出 `ConvertReport`，
不碰任何全局状态、不弹窗、不打印。好处：

- 可以单独 `import` 进别的脚本
- 可以在 CI 里跑断言
- 界面只是它的一个调用方，不是它的宿主

***

## 常见问题

<details>
<summary><b>打开后输入框里是一个不存在的路径（<code>C:\Users\99259\...</code>）</b></summary>

这不是 bug。那两个输入框的初值是代码顶部的常量，而仓库里保留的是**开发者本人的本地目录**，
在你的机器上当然不存在。

处理方式：点「浏览」选自己的目录，或者直接改 `dataset_tool.py` 开头的
`DEFAULT_SOURCE` / `DEFAULT_OUTPUT` 两行。

**不影响功能**——路径只在点「开始转换」时才会被真正使用，届时会做存在性校验并给出明确报错。
</details>

<details>
<summary><b>X-AnyLabeling 训练面板里的 Data 该填什么？</b></summary>

填本工具生成的 `data.yaml` 的完整路径，例如：

```
C:\Users\you\Desktop\dataset\yolo\data.yaml
```

注意：**这个字段不能留空。** X-AnyLabeling 内部会无条件调用 `validate_data_file("")`，
空字符串会直接报 `Failed to parse data file:`（冒号后面是空的，很有迷惑性）。

</details>

<details>
<summary><b>类别名能用中文吗？</b></summary>

能用，但不建议。本工具写 `data.yaml` 时用的是显式 UTF-8，Ultralytics 也是按 UTF-8 读的，
只要路径本身是纯 ASCII 一般没问题。

风险出在链路更长的时候（第三方工具、不同编辑器保存、终端编码）。
**转换页的「类别重命名」就是为此准备的**——`物资箱=supply_box`，一行搞定。

另外注意：X-AnyLabeling 里 GroundingDINO 一类的文本提示模型用的是英文文本编码器，
**中文提示词基本无效**。这跟类别名是两回事。

</details>

<details>
<summary><b>为什么有些标签文件是 0 字节？</b></summary>

那是**背景图**——源标注里这张图存在 json 但没有任何形状。
生成空 `.txt` 是 YOLO 的标准做法，能让模型学会「这类区域里什么都没有」。不是 bug。

</details>

<details>
<summary><b>提示「输出目录已存在且非空」怎么办？</b></summary>

这是刻意的保护。三选一：

1. 换一个输出目录（推荐）
2. 手动清空后重试
3. 确认无风险后勾选「覆盖已存在的输出目录」（会先删除整个目录）

</details>

<details>
<summary><b>界面报 <code>image "pyimageN" doesn't exist</code></b></summary>

这是 CustomTkinter 的 `CTkImage` 在频繁重绘时的已知问题。本工具的查看页已经不使用它了，
见[关键设计取舍](#1-图像用原生-tklabel而不是-ctklabel)。

如果你在其他 CustomTkinter 项目里遇到，思路是一样的：改用原生 `tk.Label`
并自己持有 `ImageTk.PhotoImage` 的引用。

</details>

<details>
<summary><b><code>--check</code> 能用在 CI 里吗？</b></summary>

可以，退出码即结果：`0` 表示结构正常，`1` 表示没找到 `images/` 或 `labels/`。
输出里每行「对 labels/\<split>」的统计可以直接 grep。

典型的健康结果长这样：

```
[images/train] 21 张图
  对 labels/train: 框 21 个, 缺标签 0 张, 空标签 0 张, 解析告警 0 条
  对 labels/val:   框 0 个,  缺标签 21 张, 空标签 0 张, 解析告警 0 条
```

交叉组合（train 图 × val 标签）全部「缺标签」是**正确**的，说明两个分支没有混。

</details>

***

## 已知限制

- **只做目标检测**。分类、分割、姿态、旋转框暂不支持。
- **转换是单向的**。本工具只做 XLABEL → YOLO；反向（把 YOLO 标签导回 X-AnyLabeling）
  请用 X-AnyLabeling 自带的「上传」功能。
- **验证集过小时会提示但不阻止**。样本量小于 5 仍可能划分出空验证集，
  日志里会提醒调大比例或补数据。
- **界面已实测，模型相关链路未全部实测**。转换与查看功能经过多轮自检与压力测试；
  但「生成的 `data.yaml` 能否在你特定的 X-AnyLabeling 版本上跑通训练」，
  受其自身版本与 EXE 打包状态影响，请以实测为准。
- **大目录首次扫描为同步操作**。转换本身在子线程跑，但初始目录扫描是同步的，
  几万张图时启动会有短暂停顿。

***

## 项目结构

```
yolo-dataset-toolkit/
├── dataset_tool.py      # 主程序：转换页 + 查看页 + 命令行（单文件，无子模块）
├── README.md            # 中文说明（本文件）
├── README.en.md         # English documentation
├── LICENSE              # MIT
└── 演示.gif              # 使用演示
```

**故意做成单文件。** 这是个几百行的工具，拆成包只会增加 `import` 路径的麻烦；
想二次开发的话，直接 `from dataset_tool import convert_dataset` 即可。

***

## 路线图

- [ ] 支持 `--task segment` 输出 YOLO 分割格式（多边形点集）
- [ ] 支持旋转框（OBB）转换
- [ ] 查看页增加「只看有问题的图」过滤模式
- [ ] 查看页支持缩放与拖拽
- [ ] 转换页支持记住上次使用的目录
- [ ] 打包为单文件可执行程序

欢迎提 Issue 认领。

***

## 贡献

1. Fork 并新建分支
2. 提交前请确保这两个自检通过：
   ```bash
   python dataset_tool.py --check <一个测试数据集>
   python dataset_tool.py --smoke
   ```
3. 代码风格：
   - **所有 Python 函数必须带类型注解**
   - 行宽 79
4. 提交 Pull Request，说明改了什么、为什么

***

## 许可

本项目采用 **MIT License** —— 详见 [LICENSE](LICENSE)。

> **关于 X-AnyLabeling**：本工具读取其 XLABEL JSON 格式并生成其训练面板可用的
> `data.yaml`，属于格式兼容，不包含也不链接其源代码。X-AnyLabeling 本身为 GPL-3.0，
> 如果你打算把两者打包分发，请自行确认许可兼容性。

***

## 致谢

- [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) —— 标注工具与数据格式
- [Ultralytics](https://github.com/ultralytics/ultralytics) —— YOLO 数据集约定
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) —— 界面框架

