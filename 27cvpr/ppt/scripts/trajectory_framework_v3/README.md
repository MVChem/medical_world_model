# Trajectory framework v3：组件独立编辑版

使用 Python 的 `python-pptx` 调整用户指定 v2 的实际 PPT 对象结构。输入快照保存在 `assets/source_v2.pptx`，不重新调用旧版画图脚本。v2 原文件保留。

- [可编辑 PPT](../../ppt/trajectory_framework_v3.pptx)
- [从最终 PPT 渲染的预览](../../ppt/trajectory_framework_v3.png)
- `build.py`：重新生成与渲染入口。
- `components.json`：所有顶层组件的名称、类型、位置和子对象清单。

## 怎么编辑

共 **49 个顶层组件**。在 PowerPoint 的“选择窗格”中按以下前缀查找对象，组件名称包含中文用途说明。

| 对象前缀 | 编辑方式 |
|---|---|
| `TEXT` | 7 个独立文本框：6 行标题和速度符号 `v`。没有任何文字藏在分组中，可直接改字、字体、字号与位置。 |
| `ARROW 01.1` | 蓝色轨迹杆是原生开放 Bézier 曲线，可以改颜色、线宽及“编辑顶点”。 |
| `ARROW 01.2` | 蓝色轨迹的箭头头部是独立自由形状，可单独移动、缩放或编辑顶点。 |
| `ARROW 02.1`、`02.2` | 两支循环箭头各为独立原生自由形状，保留 v2 的渐变和曲线轮廓；通过“编辑顶点”调整形状。它们不是带黄色手柄的预设弧形箭头。 |
| 其余 `ARROW` | 12 支放射箭头与 2 支双向箭头为 PowerPoint 原生预设形状，可移动、旋转、改色，并用黄色手柄调节杆宽与头部比例。 |
| `CARD` | 每张卡片各自保留一层小分组。可整体移动，也可进入分组或取消组合编辑表面、边框、山峰、太阳等。 |
| `POINT` | 三个时间点分别组成一层小分组，可分别移动、缩放。 |
| `TARGET` | 四个同心圆彼此独立，可单独改色、大小和位置。 |
| `FX` | 背景柔光、轨迹白色柔光和速度短线。可以在选择窗格中隐藏、移动或删除，不影响文字和箭头主体。 |

没有面板级的大分组，也没有多层嵌套组件分组。所有可见内容均为原生 PowerPoint 对象，没有嵌入 PNG 或 SVG。

装饰柔光由多条半透明曲线组成，仍保留为一个独立 `FX` 小组；改变蓝色轨迹的路径后，柔光**不会自动跟随**，需要单独调整或隐藏。循环箭头的箭身与头部保留在同一个自由形状内，二者的顶点均可编辑。速度符号从描边轮廓改成了 Arial Bold Italic 的实际字符 `v`，字形与旧版存在轻微差异。

## 重建和更新预览

需要 Python 3、`python-pptx`、`lxml`、`PyMuPDF`，以及 PATH 中的 LibreOffice。字体使用 Arial。

```bash
python -m pip install python-pptx lxml PyMuPDF
python build.py --overwrite
```

**`--overwrite` 会从保存的 v2 快照重建 v3，覆盖 v3 的手动编辑。** 已经在 PowerPoint 中改过 v3 时，只更新预览：

```bash
python build.py --render-only
```

脚本只通过自身位置解析路径。移动文件夹时保持 `ppt/` 与 `scripts/trajectory_framework_v3/` 的相对位置。输入快照也随脚本提供。

结构检查包括：文字均位于顶层、没有组件嵌套、14 支原生预设箭头、对象名称和 ID 唯一、没有嵌入图片。预览由实际 PPT 经 LibreOffice 导出并渲染，尺寸为 3072 × 1024，比例 3:1。
