# Figure 3 — editable CXR version

根据 `assets/reference.png` 重建的单页 PowerPoint，保留原图 2018:779 比例、两个面板的布局及粉／蓝／黄色分支。按要求将所有脑部 MRI 替换为本地 MIMIC-CXR 正位胸片，同步调整报告摘录和右侧特征图。

交付文件：

- `../../ppt/fig3_v3.pptx`：可编辑 PPT。
- `../../ppt/fig3_v3.png`：从最终 PPT 经 LibreOffice 渲染的 4036 × 1558 预览。
- `build.py`：重建与渲染入口；路径均相对脚本解析。
- `assets/cxr.png`：随附的胸片素材，重建不依赖原始数据目录。
- `assets/provenance.json`：胸片、报告的仓库相对来源路径及裁剪信息。

## 重建

Python 3.10+，依赖 `python-pptx`、`Pillow`、`numpy`；预览另需 LibreOffice 和 Poppler 的 `pdftoppm`。

```bash
python -m pip install python-pptx Pillow numpy
python build.py --render
```

默认输出到上述 `ppt/` 子目录，重新运行会覆盖该脚本生成的同名结果。手动编辑 PPT 后请另存，或指定不同输出：

```bash
python build.py --output /path/to/fig3_custom.pptx --render
```

整个 `27cvpr/ppt` 文件夹可以一起搬移后重建。只复制本脚本文件夹时，使用 `--output` 指定保存位置即可。渲染过程中的 PDF 和 LibreOffice 配置写入临时目录并自动删除，不会加入 Git。

## 编辑方式

- 标题、标签、报告和公式均为独立原生文本框。公式使用普通文本及上下标，不是位图或轮廓路径。
- 所有模块、slot、背景、括号、箭头均为原生对象。梯形是可编辑顶点的自由形状；直线箭头是带线端箭头的原生连接线；折线箭头可编辑顶点，箭头随路径调整。
- 使用 PowerPoint 的“选择窗格”按名称选取对象，例如 `Panel A title`、`Readout 1 label`、`Slots to spatial reconstruction`。
- 仅对单个特征平面（纹理、边缘、厚度）、坐标网格、雪花图标做浅层分组。没有整个面板或整页的大分组；文字不与背景绑定。
- 一组 `Reconstructed plane 1` 或 `Target plane 1` 可独立移动。展开组可替换纹理或编辑边缘。若只改变纹理大小，需同步调整同组轮廓；整体缩放该平面小组可保持一致。
- 直线连接线可独立调整端点。它们使用显式端点位置，没有自动吸附到模块；移动模块后需按需要移动对应标签和连接线。
- 胸片、两张真实裁剪图，以及六个特征平面内的密集纹理为位图。其余图形和文字可直接编辑；没有整图贴底，也没有嵌入 SVG。

## 素材和近似说明

胸片来自本地 `code/data/MIMIC_CXR` 中的 PA 样本。报告显示的是相应报告的缩略摘录：`No focal consolidation` 和 `No acute ...`；完整对应原文及来源见 `assets/provenance.json`。

**右侧彩色图是流程示意素材，不是模型实验结果。** `feature_fields()` 使用胸片纹理、解析空间函数和固定的颜色映射生成 target/reconstructed 两组示意特征场，无随机数、无模型推理。两组图共享空间结构并有轻微差异，以说明对齐过程；不能将其作为实际 attention、分割、teacher embedding 或重建效果报告。此说明也写入了 PPT 备注。

`feature_plane()` 给每个纹理施加透视，并保留独立的原生边缘。可以在 `feature_fields()` 中替换为真实模型导出的二维数组，或修改 `build()` 中对应的布局与文字。

Arial 用于正文，Times New Roman 用于数学变量；花体 ℒ 使用本机可用的 Latin Modern Math。相较参考图，图像内容、特征纹理和花体符号为近似，背景采用纯色，其他排版按参考图重建。

预览应以最终 PPT 渲染为准。修改脚本后运行 `--render`；手动修改 PPT 后请直接在 PowerPoint 中重新导出预览，避免运行脚本覆盖手动修改。
