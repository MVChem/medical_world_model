# fig2_v1

根据用户参考图重建的一页可编辑 PPT。原图按 `fig2_v1.png` 保存于
`../../source/`，PPT、最终 PPT 渲染预览和 PDF 位于 `../../ppt/`。
页面保留参考图的 2061 × 763 比例；`fig2_v1_paper.pdf` 仅裁去外围留白，
供文章引用。

依赖：Python 3、python-pptx、Pillow、PyMuPDF，以及系统 LibreOffice。
字体为 Comic Sans MS 和 Times New Roman；当前环境已安装。参考图未提供
字体元数据，手写文字用 Comic Sans MS 近似，原图细微纹理以纯色填充近似。

在本目录运行：

```bash
python build.py
```

脚本相对于自身定位素材与输出；整个 `ppt/` 文件夹移动后仍可运行。
默认重新生成 PPT，会覆盖同名文件。手动编辑 PPT 后，仅更新导出：

```bash
python build.py --render-only
```

在 `27cvpr/` 下更新论文图并编译：

```bash
cp ppt/ppt/fig2_v1_paper.pdf imgs/fig2_v1.pdf
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

所有文字、模块、24 个输入 token、8 个输出 slot、箭头和汇合括号均为
PowerPoint 原生对象；无嵌入 SVG 或位图，原图只作为外部参考保存。
文字为独立文本框，`task` 为原生下标文字。通过“选择窗格”中的 `Label:`、
`Equation:`、`Panel:`、`Module:`、`Token:`、`Slot:`、`Arrow:` 和 `Bracket:`
名称定位对象。每条折线箭头为单个可编辑顶点的开放自由曲线，直箭头为
原生连接线，箭头端点随线移动。对象没有整页分组；模块移动时按需同时选中
背景、标签及相应连线，它们不自动吸附。预览来自最终 PPT 的 LibreOffice
PDF 导出，论文版保持矢量图形和文字。
