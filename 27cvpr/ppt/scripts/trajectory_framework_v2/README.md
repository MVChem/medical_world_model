# Trajectory framework：参考图重建为可编辑 PPT

根据用户提供的三栏示意图重建，沿用项目 `27cvpr/ppt` 的交付结构：

- `../../ppt/trajectory_framework_v2.pptx`：单页可编辑 PPT，页面比例为 3:1。
- `../../ppt/trajectory_framework_v2.png`：由最终 PPT 经 LibreOffice 渲染的 3072 × 1024 预览。
- `build.py`：重建和渲染入口。
- `assets/reference.png`：用户原始透明 PNG 的副本，2172 × 724；仅用于对照，不嵌入 PPT。

## 本次修订

按原图重新描摹第一栏七张卡片的透视四边形、圆角与下缘；改用随距离变浅的玻璃底色，减少后层边界透出的干扰。补回中间卡片下方的淡色深度边缘，并用连续透明描边近似蓝色轨迹周围的白色柔光。前景相框下缘也逐渐淡化。原始 v1 文件保留，便于比较。

## 编辑能力

幻灯片包含 **191 个原生形状和文本框**，其中有 106 个采用三次 Bézier 曲线的自由形状。没有嵌入 SVG 或位图。三个面板分别组成命名分组；在 PowerPoint 中可用“选择窗格”、进入分组或取消组合来选中卡片、图标、环形箭头、时间点、同心圆和标题。

- 六行标题均为原生可编辑文本框，显式采用 Arial Bold。当前渲染环境安装了 Arial，没有使用字体替代。第二栏标题按原图为 **Trajectory Consistency Evolution**。
- 卡片、圆形、山峰、太阳、双向箭头、轨迹和指示箭头均为原生对象。自由形状可以使用“编辑顶点”调整，颜色、线条和渐变可以分别修改。
- 蓝色速度符号采用自由形状描摹，外观可以编辑，但它不是公式或文本字符。
- 玻璃质感、渐变和柔光用原生渐变、透明度及同心椭圆近似。没有逐像素复刻原图的模糊噪点；这些细节可能随 PowerPoint / LibreOffice 版本略有变化。
- 原图背景透明，PPT 采用白色页面背景。页面背景可在 PowerPoint 中修改。

## 重新生成

依赖 Python 3、`python-pptx`、`lxml`、`PyMuPDF`；预览还需要 PATH 中能找到 `libreoffice` 或 `soffice`。建议安装 Arial 字体，以获得相同的文本宽度。

```bash
python -m pip install python-pptx lxml PyMuPDF
python build.py --overwrite
```

脚本以自身路径查找输出位置，不依赖工作目录或机器上的固定 Python 路径。移动文件时保持 `ppt/` 与 `scripts/trajectory_framework_v2/` 的相对关系。首次生成不需要 `--overwrite`。

**`--overwrite` 会根据脚本重建 PPT，覆盖后续在 PowerPoint 中的手动修改。** 如果已经手动编辑，只更新预览：

```bash
python build.py --render-only
```

仅生成 PPT、不渲染预览：

```bash
python build.py --overwrite --no-render
```

## 验证

脚本检查 PPT 中没有 `ppt/media/` 或图片对象，确认三组图形及全部六行标题；渲染时检查单页、3:1 页面比例和标题文本。PNG 来自 PPT 导出的临时 PDF，临时 PDF 不作为交付物保留。

人工对照原图检查了整体布局、卡片层叠、三处轨迹时间点、上下循环方向、两处双向连接、八个向外箭头、四个向内箭头及标题。柔光与半透明卡片属于视觉近似，主要构图和所有文字保持原意。
