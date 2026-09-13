# 历史任务原文（原根目录 TODO.md）

本文件保存各日期的需求记录，论文资源路径已统一为当前目录，过时的投稿事项已移除；不作为当前待办或执行授权入口。[TODO.md](../TODO.md) 已停用，仅保留历史快照；后续计划和决定见 [研究笔记](../README.md)，项目结构见 [README.md](../../README.md)。

# 0811 construct mimic example
I have mimic data under /home/data1/data/MIMIC

See /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr, now i want a example to see the future state, i need to construct some data like this:

current state -> future state

current state can contain text, medical images
future state can contain this as well

code under /home/data2/chk/workspace/2026/08/04/medical_world_model/code/MIMIC_example

# 0812 vla-jepa to medical world model (plan mode)
Can you try to do the training? We would write a plan file started with date like xxxx_<plan_name>.md under folder /home/data2/chk/workspace/2026/08/04/medical_world_model/plans. We would do this in a small scale first, we would follow the method describe in https://arxiv.org/pdf/2602.10098

We would do those things:
1. we would follow the method described in https://arxiv.org/pdf/2602.10098
2. we would use a pretrained vjepa2 architecture, nature images one is ok, since vjepa2 input is multiple images, if we only have one, stack them (This would be a baseline now)
3.  we would use this vlm as the bottleneck models--Qwen--Qwen3.5-4B
4. we can use all the gpu cards
5. i have not define the downstream tasks now, what might be suitable? we would create a plan for this
6. what about the predictor? or we would just use the vla jepa style
7. We would use MIMIC dataset, see /home/data2/chk/workspace/2026/08/04/medical_world_model/code/MIMIC_example

# 0813 draw data pipeline (useless 08141501)

I want you to draw a prompt for image2 to generate images, the prompt should be short and precise and be a publication ready prompt, we want nice layout, this would be a appendix image i think

Bascially, i want to show the data we use, see project /home/data2/chk/workspace/2026/08/04/medical_world_model/code/MIMIC_example and /home/data2/chk/workspace/2026/08/04/medical_world_model/code/mimic_vla_jepa, we may also copy some asserts from /home/data1/data/MIMIC, prompt you written would be in /home/data2/chk/workspace/2026/08/04/medical_world_model/plans, and asserts you copied should also be there in proper place or folder, etc.

# 0814 choose good data

See /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr and /home/data2/chk/workspace/2026/08/04/medical_world_model/code/mimic_vla_jepa, i want make some data into /home/data2/chk/workspace/2026/08/04/medical_world_model/research_notes/assets/mimic_data_appendix (you can delete the old one)

Should contain of those examples:

1. MIMIC PNG with current state (image+text) and future state (image+text) with time interval, example * 3
2. like 1, but time iterval can be several, this also be 3

You should wirte short and clean readme say how they pair

# 0817 how is the training going on?

How is the training going on?
/home/data2/chk/workspace/2026/08/04/medical_world_model/code/mimic_vla_jepa

# 0818 clone Clin-JEPA

clone https://github.com/Kamaleswaran-Lab/Clin-JEPA to /home/data2/chk/workspace/2026/08/04/medical_world_model/code
and download paper to /home/data2/chk/workspace/2026/08/04/medical_world_model/related_works

try to run that repo (very demo run is ok)

# 0818 insert the method figure

We would convert A to pdf and then insert that to proper place in the main content of the B

A is /home/data2/chk/workspace/2026/08/04/medical_world_model/imgs/ppt/demo_fig1_v1.pptx

and B is /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr

# 0824 rewrite the paper

https://chatgpt.com/share/6a8d8b71-f530-83e8-b967-7b47caf67151

看一下这个链接

然后重写一下我们的/home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr

用ccf skills

然后，可以夸大一下我们的claim，我们要把novelty sell出去

主要两点

1. 很好的future state prediction
2. 很好的genral embeddeing with the power combined with vlm and jepa

这非常重要

然后实验的表先帮我排版好，可以先空着 我要metric这样

现在的草稿你可以随便修改，随便删除，很多东西根本没用

appendix基本也能都删了 全是废话

# 0827 find mimic example

我有mimic的数据集/home/data2/chk/data/MIMIC

然后我记得MIMIC_CXR和mimic-iv-3.1似乎是有关联的

/home/data2/chk/workspace/2026/08/04/medical_world_model/code/MIMIC_example有写这样的例子吗？

找几个有代表性的联系一下这几个数据库，我估计我之后要找一下这个数据然后画在论文的appendix里面

可以看一下/home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr，不过这也是个草稿

如果没有这个例子，你可以随便修改或者彻底重构/home/data2/chk/workspace/2026/08/04/medical_world_model/code/MIMIC_example搞一个例子

/home/data2/chk/workspace/2026/.venv可以用这个python

# 0827 draw figures

We would draw image A, and the ppt path is B (which is the target file we want), script under C (if we use python to draw things, this is optional)

We would use svg as much as possible when drawing instead of png/jpg, some asserts can be found via web if we need jpg/png/video. Python is under /home/data2/chk/workspace/2026/.venv. C should with simple and clean md files, keep as much as the same layout, font, style and color as the original one

We want things editable in ppt and use native ppt components to draw, if ppt is hard to draw we would use python

Code you write should be simple and short, do not overdesign

A: /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/source/<name>.<extension>
B: /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/ppt/<name>.pptx
C: /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/scripts/<name>

name is Codex Image Aug 29, 2026, 06_33_48 AM.png






```txt
我还有个idea，其实vlajepa应该是

它可以对qwen模型学不同的leanable token，我觉得medical任务也可以这样做

有一些future token，seg token，还有比如disease_token 病灶_token

有点类似于moe

然后这些token聚合就可以是很好的embedding


我们可以做细粒度的fine grain这样我觉得可能是有用的
```

# 0828 gpt image2 prompt

我需要一个gpt image2的prompt，之后我自己喂给gpt image2去生成图片，要英文版本的

读一下/home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr

我们主要画这个图的fig1展示pipeline，包括网络架构，token使用之类

要好看有设计感，nature系列配色，好好结合文章写prompt，不要过多文字块（因为要放在论文正文里面，后续我会监督修改，prompt可以细致一点）


写完的prompt放在/home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/prompt下

# 0829 downstream tasks
/home/data2/chk/workspace/2026/08/04/medical_world_model/code/vjepa2 我们要看一下我们训练的jepa效果怎么样

我们看两个任务 一个是ixi的age regression

一个是图像重建任务

寻一个regressor和一个decorder我们看一下

在/home/data2/chk/workspace/2026/08/04/medical_world_model/code/vjepa2新建一个folder我们专门做下游任务的test

# 0830 mimic image

请参考/home/data2/chk/workspace/2026/08/04/medical_world_model/code/MIMIC_example，里面有我们想要展示的数据

我们要prompt和一些资源画图，你现在写一个英文的prompt放在/home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/prompt，一些mimic的资源放在/home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/source（新建一个folder）

我们要一个image2的prompt来展示我们的数据这样，要好看，不过对于prompt本身不要做太多限制（让image2自己去发挥），充分展示我们的数据即可，这个image要放在appendix里面估计

我后面会把prompt和图片一起给image2

# 0830 image2 draw

We would draw image A, and the path is B, script under C (if we use python to draw things), we would use svg as much as possible when drawing (we would keep this high resolution for paper submission), some asserts can be found via web if we need jpg/png. Python is under /home/data2/chk/workspace/2026/.venv, C should with simple and clean md files, keep as much as the same layout, font, style and color as the original one. We would use native ppt as much as possible, if very complex components, we would use python. (What the python draw should be tight, without much white space)

code you write should be simple and short, do not overdesign

A: /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/source/<name>.png/jpg
B: /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/ppt/<name>.pptx
C: /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/scripts/<name>

where name is fig1_v5


# 0831 mimic image v2

请参考/home/data2/chk/workspace/2026/08/04/medical_world_model/code/MIMIC_example，里面有我们想要展示的数据

我们要prompt和一些资源画图，你现在写一个英文的prompt放在/home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/prompt，一些mimic的资源放在/home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/source（新建一个folder）

我们要一个image2的prompt来展示我们的数据这样，要好看，不过对于prompt本身不要做太多限制（让image2自己去发挥），充分展示我们的数据即可，这个image要放在appendix里面估计

我后面会把prompt和图片一起给image2

# 0831 mimic example


/home/data2/chk/data/MIMIC
/home/data2/chk/workspace/2026/08/04/medical_world_model/code/MIMIC_example


# 0903 appendix fig1 v1

/home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/source/appendix_fig1_v1.png with sources under /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/source/mimic_appendix_cases (data from /home/data2/chk/workspace/2026/08/04/medical_world_model/code/MIMIC_example)

I find this image messy, can you make this more organized?

I need a image2 prompt for that under /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/prompt named appendix_fig1_v2, and try to make this figure nice

# 0903 fig1 prompt

Please read /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/main.pdf and /home/data2/chk/workspace/2026/08/04/medical_world_model/related_works/26-Arxiv-VLA-JEPA.pdf carefully

Now we need a image2 prompt to generate the fig1 for /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/main.pdf, do not make too much restriction (let image2 to manage the style i think, we would only say what we want to draw)

No need to draw too fucking much details

Draw the main idea and training pipeline and data flow i think would be ok

Make the style maybe like current fig1

We would write prompt under /home/data2/chk/workspace/2026/08/04/medical_world_model/27cvpr/ppt/prompt called fig1_v1.txt please

# 0904 chest seg

https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation.git clone this to /home/data2/chk/workspace/2026/08/04/medical_world_model/code and then can you write a ipynb to do the segmentation?

Use a mimic image (/home/data1/data/MIMIC)
