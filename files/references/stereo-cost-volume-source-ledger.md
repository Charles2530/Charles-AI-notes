# 双目匹配与 Cost Volume 来源台账

这份台账服务于 [双目匹配与 Cost Volume](../embodied-ai/stereo-matching-and-cost-volume.md)。正文的写法刻意参考高质量技术博客和中文公众号常用的讲解节奏：先从一个可感知的问题进入，再把几何、张量、失败区域和系统部署逐层展开。事实、公式、论文贡献和 benchmark 口径仍以官方文档、论文和评测站为准。

## 使用方式

第一类来源是双目几何和工程教程。OpenCV depth map 教程、LearnOpenCV epipolar geometry 文章和 Wikimedia stereo camera 图，支撑 baseline、focal length、disparity、depth、rectification 和对应点搜索的基本讲法。

第二类来源是深度 stereo 论文。GC-Net、PSMNet、Cascade Cost Volume、RAFT-Stereo 和 FoundationStereo 支撑 cost volume 构造、3D regularization、多尺度上下文、coarse-to-fine volume、recurrent refinement 和 zero-shot stereo 的演化线索。

第三类来源是系统案例。Fast-FoundationStereo 提供 feature backbone、cost filtering、disparity refinement 三段式 pipeline 的真实图，帮助正文把“强泛化”和“实时化”拆成两张账。

第四类来源是评测基准。Middlebury、ETH3D 和 KITTI 支撑 EPE、bad pixel、D1、遮挡、边界、弱纹理、真实采集场景和自动驾驶场景的差异。

第五类来源是讲法参考。中文博客、知乎专栏和微信公众号类文章只用于学习节奏：先问“为什么双目不是单目深度”，再引入 cost volume，最后落到失败区域和部署判断。稳定性不足、版权不清或缺少原始出处的图和结论不进入正文。

## 写作边界

正文不把 stereo 等同于普通单目深度。Stereo 的优势来自两视图几何和对应点搜索；monocular foundation prior 可以增强泛化，但不能替代 baseline、rectification 和 disparity 这条物理链。

正文不把 cost volume 写成抽象张量名。它是每个像素在多个候选 disparity 下的匹配证据表，计算和显存成本来自额外的 \(D\) 维。

正文不把 benchmark 排名当作唯一结论。Stereo 需要分桶看弱纹理、重复纹理、遮挡、透明/反光、边界、远距离和真实相机同步标定问题。

正文不把 Fast-FoundationStereo 只当加速论文。它的图也适合说明 stereo pipeline 本身：feature backbone、cost volume / cost filtering、initial disparity、refinement 和 runtime 热路径。

正文没有使用 image2 或其他生成式工具作图。所有图片都来自仓库已有资产，并在正文标出原始来源。

## 图片使用

| 图片 | 原始来源 | 本站使用方式 |
| --- | --- | --- |
| `stereo-camera-model-wikimedia.jpg` | [Wikimedia Stereo Camera Model](https://commons.wikimedia.org/wiki/File:Stereo-camera-model.jpg) | 说明 \(z=fB/d\)、baseline、focal length 和左右投影点。 |
| `figure-3-framework-padded.png` | [Fast-FoundationStereo](https://arxiv.org/abs/2512.11130) Figure 3 | 说明 feature backbone、cost filtering、disparity refinement 三段式 pipeline。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [OpenCV: Depth Map from Stereo Images](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html) | 官方教程 | 支撑 disparity、baseline、focal length 与 depth 公式，以及 StereoBM 参数背后的工程问题。 | 几何 / 工程 |
| [LearnOpenCV: Epipolar Geometry and Stereo Vision](https://learnopencv.com/introduction-to-epipolar-geometry-and-stereo-vision/) | 技术博客 | 借鉴从人眼直觉到 epipolar constraint、point correspondence 和 depth 的讲解方式。 | 几何入门 |
| [Wikimedia Stereo Camera Model](https://commons.wikimedia.org/wiki/File:Stereo-camera-model.jpg) | 图源 | 使用 stereo camera 几何图解释 \(z=fB/d\)。 | 图片来源 |
| [GC-Net](https://arxiv.org/abs/1703.04309) | 论文 | 支撑 deep feature cost volume、3D convolution regularization 和 differentiable disparity regression 的历史位置。 | Cost volume |
| [PSMNet](https://arxiv.org/abs/1803.08669) | 论文 | 支撑 spatial pyramid pooling、3D CNN 和 stacked hourglass regularization 的讲法。 | Cost filtering |
| [Cascade Cost Volume](https://arxiv.org/abs/1912.06378) | 论文 | 支撑 coarse-to-fine / cascade volume 降低高分辨率 cost volume 成本的解释。 | 高效 cost volume |
| [RAFT-Stereo](https://arxiv.org/abs/2109.07547) | 论文 | 支撑 recurrent update、multilevel cost / correlation 和 iterative disparity refinement 的讲法。 | Refinement |
| [FoundationStereo](https://arxiv.org/abs/2501.09898) | 论文 | 支撑 zero-shot stereo、monocular foundation prior、large synthetic data 和 long-range cost filtering。 | Foundation stereo |
| [Fast-FoundationStereo](https://arxiv.org/abs/2512.11130) | 论文 | 使用 Figure 3 说明 stereo pipeline 和三段热路径；也关联实时化策略。 | 系统案例 |
| [Middlebury Stereo Evaluation v3](https://vision.middlebury.edu/stereo/eval3/MiddEval3-newFeatures.html) | Benchmark 文档 | 支撑 bad pixel、average absolute error、RMS、quantile 等指标解释。 | 评测 |
| [ETH3D two-view benchmark](https://www.eth3d.net/low_res_two_view.php?coverage=0&mask=all&metric=bad-1-0&set=test&sortby=d52) | Benchmark 文档 | 支撑真实采集 stereo benchmark 和 smaller-is-better 指标口径。 | 评测 |
| [KITTI Stereo Evaluation](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) | Benchmark 文档 | 支撑自动驾驶场景、D1 / outlier 类指标和 benchmark 口径。 | 评测 |
