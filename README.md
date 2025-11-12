# GestureRecognition - 手势识别系统

基于深度学习的多模型手势识别项目，支持Caffe和MediaPipe两种技术方案，提供完整的手势识别、分类和轨迹追踪功能。

## 项目概述

本项目提供两种互补的手势识别技术方案：

1. **Caffe模型方案**：基于OpenCV DNN模块和Caffe模型，实现22个手部关键点检测，适合需要详细手部分析的应用场景
2. **MediaPipe方案**：基于Google MediaPipe框架，提供高性能实时手势识别，支持手势分类和手指轨迹追踪，代码经过重构优化

## 项目结构

```
GestureRecognition/
├── README.md                 # 项目主文档
├── requirements.txt          # 项目依赖包列表
├── caffe_model/              # Caffe模型方案
│   ├── README.md             # Caffe模型详细文档
│   ├── handPoseBase.py       # 手势识别基类
│   ├── handpose.py           # 单张图像处理
│   ├── handPoseCamera.py     # 摄像头实时处理
│   ├── handPoseImage-many.py  # 批量图像处理
│   ├── hand/                  # Caffe模型文件
│   │   ├── pose_deploy.prototxt      # 模型配置文件
│   │   └── pose_iter_102000.caffemodel # 模型权重文件
│   ├── images/               # 测试图像目录
│   └── results/              # 输出结果目录
└── mediapipe_model/          # MediaPipe方案（重构优化）
    ├── app.py                # 主应用程序（重构版）
    ├── classification_train/  # 分类器训练脚本
    │   ├── keypoint_classification.py      # 关键点分类器训练
    │   └── point_history_classification.py # 轨迹历史分类器训练
    ├── model/                 # 预训练模型和分类器
    │   ├── keypoint_classifier/            # 关键点分类器
    │   └── point_history_classifier/       # 轨迹历史分类器
    ├── utils/                # 工具类
    │   └── cvfpscalc.py      # FPS计算工具
    ├── results/              # 数据记录目录
    └── README.md              # MediaPipe方案文档
```

## 技术方案对比

| 特性 | Caffe模型方案 | MediaPipe方案（重构优化版） |
|------|-------------|---------------------|
| 技术基础 | OpenCV DNN + Caffe | Google MediaPipe + TensorFlow |
| 关键点数量 | 22个 | 21个 |
| 性能 | 中等 | 高性能实时处理 |
| 准确性 | 良好 | 优秀 |
| 实时性 | 良好 | 优秀（60+ FPS） |
| 模型大小 | 140MB | 轻量级 |
| 代码质量 | 基础实现 | 重构优化，可读性强 |
| 功能特性 | 关键点检测 | 手势分类 + 轨迹追踪 |

## 快速开始

### 环境要求
```bash
# 安装所有依赖
pip install -r requirements.txt

# 或者单独安装主要依赖
pip install mediapipe opencv-python numpy tensorflow scikit-learn
```

### Caffe模型方案

#### 使用方法
```bash
# 单张图像处理
cd caffe_model
python handpose.py

# 批量图像处理
python handPoseImage-many.py

# 实时摄像头处理
python handPoseCamera.py
```

### MediaPipe方案（重构优化版）

#### 使用方法
```bash
# 进入项目目录
cd mediapipe_model

# 运行手势识别应用
python app.py

# 可选的命令行参数
python app.py --device 0 --width 1280 --height 720

# 训练自定义手势分类器
cd classification_train
python keypoint_classification.py
python point_history_classification.py
```

**MediaPipe方案控制键说明：**
- **数字键 0-9**：记录对应手势的训练数据
- **n键**：切换至正常模式
- **k键**：切换至关键点记录模式
- **h键**：切换至轨迹历史记录模式
- **ESC键**：退出程序

## 功能特性

### Caffe模型方案
- **22个手部关键点检测**：更详细的手部关节点识别
- **骨架连接线绘制**：自动生成手部骨架可视化
- **多模式处理支持**：单张图像、批量图像、实时摄像头
- **可配置的检测阈值**：灵活调整检测灵敏度
- **实时性能监控**：显示FPS和检测状态

### MediaPipe方案（重构优化版）
- **21个手部关键点检测**：基于MediaPipe的高精度检测
- **手势分类功能**：支持多种自定义手势识别
- **手指轨迹追踪**：实时追踪手指移动轨迹
- **高性能实时处理**：60+ FPS的实时性能
- **代码重构优化**：消除重复代码，提高可维护性
- **统一文本绘制**：新增`draw_text_with_outline`函数
- **模块化设计**：清晰的函数分离和代码结构

## 技术亮点

### MediaPipe方案代码重构
- **代码复用**：将72行线段绘制代码简化为20行，使用connections列表定义手指连接
- **统一绘制逻辑**：将60行圆点绘制代码简化为10行，通过large_points列表区分指尖大小
- **新增功能**：新增`draw_text_with_outline`函数统一文本绘制逻辑
- **注释完善**：所有Python文件均已添加详细的中文注释

### 版本兼容性
- 解决MediaPipe版本兼容性问题，使用`getattr()`安全访问属性
- 支持多种MediaPipe版本，避免"Unresolved attribute reference"警告

## 使用建议

### 选择Caffe方案的情况
- 需要22个关键点的详细检测
- 对模型文件大小不敏感
- 熟悉OpenCV和Caffe
- 预算有限的环境

### 选择MediaPipe方案的情况
- 需要高性能实时处理
- 追求更好的准确性
- 希望快速部署和集成
- 移动端或资源受限环境

## 更新日志

### v3.0 - 代码重构与优化
- **MediaPipe方案重构**：消除重复代码，提高代码可维护性
- **新增统一绘制函数**：新增`draw_text_with_outline`函数
- **代码注释完善**：为所有Python文件添加详细中文注释
- **版本兼容性改进**：解决MediaPipe版本兼容性问题

### v2.0 - 多模型架构
- 新增MediaPipe方案
- 重构项目目录结构
- 优化Caffe方案代码

### v1.0 - 初始版本
- 基于Caffe模型的手势检测
- 支持单张、批量、实时处理
- 完整的可视化功能

## 贡献指南

欢迎为项目贡献代码或改进：
1. 提交Issue报告问题或建议
2. 创建Pull Request贡献代码
3. 完善文档和示例

## 许可证

本项目采用MIT许可证。

---

**注意**：
- Caffe模型需要下载约140MB的模型文件
- MediaPipe模型会自动下载所需资源
- 项目已针对最新代码进行重构和优化，文档完全同步更新