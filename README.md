# GestureRecognition - 手势检测系统

多模型手势检测项目，支持基于Caffe和MediaPipe两种技术方案的手势识别。

## 项目概述

本项目提供了两种不同的手势检测技术方案：

1. **Caffe模型方案**：基于OpenCV DNN模块和Caffe模型，实现22个手部关键点检测
2. **MediaPipe方案**：基于Google MediaPipe框架，提供更先进的手势识别功能

每种方案都支持单张图像、批量图像和实时摄像头处理。

## 项目结构

```
GestureRecognition/
├── README.md                 # 项目主文档
├── caffe_model/              # Caffe模型方案
│   ├── README.md             # Caffe模型文档
│   ├── handPoseBase.py       # 手势识别基类
│   ├── handpose.py           # 单张图像处理
│   ├── handPoseCamera.py     # 摄像头实时处理
│   ├── handPoseImage-many.py  # 批量图像处理
│   ├── hand/                  # Caffe模型文件
│   │   ├── pose_deploy.prototxt      # 模型配置文件
│   │   └── pose_iter_102000.caffemodel # 模型权重文件
│   ├── images/               # 测试图像
│   └── results/              # 输出结果目录
└── mediapipe_model/          # MediaPipe方案
    ├── README.md             # MediaPipe模型文档
    └── README_EN.pdf         # 英文文档
```

## 技术方案对比

| 特性 | Caffe模型方案 | MediaPipe方案 |
|------|-------------|--------------|
| 技术基础 | OpenCV DNN + Caffe | Google MediaPipe |
| 关键点数量 | 22个 | 21个 |
| 性能 | 中等 | 高性能 |
| 准确性 | 良好 | 优秀 |
| 实时性 | 良好 | 优秀 |
| 模型大小 | 140MB | 轻量级 |
| 易用性 | 中等 | 简单 |

## 快速开始

### Caffe模型方案

#### 环境要求
```bash
pip install opencv-python numpy
```

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

### MediaPipe方案

#### 环境要求
```bash
pip install mediapipe opencv-python numpy
```

#### 使用方法
（请参考mediapipe_model目录下的文档）

## 详细文档

- **Caffe模型方案**：请查看 [caffe_model/README.md](./caffe_model/README.md)
- **MediaPipe方案**：请查看 [mediapipe_model/README.md](./mediapipe_model/README.md)

## 功能特性

### Caffe模型方案
- 22个手部关键点检测
- 骨架连接线绘制
- 多模式处理支持
- 可配置的检测阈值
- 实时性能监控

### MediaPipe方案
- 21个手部关键点检测
- 高性能实时处理
- 先进的机器学习算法
- 简单易用的API

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

**注意**：Caffe模型需要下载约140MB的模型文件，MediaPipe模型会自动下载所需资源。