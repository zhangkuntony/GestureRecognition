# 基于 MediaPipe 的手势识别系统

## 项目概述

这是一个基于 Google MediaPipe 框架的实时手势识别系统，能够检测和识别静态手势以及动态手势轨迹。系统使用深度学习模型对摄像头捕获的手部动作进行实时分析，支持多种手势分类。

## 项目结构

```
mediapipe_model/
├── app.py                           # 主应用程序入口
├── classification_train/            # 模型训练脚本目录
│   ├── keypoint_classification.py      # 关键点分类器训练脚本
│   └── point_history_classification.py # 轨迹历史分类器训练脚本
├── model/                           # 模型文件目录
│   ├── __init__.py                 # 模型包初始化
│   ├── keypoint_classifier/        # 关键点分类器
│   │   ├── keypoint_classifier.py      # 关键点分类器实现
│   │   ├── keypoint_classifier.keras   # Keras 模型文件
│   │   ├── keypoint_classifier.tflite  # TensorFlow Lite 模型
│   │   └── keypoint_classifier_label.csv # 分类标签文件
│   └── point_history_classifier/    # 轨迹历史分类器
│       ├── point_history_classifier.py  # 轨迹历史分类器实现
│       ├── point_history_classifier.keras
│       ├── point_history_classifier.tflite
│       └── point_history_classifier_label.csv
└── utils/                           # 工具类目录
    ├── __init__.py                 # 工具包初始化
    └── cvfpscalc.py                # FPS 计算工具类
```

## 功能特性

### 1. 静态手势识别
- 实时检测手部21个关键点
- 支持多种静态手势分类（如OK手势、停止手势等）
- 使用深度学习模型进行手势识别

### 2. 动态手势轨迹识别
- 跟踪手指移动轨迹
- 识别动态手势（如挥手、画圈等）
- 基于历史轨迹数据的模式识别

### 3. 实时性能
- 实时FPS计算和显示
- 滑动窗口平均法提高帧率稳定性
- 高效的 TensorFlow Lite 模型推理

### 4. 可视化界面
- 实时显示手部关键点和连接线
- 显示手势识别结果
- 轨迹路径可视化
- 边框和识别信息显示

## 核心模块

### 1. 主应用程序 (app.py)
- 摄像头视频流捕获
- MediaPipe 手部检测
- 手势分类和轨迹识别
- 可视化界面渲染

### 2. 关键点分类器 (KeyPointClassifier)
- 基于手部21个关键点的静态手势识别
- 使用 TensorFlow Lite 进行模型推理
- 支持多种手势分类

### 3. 轨迹历史分类器 (PointHistoryClassifier)
- 基于手指移动轨迹的动态手势识别
- 处理连续16帧的历史轨迹数据
- 深度学习模型进行轨迹模式识别

### 4. 训练模块 (classification_train/)
- 关键点分类器训练脚本
- 轨迹历史分类器训练脚本
- 模型评估和混淆矩阵可视化

## 使用方法

### 1. 运行应用程序
```bash
python app.py
```

### 2. 控制指令
- **数字键 0-9**: 切换手势分类模式
- **N 键**: 进入关键点记录模式
- **K 键**: 进入轨迹历史记录模式
- **H 键**: 切换手势识别模式
- **ESC 键**: 退出程序

### 3. 模型训练
```bash
# 训练关键点分类器
python classification_train/keypoint_classification.py

# 训练轨迹历史分类器
python classification_train/point_history_classification.py
```

## 技术栈

- **MediaPipe**: Google 的实时感知框架，用于手部关键点检测
- **TensorFlow/Keras**: 深度学习模型训练和推理
- **TensorFlow Lite**: 轻量级模型部署和移动端优化
- **OpenCV**: 图像处理和视频流处理
- **Scikit-learn**: 机器学习工具和评估指标

## 依赖环境

### 核心依赖
- Python 3.8+
- OpenCV-Python
- MediaPipe
- TensorFlow 2.x
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### 安装依赖
```bash
pip install -r requirements.txt
```

## 模型说明

### 关键点分类器
- 输入: 21个手部关键点的归一化坐标（42维特征）
- 输出: 静态手势分类结果
- 模型结构: 多层感知机（MLP）
- 激活函数: ReLU + Softmax

### 轨迹历史分类器
- 输入: 16帧历史轨迹点（32维特征）
- 输出: 动态手势分类结果
- 模型结构: 全连接神经网络
- 支持 LSTM 模式（可选）

## 性能优化

- 使用 TensorFlow Lite 进行模型推理，提高运行效率
- 滑动窗口FPS计算，提供稳定的帧率显示
- 优化的图像处理流程，减少计算开销
- 支持多线程模型推理

## 扩展性

项目具有良好的模块化设计，便于：
- 添加新的手势分类
- 扩展动态手势识别功能
- 集成更多计算机视觉算法
- 部署到移动设备或嵌入式系统

## 注意事项

- 确保摄像头设备正常工作
- 保持良好的光照条件以获得最佳识别效果
- 训练模型时确保有足够的手势数据样本
- 在性能较弱的设备上可能需要调整视频分辨率

## 许可证

本项目基于MIT许可证开源。

---

*该项目专注于实时手势识别技术，适用于人机交互、虚拟现实、智能家居等多种应用场景。*