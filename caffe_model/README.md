# Caffe模型手势检测方案

基于OpenCV DNN模块和Caffe模型的手势检测系统，支持22个手部关键点检测。

## 项目概述

本方案使用Caffe深度学习模型实现手势检测功能，能够检测图像或视频中的22个手部关键点，并绘制手部骨架。系统采用模块化设计，代码结构清晰，易于扩展和使用。

## 功能特性

- **22个关键点检测**：详细的手部关键点识别
- **多模式支持**：单张图像、批量图像、实时摄像头处理
- **高性能**：基于OpenCV DNN模块，支持GPU加速
- **模块化设计**：统一的基类设计，代码复用性高
- **可视化结果**：支持关键点标记和骨架绘制
- **可配置参数**：阈值、模型路径等参数可配置

## 目录结构

```
caffe_model/
├── README.md                 # 本文档
├── handPoseBase.py          # 手势识别基类
├── handpose.py              # 单张图像处理
├── handPoseCamera.py        # 摄像头实时处理
├── handPoseImage-many.py    # 批量图像处理
├── hand/                    # 模型文件
│   ├── pose_deploy.prototxt      # 模型配置文件
│   └── pose_iter_102000.caffemodel # 模型权重文件
├── images/                  # 测试图像目录
│   └── *.jpeg
└── results/                 # 输出结果目录
    ├── singleImage/         # 单张图像结果
    ├── multipleImage/       # 批量图像结果
    └── cameraOutput/        # 摄像头输出结果
```

## 快速开始

### 环境要求

- Python 3.6+
- OpenCV 4.0+
- NumPy

安装依赖：
```bash
pip install opencv-python numpy
```

### 模型文件

确保 `hand/` 目录下包含以下模型文件：
- `pose_deploy.prototxt` - 模型配置文件
- `pose_iter_102000.caffemodel` - 模型权重文件

### 使用方法

#### 1. 单张图像处理
```python
# handpose.py
from handPoseBase import HandPoseBase

# 初始化手势识别器
hand_pose = HandPoseBase(threshold=0.1)

# 处理单张图像
image_path = 'images/00000.jpeg'
original_frame, keypoints_image, skeleton_image, points = hand_pose.process_single_image(image_path)

print(f"检测到 {len([p for p in points if p is not None])} 个关键点")
```

运行：
```bash
python handpose.py
```

#### 2. 批量图像处理
```python
# handPoseImage-many.py
from handPoseBase import HandPoseBase

hand_pose = HandPoseBase(threshold=0.1)
# 自动处理images/目录下的所有图像
```

运行：
```bash
python handPoseImage-many.py
```

#### 3. 实时摄像头处理
```python
# handPoseCamera.py
from handPoseCamera import HandPoseCamera

# 创建手势识别器
hand_pose_camera = HandPoseCamera(input_source=0, threshold=0.2)

# 运行手势识别
hand_pose_camera.run()
```

运行：
```bash
python handPoseCamera.py
```

## API 文档

### HandPoseBase 类

手势识别基础类，包含所有公共功能。

#### 初始化
```python
hand_pose = HandPoseBase(
    proto_file="hand/pose_deploy.prototxt",
    weights_file="hand/pose_iter_102000.caffemodel", 
    threshold=0.1
)
```

#### 主要方法

- `process_single_image(image_path, target_height=368)` - 处理单张图像
- `detect_keypoints(image, target_height=368)` - 检测关键点
- `draw_keypoints(image, points, color=(0, 255, 255), radius=8)` - 绘制关键点
- `draw_skeleton(image, points, line_color=(0, 255, 255), line_thickness=2)` - 绘制骨架
- `preprocess_image(image, target_height=368)` - 图像预处理

### HandPoseCamera 类

摄像头手势识别类，继承自HandPoseBase。

```python
camera = HandPoseCamera(input_source=0, threshold=0.2)
camera.run()  # 开始实时检测
```

## 关键点说明

系统检测22个手部关键点，关键点连接关系如下：

- **0**：手腕中心
- **1-4**：大拇指（从根部到指尖）
- **5-8**：食指（从根部到指尖）
- **9-12**：中指（从根部到指尖）
- **13-16**：无名指（从根部到指尖）
- **17-20**：小指（从根部到指尖）

骨架连接关系：
- 0-1-2-3-4：大拇指
- 0-5-6-7-8：食指  
- 0-9-10-11-12：中指
- 0-13-14-15-16：无名指
- 0-17-18-19-20：小指

## 性能优化

### 参数调整

- **阈值调整**：通过修改 `threshold` 参数控制检测灵敏度
  - 较低值：检测更多关键点，但可能包含噪声
  - 较高值：检测更准确，但可能漏检某些关键点

- **目标高度**：调整 `target_height` 影响处理速度和精度
  - 较小值：处理更快，但精度可能降低
  - 较大值：精度更高，但处理速度变慢

### GPU加速

如果系统支持CUDA，可启用GPU加速：
```python
import cv2

# 设置使用GPU
cv2.cuda.setDevice(0)
```

## 输出示例

处理结果保存在 `results/` 目录下：

### 单张图像处理
- `results/singleImage/Keypoints.jpg` - 关键点标记图像
- `results/singleImage/lines.jpg` - 骨架连接图像

### 批量图像处理
- `results/multipleImage/*.jpeg` - 所有处理后的图像

### 摄像头处理
- `results/cameraOutput/output.avi` - 录制的视频文件

## 故障排除

### 常见问题

1. **无法加载模型**
   - 检查 `hand/` 目录下模型文件是否存在
   - 验证文件路径是否正确

2. **摄像头无法打开**
   - 检查摄像头设备号是否正确
   - 验证摄像头是否被其他程序占用

3. **图像处理失败**
   - 检查图像文件路径和格式
   - 确保图像文件可正常读取

4. **性能问题**
   - 调整 `target_height` 参数
   - 降低检测阈值
   - 考虑启用GPU加速

### 错误处理

系统包含完整的错误处理机制，会在控制台输出详细的错误信息。常见的错误信息包括：

- `"无法从摄像头读取视频"` - 摄像头初始化失败
- `"无法读取图像: {image_path}"` - 图像文件读取失败
- `"检测到 {n} 个关键点"` - 成功检测的关键点数量

## 扩展开发

### 添加新功能

1. **自定义关键点绘制**
```python
# 在HandPoseBase类中添加方法
def custom_draw(self, image, points):
    # 实现自定义绘制逻辑
    pass
```

2. **实时手势识别**
```python
# 扩展HandPoseCamera类
class GestureRecognizer(HandPoseCamera):
    def detect_gesture(self, points):
        # 实现手势识别逻辑
        pass
```

3. **性能监控**
```python
import time

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
    
    def log_performance(self):
        elapsed = time.time() - self.start_time
        print(f"处理时间: {elapsed:.3f}秒")
```

## 许可证

本方案采用MIT许可证。

---

**注意**：使用前请确保已正确安装OpenCV和NumPy，并下载了相应的模型文件。