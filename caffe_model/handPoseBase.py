import cv2
import numpy as np

def preprocess_image(image, target_height=368):
    """
    预处理图像，准备输入网络

    Args:
        image: 输入图像
        target_height: 目标高度

    Returns:
        tuple: (预处理后的blob, 图像的宽高比, 原始图像宽度, 原始图像高度)
    """
    frame_height = image.shape[0]
    frame_width = image.shape[1]
    aspect_ratio = frame_width / frame_height

    # 计算目标宽度，保持宽高比
    target_width = int(((aspect_ratio * target_height) * 8) // 8)

    # 创建输入blob
    inp_blob = cv2.dnn.blobFromImage(
        image, 1.0 / 255, (target_width, target_height),
        (0, 0, 0), swapRB=False, crop=False
    )

    return inp_blob, aspect_ratio, frame_width, frame_height


def draw_keypoints(image, points, color=(0, 255, 255), radius=8, thickness=-1):
    """
    在图像上绘制关键点

    Args:
        image: 输入图像
        points: 关键点列表
        color: 关键点颜色
        radius: 关键点半径
        thickness: 线条粗细

    Returns:
        numpy.ndarray: 绘制了关键点的图像
    """
    image_copy = np.copy(image)

    for i, point in enumerate(points):
        if point:
            cv2.circle(image_copy, point, radius, color, thickness, lineType=cv2.FILLED)
            cv2.putText(image_copy, str(i), point, cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    return image_copy


class HandPoseBase:
    """手势识别基础类，包含公共的配置和功能"""
    
    def __init__(self, proto_file="hand/pose_deploy.prototxt", 
                 weights_file="hand/pose_iter_102000.caffemodel",
                 threshold=0.1):
        """
        初始化手势识别器
        
        Args:
            proto_file: 模型配置文件路径
            weights_file: 模型权重文件路径
            threshold: 关键点检测阈值
        """
        self.proto_file = proto_file
        self.weights_file = weights_file
        self.threshold = threshold
        
        # 手势关键点配置
        self.n_points = 22
        self.pose_pairs = [
            [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
            [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
            [0,17],[17,18],[18,19],[19,20]
        ]
        
        # 加载模型
        self.net = cv2.dnn.readNetFromCaffe(self.proto_file, self.weights_file)

    def detect_keypoints(self, image, target_height=368):
        """
        检测手势关键点
        
        Args:
            image: 输入图像
            target_height: 目标高度
            
        Returns:
            tuple: (网络输出, 关键点列表)
        """
        # 预处理图像
        inp_blob, aspect_ratio, frame_width, frame_height = preprocess_image(image, target_height)
        
        # 网络前向传播
        self.net.setInput(inp_blob)
        output = self.net.forward()
        
        # 检测关键点
        points = []
        for i in range(self.n_points):
            prob_map = output[0, i, :, :]
            prob_map = cv2.resize(prob_map, (frame_width, frame_height))
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(prob_map)
            
            if max_val > self.threshold:
                points.append((int(max_loc[0]), int(max_loc[1])))
            else:
                points.append(None)
        
        return output, points

    def draw_skeleton(self, image, points, line_color=(0, 255, 255), line_thickness=2,
                      point_color=(0, 0, 255), point_radius=8):
        """
        在图像上绘制骨架连接线
        
        Args:
            image: 输入图像
            points: 关键点列表
            line_color: 连接线颜色
            line_thickness: 连接线粗细
            point_color: 关键点颜色
            point_radius: 关键点半径
            
        Returns:
            numpy.ndarray: 绘制了骨架的图像
        """
        image_copy = np.copy(image)
        
        for pair in self.pose_pairs:
            part_a = pair[0]
            part_b = pair[1]
            
            if points[part_a] and points[part_b]:
                cv2.line(image_copy, points[part_a], points[part_b], 
                        line_color, line_thickness, lineType=cv2.LINE_AA)
                cv2.circle(image_copy, points[part_a], point_radius, 
                          point_color, thickness=-1, lineType=cv2.FILLED)
                cv2.circle(image_copy, points[part_b], point_radius, 
                          point_color, thickness=-1, lineType=cv2.FILLED)
        
        return image_copy
    
    def process_single_image(self, image_path, target_height=368):
        """
        处理单张图像
        
        Args:
            image_path: 图像文件路径
            target_height: 目标高度
            
        Returns:
            tuple: (原始图像, 关键点图像, 骨架图像, 关键点列表)
        """
        # 读取图像
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 检测关键点
        output, points = self.detect_keypoints(frame, target_height)
        
        # 绘制关键点
        keypoints_image = draw_keypoints(frame, points)
        
        # 绘制骨架
        skeleton_image = self.draw_skeleton(frame, points)
        
        return frame, keypoints_image, skeleton_image, points