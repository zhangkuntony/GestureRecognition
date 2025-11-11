import cv2
import time
import os
from handPoseBase import HandPoseBase

def process_multiple_images():
    """处理多张图像"""
    # 初始化手势识别器
    hand_pose = HandPoseBase(threshold=0.1)
    
    # 图像文件夹路径
    train_data_path = 'images/'
    output_path = 'results/multipleImage/'
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(train_data_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到 {len(image_files)} 张图像")
    
    for img_file in image_files:
        start_at = time.time()
        
        # 构建完整路径
        image_path = os.path.join(train_data_path, img_file)
        
        try:
            # 处理单张图像
            original_frame, keypoints_image, skeleton_image, points = hand_pose.process_single_image(image_path)
            
            # 检测到的关键点数量
            detected_points = len([p for p in points if p is not None])
            
            # 保存骨架图像
            output_file = os.path.join(output_path, img_file)
            cv2.imwrite(output_file, skeleton_image)
            
            processing_time = time.time() - start_at
            
            print(f"图像 {img_file}: 检测到 {detected_points} 个关键点, 处理时间: {processing_time:.3f}秒")
            
        except Exception as e:
            print(f"处理图像 {img_file} 时发生错误: {e}")

if __name__ == "__main__":
    start_time = time.time()
    
    process_multiple_images()
    
    total_time = time.time() - start_time
    print(f"所有图像处理完成，总耗时: {total_time:.3f}秒")
