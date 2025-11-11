import cv2
import time
from handPoseBase import HandPoseBase, draw_keypoints

class HandPoseCamera:
    """基于摄像头的手势识别"""
    
    def __init__(self, input_source=0, threshold=0.2):
        """
        初始化摄像头手势识别器
        
        Args:
            input_source: 摄像头输入源 (0为默认摄像头)
            threshold: 关键点检测阈值
        """
        self.input_source = input_source
        self.hand_pose = HandPoseBase(threshold=threshold)
        self.cap = cv2.VideoCapture(self.input_source)
        
        # 获取摄像头参数
        has_frame, frame = self.cap.read()
        if not has_frame:
            raise ValueError("无法从摄像头读取视频")
        
        self.frame_width = frame.shape[1]
        self.frame_height = frame.shape[0]
        
        # 初始化视频写入器
        self.vid_writer = cv2.VideoWriter('results/cameraOutput/output.avi', 
                                         cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'),
                                         15, (self.frame_width, self.frame_height))
    
    def process_frame(self, frame):
        """
        处理单个视频帧
        
        Args:
            frame: 输入视频帧
            
        Returns:
            tuple: (处理后的骨架图像, 关键点图像, 关键点列表)
        """
        start_time = time.time()
        
        # 检测关键点
        output, points = self.hand_pose.detect_keypoints(frame)
        
        # 绘制关键点
        keypoints_image = draw_keypoints(frame, points, radius=6)
        
        # 绘制骨架
        skeleton_image = self.hand_pose.draw_skeleton(frame, points, 
                                                    line_thickness=2, point_radius=5)
        
        processing_time = time.time() - start_time
        print(f"帧处理时间: {processing_time:.3f}秒")
        
        return skeleton_image, keypoints_image, points
    
    def run(self):
        """运行摄像头手势识别"""
        k = 0
        print("开始手势识别，按ESC键退出...")
        
        while True:
            k += 1
            start_time = time.time()
            
            # 读取帧
            has_frame, frame = self.cap.read()
            if not has_frame:
                break
            
            # 处理帧
            skeleton_image, keypoints_image, points = self.process_frame(frame)
            
            # 显示结果
            cv2.imshow('Hand Pose - Skeleton', skeleton_image)
            
            # 检测按键
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                break
            
            # 写入视频
            self.vid_writer.write(skeleton_image)
            
            total_time = time.time() - start_time
            print(f"总帧处理时间: {total_time:.3f}秒")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'vid_writer'):
            self.vid_writer.release()
        cv2.destroyAllWindows()

# 主程序
if __name__ == "__main__":
    try:
        # 创建手势识别器
        hand_pose_camera = HandPoseCamera(input_source=0, threshold=0.2)
        
        # 运行手势识别
        hand_pose_camera.run()
        
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print("程序结束")



