import cv2
from handPoseBase import HandPoseBase

# 初始化手势识别器
hand_pose = HandPoseBase(threshold=0.1)

# 处理单张图像
image_path = 'images/00000.jpeg'
original_frame, keypoints_image, skeleton_image, points = hand_pose.process_single_image(image_path)

print(f"检测到 {len([p for p in points if p is not None])} 个关键点")

# 保存图片
result_path = 'results/singleImage/'
cv2.imwrite(result_path + 'Keypoints.jpg', keypoints_image)
cv2.imwrite(result_path + 'lines.jpg', skeleton_image)

print("处理完成，结果已保存到", result_path)