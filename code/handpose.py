import cv2
import numpy as np

# 1. 模型加载
protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 2. 图片预处理
frame = cv2.imread('images/00000.jpeg')
frame_copy = np.copy(frame)
frame_height = frame.shape[0]
frame_width = frame.shape[1]
aspect_ratio = frame_width / frame_height
in_height = 368
in_width = int(aspect_ratio * in_height)
inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height), (0, 0, 0), swapRB=False, crop=False)

# 3. 模型前向推理
net.setInput(inp_blob)
output = net.forward()
print(output.shape)

# 4. 寻找关键点
points = []
nPoints = 22
threshold = 0.1

for i in range(nPoints):
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (frame_width, frame_height))
    min_val, max_val, min_index, max_index = cv2.minMaxLoc(probMap)

    # 5. 绘制关键点和编号
    if max_val > threshold:
        cv2.circle(frame_copy, (max_index[0], max_index[1]), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame_copy, "{}".format(i), (max_index[0], max_index[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        points.append((max_index[0], max_index[1]))
    else:
        print(max_val, i)
        points.append(None)

# 6. 绘制关键点条线
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

result_path = 'results/singleImage/'

# 保存图片
cv2.imwrite(result_path + 'Keypoints.jpg', frame_copy)
cv2.imwrite(result_path + 'lines.jpg', frame)