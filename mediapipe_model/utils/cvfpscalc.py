# FPS（帧率）计算工具类
# 用于计算视频流的实时帧率，使用滑动窗口平均法提高稳定性

# FPS（帧率）计算工具类
# 用于计算视频流的实时帧率，使用滑动窗口平均法提高稳定性

from collections import deque
import cv2

class CvFpsCalc(object):
    """
    FPS（帧率）计算工具类
    
    使用滑动窗口平均法计算视频流的实时帧率，提高稳定性
    通过计算连续帧之间的时间差，使用滑动窗口平均来平滑帧率波动
    """
    
    def __init__(self, buffer_len=1):
        """
        初始化FPS计算器
        
        参数:
        buffer_len: 滑动窗口大小，用于计算平均FPS
        """
        # 记录开始时间（以时钟节拍为单位）
        self._start_tick = cv2.getTickCount()
        
        # 计算时钟频率的倒数，用于将节拍转换为毫秒
        self._freq = 1000.0 / cv2.getTickFrequency()
        
        # 使用双端队列存储时间差，限制队列长度
        self._diff_times = deque(maxlen=buffer_len)

    def get(self):
        """
        获取当前FPS值
        
        返回:
        fps_rounded: 四舍五入到两位小数的FPS值
        """
        # 获取当前时间节拍
        current_tick = cv2.getTickCount()
        
        # 计算当前帧与上一帧的时间差（毫秒）
        different_time = (current_tick - self._start_tick) * self._freq
        
        # 更新开始时间为当前时间
        self._start_tick = current_tick

        # 将时间差添加到滑动窗口
        self._diff_times.append(different_time)

        # 计算平均时间间隔，然后转换为FPS
        # FPS = 1000 / 平均帧间隔（毫秒）
        fps = 1000.0 / (sum(self._diff_times) / len(self._diff_times))
        
        # 四舍五入到两位小数
        fps_rounded = round(fps, 2)

        return fps_rounded