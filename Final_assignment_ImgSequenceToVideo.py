import cv2
import os

"""

### 实例代码中文件路径的相关说明 ###

# 图像序列所在的目录路径
img_sequence_dir = 'C:\img_sequence'

# 输出视频的文件名和保存路径
output_video_path = r"C:\output_video\ video4.avi"

"""


# 定义fps计算函数
def calculate_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    return FPS


# 定义图片序列转视频函数
def img_sequence_to_video(img_sequence_dir, output_video_path):
    # 获取图像序列中的图像列表
    img_files = sorted(os.listdir(img_sequence_dir))

    # 获取第一张图像的宽度和高度
    first_img_path = os.path.join(img_sequence_dir, img_files[0])
    first_img = cv2.imread(first_img_path)
    height, width, _ = first_img.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 可根据需要更改编解码器
    fps = calculate_fps(r"C:\face\test2.mp4")  # 帧率
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 将图像序列逐帧写入视频对象
    for img_file in img_files:
        print(img_file)
        img_path = os.path.join(img_sequence_dir, img_file)
        img = cv2.imread(img_path)
        output_video.write(img)

    # 释放视频写入对象
    output_video.release()


if __name__ == '__main__':

    # 调用图片序列转视频函数
    img_sequence_to_video(r"C:\img_sequence_decoding", r"C:\output_video\video(decoding-5.22(60%)).avi")
    pass
