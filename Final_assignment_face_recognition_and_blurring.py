# 对视频的人脸识别与打码

# 导入依赖
import cv2 as cv
import json
import gzip
import numpy as np


# json文件压缩函数
def compress_json(data, filename):
    compressed_file = filename + '.gz'
    with gzip.GzipFile(compressed_file, 'w') as f:
        json_data = json.dumps(data, default=int).encode('utf-8')
        f.write(json_data)
    return compressed_file


cou = 0


def do_mosaic(img, x, y, w, h, count_do_mosaic, neighbor=9):
    global cou
    cou += 1
    print(cou)
    """
    :param int x :  马赛克左顶点
    :param int y:  马赛克左顶点
    :param int w:  马赛克宽
    :param int h:  马赛克高
    :param int neighbor:  马赛克每一块的宽
    """
    # 密钥生成部分
    mosaic_cut = img[y:y + h, x:x + w]

    key = {
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'img_cut': np.array(mosaic_cut, dtype=int).tolist()
    }

    key_file = fr'C:\key_file\mosaic_key.json_{count_do_mosaic}'
    compress_json(key, key_file)

    # 打码部分
    for i in range(0, h, neighbor):
        for j in range(0, w, neighbor):
            rect = [j + x, i + y]
            color = img[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            x2 = rect[0] + neighbor - 1  # 关键点2 减去一个像素
            y2 = rect[1] + neighbor - 1
            if x2 > x + w:
                x2 = x + w
            if y2 > y + h:
                y2 = y + h
            right_down = (x2, y2)
            cv.rectangle(img, left_up, right_down, color, -1)  # rectangle为矩形画图函数

    return img


count = 0
count_img = 0  # 当前图片数量位置
dic_count_img_faces = {}  # 此字典检查并统计每张图片调用打码函数的次数（faces的值（测试理论值为：step = 5【实际有误差】））


def face_detect_demo(img, imgs):
    global count
    global count_img
    global dic_count_img_faces
    count_img += 1
    # 将图片灰度
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 加载特征数据
    face_detector = cv.CascadeClassifier(r"D:\anzhuang\SystemFiles\Data\Haarcascades\frontalface.xml")
    # 进行检测人脸操作(参数也可以不写)
    faces = face_detector.detectMultiScale(gray)

    for x, y, w, h in faces:
        img_mosaic = do_mosaic(img, x, y, w, h, neighbor=15, count_do_mosaic=count)
        count += 1
        # 显示图片(这时的照片会一闪而过)
        cv.imshow('result', img_mosaic)
    # count值写入字典
    dic_count_img_faces[count_img] = count

    imgs.append(img)  # 保存图片到列表中
    cv.imshow('result', img)


# 定义主函数
def face_recognition_and_blurring(Filename=r"C:\face\test2.mp4", Image_compression_rate=60):
    # 获取压缩率
    # Image_compression_rate = int(input("请输入压缩率（0-100）："))
    # 判断压缩率是否在0-100之间
    if Image_compression_rate < 0 or Image_compression_rate > 100:
        print("压缩率不在0-100之间")
        return

    # 读取视频
    cap = cv.VideoCapture(Filename)
    # 创建一个列表用于保存图片
    img_lst = []
    # 播放进行读取（一帧一帧的走）
    while True:
        # flag表示是否在播放（布尔类型）
        flag, frame = cap.read()
        # 判断是否在播放
        if not flag:
            break
        face_detect_demo(frame, img_lst)
        # 输入q的时候进行关闭
        if ord('q') == cv.waitKey(10):
            break

    # 输出统计字典
    print(dic_count_img_faces)
    # 保存字典文件
    tf = open(r"C:\key_file_count\dic_count_img_faces.json", "w")
    json.dump(dic_count_img_faces, tf)  # dump()将数据写入json文件中
    tf.close()  # 关闭文件

    # 释放内存
    cv.destroyAllWindows()
    # 释放视频的空间
    cap.release()

    # 定义图像生成函数（加压缩）
    def save_compressed_frames():  # 保存逐帧图像并进行压缩
        for i, img in enumerate(img_lst):
            fn = '%s/shot_%03d.jpg' % (r"C:\img_sequence", i)  # 选择JPEG格式保存
            cv.imwrite(fn, img, [cv.IMWRITE_JPEG_QUALITY, Image_compression_rate])  # 调整JPEG质量参数来控制压缩比例
            print(fn, 'saved')

    save_compressed_frames()


# 调用主函数
if __name__ == '__main__':
    # face_recognition_and_blurring()
    pass
