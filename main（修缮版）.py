#######################################
# -*-coding:utf-8-*-
# Project: Face-Mosaic-System
# Author: Rike Fifth Team
# Version: 2.0.0
#######################################

import tkinter.messagebox
from tkinter.messagebox import showinfo
from tkinter.filedialog import *
from tkinter import *
from PIL import Image as im
from PIL import ImageTk
import time
import shutil
import Pmw
import cv2
import os
import threading
import numpy as np
import json
import gzip

# 全局变量

filename = ''
output = ''
admit = True
paused = True
speed = 1.0
current_frame = 1
cap = cv2.VideoCapture()
mosaic = False
mosaic_speed = 1
mosaic_value = 15


def do_mosaic(img, _x, _y, w, h, neighbor):
    for _i in range(0, h, neighbor):  # 关键点0 从0开始
        for _j in range(0, w, neighbor):
            rect = [_j + _x, _i + _y]  # 关键点1  矩形取点的位置
            color = img[_i + _y][_j + _x].tolist()
            left_up = (rect[0], rect[1])
            x2 = rect[0] + neighbor - 1
            y2 = rect[1] + neighbor - 1
            if x2 > _x + w:
                x2 = _x + w
            if y2 > _y + h:
                y2 = _y + h
            right_down = (x2, y2)
            cv2.rectangle(img, left_up, right_down, color, -1)
    return img


def mosaic_val_to_neighbour(value):
    if value == 0:
        return 1
    else:
        return value


def face_detect_mosaic(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(r"./frontalface.xml")
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
    for _x, _y, w, h in faces:
        img = do_mosaic(img, _x, _y, w, h, neighbor=mosaic_val_to_neighbour(mosaic_value))
    return img


def face_mosaic(*args):
    global filename, mosaic, mosaic_speed
    args = list(args)
    args.reverse()
    if filename[-4:] == '.mp4':
        if not mosaic:
            mosaic = True
            mosaic_speed = 8
            scale.configure(state='normal')
            bt1.configure(image=bt_mosaic)
        else:
            mosaic = False
            mosaic_speed = 1
            scale.configure(state='disabled')
            bt1.configure(image=bt_un_mosaic)
    else:
        showinfo('提示', '请选择视频文件')
        my_open()


def face_mosaic_output():
    global mosaic
    mosaic = False
    if filename[-4:] == '.mp4':
        file_reset()
        dic_count_img_faces = {}  # 此字典检查并统计每张图片调用打码函数的次数（faces的值（测试理论值为：step = 5【实际有误差】））
        _cap = cv2.VideoCapture(filename)  # 读取视频
        total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数和帧率
        img_lst = [None for _ in range(total_frames)]  # 创建一个列表用于保存图片

        def compress_json(data, js_filename):  # json文件压缩函数
            compressed_file = js_filename + '.gz'
            with gzip.GzipFile(compressed_file, 'w') as f:
                json_data = json.dumps(data, default=int).encode('utf-8')
                f.write(json_data)
            return compressed_file

        def do_mosaic_output(img, i_img, cou, _x, _y, w, h, neighbor):  # 单张图片单个人脸的打码函数
            # 密钥生成部分
            mosaic_cut = img[_y:_y + h, _x:_x + w]
            key = {
                'x': _x,
                'y': _y,
                'w': w,
                'h': h,
                'img_cut': np.array(mosaic_cut, dtype=int).tolist()
            }
            dic_count_img_faces[f'{i_img}'] = f'{i_img}_{cou}'  # 将当前图片的索引和当前人脸的索引存入字典
            key_file = fr'./key_file\mosaic_key.json_{i_img}_{cou}'  # 生成密钥文件
            compress_json(key, key_file)  # 压缩密钥文件
            # 打码部分
            for _i in range(0, h, neighbor):
                for _j in range(0, w, neighbor):
                    rect = [_j + _x, _i + _y]
                    color = img[_i + _y][_j + _x].tolist()  # 关键点1 tolist
                    left_up = (rect[0], rect[1])
                    x2 = rect[0] + neighbor - 1  # 关键点2 减去一个像素
                    y2 = rect[1] + neighbor - 1
                    if x2 > _x + w:
                        x2 = _x + w
                    if y2 > _y + h:
                        y2 = _y + h
                    right_down = (x2, y2)
                    cv2.rectangle(img, left_up, right_down, color, -1)  # rectangle为矩形画图函数

            return img  # 返回打码后的图片

        def face_recognition_and_blurring(img, i_img):  # img是视频帧，i_img是视频帧的索引
            global mosaic_value
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片灰度
            face_detector = cv2.CascadeClassifier(r"./frontalface.xml")  # 加载特征数据
            faces = face_detector.detectMultiScale(gray)  # 进行检测人脸操作(参数也可以不写)
            for count_face, (_x, _y, w, h) in enumerate(faces):  # count_face是第几个人脸
                do_mosaic_output(img, i_img, count_face, _x, _y, w, h, neighbor=mosaic_val_to_neighbour(mosaic_value))
            img_lst[i_img] = img  # 将打码后的图片存入列表

        def save_dict(dic):  # 保存字典文件
            tf = open(r"./key_file_count\dic_count_img_faces.json", "w")
            json.dump(dic, tf)  # dump()将数据写入json文件中
            tf.close()  # 关闭文件

        def process_save_compressed_frames(lst):
            l = len(lst)
            threads = []

            def save_frame(index, frame):
                format_index = str(index).zfill(3)
                cv2.imwrite(r"./img_sequence_encoding\{}.jpg".format(format_index), frame)

            for _ in range(l):
                thread = threading.Thread(target=save_frame, args=(_, lst[_]))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

        def process_video(Cap, Total_frames):  # 并发处理视频
            threads = []
            for _i in range(Total_frames):  # i代表当前帧数（从0开始）
                ret, frame = Cap.read()
                if not ret:
                    break

                thread = threading.Thread(target=face_recognition_and_blurring, args=(frame, _i))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

        def img_to_video():
            def img_sequence_to_video(img_sequence_dir, output_video_path):
                # 获取图像序列中的图像列表
                img_files = sorted(os.listdir(img_sequence_dir))

                # 获取第一张图像的宽度和高度

                first_img_path = os.path.join(img_sequence_dir, img_files[0])
                first_img = cv2.imread(first_img_path)  # 读取第一张图片
                img_height, img_width, _ = first_img.shape  # 获取图像高度、宽度和通道数

                # 创建视频写入对象

                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 可根据需要更改编解码器
                fps = calculate_fps(filename)  # 帧率
                output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (img_width, img_height))  # 创建视频写入对象

                # 将图像序列逐帧写入视频对象

                for img_file in img_files:
                    img_path = os.path.join(img_sequence_dir, img_file)
                    img = cv2.imread(img_path)
                    output_video.write(img)

                # 释放视频写入对象

                output_video.release()

            # 调用图片序列转视频函数
            img_sequence_to_video(r"./img_sequence_encoding", r"./output_video\video.mp4")
        return_val = tkinter.messagebox.askquestion('生成打码视频', '是否生成视频？')
        if return_val == 'yes':
            process_video(_cap, total_frames)  # 并发处理视频
            process_save_compressed_frames(img_lst)  # 并发处理视频
            _cap.release()  # 释放视频资源
            save_dict(dic_count_img_faces)  # 保存统计字典
            img_to_video()
            showinfo('提示', '视频生成成功！')
            save_as()
    else:
        return_val = tkinter.messagebox.askquestion('提示', '请选择视频！')
        if return_val == 'yes':
            my_open()
            face_mosaic_output()


def face_reverse_output(*args):
    args = list(args)
    args.reverse()
    if os.listdir(r'./img_sequence_encoding') and os.listdir(r'./key_file') and os.listdir(r'./key_file_count'):
        # json文件解压缩函数

        def decompress_json(compressed_file):
            with gzip.GzipFile(compressed_file, 'r') as f:
                json_data = f.read().decode('utf-8')
                data = json.loads(json_data)
            return data

        # 定义解码函数

        def decode_mosaic(mosaic_img, key_file):
            key = decompress_json(key_file)  # 解压缩json文件
            _x = key['x']
            _y = key['y']
            w = key['w']
            h = key['h']  # 获取坐标和宽高
            img_cut = key['img_cut']  # 获取密钥
            mosaic_img[_y:_y + h, _x:_x + w] = img_cut
            return mosaic_img

        # 定义视频解码函数

        def video_decoding(img_sequence_dir, output_img_path):
            # 获取图像序列中的图像列表
            img_files = sorted(os.listdir(img_sequence_dir))

            # 读取密钥统计字典文件
            tf = open(r"./key_file_count\dic_count_img_faces.json", "r")
            dic_count_img_faces = json.load(tf)
            tf.close()  # 关闭文件

            for index, img_file in enumerate(img_files):
                img_path = os.path.join(img_sequence_dir, img_file)  # 图像路径
                mosaic_img = cv2.imread(img_path)
                if str(index) not in dic_count_img_faces.keys():
                    # 保存解码后的图像
                    fn = '%s/%03d.jpg' % (output_img_path, index)
                    cv2.imwrite(fn, mosaic_img)
                else:  # 如果该帧图像中有人脸
                    # 调用解码函数
                    for _i in range(int(dic_count_img_faces[str(index)][-1]) + 1):
                        key_file = r'./key_file\mosaic_key.json_' + str(index) + f"_{_i}" + ".gz"
                        mosaic_img = decode_mosaic(mosaic_img, key_file=key_file)
                        # 保存解码后的图像
                        fn = '%s/%03d.jpg' % (output_img_path, index)
                        cv2.imwrite(fn, mosaic_img)

        def img_to_video():
            def img_sequence_to_video(img_sequence_dir, output_video_path):
                # 获取图像序列中的图像列表
                img_files = sorted(os.listdir(img_sequence_dir))
                # 获取第一张图像的宽度和高度
                first_img_path = os.path.join(img_sequence_dir, img_files[0])
                first_img = cv2.imread(first_img_path)
                img_height, img_width, _ = first_img.shape  # 获取图像高度和宽度

                # 创建视频写入对象

                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 可根据需要更改编解码器
                # fps = calculate_fps(filename)  # 帧率
                fps = 30
                output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (img_width, img_height))

                # 将图像序列逐帧写入视频对象

                for img_file in img_files:
                    img_path = os.path.join(img_sequence_dir, img_file)
                    img = cv2.imread(img_path)
                    output_video.write(img)  # 写入视频对象

                # 释放视频写入对象

                output_video.release()

            # 调用图片序列转视频函数
            img_sequence_to_video(r"./img_sequence_decoding", r"./output_video\video2.mp4")

        return_val = tkinter.messagebox.askquestion('生成解码视频', '是否生成视频？')
        if return_val == 'yes':
            video_decoding(r"./img_sequence_encoding", r"./img_sequence_decoding")
            img_to_video()
            save_as()
            file_reset()
    else:
        showinfo('提示', '请先进行人脸解码！')
        my_open()


def clear(*args):
    global filename, admit, stream, current_frame
    if not args:
        admit = False  # 重置admit
        filename = ''  # 重置文件名
        stream.destroy()  # 销毁stream
        stream = Label(app, bg='grey', width=960, height=540)  # 重新创建stream
        stream.grid()  # 重新创建stream
        current_frame = 1
        get_filename()  # 重新获取文件名


def calculate_fps(video_path):
    v = cv2.VideoCapture(video_path)
    fps = v.get(cv2.CAP_PROP_FPS)
    return fps


def play_pause_video(*args):
    global paused, current_frame, cap
    args = list(args)
    args.reverse()
    paused = not paused
    if current_frame == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        my_open()
    if filename[-4:] == '.mp4':
        cap = cv2.VideoCapture(filename)
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == current_frame:
            my_open()
            current_frame = 0
        if mosaic:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 1)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        def video_stream():  # 视频流
            global speed, paused, current_frame, mosaic_speed
            if admit and not paused:
                _, frame = cap.read()
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = im.fromarray(cv2image)
                if mosaic:
                    img = im.fromarray(face_detect_mosaic(cv2image))
                img = img.resize((960, 540))
                imgtk = ImageTk.PhotoImage(image=img)
                stream.imgtk = imgtk
                stream.configure(image=imgtk)
                if current_frame < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                    if mosaic:
                        stream.after(int(1000 / calculate_fps(filename) / mosaic_speed), video_stream)
                        current_frame += 1
                    else:
                        stream.after(int(1000 / calculate_fps(filename) / speed), video_stream)
                current_frame += 1
        video_stream()
    else:
        return_val= tkinter.messagebox.askquestion('提示', '打开视频!')
        if return_val == 'yes':
            my_open()
            paused = True
            play_pause_video()
        else:
            paused = not paused
            change_play_button()
    change_play_button()


def back(*args):
    global current_frame, filename, cap
    args = list(args)
    args.reverse()
    if filename[-4:] == '.mp4':
        if current_frame < 2 * calculate_fps(filename):
            current_frame = 1
        else:
            current_frame -= 2 * calculate_fps(filename)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)


def forward(*args):
    global current_frame, filename, cap
    args = list(args)
    args.reverse()
    if filename[-4:] == '.mp4':
        if current_frame > cap.get(cv2.CAP_PROP_FRAME_COUNT) - 2 * calculate_fps(filename):
            current_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        else:
            current_frame += 2 * calculate_fps(filename)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)


def change_play_button():
    global paused
    if paused:
        bt3.configure(image=bt_play)
    else:
        bt3.configure(image=bt_stop)


def copy():
    global filename
    os.system('echo ' + filename + ' | clip')
    showinfo('提示', '已复制')


def get_filename():
    global filename
    screen.configure(text=filename)


def my_open(*args):
    global filename, admit, current_frame, paused
    args = list(args)
    args.reverse()
    clear()
    default_dir = r'D:\python'
    filename = askopenfilename(title=u'选择视频文件', defaultextension='.mp4',
                               initialdir=(os.path.expanduser(default_dir)))
    get_filename()
    admit = True
    if not paused:
        paused = True
        change_play_button()


def save_as(*args):
    args = list(args)
    args.reverse()
    if os.listdir('./output_video'):
        pre_video = './output_video/' + os.listdir('./output_video')[0]
        f = asksaveasfilename(initialfile='未命名.mp4', defaultextension='.mp4')  # 打开文件对话框并获取文件路径保存
        shutil.move(pre_video, f)
        showinfo('提示', '已保存')
    elif filename:
        pre_video = filename
        f = asksaveasfilename(initialfile='未命名.mp4', defaultextension='.mp4')  # 打开文件对话框并获取文件路径保存
        shutil.copyfile(pre_video, f)
    else:
        showinfo('警告', '请选择视频文件')
        my_open()
        save_as()


def speed_change(*args):
    global speed
    args = list(args)
    args.reverse()
    if not mosaic:
        if speed < 4:
            speed *= 2
        else:
            speed = 0.25
        show_speed()
    else:
        showinfo('提示', '请先关闭mosaic')


def speed_up(*args):
    global speed
    args = list(args)
    args.reverse()
    if not mosaic:
        if speed < 4:
            speed *= 2
        else:
            speed = 4.0
        show_speed()
    else:
        showinfo('提示', '请先关闭mosaic')


def speed_down(*args):
    global speed
    args = list(args)
    args.reverse()
    if not mosaic:
        if speed > 0.25:
            speed /= 2
        else:
            speed = 0.25
        show_speed()
    else:
        showinfo('提示', '请先关闭mosaic')


def show_speed():
    global speed
    t = Message(app, text=str(speed) + '倍速', width=300, aspect=100, bg='white', font=('黑体', 15))
    t.grid(row=0, column=0, columnspan=2)
    t.after(1000, t.destroy)
    t.mainloop()


def get_help():
    showinfo('帮助',
             '点击打开导入视频，\n'
             '点击保存导出视频，\n'
             '点击播放暂停视频，\n'
             '点击加速减速视频，\n'
             '点击退出程序，\n'
             '点击作者查看作者信息，\n'
             '点击版权所有查看版权信息，\n'
             '点击版本号查看版本信息')


def author():
    showinfo('作者团队信息', 'Rike-Fifth 团队')


def version():
    showinfo('版本号：', 'v-2.0.0')


def mosaic_val(value):
    global mosaic_value
    mosaic_value = int(value)


def file_reset():
    shutil.rmtree('./output_video')
    os.mkdir('./output_video')
    shutil.rmtree('./img_sequence_encoding')
    os.mkdir('./img_sequence_encoding')
    shutil.rmtree('./img_sequence_decoding')
    os.mkdir('./img_sequence_decoding')
    shutil.rmtree('./key_file')
    os.mkdir('./key_file')
    shutil.rmtree('./key_file_count')
    os.mkdir('./key_file_count')


if __name__ == '__main__':

    # 启动窗口

    prior = Tk()

    width = 494
    height = 294
    screen_width = prior.winfo_screenwidth()
    screen_height = prior.winfo_screenheight()
    x = int(screen_width / 2 - width / 2)
    y = int(screen_height / 2 - height / 2)
    size = '{}x{}+{}+{}'.format(width, height, x, y)
    prior.geometry(size)
    prior.overrideredirect(True)  # 去除窗口边框
    prior.wm_attributes("-toolwindow", True)  # 置为工具窗口(没有最大最小按钮)
    prior.wm_attributes("-topmost", True)  # 永远处于顶层
    prior_bg = PhotoImage(file=r'./bt_img\prior.png')
    prior_show = Label(prior, image=prior_bg, width=490, height=290)
    prior_show.grid()
    i = 0.0
    while i < 0.8:
        prior.attributes("-alpha", i)
        time.sleep(0.03)
        i += 0.02
        prior.update()
    time.sleep(1)
    j = 0.8
    while j > 0.0:
        prior.attributes("-alpha", j)
        time.sleep(0.03)
        j -= 0.02
        prior.update()
    prior.after(0, prior.destroy)
    prior.mainloop()

    root = Tk()
    root.title('锐克五队--人脸加密器')

    width = 960
    height = 600

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = int(screen_width / 2 - width / 2)
    y = int(screen_height / 2 - height / 2)
    size = '{}x{}+{}+{}'.format(width, height, x, y)
    root.geometry(size)
    root.resizable(False, False)
    root.iconphoto(False, PhotoImage(file=r'./bt_img\icon.png'))
    root.grid_rowconfigure(0, weight=10, uniform="group1")
    root.grid_rowconfigure(1, weight=1, uniform="group1")
    root.grid_columnconfigure(0, weight=1)
    root.wm_attributes("-topmost", True)

    # 功能栏

    buttons = Frame(root, bg='silver', cursor='cross')
    buttons.grid(row=1, column=0, sticky="nsew")

    app = Frame(root, bg="white", cursor='cross', width=960, height=540)
    app.grid(row=0, column=0, sticky="nsew")

    stream = Label(app, bg='grey', width=960, height=540)
    stream.grid()

    # 功能按钮

    bt_un_mosaic = PhotoImage(file=r'./bt_img\bt_un_mosaic.png')
    bt_mosaic = PhotoImage(file=r'./bt_img\bt_mosaic.png')
    bt_play = PhotoImage(file=r'./bt_img\play.png')
    bt_stop = PhotoImage(file=r'./bt_img\stop.png')
    bt_forward = PhotoImage(file=r'./bt_img\forward.png')
    bt_back = PhotoImage(file=r'./bt_img\back.png')
    bt_clear = PhotoImage(file=r'./bt_img\clear.png')
    bt_up = PhotoImage(file=r'./bt_img\speed_up.png')
    bt_down = PhotoImage(file=r'./bt_img\speed_down.png')
    bt_copy = PhotoImage(file=r'./bt_img\copy.png')
    bt_input = PhotoImage(file=r'./bt_img\input.png')
    bt_output = PhotoImage(file=r'./bt_img\output.png')

    bt1 = Button(buttons, image=bt_un_mosaic, relief='groove', width=50, height=40, command=face_mosaic)
    bt1.grid(row=0, column=0)
    bt1_tip = Pmw.Balloon(root)
    bt1_tip.bind(bt1, '打码预览')

    bt2 = Button(buttons, image=bt_input, relief='groove', width=50, height=40, command=face_reverse_output)
    bt2.grid(row=0, column=1)
    bt2_tip = Pmw.Balloon(root)
    bt2_tip.bind(bt2, '解码')

    sep1 = Label(buttons, text='|', bg='silver', width=2)
    sep1.grid(row=0, column=2)

    bt_back1 = Button(buttons, image=bt_back, relief='flat', width=50, height=40, command=back)
    bt_back1.grid(row=0, column=3)
    bt_back1_tip = Pmw.Balloon(root)
    bt_back1_tip.bind(bt_back1, '快退')

    bt3 = Button(buttons, image=bt_play, relief='flat', width=50, height=40, command=play_pause_video)
    bt3.grid(row=0, column=4)
    bt3_tip = Pmw.Balloon(root)
    bt3_tip.bind(bt3, '播放/暂停')

    bt_forward1 = Button(buttons, image=bt_forward, relief='flat', width=50, height=40, command=forward)
    bt_forward1.grid(row=0, column=5)
    bt_forward1_tip = Pmw.Balloon(root)
    bt_forward1_tip.bind(bt_forward1, '快进')

    sep2 = Label(buttons, text='|', bg='silver', width=2)
    sep2.grid(row=0, column=6)

    speeds = Frame(buttons, bg='silver', width=50, height=40)
    speeds.grid(row=0, column=7, sticky="nsew")

    bt_speed_up = Button(speeds, image=bt_up, relief='flat', width=30, height=18, command=speed_up)
    bt_speed_up.grid(row=0, column=0)
    up_tip = Pmw.Balloon(root)
    up_tip.bind(bt_speed_up, '升倍速')

    bt_speed_down = Button(speeds, image=bt_down, relief='flat', width=30, height=18, command=speed_down)
    bt_speed_down.grid(row=1, column=0)
    down_tip = Pmw.Balloon(root)
    down_tip.bind(bt_speed_down, '降倍速')

    sep3 = Label(buttons, text='|', bg='silver', width=2)
    sep3.grid(row=0, column=8)

    bt4 = Button(buttons, image=bt_clear, relief='flat', width=50, height=40, command=clear)
    bt4.grid(row=0, column=9)
    bt4_tip = Pmw.Balloon(root)
    bt4_tip.bind(bt4, '清空')

    s = Label(buttons, bg='silver', text='文件路径:', width=8, height=2)
    s.grid(row=0, column=10)

    screen = Label(buttons, text=filename, width=40, height=2, bg='grey')
    screen.grid(row=0, column=11)

    bt5 = Button(buttons, image=bt_copy, width=50, height=40, command=copy)
    bt5.grid(row=0, column=12)
    bt5_tip = Pmw.Balloon(root)
    bt5_tip.bind(bt5, '复制')

    bt_out = Button(buttons, image=bt_output, relief='flat', width=55, height=40, command=face_mosaic_output)
    bt_out.grid(row=0, column=13)
    bt_out_tip = Pmw.Balloon(root)
    bt_out_tip.bind(bt_out, '输出打码视频')

    # 滑动条

    mosaic_body = Frame(root, bg='silver', cursor='cross')
    mosaic_body.grid(row=0, column=1, sticky="nsew")

    doubleVar = DoubleVar()

    scale = Scale(mosaic_body,
                  from_=40,  # 设置最大值
                  to=0,  # 设置最小值
                  resolution=2,  # 设置步长
                  length=480,  # 设置轨道的长度
                  width=25,  # 设置轨道的宽度
                  highlightcolor='blue',  # 设置轨道的颜色
                  digits=1,  # 设置十位有效数字，即显示的数字个数。
                  command=mosaic_val,  # 绑定事件处理函数。
                  variable=doubleVar,  # 绑定变量
                  showvalue=False,  # 是否显示当前值
                  tickinterval=5,  # 设置刻度
                  state=DISABLED,  # 设置状态
                  )
    scale.grid(row=0, column=1, sticky="nsew")

    Scale_show = Label(mosaic_body, text='Mosaic', width=8, height=3, bg='silver')
    Scale_show.grid(row=1, column=1)

    info = Frame(root, bg='gray', cursor='cross')
    info.grid(row=1, column=1, sticky="nsew")
    info1 = Label(info, text='Rike Fifth', width=8, height=3)
    info1.grid()

    # 菜单栏

    menubar = Menu(root)
    menu = Menu(menubar, tearoff=False, activeborderwidth=8, borderwidth=20)
    menu.add_command(label='打开', accelerator='Ctrl + O', command=my_open)
    menu.add_command(label='另存为', accelerator='Ctrl + Shift + S', command=save_as)
    menubar.add_cascade(label='文件', menu=menu)

    function = Menu(menubar, tearoff=False, activeborderwidth=8, borderwidth=20)
    function.add_command(label='打码预览', command=face_mosaic, accelerator='Ctrl + P')
    function.add_command(label='打码', command=face_mosaic_output, accelerator='Ctrl + M')
    function.add_command(label='解码', command=face_reverse_output, accelerator='Ctrl + R')
    function.add_command(label='重置', command=clear, accelerator='Ctrl + C')
    function.add_command(label='倍速', command=speed_change, accelerator='Ctrl + S')
    menubar.add_cascade(label='功能', menu=function)

    helper = Menu(menubar, tearoff=False, activeborderwidth=8, borderwidth=20)
    helper.add_command(label='使用说明', command=get_help)
    menubar.add_cascade(label='帮助', menu=helper)

    about = Menu(menubar, tearoff=False, activeborderwidth=8, borderwidth=20)
    about.add_command(label='作者', command=author)
    about.add_command(label='版本', command=version)
    about.add_command(label='清除缓存', command=file_reset)
    menubar.add_cascade(label='关于', menu=about)
    editmenu = Menu(menubar, cursor='cross')
    root.config(menu=menubar)

    # 快捷键

    root.bind('<KeyPress-Left>', back)
    root.bind('<KeyPress-Right>', forward)
    root.bind('<KeyPress-Up>', speed_up)
    root.bind('<KeyPress-Down>', speed_down)
    root.bind('<KeyPress-space>', play_pause_video)
    root.bind('<Control-o>', my_open)
    root.bind('<Control-Shift-S>', save_as)
    root.bind('<Control-p>', face_mosaic)
    root.bind('<Control-m>', face_mosaic_output)
    root.bind('<Control-r>', face_reverse_output)
    root.bind('<Control-c>', clear)
    root.bind('<Control-s>', speed_change)

    root.mainloop()
