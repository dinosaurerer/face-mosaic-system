# Facial-coding-and-decoding-in-videos

项目介绍：此项目基于opencv库实现对人脸（主要是正脸）的抓取与识别，使用正规马赛克进行人脸打码处理，打码结束后自动生成密钥隐藏文件，可以通过对应的密钥文件实现对打码视频的还原。

项目背景：来自三名大学生的数据结构课程的期末大作业。--锐克五队团队出品

项目文件功能简介：

versio 1.0：（无可视化操作）

注：1.0版本的项目文件比较冗杂，共有四个文件，其中包含一个主文件（Final_assignment_main）和三个功能文件。
项目文件的详细功能在下面的具体文件介绍中有详细解释，这里不再赘述；
特别注意的是，1.0版本的函数运作并不具有高效性（这是没有充分理利用计算机算力而导致的，在下面的2.0版本中这一现象将会得到较大的改变。）

Final_assignment_face_recognition_and_blurring：对传入的视频进行正规马赛克处理（区别于传统的高斯马赛克），运行结果为马赛克图片序列文件夹


Final_assignment_ImgSequenceToVideo：此程序负责读取指定文件夹的图片序列再转为视频文件，注意，fps = calculate_fps(r"C:\face\test2.mp4") 一般修改帧率时要改路径


Final_assignment_VideoFaceRestoration：基于马赛克文件夹结合密钥文件对其进行解码，处理结果为decoding文件夹


Final_assignment_main：项目主函数通过import上述三个文件项目子文件，完整实现打码与解码的过程。



versio 2.0：（添加可视化操作）

注：该版本中的main函数（main_v2.0）已经涵盖了从打码到解码部分的所有的代码并且添加了可视化框架（主要功能已绑定了快捷键），鼠标在按钮上的悬浮操作会有飘窗解释。

请注意，在2.0版本中我们着重解决了老版本中由于性能利用不充分而导致的运行效率地的问题，这场情况下在threading多线程的运行下cpu的占有率将达到95%作用（这只是为了打码函数运作时更快的进行硬盘的读写操作以生成密钥文件）；

另外我们增添了实时渲染的功能以及可以随时更改打码精细度的滑块，实时渲染由于并不发送读写操作，基本上可以在渲染时达到原视频塑料的80%（测试数据为视频中有5张人脸的条件下获得），在此特别感谢负责可视化部分的人员@ Future-Elite！


环境配置：在python3环境下，使用以下模块与库：
	1.time
	2.os
	3.gzip
	4.shutil
	5.tkinter----8.6
	6.PIL----9.5.0
	7.Pmw----2.1.1
	8.opencv-python----4.7.0.72
	9.threading
	10.numpy----1.24.3
	11.json----2.0.9

运行方法与效果：
安装压缩包后，双击运行exe文件即可运行；
运行开始时会出现两秒左右的程序预启动动画，结束后跳出具体的视频播放窗口及相关按键，这些按键除了暂停，快进，快退，倍速等基础功能外，还具有打码预览（实时渲染），生成人脸打码、解码视频，以及调节马赛克精细度等功能。较为直观简洁的用户交互界面，方便用户进行具体的打码、解码操作。
