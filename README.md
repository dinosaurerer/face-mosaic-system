# Facial-coding-and-decoding-in-videos

项目介绍：此项目基于opencv库实现对人脸（主要是正脸）的抓取与识别，使用正规马赛克进行人脸打码处理，打码结束后自动生成密钥隐藏文件，可以通过对应的密钥文件实现对打码视频的还原。

项目背景：来自三名大学生的数据结构课程的期末大作业。

项目文件功能简介：


Final_assignment_face_recognition_and_blurring：对传入的视频进行正规马赛克处理（区别于传统的高斯马赛克），运行结果为马赛克图片序列文件夹


Final_assignment_ImgSequenceToVideo：此程序负责读取指定文件夹的图片序列再转为视频文件，注意，fps = calculate_fps(r"C:\face\test2.mp4") 一般修改帧率时要改路径


Final_assignment_VideoFaceRestoration：基于马赛克文件夹结合密钥文件对其进行解码，处理结果为decoding文件夹


Final_assignment_main：项目主函数通过import上述三个文件项目子文件，完整实现打码与解码的过程。







