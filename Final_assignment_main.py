from Final_assignment_face_recognition_and_blurring import face_recognition_and_blurring
from Final_assignment_VideoFaceRestoration import VideoDecoding
from Final_assignment_ImgSequenceToVideo import img_sequence_to_video

face_recognition_and_blurring(Filename=r"C:\face\test2.mp4", Image_compression_rate=60)  # 人脸识别与模糊
img_sequence_to_video(r"C:\img_sequence_encoding", r"C:\output_video\video(encoding-5.22(60%)).avi")  # 图片序列转视频

VideoDecoding(r"C:\output_video\video(encoding-5.22(60%)).avi", r"C:\img_sequence_decoding")  # 视频解码
img_sequence_to_video(r"C:\img_sequence_decoding", r"C:\output_video\video(decoding-5.22(60%)).avi")  # 图片序列转视频
