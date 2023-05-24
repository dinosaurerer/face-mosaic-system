# 导入依赖
import cv2 as cv
import os
import gzip
import json

"""
# 打码图像序列所在的目录路径
    img_sequence_dir = C:\img_sequence
"""


# json文件解压缩函数
def decompress_json(compressed_file):
    with gzip.GzipFile(compressed_file, 'r') as f:
        json_data = f.read().decode('utf-8')
        data = json.loads(json_data)
    return data


# 定义解码函数
def decode_mosaic(mosaic_img, key_file):
    """
    :param np.ndarray mosaic_img: 打码后的图像
    :param str key_file: 加密密钥文件名
    :return: 解码后的图像
    :rtype: np.ndarray
    """
    key = decompress_json(key_file)

    x = key['x']
    y = key['y']
    w = key['w']
    h = key['h']

    img_cut = key['img_cut']

    mosaic_img[y:y + h, x:x + w] = img_cut

    return mosaic_img


# 定义视频解码函数(主函数)
def VideoDecoding(img_sequence_dir, output_img_path):
    # 获取图像序列中的图像列表
    img_files = sorted(os.listdir(img_sequence_dir))

    # 读取密钥统计字典文件
    tf = open(r"C:\key_file_count\dic_count_img_faces.json", "r")
    dic_count_img_faces = json.load(tf)
    tf.close()  # 关闭文件
    # 为dic_count_img_faces添加第0帧的值（防止下面的做差报错）
    dic_count_img_faces["0"] = 0
    print(dic_count_img_faces)  # 打印密钥统计字典

    count_json = 0

    for index, img_file in enumerate(img_files):
        img_path = os.path.join(img_sequence_dir, img_file)  # 图像路径
        mosaiced_img = cv.imread(img_path)
        # 调用解码函数
        times = index + 1
        for i in range(dic_count_img_faces[str(times)] - dic_count_img_faces[str(times - 1)]):
            key_file = r'C:\key_file\mosaic_key.json_' + str(count_json) + ".gz"
            count_json += 1
            mosaiced_img = decode_mosaic(mosaiced_img, key_file=key_file)

        # 保存解码后的图像
        # 释放内存（由于底层是c++写的，所以将底层里面的空间进行释放）
        cv.destroyAllWindows()
        fn = '%s/decoding_img_%03d.jpg' % (output_img_path, index)
        cv.imwrite(fn, mosaiced_img)
        print(fn, "saved")  # 输出保存提示


if __name__ == '__main__':
    # 调用视频解码函数
    # VideoDecoding(r"C:\img_sequence_encoding", r"C:\img_sequence_decoding")
    pass
