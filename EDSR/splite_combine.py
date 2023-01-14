import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import subprocess


def get_video_frame(path, savepath):
    # path = "3840x1920_28M_25FPS_265_4K_diving_1.mp4"
    # savepath = "test/diving_1"

    cap = cv2.VideoCapture()
    cap.open(path)
    i = 0
    while True:
        ret, hr_frame = cap.read()
        if ret and i < 3:
            s = "%04d" % i
            filename = os.path.join(savepath, str(s) + ".png")
            print(filename)
            cv2.imwrite(filename, hr_frame)
            i += 1
        else:
            cap.release()
            break


def make_video(path):
    # path = "diving1"
    tp = cv2.imread(path + "/0001.png")
    size = (tp.shape[1], tp.shape[0])
    fps = 25
    vidwri = cv2.VideoWriter(
        "result_video.avi",
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps,
        size
    )
    fillist = os.listdir(path)
    files = [os.path.join(path, i) for i in fillist]
    files.sort(key=lambda x: (x[:-4]))

    for item in files:
        # print(item)
        img = cv2.imread(item)
        vidwri.write(img)

    vidwri.release()
    print("finish making video....")


def image_splite(pic_path):
    # pic_path = 'test/diving_1/0002.png'
    pic_target = './splite_image/'  # savepath
    if not os.path.exists(pic_target):
        os.makedirs(pic_target)
    # ERP

    picture = cv2.imread(pic_path)
    (width, length, depth) = picture.shape
    cut_width = width // 2 // 2
    cut_length =length // 2 // 2
    pic = np.zeros((cut_width, cut_length, depth))

    num_width = int(width / cut_width)
    num_length = int(length / cut_length)

    for i in range(0, num_width):
        for j in range(0, num_length):
            pic = picture[i * cut_width: (i + 1) * cut_width, j * cut_length: (j + 1) * cut_length, :]
            result_path = pic_target + '{}_{}.png'.format(i + 1, j + 1)
            cv2.imwrite(result_path, pic)

    print("splite done!!!")


# image_splite('test/diving_1/0001.png')


def image_combine(path, savename):
    # pic_path = './experiment/test/results-Demo/'
    pic_path = path


    num_width_list = []
    num_lenght_list = []

    picture_names = os.listdir(pic_path)
    if len(picture_names) == 0:
        print("no file!!!!")
    else:
        img_1_1 = cv2.imread(pic_path + '1_1.png')
        (width, length, depth) = img_1_1.shape
        # print(img_1_1.shape)

        for picture_name in picture_names:
            num_width_list.append(int(picture_name.split("_")[0]))
            num_lenght_list.append(int((picture_name.split("_")[-1]).split(".")[0]))

        num_width = max(num_width_list)
        num_length = max(num_lenght_list)

        splicing_pic = np.zeros((num_width * width, num_length * length, depth))
        print(splicing_pic.shape)
        # get pic
        for i in range(1, num_width + 1):
            for j in range(1, num_length + 1):
                img_part = cv2.imread(pic_path + '{}_{}.png'.format(i, j))
                splicing_pic[width * (i - 1): width * i, length * (j - 1): length * j, :] = img_part

        pic = splicing_pic
        cv2.imwrite(savename, pic)
        print("after combination:", pic.shape)
        print("combine done!!!")

# image_combine("./split_images/", "result.png")


def test_one_psnr(sr, gt):
    gtframe = cv2.imread(gt)
    srframe = cv2.imread(sr)
    print(gtframe.shape, srframe.shape)
    p = psnr(gtframe, srframe)
    print("psnr is: ", p)


def split_test_combine(testpath):
    # path = "test_data/ERP_x2_downsample"
    path = testpath
    print(path)

    fillist = os.listdir(path)
    files = [os.path.join(path, i) for i in fillist]
    for item in files:
        print(item.split('/')[2])
        # if item.split('/')[2] != '0000.png' or item.split('/')[2] != '0001.png' or item.split('/')[2] != '0002.png':
        #     continue
        # print(item)
        image_splite(item)
        os.system("src/demo.sh")
        # subprocess.call("sh ./src/demo.sh")
        image_combine('./experiment/test/results-Demo/', './RCAN_result/' + item.split('/')[2])

split_test_combine("test_data/ERP_x2_downsample")
