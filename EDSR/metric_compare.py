import json
import math
import os

import imageio
import imgproc
import lpips
from pathlib import Path
import numpy as np
from PIL import Image
import py360convert
import torch

import cv2
from py360convert import cube_list2h, utils
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def make_dowmsample():
    # path = "test/Cubemap_8K_musician"
    # savepath = "test/Cubemap_8K_musician_x3/"
    # path = "test/8K_musician_1"
    # savepath = "test/8K_musician_x4_downsample/"
    path = "test/ERP_to_CMP_GT"
    savepath = "test/CMP_x4_downsample/"
    scale = 4
    # path = "test/diving_1"
    # savepath = "test/diving_x8_downsample/"
    # path = "ODI_cube_HR"
    # savepath = "test/ODI_cube_X3/"
    # path = "DIV2K_train_HR"
    # savepath = "DIV2K_train_LRx8/"
    #
    fillist = os.listdir(path)
    files = [os.path.join(path,i) for i in fillist]
    for item in files:
        print(item)
        # print(item.split('/')[1])
        frame = cv2.imread(item)
        # frame = frame[:frame.shape[0]//3*3,:frame.shape[1]//3*3]
        print(frame.shape)
        bicubic_frame = cv2.resize(frame, (frame.shape[1]//scale,frame.shape[0]//scale), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite(savepath+item.split('/')[1], bicubic_frame)

        cv2.imwrite(savepath+item.split('/')[2], bicubic_frame)

    print("finishing..... .... ")

# make_dowmsample()

def cube_dict2h(cube_dict, face_k=['F', 'R', 'B', 'L', 'U', 'D']):
    # assert len(k) == 6
    return cube_list2h([cube_dict[k] for k in face_k])
def c2e(cubemap, h, w, mode='bilinear', cube_format='dice'):
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = utils.cube_list2h(cubemap)
    elif cube_format == 'dict':
        cubemap = cube_dict2h(cubemap)
    elif cube_format == 'dice':
        cubemap = utils.cube_dice2h(cubemap)
    else:
        raise NotImplementedError('unknown cube_format')
    assert len(cubemap.shape) == 3
    assert cubemap.shape[0] * 6 == cubemap.shape[1]
    assert w % 8 == 0
    face_w = cubemap.shape[0]

    uv = utils.equirect_uvgrid(h, w)
    u, v = np.split(uv, 2, axis=-1)
    u = u[..., 0]
    v = v[..., 0]
    cube_faces = np.stack(np.split(cubemap, 6, 1), 0)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = utils.equirect_facetype(h, w)
    coor_x = np.zeros((h, w))
    coor_y = np.zeros((h, w))

    for i in range(4):
        mask = (tp == i)
        coor_x[mask] = 0.5 * np.tan(u[mask] - np.pi * i / 2)
        coor_y[mask] = -0.5 * np.tan(v[mask]) / np.cos(u[mask] - np.pi * i / 2)

    mask = (tp == 4)
    c = 0.5 * np.tan(np.pi / 2 - v[mask])
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = c * np.cos(u[mask])

    mask = (tp == 5)
    c = 0.5 * np.tan(np.pi / 2 - np.abs(v[mask]))
    coor_x[mask] = c * np.sin(u[mask])
    coor_y[mask] = -c * np.cos(u[mask])

    # Final renormalize
    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

    equirec = np.stack([
        utils.sample_cubefaces(cube_faces[..., i], tp, coor_y, coor_x, order=order)
        for i in range(cube_faces.shape[3])
    ], axis=-1)

    return equirec

def makecube(path):
    e = np.array(Image.open(path))

    # You can make convertion between supported cubemap format

    cube = py360convert.e2c(e,face_w=1920) #512 # 1920
    cube_h = py360convert.cube_dice2h(cube)  # the inverse is cube_h2dice
    cube_dict = py360convert.cube_h2dict(cube_h)  # the inverse is cube_dict2h
    # cube_list = py360convert.cube_h2list(cube_h)  # the inverse is cube_list2h
    # Image.fromarray(cube).save("0001_cubemap.png")
    # Image.fromarray(cube_h).save("0001_cube_h.png")
    # dict_keys(['F', 'R', 'B', 'L', 'U', 'D'])
    t = np.zeros((3840, 5760, e.shape[2]))
    # t = np.zeros((1024, 1536, e.shape[2]))

    print(t.shape)
    Image.fromarray(np.uint8(cube_dict["U"])).save("u.png")
    Image.fromarray(np.uint8(cube_dict["R"])).save("r.png")
    Image.fromarray(np.uint8(cube_dict["B"])).save("b.png")


    u = np.array(Image.open("u.png").transpose(Image.FLIP_TOP_BOTTOM))
    r = np.array(Image.open("r.png").transpose(Image.FLIP_LEFT_RIGHT))
    b = np.array(Image.open("b.png").transpose(Image.FLIP_LEFT_RIGHT))

    w,h = cube_dict["F"].shape[1],cube_dict["F"].shape[0]
    t[:w,:h,] = r
    t[:w, h:h*2,] = cube_dict["L"]
    t[:w, h*2:,] = u
    t[w:, :h,] = cube_dict["D"]
    t[w:, h:h*2,] = cube_dict["F"]
    t[w:, h*2:,] = b
    # print(type(t))
    # Image.fromarray(np.uint8(t)).save("test/To_Cube_GT_musician/"+path.split('/')[2])
    # Image.fromarray(np.uint8(t)).save("ODI_cube_HR/"+path[5:])
    Image.fromarray(np.uint8(t)).save("tmp.png")
    # cv2.imwrite("0001_cube.png",t)
    # return np.uint8(t)

# makecube("test/8K_musician_1/0301.png")

def to_cube_filelist():
    gtpath = "test/8K_musician_1"
    # gtpath = "test/Cubemap_8K_musician"
    #

    fillist = os.listdir(gtpath)
    files = [os.path.join(gtpath, i) for i in fillist]
    for item in files:
        print(item)

        # print(item.split('/')[2])
        makecube(item)

# to_cube_filelist()

def make_ODI_cube():
    path = "ODIHR"
    savepath = "ODI_cube_HR/"
    fillist = os.listdir(path)
    files = [os.path.join(path, i) for i in fillist]
    for item in files:
        print(item)
        makecube(item)
    print('finishing.....')
# make_ODI_cube()

def makeERP(path):

    cubemap = np.array(Image.open(path))
    print(cubemap.shape)
    # You can make convertion between supported cubemap format

    cube_dict = {}

    w,h = 1920,1920
    # r = cubemap[:w,:h,]
    Image.fromarray(np.uint8(cubemap[:w,:h,])).save("r.png")
    cube_dict["R"] = np.array(Image.open("r.png").transpose(Image.FLIP_LEFT_RIGHT))

    cube_dict["L"] = cubemap[:w, h:h*2,]

    # u = cubemap[:w, h*2:,]
    Image.fromarray(np.uint8(cubemap[:w, h*2:,])).save("u.png")
    cube_dict["U"] = np.array(Image.open("u.png").transpose(Image.FLIP_TOP_BOTTOM))


    cube_dict["D"] = cubemap[w:, :h,]
    cube_dict["F"] = cubemap[w:, h:h*2,]

    # b = cubemap[w:, h*2:,]
    Image.fromarray(np.uint8(cubemap[w:, h*2:,])).save("b.png")
    cube_dict["B"] = np.array(Image.open("b.png").transpose(Image.FLIP_LEFT_RIGHT))

    erp = c2e(cube_dict,h=3840,w=7680,cube_format='dict')

    Image.fromarray(np.uint8(erp)).save("tmp.png")
    # cv2.imwrite("0001_cube.png",t)
    # return np.uint8(t)
# makeERP("test/Cubemap_8K_musician/0000.png")

def convert_3840x2880(path):
    t = np.array(Image.open(path)) # 2880x1920   each cube:960
    # out = np.zeros((2880,3840, t.shape[2]))
    out = np.array(Image.open("ERP_to_CMP_8K_musician_X2/0000.png"))
    w,h = 960,960
    r = t[:w, :h, ]
    l = t[:w, h:h * 2, ]
    u = t[:w, h * 2:, ]
    Image.fromarray(np.uint8(u)).save("u.png")
    d = t[w:, :h, ]
    Image.fromarray(np.uint8(d)).save("d.png")

    f = t[w:, h:h * 2, ]
    b = t[w:, h * 2:, ]
    t_u = np.array(Image.open("u.png").transpose(Image.ROTATE_270))
    t_d = np.array(Image.open("d.png").transpose(Image.ROTATE_90))


    out[w:w*2,h:h*2,:]= r
    out[w:w*2, h*3:,:] = l
    out[:w, h:h * 2,:] = t_u
    out[w*2:, h:h * 2,:] = t_d
    out[w:w*2, :h,:] = f
    out[w:w*2, h*2:h*3, :] = b

    Image.fromarray(np.uint8(out)).save(path)

def dir_con():
    path = "test/EAC_to_CMP_3840x2880"
    fillist = os.listdir(path)
    files = [os.path.join(path, i) for i in fillist]
    for item in files:
        print(item)
        convert_3840x2880(item)

# convert_3840x2880("EAC_to_CMP_3840x2880_X2/0004.png")
# dir_con()
def cal_psnr(hr, sr):
    # to y
    # hr = cv2.cvtColor(hr,cv2.COLOR_BGR2YCR_CB)
    # sr = cv2.cvtColor(sr,cv2.COLOR_BGR2YCR_CB)

    diff = (sr / 1.0 - hr / 1.0) / 255.0
    # print(diff.shape)
    # shave = 2
    # # shave = 0
    #
    # valid = diff[shave:-shave, shave:-shave, ...]
    # mse = np.mean(np.power(valid, 2))
    mse = np.mean(diff ** 2)

    return -10 * math.log10(mse)

def test_psnr():
    # gtpath = "test/8K_musician_1"
    # gtpath = "test/To_Cube_GT_musician/"
    # gtpath = "test/ERP_to_CMP_8K_musician/"
    # gtpath = "ODIHR/"
    # gtpath = "test/ERP_8K_musician/"
    # gtpath = "test/ISP_8K_musician/"
    gtpath = "test/CMP_8K_musician/"
    # gtpath = "test/Cubemap_8K_musician/"
    # srpath = "experiment/EDSR_result/ODI_ERP_X2/results-Demo/"
    # srpath = "experiment/RCAN_result/ODI_ERP_X2/results-Demo/"

    scale = '4'
    # srpath = "RCAN_result/ERP_x2/"
    # srpath = "RCAN_result/ERP_x3/"
    # srpath = "RCAN_result/ERP_x4/"
    # srpath = "RCAN_result/ISP_x4/"
    srpath = "RCAN_result/CMP_x"+scale+"/"

    # srpath = "EDSR_result/ERP_x4/"
    # srpath = "experiment/test/results-Demo/"

    # result_file = Path(f'result_RCAN_ERP_X4.txt').open('w', encoding='utf8')
    result_file = Path(f'result_RCAN_CMP_X'+scale+'.txt').open('w', encoding='utf8')

    # result_file = Path(f'result_ERP_X2.txt').open('w', encoding='utf8')

    # gtpath = "test/8K_musician_1"
    # srpath = "8K_musician_1/"
    avg = []
    fillist = os.listdir(srpath)
    files = [os.path.join(srpath, i) for i in fillist]
    # print(len(files))
    for item in files:
        print(item)
        print(gtpath + item.split('/')[2])
        # print(gtpath + item.split('/')[1])

        try:
            # srframe = cv2.cvtColor(cv2.imread(srpath+ + item.split('/')[1]), cv2.COLOR_BGR2RGB)
            # gtframe = cv2.imread(item)
            srframe = cv2.imread(item)
            # gtframe = gtframe[:gtframe.shape[0]//3*3,:gtframe.shape[1]//3*3]
            # print(srpath + item.split('/')[1])
            # makecube(srpath + item.split('/')[2])
            # srframe = cv2.cvtColor(cv2.imread('tmp.png'), cv2.COLOR_BGR2RGB)
            # srframe = cv2.imread(srpath + item.split('/')[1][:3]+'.png') # ODI
            gtframe =cv2.imread(gtpath + item.split('/')[2]) # musician
            if int(scale)==3:
                gtframe = gtframe[:gtframe.shape[0]//3*3,:gtframe.shape[1]//3*3]


            print(gtframe.shape,srframe.shape)
            # gtframe = imgproc.bgr2ycbcr(gtframe, True)
            # srframe = imgproc.bgr2ycbcr(srframe, True)
            # p = psnr(gtframe,srframe)
            p = cal_psnr(gtframe,srframe)
            # s = ssim(gtframe, srframe, multichannel=True)

            print(p)
            avg.append(p)
            # avg_s.append(s)
            result = {
                'filename': item,
                'psnr': p

            }
            result_file.write(json.dumps(result) + ',\n')
            result_file.flush()
        except:
            print(item)
            break

    print('avg psnr is : ', sum(avg)/len(avg))
    result_file.write('avg psnr is : '+str( sum(avg)/len(avg))+ ',\n')
    # result_file.write('avg ssim is : '+str( sum(avg_s)/len(avg_s))+ ',\n')
    result_file.flush()
    print("finishing..... .... ")

# test_psnr()

def test_bicubic_psnr(gtpath,scale):

        frame =cv2.imread(gtpath)
        if int(scale)==3:
            frame = frame[:frame.shape[0]//3*3,:frame.shape[1]//3*3]

        bicubic_frame = cv2.resize(frame, (frame.shape[1]//scale,frame.shape[0]//scale), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('tmp.jpg',bicubic_frame)
        # bicubic_frame = cv2.resize(frame, (bicubic_frame.shape[1]//scale,bicubic_frame.shape[0]//scale), interpolation=cv2.INTER_CUBIC)
        # bicubic_frame = cv2.resize(bicubic_frame, (bicubic_frame.shape[1]*scale,bicubic_frame.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
        bframe = bicubic_frame
        srframe = cv2.resize(bframe, (bframe.shape[1]*scale,bframe.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
        # cv2.imwrite('tmp_i.jpg',srframe)

        print(frame.shape,frame.shape)
        # gtframe = imgproc.bgr2ycbcr(gtframe, True)
        # srframe = imgproc.bgr2ycbcr(srframe, True)
        # p = psnr(gtframe,srframe)
        p = cal_psnr(frame,srframe)
        # s = ssim(gtframe, srframe, multichannel=True)
        print(p)


test_bicubic_psnr("./test/CMP_8K_musician/0000.png",2)
# test_bicubic_psnr("./ODIHR/000.jpg",2)
# test_bicubic_psnr("./ODIHR/004.jpg",2)


def test_one_psnr(sr,gt):
    gtframe = cv2.cvtColor(cv2.imread(gt), cv2.COLOR_BGR2RGB)
    srframe = cv2.cvtColor(cv2.imread(sr), cv2.COLOR_BGR2RGB)
    # gtframe = np.array(Image.open(gt))[:2000,:]
    # srframe = np.array(Image.open(sr))[:2000,:]
    # gtframe1 = np.array(Image.open(gt))[2000:, :]
    # srframe1 = np.array(Image.open(sr))[2000:, :]
    print(gtframe.shape, srframe.shape)

    # gtframe = torch.from_numpy(gtframe).unsqueeze(0)
    # srframe = torch.from_numpy(srframe).unsqueeze(0)

    p = psnr(gtframe, srframe)
    # p = cal_ws_psnr(gtframe, srframe)
    # p1 = cal_psnr(gtframe1, srframe1)
    # s = ssim(gtframe,srframe,multichannel=True)
    print("mse is ",np.mean((gtframe/1.0-srframe/1.0)**2))
    print("mae is ",np.mean(np.absolute(gtframe/1.0-srframe/1.0)))
    print("psnr is: ",(p))
    # print("ssim is: ",s)

def test_one_ssim(sr,gt,scale):
    gtframe = cv2.cvtColor(cv2.imread(gt), cv2.COLOR_BGR2RGB)
    srframe = cv2.cvtColor(cv2.imread(sr), cv2.COLOR_BGR2RGB)

    gtframe =gtframe[:gtframe.shape[0]//scale*scale,:gtframe.shape[1]//scale*scale]
    srframe = srframe[:srframe.shape[0]//scale*scale,:srframe.shape[1]//scale*scale]

    print(gtframe.shape, srframe.shape)

    p = ssim(gtframe, srframe,multichannel=True)
    # p = cal_ws_psnr(gtframe, srframe)
    # p1 = cal_psnr(gtframe1, srframe1)
    # s = ssim(gtframe,srframe,multichannel=True)
    print("ssim is: ",(p))
    return p

def test_ssim():
    # gtpath = "test/EAC_to_CMP_3840x2880/"
    # srpath = "EAC_to_CMP_3840x2880_X2/"
    # gtpath = "test/SSP_to_CMP_8K_musician/"
    # srpath = "SSP_to_CMP_8K_musician_X4/"
    # gtpath = "test/TSP_to_CMP_8K_musician/"
    # srpath = "TSP_to_CMP_8K_musician_X2/"
    # gtpath = "test/OHP_to_CMP_8K_musician/"
    # srpath = "OHP_to_CMP_8K_musician_X4/"
    # gtpath = "test/ISP_to_CMP_8K_musician/"
    # srpath = "ISP_to_CMP_8K_musician_X2/"
    gtpath = "test/ERP_to_CMP_8K_musician/"
    srpath = "ERP_to_CMP_8K_musician_X4/"

    # result_file = Path(f'result_EAC_to_CMP_musician_X2_ssim.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_SSP_to_CMP_musician_X4_ssim.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_TSP_to_CMP_musician_X2_ssim.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_OHP_to_CMP_musician_X4_ssim.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_ISP_to_CMP_musician_X2_ssim.txt').open('w', encoding='utf8')
    result_file = Path(f'result_ERP_to_CMP_musician_X4_ssim.txt').open('w', encoding='utf8')

    avg = []
    avg_s = []
    fillist = os.listdir(srpath)
    files = [os.path.join(srpath, i) for i in fillist]
    for item in files:
        print(item)
        print(gtpath + item.split('/')[1])
        try:
            srframe = cv2.cvtColor(cv2.imread(item), cv2.COLOR_BGR2RGB)
            # gtframe = gtframe[:gtframe.shape[0]//3*3,:gtframe.shape[1]//3*3]
            # print(srpath + item.split('/')[1])
            # makecube(srpath + item.split('/')[2])
            # srframe = cv2.cvtColor(cv2.imread('tmp.png'), cv2.COLOR_BGR2RGB)
            # srframe = cv2.imread(srpath + item.split('/')[1][:3]+'.png') # ODI
            gtframe =cv2.cvtColor(cv2.imread(gtpath + item.split('/')[1]), cv2.COLOR_BGR2RGB) # musician

            print(gtframe.shape,srframe.shape)
            p = ssim(gtframe,srframe,multichannel=True)
            # s = ssim(gtframe, srframe, multichannel=True)

            print(p)
            avg.append(p)
            # avg_s.append(s)
            result = {
                'filename': item,
                'ssim': p

            }
            result_file.write(json.dumps(result) + ',\n')
            result_file.flush()
        except:
            print(item)
            break

    print('avg psnr is : ', sum(avg)/len(avg))
    result_file.write('avg ssim is : '+str( sum(avg)/len(avg))+ ',\n')
    # result_file.write('avg ssim is : '+str( sum(avg_s)/len(avg_s))+ ',\n')
    result_file.flush()
    print("finishing..... .... ")

# test_ssim()
# test_one_psnr("ERP_8K_musician_X8/0000.png","tmp.png")
# test_one_psnr("ERP_8K_musician_X4/0000.png","tmp.png")
# test_one_psnr("ERP_8K_musician_X3/0000.png","tmp.png")
# test_one_psnr("ERP_8K_musician_X2/0000.png","tmp.png")
# test_one_psnr("test/8K_musician_1/0000.png","tmp.png")

def test_lpips(sr,gt,scale):
    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

    if scale==3:
        gtframe = cv2.imread(gt)
        srframe = cv2.imread(sr)

        gtframe = gtframe[:gtframe.shape[0] // scale * scale, :gtframe.shape[1] // scale * scale]
        srframe = srframe[:srframe.shape[0] // scale * scale, :srframe.shape[1] // scale * scale]
        cv2.imwrite("tmpgt.png", gtframe)
        cv2.imwrite("tmpsr.png", srframe)

        gtframe = lpips.im2tensor(lpips.load_image("tmpgt.png"))
        srframe = lpips.im2tensor(lpips.load_image("tmpsr.png"))
    else:
        gtframe = lpips.im2tensor(lpips.load_image(gt))
        srframe = lpips.im2tensor(lpips.load_image(sr))

    # img0 = torch.zeros(1, 3, 64, 64)  # image should be RGB, IMPORTANT: normalized to [-1,1]
    # img1 = torch.zeros(1, 3, 64, 64)
    d = loss_fn_alex(gtframe, srframe)
    print(d)
    return d

def test_all_lpips():
    # gtpath = "test/EAC_to_CMP_3840x2880/"
    # srpath = "EAC_to_CMP_3840x2880_X4/"
    # gtpath = "test/SSP_to_CMP_8K_musician/"
    # srpath = "SSP_to_CMP_8K_musician_X2/"
    # gtpath = "test/TSP_to_CMP_8K_musician/"
    # srpath = "TSP_to_CMP_8K_musician_X4/"
    # gtpath = "test/OHP_to_CMP_8K_musician/"
    # srpath = "OHP_to_CMP_8K_musician_X2/"
    # gtpath = "test/ISP_to_CMP_8K_musician/"
    # srpath = "ISP_to_CMP_8K_musician_X4/"
    # gtpath = "test/ERP_to_CMP_8K_musician/"
    # srpath = "ERP_to_CMP_8K_musician_X3/"
    gtpath = "test/8K_musician_1/"
    srpath = "ERP_8K_musician_X2/"

    # result_file = Path(f'result_SSP_to_CMP_musician_X2_lpips.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_TSP_to_CMP_musician_X4_lpips.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_OHP_to_CMP_musician_X2_lpips.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_ISP_to_CMP_musician_X4_lpips.txt').open('w', encoding='utf8')
    # result_file = Path(f'result_ERP_to_CMP_musician_X3_lpips.txt').open('w', encoding='utf8')
    result_file = Path(f'result_ERP_musician_X2_lpips.txt').open('w', encoding='utf8')



    avg = []
    fillist = os.listdir(srpath)
    files = [os.path.join(srpath, i) for i in fillist]
    for item in files:
        print(item)
        print(gtpath + item.split('/')[1])
        l = test_lpips(sr=item,gt=gtpath + item.split('/')[1]).item()
        avg.append(l)
        result = {
            'filename': item,
            'lpips': l

        }
        result_file.write(json.dumps(result) + ',\n')
        result_file.flush()

    print('avg lpips is : ', sum(avg)/len(avg))
    result_file.write('avg lpips is : ' + str(sum(avg) / len(avg)) + ',\n')
    result_file.flush()
    print("finishing..... .... ")

# test_all_lpips()

def test_tmp():
    result_file = Path(f'result_ERPP_musician_lpips.txt').open('w', encoding='utf8')

    srname = "ERP_8K_musician"

    gt0 = "test/"+srname+"/0000.png"
    gt1 = "test/"+srname+"/0001.png"
    gt2 = "test/"+srname+"/0002.png"
    gt3 = "test/"+srname+"/0003.png"
    gt4 = "test/"+srname+"/0004.png"


    l0 = test_lpips(srname + "_X2/0000.png", gt0,2)
    l1 = test_lpips(srname + "_X2/0001.png", gt1,2)
    l2 = test_lpips(srname + "_X2/0002.png", gt2,2)
    l3 = test_lpips(srname + "_X2/0003.png", gt3,2)
    l4 = test_lpips(srname + "_X2/0004.png", gt4,2)
    rx2 = (l0 + l1 + l2 + l3 + l4).item() / 5
    print(rx2)
    result_X2 = {
        'scale': "X2",
        'lpips': rx2

    }
    result_file.write(json.dumps(result_X2) + ',\n')
    result_file.flush()

    l0 = test_lpips(srname + "_X3/0000.png", gt0,3)
    l1 = test_lpips(srname + "_X3/0001.png", gt1,3)
    l2 = test_lpips(srname + "_X3/0002.png", gt2,3)
    l3 = test_lpips(srname + "_X3/0003.png", gt3,3)
    l4 = test_lpips(srname + "_X3/0004.png", gt4,3)
    rx3 = (l0 + l1 + l2 + l3 + l4).item() / 5
    print(rx3)
    result_X3 = {
        'scale': "X3",
        'lpips': rx3

    }
    result_file.write(json.dumps(result_X3) + ',\n')
    result_file.flush()

    l0 = test_lpips(srname+"_X4/0000.png", gt0,4)
    l1 = test_lpips(srname+"_X4/0001.png", gt1,4)
    l2 = test_lpips(srname+"_X4/0002.png", gt2,4)
    l3 = test_lpips(srname+"_X4/0003.png", gt3,4)
    l4 = test_lpips(srname+"_X4/0004.png", gt4,4)
    rx4 = (l0 + l1 + l2 + l3 + l4).item() / 5
    print(rx4)
    result_X4 = {
        'scale': "X4",
        'lpips': rx4

    }
    result_file.write(json.dumps(result_X4) + ',\n')
    result_file.flush()

# test_tmp()

def test_tmp_ssim():
    result_file = Path(f'result_ERP_musician_ssim.txt').open('w', encoding='utf8')

    srname = "ERP_8K_musician"

    gt0 = "test/" + srname + "/0000.png"
    gt1 = "test/" + srname + "/0001.png"
    gt2 = "test/" + srname + "/0002.png"
    gt3 = "test/" + srname + "/0003.png"
    gt4 = "test/" + srname + "/0004.png"

    l0 = test_one_ssim(srname+"_X2/0000.png",gt0,2)
    l1 = test_one_ssim(srname+"_X2/0001.png",gt1,2)
    l2 = test_one_ssim(srname+"_X2/0002.png",gt2,2)
    l3 = test_one_ssim(srname+"_X2/0003.png",gt3,2)
    l4 = test_one_ssim(srname+"_X2/0004.png",gt4,2)
    rx2 = (l0 + l1 + l2 + l3 + l4) / 5
    print(rx2)
    result_X2 = {
        'scale': "X2",
        'lpips': rx2

    }
    result_file.write(json.dumps(result_X2) + ',\n')
    result_file.flush()

    l0 = test_one_ssim(srname+"_X3/0000.png", gt0,3)
    l1 = test_one_ssim(srname+"_X3/0001.png", gt1,3)
    l2 = test_one_ssim(srname+"_X3/0002.png", gt2,3)
    l3 = test_one_ssim(srname+"_X3/0003.png", gt3,3)
    l4 = test_one_ssim(srname+"_X3/0004.png", gt4,3)
    rx3 = (l0 + l1 + l2 + l3 + l4) / 5
    print(rx3)
    result_X3 = {
        'scale': "X3",
        'lpips': rx3

    }
    result_file.write(json.dumps(result_X3) + ',\n')
    result_file.flush()

    l0 = test_one_ssim(srname+"_X4/0000.png", gt0,4)
    l1 = test_one_ssim(srname+"_X4/0001.png", gt1,4)
    l2 = test_one_ssim(srname+"_X4/0002.png", gt2,4)
    l3 = test_one_ssim(srname+"_X4/0003.png", gt3,4)
    l4 = test_one_ssim(srname+"_X4/0004.png", gt4,4)
    rx4 = (l0 + l1 + l2 + l3 + l4) / 5
    print(rx4)
    result_X4 = {
        'scale': "X4",
        'lpips': rx4

    }
    result_file.write(json.dumps(result_X4) + ',\n')
    result_file.flush()

# test_tmp_ssim()

# print("X4")
# test_one_psnr("ERP_8K_musician_X4/0000.png","test/8K_musician_1/0000.png")
# print("X8")
# test_one_psnr("ERP_8K_musician_X8/0000.png","test/8K_musician_1/0000.png")
# print("X3")
# test_one_psnr("ERP_8K_musician_X3/0000.png","test/8K_musician_1/0000.png")
# print("X2")
# test_one_psnr("ERP_8K_musician_X2/0000.png","test/8K_musician_1/0000.png")

# print("X8")
# test_one_psnr("Cubemap_8K_musician_X8/0003.png","test/Cubemap_8K_musician/0003.png")
# print("X4")
# test_one_psnr("Cubemap_8K_musician_X4/0003.png","test/Cubemap_8K_musician/0003.png")
# print("X3")
# test_one_psnr("Cubemap_8K_musician_x3/0003.png","test/Cubemap_8K_musician/0003.png")
# print("X2")
# test_one_psnr("Cubemap_8K_musician/0003.png","test/Cubemap_8K_musician/0003.png")

