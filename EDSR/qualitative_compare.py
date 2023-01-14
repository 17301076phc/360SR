import math
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

def calc_psnr(hr, sr):
    # scale = 2
    # diff = (sr/1.0 - hr/1.0) / 255.0
    # # print(diff)
    # shave = scale + 6
    #
    # valid = diff[shave:-shave, shave:-shave, ...]
    # mse = np.mean(np.power(valid,2))
    # # print(mse)
    #
    # return -10 * math.log10(mse)
    return 10. * math.log10(255.0**2 / np.mean((hr/1.0 - sr/1.0) ** 2 ))
	
def main():
# get min psnr frame
    cap = cv2.VideoCapture()
    cap.open(path)
    i = 0
    scale = 2
    minpsnr = 99
    while True:
        ret, frame = cap.read()
        if ret:
            for _ in range(20):
                _, frame = cap.read()
            hr_frame = cv2.resize(frame, (frame.shape[1]//scale,frame.shape[0]//2), interpolation=cv2.INTER_CUBIC)
            # bicubic_frame = cv2.resize(lr, (lr.shape[1]*scale,lr.shape[0]*2), interpolation=cv2.INTER_CUBIC)
            lr = cv2.resize(hr_frame, (hr_frame.shape[1]//scale,hr_frame.shape[0]//2), interpolation=cv2.INTER_CUBIC)

            sr_tensor = F.interpolate(
                torch.from_numpy(lr / 255).float().permute(2, 0, 1).unsqueeze(0),  # (1,3,W,H)
                scale_factor=scale,
                mode='bicubic',
                align_corners=True) * 255

            sr_tensor = sr_tensor.squeeze().permute(1, 2, 0)  # (W,H,3)
            bicubic_frame = sr_tensor.numpy()
            p = calc_psnr(hr_frame, bicubic_frame)
            if p<minpsnr:
                minpsnr = p
                print(minpsnr)
            # s = "%04d"%i
            # filename = os.path.join(savepath,str(s)+".png")
                cv2.imwrite(savepath+"/hr.png",hr_frame)
                cv2.imwrite(savepath+"/lr.png",bicubic_frame)

            i += 1
        else:
            cap.release()
            break

def get_minpsnr_patch():
    savepath = "qualtative_compare/projections"
    gt = cv2.imread("qualtative_compare/projections/erp.png")
    # sr = cv2.imread("qualtative_compare/musician/lr.png")
    lr = cv2.resize(gt, (gt.shape[1] // 2, gt.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

    sr_tensor = F.interpolate(
        torch.from_numpy(lr / 255).float().permute(2, 0, 1).unsqueeze(0),  # (1,3,W,H)
        scale_factor=2,
        mode='bicubic',
        align_corners=True) * 255

    sr_tensor = sr_tensor.squeeze().permute(1, 2, 0)  # (W,H,3)
    sr = sr_tensor.numpy()

    (width, length, depth) = gt.shape
    print(gt.shape)
    
    fi = 0
    fj = 0
    minpsnr = 99
    patchsize = 96
    # cv2.rectangle(gt, (969,218), (969 + patchsize, 218 + patchsize), (0, 0, 255), 3)

    for i in range(0,width-patchsize,8):
        for j in range(0,length-patchsize,8):
            gt_pic = gt[i:i+patchsize, j: j+patchsize, :]
            sr_pic = sr[i:i+patchsize, j: j+patchsize, :]
            p = calc_psnr(gt_pic, sr_pic)
            if p < minpsnr:
                minpsnr = p
                print(minpsnr)
                print(fi, fj)
                fi = i
                fj = j
    print(fi,fj)
    gggg= gt[fi:fi+patchsize, fj: fj+patchsize, :]
    cv2.imwrite("qualtative_compare/projection_lr" + "/erp_lr_patch.png", cv2.resize(gggg, (gggg.shape[1] // 2, gggg.shape[0] // 2), interpolation=cv2.INTER_CUBIC))
    cv2.imwrite(savepath + "/erp_hr_patch.png", gt[fi:fi+patchsize, fj: fj+patchsize, :])
    cv2.rectangle(gt, (fj, fi), (fj + patchsize, fi + patchsize), (0, 0, 255), 3)
    cv2.imwrite(savepath + "/erp_hr_rect.png", gt)

def bic_sr():
    hr = cv2.imread("qualtative_compare/fig2/erp_erp_patch.png")
    lr = cv2.resize(hr, (hr.shape[1] // 2, hr.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
    sr_tensor = F.interpolate(
        torch.from_numpy(lr / 255).float().permute(2, 0, 1).unsqueeze(0),  # (1,3,W,H)
        scale_factor=2,
        mode='bicubic',
        align_corners=True) * 255

    sr_tensor = sr_tensor.squeeze().permute(1, 2, 0)  # (W,H,3)
    bicubic_frame = sr_tensor.numpy()

    cv2.imwrite("qualtative_compare/" + "/bicubic_sr.png", bicubic_frame)


def get_patch(fi, fj):
    # fi, fj=144,904
    # fi, fj = 480, 1528
    patchsize=96
    hr = cv2.imread("qualtative_compare/all_erp/erp.png")
    cv2.imwrite("qualtative_compare/fig2/" + "/erp_erp_patch.png", hr[fi:fi+patchsize, fj: fj+patchsize, :])
	
bic_sr()
get_minpsnr_patch()
# get_patch()
main()