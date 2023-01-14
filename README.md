# 360SR
This repository is used for the 360SR task.

## Projection transformation
### 1.ffmpeg
download

	git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg

#### ffmpeg v360 usage

	./ffmpeg -i EAC/0000.png -vf "v360=eac:c3x2" EAC_to_CMP/0000.png
	./ffmpeg -i EAC/0000.png -vf "v360=eac:e" scale=7680:3840 EAC_to_ERP/0000.png
	./ffmpeg -i input.png -s 4928x2328 -pix_fmt yuv420p output.yuv

 [for more information about ffmpeg](https://ffmpeg.org/ffmpeg.html)

### 2.360tools
360tools_conv is used for conversion between different projection formats  
360tools_metric implements various quality metrics for vr quality evaluation

#### Example usage
    ./360tools_conv -i [file] -o [file] -f [int] -w [int] -h [int] -l [int] -m [int] -x [int] -y [int]  
    ./360tools_conv -i input_3840x1920.yuv -w 3840 -h 1920 -x 1 -o output_isp_4268x2016.yuv -l 4268 -m 2016 -y 1 -f 5 -n 1    
    
    -f  "convfmt"
	 1:  ERP  to ISP
	 2:  ISP  to ERP
     3:  ERP  to CMP
     4:  CMP  to ERP
     5:  ERP  to OHP
     6:  OHP  to ERP
     7:  ERP  to TSP
     8:  TSP  to ERP
     9:  ERP  to SSP
     10: SSP  to ERP
[for more information about 360tools](https://github.com/Samsung/360tools)

### 3.py360convert
#### install
	pip install py360convert
	
#### usage
	e2c(e_img, face_w=256, mode='bilinear', cube_format='dice') # Convert the given equirectangular to cubemap
	e2p(e_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear') # Take perspective image from given equirectangular
	c2e(cubemap, h, w, cube_format='dice') # Convert the given cubemap to equirectangular
## Dependencies
+ anaconda 4.10.3
+ python 3.7
+ pytorch 1.11.0
+ numpy 1.21
+ skimage 0.19
+ imageio 2.16
+ matplotlib 3.5
+ tqdm 4.63
+ cv2 4.5
+ cuda 11.3
+ nvcc 8.2

## SR model
1. [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)
2. [RCAN](https://github.com/sanghyun-son/EDSR-PyTorch)
3. [SwinIR](https://github.com/JingyunLiang/SwinIR)
