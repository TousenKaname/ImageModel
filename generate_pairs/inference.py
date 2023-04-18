import torch
from ControlNet.interface import Interface
from PIL import Image
import numpy as np
import matplotlib
from torch.autograd import Variable
from warp_pic import warp_pic

def to_numpy_image(x):
    return (x.detach().permute(0,2,3,1).numpy()*255).astype(np.uint8)

prompt = "real clothes, coloring, white background, succinct"
a_prompt = "best quality, extremely detailed"
n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
num_samples = 1 
image_resolution = 512
ddim_steps = 20
guess_mode = False
strength = 1
scale = 9
seed = 369442106
eta = 0.0
low_threshold = 100
high_threshold = 200

img_path = 'data/'
img_name = []
result_name_list = []
warp_name = []
for i in range(1, 11):
    img_name.append(img_path+"{:03d}".format(i)+".jpg")

np_img_list = []
for name in img_name:
    img = Image.open(name)
    np_img = np.array(img)
    np_img_list.append(np_img)
    # results[0] canny_result 
    # results[1] result
    results = Interface.process(np_img, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
    result_name = "result_" + name
    result_name_list.append("result_" + name)
    matplotlib.image.imsave(result_name, results[1])

for name in result_name_list:
    source_image = Image.open(name).convert(mode = 'RGB')
    source_image = np.array(source_image).astype('float32')
    source_image = np.expand_dims(source_image.swapaxes(2, 1).swapaxes(1, 0), 0)
    source_image = Variable(torch.from_numpy(source_image))

    warp_name = "warped_" + name
    warp_pic(source_image, warp_name)
