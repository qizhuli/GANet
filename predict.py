from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from tqdm import tqdm
import os
import torch
import torch.nn.parallel
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import warnings
warnings.simplefilter("ignore")


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch GANet Example")
    parser.add_argument(
        "--crop_height", type=int, required=True, help="crop height"
    )
    parser.add_argument(
        "--crop_width", type=int, required=True, help="crop width"
    )
    parser.add_argument(
        "--max_disp", type=int, default=192, help="max disp"
    )
    parser.add_argument(
        "--resume", type=str, default="", help="resume from saved model"
    )
    parser.add_argument(
        "--cuda", type=bool, default=True, help="use cuda?"
    )
    parser.add_argument(
        "--kitti", type=int, default=0, help="kitti dataset? Default=False"
    )
    parser.add_argument(
        "--kitti2015", type=int, default=0, help="kitti 2015? Default=False"
    )
    parser.add_argument(
        "--cityscapes", type=int, default=0, help="Cityscapes? Default=False"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="data root"
    )
    parser.add_argument(
        "--test_list", type=str, required=True, help="training list"
    )
    parser.add_argument(
        "--save_path", type=str, default="./result/", help="location to save result"
    )
    parser.add_argument(
        "--model", type=str, default="GANet_deep", help="model to train"
    )
    parser.add_argument("--local_rank", type=int, default=0)

    opt = parser.parse_args()
    print(opt)
    return opt


def test_transform(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], "float32")
        temp_data[:, crop_height - h : crop_height, crop_width - w : crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[
            :, start_y : start_y + crop_height, start_x : start_x + crop_width
        ]
    left = np.ones([1, 3, crop_height, crop_width], "float32")
    left[0, :, :, :] = temp_data[0:3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], "float32")
    right[0, :, :, :] = temp_data[3:6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w


def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], "float32")
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]
    # r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data


def test(model, leftname, rightname, savename, crop_height, crop_width, cuda):
    # count=0

    input1, input2, height, width = test_transform(
        load_data(leftname, rightname), crop_height, crop_width
    )

    input1 = Variable(input1, requires_grad=False)
    input2 = Variable(input2, requires_grad=False)

    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)

    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= crop_height and width <= crop_width:
        temp = temp[
            0,
            crop_height - height : crop_height,
            crop_width - width : crop_width,
        ]
    else:
        temp = temp[0, :, :]
    skimage.io.imsave(savename, (temp * 256).astype("uint16"))


def main():
    opt = parse_args()

    if opt.model == "GANet11":
        from models.GANet11 import GANet
    elif opt.model == "GANet_deep":
        from models.GANet_deep import GANet
    else:
        raise Exception("No suitable model found...")

    cuda = opt.cuda
    # cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        assert cuda, "Distributed inference only works with GPUs"
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        dist.barrier()

    # torch.manual_seed(opt.seed)
    # if cuda:
    #     torch.cuda.manual_seed(opt.seed)
    # print('===> Loading datasets')

    print("===> Building model")
    model = GANet(opt.max_disp)

    if distributed:
        model.to('cuda')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.local_rank], output_device=[opt.local_rank])
    elif cuda:
        model = torch.nn.DataParallel(model)
        model.to('cuda')

    model.eval()

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint["state_dict"], strict=False)

        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Inference
    file_path = opt.data_path
    file_list = opt.test_list
    with open(file_list, "r") as f:
        filelist = [line.strip() for line in f.readlines()]

    if distributed:
        filelist = filelist[opt.local_rank::num_gpus]

    for current_file in tqdm(filelist):
        if opt.kitti2015:
            leftname = os.path.join(file_path, "image_2", current_file)
            rightname = os.path.join(file_path, "image_3", current_file)
            savename = os.path.join(opt.save_path, current_file)
        if opt.kitti:
            leftname = os.path.join(file_path, "colored_0", current_file)
            rightname = os.path.join(file_path, "colored_1", current_file)
            savename = os.path.join(opt.save_path, current_file)
        if opt.cityscapes:
            file_id = current_file.split("_leftImg8bit.png")[0]
            leftname = os.path.join(
                file_path, "leftImg8bit", file_id + "_leftImg8bit.png"
            )
            rightname = os.path.join(
                file_path, "rightImg8bit", file_id + "_rightImg8bit.png"
            )
            savename = os.path.join(
                opt.save_path, os.path.basename(file_id) + "_Disp16bit.png"
            )

        test(model, leftname, rightname, savename, opt.crop_height, opt.crop_width, cuda)


if __name__ == "__main__":
    main()
