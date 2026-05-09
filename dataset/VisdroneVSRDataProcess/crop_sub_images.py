# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import re
import sys
from multiprocessing import Pool

import cv2
import mmcv
import mmengine
import numpy as np
from mmagic.datasets.transforms import MATLABLikeResize, blur_kernels
from skimage import img_as_float
from skimage.io import imread, imsave
import math
import lmdb

def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is smaller
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    iscenter=opt['center']                  #是否执行中心裁切
    thresh_size = opt['thresh_size']
    sequence, img_name = re.split(r'[\\/]', path)[-2:]
    img_name, extension = osp.splitext(osp.basename(path))
    extension = ".png"
    img = mmcv.imread(path, flag='unchanged')

    if img.ndim == 2 or img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')
    if iscenter:
        #h w 480 640
        x=(h-crop_size[0])//2
        y=(w-crop_size[1])//2
        cropped_img = img[x:x + crop_size[0], y:y + crop_size[1], ...]
        # sub_folder = osp.join(opt['save_folder'],
        #                       f'{sequence}_s{index:03d}')
        sub_folder = os.path.join(opt['save_folder'],sequence)
        mmengine.mkdir_or_exist(sub_folder)
        if opt['compression_level']>0:
            cv2.imwrite(
                osp.join(sub_folder, f'{img_name}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
        else:
            cv2.imwrite(
                osp.join(sub_folder, f'{img_name}{extension}'), cropped_img)
    else:
        #按照步长裁切
        h_space = np.arange(0, h - crop_size + 1, step)
        if h - (h_space[-1] + crop_size) > thresh_size:
            h_space = np.append(h_space, h - crop_size)
        w_space = np.arange(0, w - crop_size + 1, step)
        if w - (w_space[-1] + crop_size) > thresh_size:
            w_space = np.append(w_space, w - crop_size)

        index = 0
        for x in h_space:
            for y in w_space:
                index += 1
                cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
                sub_folder = osp.join(opt['save_folder'],
                                      f'{sequence}_s{index:03d}')
                mmengine.mkdir_or_exist(sub_folder)
                if opt['compression_level'] > 0:
                    cv2.imwrite(
                        osp.join(sub_folder, f'{img_name}{extension}'), cropped_img,
                        [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
                else:
                    cv2.imwrite(
                        osp.join(sub_folder, f'{img_name}{extension}'), cropped_img)
    process_info = f'Processing {img_name} ...'
    return process_info


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    #创建文件夹
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        # sys.exit(1)

    img_list = list(mmengine.scandir(input_folder, recursive=True))

    img_list = [osp.join(input_folder, v) for v in img_list]
    prog_bar = mmengine.ProgressBar(len(img_list))
    pool = Pool(opt['n_thread'])
    for path in img_list:
        # worker(path,opt)
        pool.apply_async(
            worker, args=(path, opt), callback=lambda arg: prog_bar.update())
    pool.close()
    pool.join()
    print('All processes done.')

def imresize(img_path, output_path, scale=None, output_shape=None):
    """Resize the image using MATLAB-like downsampling.

    Args:
        img_path (str): Input image path.
        output_path (str): Output image path.
        scale (float | None, optional): The scale factor of the resize
            operation. If None, it will be determined by output_shape.
            Default: None.
        output_shape (tuple(int) | None, optional): The size of the output
            image. If None, it will be determined by scale. Note that if
            scale is provided, output_shape will not be used.
            Default: None.
    """

    matlab_resize = MATLABLikeResize(
        keys=['data'], scale=scale, output_shape=output_shape)
    img = imread(img_path)
    img = img_as_float(img)
    data = {'data': img}
    output = matlab_resize(data)['data']
    output = np.clip(output, 0.0, 1.0) * 255
    output = np.around(output).astype(np.uint8)
    imsave(output_path, output)


def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.

    Args:
        kernel_size (int): The size of the kernel.

    Returns:
        xy_grid (np.ndarray): stacked xy coordinates with shape
            (kernel_size, kernel_size, 2).
    """
    range_ = np.arange(-(kernel_size - 1.) / 2., (kernel_size - 1.) / 2. + 1.)
    x_grid, y_grid = np.meshgrid(range_, range_)
    xy_grid = np.hstack((x_grid.reshape((kernel_size * kernel_size, 1)),
                         y_grid.reshape(kernel_size * kernel_size,
                                        1))).reshape(kernel_size, kernel_size,
                                                     2)

    return xy_grid


def bd_downsample(img_path, output_path, sigma=1.6, scale=4):
    """Downsampling using BD degradation(Gaussian blurring and downsampling).

    Args:
        img_path (str): Input image path.
        output_path (str): Output image path.
        sigma (float): The sigma of Gaussian blurring kernel. Default: 1.6.
        scale (int): The scale factor of the downsampling. Default: 4.
    """

    # Gaussian blurring
    kernelsize = math.ceil(sigma * 3) * 2 + 2
    kernel = blur_kernels.bivariate_gaussian(
        kernelsize, sigma, grid=mesh_grid(kernelsize))
    img = cv2.imread(img_path)
    img = img_as_float(img)
    output = cv2.filter2D(
        img,
        -1,
        kernel,
        anchor=((kernelsize - 1) // 2, (kernelsize - 1) // 2),
        borderType=cv2.BORDER_REPLICATE)

    # downsampling
    output = output[int(scale / 2) - 1:-int(scale / 2) + 1:scale,
                    int(scale / 2) - 1:-int(scale / 2) + 1:scale, :]

    output = np.clip(output, 0.0, 1.0) * 255
    output = output.astype(np.float32)
    output = np.floor(output + 0.5)
    cv2.imwrite(output_path, output)


def worker_down(clip_path, args):
    """Worker for each process.

    Args:
        clip_name (str): Path of the clip.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """

    gt_dir = osp.join(args['root_dir'], 'GT', clip_path)
    bi_dir = osp.join(args['root_dir'], 'BIx4', clip_path)
    bd_dir = osp.join(args['root_dir'], 'BDx4', clip_path)
    mmengine.utils.mkdir_or_exist(bi_dir)
    mmengine.utils.mkdir_or_exist(bd_dir)
    # imresize(osp.join(gt_dir, img), osp.join(bi_dir, img), scale=1 / 4)
    # bd_downsample(osp.join(gt_dir, img), osp.join(bd_dir, img))

    img_list = sorted(os.listdir(gt_dir))
    for img in img_list:
        imresize(osp.join(gt_dir, img), osp.join(bi_dir, img), scale=1 / 4)
        bd_downsample(osp.join(gt_dir, img), osp.join(bd_dir, img))

    process_info = f'Processing {clip_path} ...'
    return process_info

def downsample_images(args):
    """Downsample images."""
    clip_list = []
    gt_dir = os.path.join(args['root_dir'],"GT")
    sequence_list = sorted(os.listdir(gt_dir))
    # for sequence in sequence_list:
    #     sequence_root = osp.join(gt_dir, sequence)
    #     clip_list.extend(
    #         [osp.join(sequence, i) for i in sorted(sequence_root)])
    #所有的图像文件路径
    prog_bar = mmengine.ProgressBar(len(clip_list))
    pool = Pool(args['n_thread'])
    for path in sequence_list:
        # worker_down(path,args)
        pool.apply_async(
            worker_down, args=(path, args), callback=lambda arg: prog_bar.update())

    pool.close()
    pool.join()
    print('All processes done.')

def make_lmdb(data_path, lmdb_path, batch=5000, compress_level=0):
    """Create lmdb for the visdrone dataset.

    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records 1)image name (with extension),
    2)image shape, and 3)compression level, separated by a white space.

    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1

    We use the image name without extension as the lmdb key.

    Args:
        mode (str): REDS dataset mode. Choices: ['train_sharp', 'train_blur',
            'train_blur_comp', 'train_sharp_bicubic', 'train_blur_bicubic'].
            They are used to identify different reds dataset for different
            tasks. Specifically:
            'train_sharp': GT frames;
            'train_blur': Blur frames for deblur task.
            'train_blur_comp': Blur and compressed frames for deblur and
                compression task.
            'train_sharp_bicubic': Bicubic downsampled sharp frames for SR
                task.
            'train_blur_bicubic': Bicubic downsampled blur frames for SR task.
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
    """
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    # if mode in ['train_sharp', 'train_blur', 'train_blur_comp']:
    #     h_dst, w_dst = 720, 1280
    # else:
    #     h_dst, w_dst = 180, 320

    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        # sys.exit(1)

    print('Reading image path list ...')
    img_path_list = sorted(
        list(mmengine.scandir(data_path, suffix='png', recursive=True)))
    keys = []
    for img_path in img_path_list:
        parts = re.split(r'[\\/]', img_path)
        type = parts[-3]
        folder = parts[-2]
        img_name = parts[-1].split('.png')[0]
        keys.append(type + "_" + folder + '_' + img_name)  # example: GT_000_00000

    # create lmdb environment
    # obtain data size for one image
    img = mmcv.imread(osp.join(data_path, img_path_list[-1]), flag='unchanged')
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    data_size_per_img = img_byte.nbytes
    print('Data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(img_path_list)
    env = lmdb.open(lmdb_path, map_size=data_size * 10)

    # write data to lmdb
    pbar = mmengine.ProgressBar(len(img_path_list))
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update()
        key_byte = key.encode('ascii')
        img = mmcv.imread(osp.join(data_path, path), flag='unchanged')
        h, w, c = img.shape

        if compress_level>0:
            _, img_byte = cv2.imencode(
            '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        else:
            _, img_byte = cv2.imencode(
            '.png', img)

        # _, img_byte = cv2.imencode(
        #     '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        # assert h == h_dst and w == w_dst and c == 3, (
        #     f'Wrong shape ({h, w}), should be ({h_dst, w_dst}).')
        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png\n')
        # txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')

def generate_anno_file(root_path, file_name='meta_info_VisDrone_train.txt'):
    """Generate anno file for REDS datasets from the ground-truth folder.

    Args:
        root_path (str): Root path for visdrone datasets.
    """
    print(f'Generate annotation files {file_name}...')
    # files=[]
    fileDirs=os.path.join(root_path,"GT")
    files = sorted(list(mmengine.scandir(fileDirs, recursive=True)))
    # seqs=sorted(os.listdir(fileDirs))
    # for seq in seqs:
    #     seqdir=os.path.join(fileDirs,seq)
    #     filenames=[os.path.join(seq,i) for i in sorted(os.listdir(seqdir))]
    #     files+=filenames
    
    txt_file = osp.join(root_path, file_name)
    mmengine.utils.mkdir_or_exist(osp.dirname(txt_file))
    #GT path 
    with open(txt_file, 'w') as f:
        for line in files:
            f.write(f"{line}\n")
        # for i in range(270):
        #     for j in range(100):
        #         f.write(f'{i:03d}/{j:08d}.png (720, 1280, 3)\n')

def main_extract_subimages(args):
    """A multi-thread tool to crop large images to sub-images for faster IO.

    It is used for visdrone dataset.

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9.
            A higher value means a smaller size and longer compression time.
            Use 0 for faster CPU decompression. Default: 3, same in cv2.

        scales (list[int]): The downsampling factors corresponding to the
            LR folders you want to process.
            Default: [].
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        For example, if scales = [4], there are two folders to be processed:
            train_sharp
            train_sharp_bicubic/X4
        After process, each sub_folder should have the same number of
        subimages. You can also specify scales by modifying the argument
        'scales'. Remember to modify opt configurations according to your
        settings.
    """

    opt = {}
    opt['n_thread'] = args.n_thread
    opt['compression_level'] = args.compression_level
    for sp in args.split:
        # HR images
        opt['input_folder'] = osp.join(args.data_root, sp)
        opt['save_folder'] = osp.join(args.save_root, sp,"GT")
        opt['crop_size'] = args.crop_size
        opt['step'] = args.step
        opt['center'] = args.center
        opt['thresh_size'] = args.thresh_size
        opt['split'] = args.split
        opt['root_dir'] = osp.join(args.save_root, sp)
        opt['n_thread'] = args.n_thread
        extract_subimages(opt)
        downsample_images(opt)
        #################################
        # generate image list anno file
        generate_anno_file(opt['root_dir'],f'meta_info_VisDrone_{sp}.txt')
        # create lmdb file
        if args.make_lmdb:
            lmdb_path = osp.join(args.save_root, f'{sp}.lmdb')
            make_lmdb(data_path=opt['root_dir'], lmdb_path=lmdb_path)
        
        # for scale in args.scales:
        #     opt['input_folder'] = osp.join(opt['save_folder'])                  #输入
        #     opt['save_folder'] = osp.join(args.data_root,                       #保存路径
        #                                   f'train_sharp_bicubic/X{scale}_sub')
        #     opt['crop_size'] = args.crop_size // scale
        #     opt['step'] = args.step // scale
        #     opt['thresh_size'] = args.thresh_size // scale
        #     extract_subimages(opt)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess visdrone datasets')
    parser.add_argument('-d','--data-root', type=str, help='root path for visdrone dataset')
    parser.add_argument('-sr', '--save-root', type=str, help='root path for save visdrone sub dataset')
    parser.add_argument("-sp",'--split',nargs='*', default=['val','test','train'], help='split for Visdrone')
    parser.add_argument(
        '-s','--scales', nargs='*', default=[4], help='scale factor list')
    parser.add_argument(
        '--crop-size',
        nargs='?',
        default=(480,640),
        help='cropped size for HR images')
    parser.add_argument(
        '--center',
        type=bool,
        nargs='?',
        default=True,
        help='crop type for HR images step or center')
    parser.add_argument(
        '--step',
        type=int,
        nargs='?',
        default=240,
        help='step size for HR images')
    parser.add_argument(
        '--thresh-size',
        type=int,
        nargs='?',
        default=0,
        help='threshold size for HR images')
    parser.add_argument(
        '--compression-level',
        type=int,
        nargs='?',
        default=0,
        help='compression level when save png images')
    parser.add_argument(
        '--n-thread',
        type=int,
        nargs='?',
        default=20,
        help='thread number when using multiprocessing')
    parser.add_argument(
        '--make-lmdb', action='store_true',default=True, help='create lmdb files')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # extract subimages
    args.scales = [int(v) for v in args.scales]
    main_extract_subimages(args)
    print(f"successfully dealed!")
