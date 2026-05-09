'''
按照顺序重命名图像名称
'''
import os
import argparse
from multiprocessing import Pool

def work(seq):
    seqqs = os.listdir(seq)
    seqqs.sort()
    seqs=[os.path.join(seq,file) for file in os.listdir(seq)]
    seqs.sort()
    #改名
    for seq_I in seqqs:
        seq_f = os.path.join(seq,seq_I)
        files = [os.path.join(seq_f, file) for file in os.listdir(seq_f)]
        files.sort()
        for i in range(len(files)):
            old_file=files[i]
            dir=os.path.dirname(old_file)
            newfile=os.path.join(dir,f"{i:05d}.jpg")
            os.rename(old_file,newfile)
            ccc=0
    #更改文件夹的名称
    for i in range(len(seqs)):
        old_file = seqs[i]
        dir = os.path.dirname(old_file)
        newfile = os.path.join(dir, f"{i:03d}")
        os.rename(old_file, newfile)
        ccc=0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess rename Visdrone datasets')
    parser.add_argument("-d",'--data-root', type=str, help='root path for Visdrone')
    parser.add_argument("-s",'--split',nargs='*', default=['train','val','test'], help='split for Visdrone')
    parser.add_argument(
        '--n-thread',
        type=int,
        nargs='?',
        default=20,
        help='thread number when using multiprocessing')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    #splits
    root=args.data_root
    thread=args.n_thread
    pool=Pool(thread)
    for sp in args.split:
        dirpath = os.path.join(root,sp)
        work(dirpath)
        # seq_dirs = [os.path.join(dirpath,dir) for dir in os.listdir(dirpath)]
        # seq_dirs.sort()
        # for seq in seq_dirs:
        #     if not os.path.isdir(seq):
        #         continue
        #     work(seq)
        #     # pool.apply_async(work,(seq,))
        ccc=0
    pool.close()
    pool.join()
    print(f"successfully done!")
    # # extract subimages
    # args.scales = [int(v) for v in args.scales]
    # main_extract_subimages(args)