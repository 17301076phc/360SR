import os
from importlib import import_module
#from dataloader import MSDataLoader
import cv2
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import torch.utils.data as data

# This is a simple wrapper function for ConcatDataset
from data import common


class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)


class Mydata(data.Dataset):
    def __init__(self,args,dir,istrain = True):
        path1 = os.listdir(dir)
        self.scale = args.scale
        self.name = "mydata"
        self.idx_scale = 0
        self.args = args
        self.train = istrain
        self.dir = dir
        self.len = len(path1)
        if istrain:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * self.len
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    def __getitem__(self, idx):
        # if self.train:
        if self.train:
            item= idx % self.len
        else:
            item=idx

        filename = self.dir+ "/"+ f'{str(item+1).zfill(4)}.png'
        # else:
        #     filename = self.dir + "/" + f'{str(item).zfill(8)}.png'
        # print(filename)
        try:
            hr = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
            lr = cv2.resize(hr, (hr.shape[1]//8,hr.shape[0]//8), interpolation=cv2.INTER_CUBIC)  # scale =8
            # lr = cv2.resize(hr, (960, 540), interpolation=cv2.INTER_CUBIC)  # scale =2
            pair = self.get_patch(lr, hr)
            pair = common.set_channel(*pair, n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

            return pair_t[0], pair_t[1], filename
        except:
            print(filename)

    def get_patch(self, lr, hr):
        scale = 8
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                input_large=False
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        # else:
        #     ih, iw = lr.shape[:2]
        #     print(lr.shape[:2])
        #     hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def __len__(self):
        if self.train:
            return self.len*self.repeat
        return self.len

    def set_scale(self, idx_scale):
        if not False:
            self.idx_scale = idx_scale
        # else:
        #     self.idx_scale = random.randint(0, len(self.scale) - 1)



class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            b1 = Mydata(args, "DIV2K_train_HR")
            datasets = [b1]
            # for d in args.data_train:
            #     module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
            #     m = import_module('data.' + module_name.lower())
            #     datasets.append(getattr(m, module_name)(args, name=d))
            print(datasets)
            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
        # testset = Mydata(args, "test_div2k", istrain=False)
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
