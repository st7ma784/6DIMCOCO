import os
from argparse import ArgumentParser
from typing import Any, Callable, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,ImageFolder

from pl_bolts.datasets import UnlabeledImagenet
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg
from pysmartdl import SmartDL
from pathlib import Path
if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
import torchvision.transforms as transforms
import io
import asyncio
import random
from PIL import Image
outQ = asyncio.Queue() 
@under_review()
class ImagenetDataModule(LightningDataModule):
    name = "imagenet"
    def __init__(
        self,
        data_dir: str,
        meta_dir: Optional[str] = None,
        num_imgs_per_val_class: int = 50,
        image_size: int = 224,
        num_workers: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            meta_dir: path to meta.bin file
            num_imgs_per_val_class: how many images per class for the validation set
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use ImageNet dataset loaded from `torchvision` which is not installed yet."
            )

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.meta_dir = meta_dir
        self.num_imgs_per_val_class = num_imgs_per_val_class
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_samples = 1281167 - self.num_imgs_per_val_class * self.num_classes

    @property
    def num_classes(self) -> int:
        """
        Return:
            1000
        """
        return 1000
    def setup(self, stage: Optional[str] = None) -> None:
        
        
        train_transform, test_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),]),transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),])
        
        train_dir = os.path.join(self.data_dir, 'ImageNet_224', 'train')
        test_dir = os.path.join(self.data_dir, 'ImageNet_224', 'train')
        self.train_dataset = ImageFolder(train_dir, transform=train_transform)
        self.test_dataset = ImageFolder(test_dir, transform=test_transform)
        
        #return train_dataset, test_dataset



    def prepare_data(self) -> None:
        '''Download and prepare data'''
        # download dataset with pysmartDL
        # download dataset
        urls=["https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
        "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
        "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar",
        "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t3.tar.gz"
        ]
        
        data_path = self.data_dir
        #check if files exist
        downloads=[]
        for url in urls:
            #get filename
            filename = url.split('/')[-1]
            #check if file exists
            #else download
            if os.path.exists(os.path.join(data_path,filename)):
                print("File {} exists".format(filename))
                #touch file
                path=Path(os.path.join(data_path,filename))
                path.touch()
            else:
                obj=SmartDL(url,os.path.join(data_path,filename),progress_bar=False, verify=False)
                obj.start() 
                downloads.append(obj)
        #check if all files are downloaded
        for obj in downloads:
            if not obj.isFinished():
                obj.wait()
                #now touch each file
                path=Path(obj.get_dest())
                path.touch()
        
        #rename devkit
        original=os.path.join(data_path,"ILSVRC2012_devkit_t3.tar.gz")
        #move to os.path.join(data_path,"devkit.tar.gz")
        os.cmd("mv {} {}".format(original,os.path.join(data_path,"devkit.tar.gz")))
        
        # make train directory and unzip files
        os.makedirs(os.path.join(data_path,"ImageNet-2012","train"),exist_ok=True)
        #extract files
        os.cmd("tar --touch -xvf {} -C {}".format(os.path.join(data_path,"ILSVRC2012_img_train.tar"),os.path.join(data_path,"ImageNet-2012","train")))
        #for file in train/*.tar;do
        for file in os.listdir(os.path.join(data_path,"ImageNet-2012","train")):
            #extract
            if file.endswith(".tar"):
                #extract tar
                #make directory
                dirname = file.split('/')[-1].split('.')[0]
                os.makedirs(os.path.join(data_path,"ImageNet-2012","train",dirname),exist_ok=True)
                os.cmd("tar --touch -xvf {} -C {}".format(file.get_dest(),os.path.join(data_path,dirname)))
        
                       
        '''
        for file in *.tar;do
        filename=$(basename $file .tar)
        if [ ! -d $filename ];then
            mkdir -pv $filename
        else
            rm -rf $filename
        fi
        tar --touch -xvf $file -C $filename
        rm $file
        done
        '''
        os.makedirs(os.path.join(data_path,"ImageNet-2012","val"),exist_ok=True)
        os.cmd("tar --touch -xvf {} -C {}".format(os.path.join(data_path,"ILSVRC2012_img_val.tar"),os.path.join(data_path,"ImageNet-2012","val")))
        
        for file in os.listdir(os.path.join(data_path,"ImageNet-2012","val")):
            if file.endswith(".tar"):
                dirname=file.split('/')[-1].split('.')[0]
                os.makedirs(os.path.join(data_path,"ImageNet-2012","val",dirname),exist_ok=True)
                os.cmd("tar --touch -xvf {} -C {}".format(file.get_dest(),os.path.join(data_path,dirname)))
        #copy APCT-master/prepare/val_prepare.sh to ImageNet-2012/prepare
        os.cmd("cp {} {}".format(os.path.join("APCT-master","prepare","val_prepare.sh"),os.path.join(data_path,"ImageNet-2012","prepare")))
        os.cmd("cd {} && bash val_prepare.sh {}".format(os.path.join(data_path,"ImageNet-2012","prepare"), os.path.join(data_path,"ImageNet-2012","val")))
        '''
        cd $$data_path/ImageNet-2012/prepare
        bash val_prepare.sh $data_path/ImageNet-2012/val
        ''' 
        #now lets resize the images

        '''
        # resize dataset
        python $file_path/prepare/resize.py --data_path $data_path
            
        '''
        #resize dataset
        self.fast_resize(os.path.join(data_path,"ImageNet-2012","train"))
        self.fast_resize(os.path.join(data_path,"ImageNet-2012","val"))
        
    def fast_resize(self,dir):
        '''resize all images in a directory to 224x224'''
        #we will use PIL to resize images to 224x224 and save them in a new directory
        #create new directory
        new_dir = dir+"_224"
        os.makedirs(new_dir,exist_ok=True)
        #now create async loop, and resize images
        loop = asyncio.get_event_loop()

        loop.run_until_complete(self.run(dir,new_dir,size=224))
        loop.close()
    async def run(self,dir,new_dir,size=224):
        queue = asyncio.Queue()
        consumer = asyncio.ensure_future(self.consume(queue, new_dir, size))
        await self.produce(queue, dir)
        await queue.join()
        consumer.cancel()

    async def produce(self,queue, dir):
        #nested directory to produce images from,
        #walk through directory and produce images
        for dir,file in os.walk(dir):
            if file.endswith(".JPEG"):
                #produce image
                await queue.put(os.path.join(dir,file))
                #await asyncio.sleep(random.random())

    async def consume(self, queue, new_dir, size=224):
        while True:
            print('consume ')
            imagepath = await queue.get()
            #open image
            buffer=io.BytesIO()
            #await reading into buffer
            await self.read(imagepath,buffer,size)
          
            #save image
            await self.save(buffer,os.path.join(new_dir,imagepath))   
            #await asyncio.sleep(random.random())
            await buffer.close()
            queue.task_done()

    async def read(self,imagepath,buffer,size):
                
        Image.open(imagepath).resize((size, size)).save(buffer, "JPEG")
    async def save(self,buffer,dest):
        with open(dest, "wb") as f:
            f.write(buffer.getbuffer())
         







if __name__ == '__main__':
    dm = ImagenetDataModule()
    print(dm.test_dataloader()[0])
    print(dm.val_dataloader()[0])
    print(dm.train_dataloader()[0])

    '''
    from torchvision.transforms import *
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder
from torch.utils.data import DataLoader
from config import *
import os

MNIST_MEAN_STD = (0.1307,), (0.3081,)
CIAFR10_MEAN_STD = [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)]
CIAFR100_MEAN_STD = [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)]
IMAGENET_MEAN_STD = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]


def set_dataloader(args, datasets=None):
    if datasets is None:
        train_dataset, val_dataset = set_dataset(args)
    else:
        train_dataset, val_dataset = datasets
    train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                prefetch_factor=4,
                persistent_workers=True)
    val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                prefetch_factor=4,
                persistent_workers=True)
    return train_loader, val_loader


def set_dataset(args):
    train_transform, test_transform = set_transforms(args)
    if args.dataset.lower() == 'mnist':
        train_dataset = MNIST(DATA_PATH, train=True, transform=train_transform, download=True)
        test_dataset = MNIST(DATA_PATH, train=False, transform=test_transform, download=True)
    elif args.dataset.lower() == 'cifar10':
        train_dataset = CIFAR10(DATA_PATH, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10(DATA_PATH, train=False, transform=test_transform, download=True)
    elif args.dataset.lower() == 'cifar100':
        train_dataset = CIFAR100(DATA_PATH, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR100(DATA_PATH, train=False, transform=test_transform, download=True)
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(DATA_PATH, 'ImageNet-sz', str(args.data_size), 'train')
        test_dir = os.path.join(DATA_PATH, 'ImageNet-sz', str(args.data_size), 'train')
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        test_dataset = ImageFolder(test_dir, transform=test_transform)
    else:
        raise NameError('No dataset named %s' % args.dataset)
    return train_dataset, test_dataset


def set_transforms(args):
    if args.dataset.lower() == 'mnist':
        train_composed = [RandomCrop(32, padding=4), ToTensor()]
        test_composed = [ToTensor()]
    elif args.dataset.lower() in ['cifar10', 'cifra100']:
        train_composed = [RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor()]
        test_composed = [transforms.ToTensor()]
    elif args.dataset.lower() == 'imagenet':
        train_composed = [ToTensor(), RandomResizedCrop((args.crop_size, args.crop_size)), RandomHorizontalFlip()]
        test_composed = [ToTensor(), RandomResizedCrop((args.crop_size, args.crop_size))]
    else:
        raise NameError('No dataset named' % args.dataset)
    return Compose(train_composed), Compose(test_composed)


def set_mean_sed(args):
    if args.dataset.lower() == 'cifar10':
        mean, std = CIAFR10_MEAN_STD
    elif args.dataset.lower() == 'cifar100':
        mean, std = CIAFR100_MEAN_STD
    elif args.dataset.lower() == 'mnist':
        mean, std = MNIST_MEAN_STD
    elif args.dataset.lower() == 'imagenet':
        mean, std = IMAGENET_MEAN_STD
    else:
        raise NameError()
    return mean, std

'''