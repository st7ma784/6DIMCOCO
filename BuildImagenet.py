import os
from argparse import ArgumentParser
from typing import Any, Callable, Optional
import asyncio
from pytorch_lightning import LightningDataModule
from multiprocessing import Pool
from pySmartDL import SmartDL
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import io
import asyncio
import random
import torch
import tarfile
from PIL import Image
outQ = asyncio.Queue() 
class ImagenetDataModule(LightningDataModule):
    name = "imagenet"
    def __init__(
        self,
        data_dir: str,
        meta_dir: Optional[str] = None,
        num_imgs_per_val_class: int = 50,
        image_size: int = 224,
        num_workers: int = 1,
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
            transforms.RandomHorizontalFlip(),transforms.ToTensor()]),transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),transforms.ToTensor()])
        
        train_dir = os.path.join(self.data_dir, 'ImageNet-2012', 'train')
        test_dir = os.path.join(self.data_dir, 'ImageNet-2012', 'val')
        self.train_dataset = ImageFolder(train_dir, transform=train_transform)
        self.test_dataset = ImageFolder(test_dir, transform=test_transform)
        
        #return train_dataset, test_dataset
    def train_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def test_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self) -> Any:
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def extract_tar(self,i):
        file,folder=i
        dirname=file.split('/')[-1].split('.')[0]
        os.makedirs(os.path.join(folder,dirname),exist_ok=True)
        #do "tar --touch -xvf {} -C {}".format(file,os.path.join(folder,dirname)))
        with tarfile.open(file,"r") as f:
            f.extractall(os.path.join(folder,dirname))
        
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
        urlsToGo=[]
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
                print("File {} does not exist".format(filename))
                if not filename=="ILSVRC2012_devkit_t3.tar.gz":

                    urlsToGo.append(url)
                else: # check if devkit exists
                    if not os.path.exists(os.path.join(data_path,"devkit.tar.gz")):
                        urlsToGo.append(url)
        if len(urlsToGo)>0:
            obj=SmartDL(urlsToGo,data_path,progress_bar=True,threads=16, verify=False)
            obj.start()                
        #rename devkit
        original=os.path.join(data_path,"ILSVRC2012_devkit_t3.tar.gz")
        #move to os.path.join(data_path,"devkit.tar.gz")
        os.system("cp {} {}".format(original,os.path.join(data_path,"devkit.tar.gz")))
        
        # make train directory and unzip files
        os.makedirs(os.path.join(data_path,"ImageNet-2012","train"),exist_ok=True)
        #extract files
        #for file in train/*.tar;do
        files= os.listdir(os.path.join(data_path,"ImageNet-2012","train"))
    
        files=list(filter(lambda x: x.endswith(".tar"),files))
        #if there are no folders in train, then extract
        if len(files)==0:
            os.system("tar --touch -xvf {} -C {}".format(os.path.join(data_path,"ILSVRC2012_img_train.tar"),os.path.join(data_path,"ImageNet-2012","train")))
            files= os.listdir(os.path.join(data_path,"ImageNet-2012","train"))
    
            files=list(filter(lambda x: x.endswith(".tar"),files))
            with Pool(16) as executor:
                executor.map(self.extract_tar,zip(files,[os.path.join(data_path,"ImageNet-2012","train")]*len(files)))

        #check if val directory exists
        os.makedirs(os.path.join(data_path,"ImageNet-2012","val"),exist_ok=True)

        files=os.listdir(os.path.join(data_path,"ImageNet-2012","val"))
        if len(files)==0:
            #os.system("tar --touch -xvf {} -C {}".format(os.path.join(data_path,"ILSVRC2012_img_val.tar"),os.path.join(data_path,"ImageNet-2012","val")))
            tarfile.open(os.path.join(data_path,"ILSVRC2012_img_val.tar"),"r").extractall(os.path.join(data_path,"ImageNet-2012","val"))
            #filter so that we only have tar files
            files=list(filter(lambda x: x.endswith(".tar"),os.listdir(os.path.join(data_path,"ImageNet-2012","val"))))
            #extract in a multithreaded way
            with Pool(16) as executor:
                executor.map(self.extract_tar,files,[os.path.join(data_path,"ImageNet-2012","val")]*len(files))
      
        if len(os.listdir(os.path.join(data_path,"ImageNet-2012","val"))) == 0:
            os.system("cp {} {}".format(os.path.join("APCT","prepare","val_prepare.sh"),os.path.join(data_path,"ImageNet-2012","val")))
            os.system("cd {} && bash val_prepare.sh {}".format(os.path.join(data_path,"ImageNet-2012","val"), os.path.join(data_path,"ImageNet-2012","val")))
        #do same for train 
        #extract files into folders like this
        '''
            for file in folder that ends with tar
                filename=$(basename $file .tar)
                if [ ! -d $filename ];then
                    mkdir -pv $filename
                else
                    rm -rf $filename
                fi
                tar --touch -xvf $file -C $filename
                rm $file
                
        '''
        asyncio.run(self.extract_files(os.path.join(data_path,"ImageNet-2012","train")))
       
        for file in os.listdir(os.path.join(data_path,"ImageNet-2012","train")):
            if file.endswith(".tar"):
                filename=file[:-4]
                if not os.path.exists(filename):
                    os.makedirs(filename,exist_ok=True)
                # else:
                #     #do os.system("rm -rf {}".format(filename)) with shutil.rmtree
                #     os.remove(filename)
                #do the following with tarfile
                #os.system("tar --touch -xvf {} -C {}".format(file,filename))
                #os.system("rm {}".format(file))
                with tarfile.open(os.path.join(data_path,"ImageNet-2012","train",file)) as t:
                    t.extractall(os.path.join(data_path,"ImageNet-2012","train",filename))
                try:
                    os.remove(os.path.join(data_path,"ImageNet-2012","train",file))
                except:
                    pass
    async def extract_files(self,dir):
        file_list = os.listdir(dir)
        #filter list by .tar
        file_list=list(filter(lambda x: x.endswith(".tar"),file_list))
        for file in file_list:
            await self.extract_tar(os.path.join(dir,file))

    async def extract_tar(self,file):
        filename=file[:-4]
        if not os.path.exists(filename):
            os.makedirs(filename,exist_ok=True)
        tarfile.open(file).extractall(filename)
        os.remove(file)

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
    #add arg parser
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Prepare ImageNet dataset')
    #add arguments data_path 
    parser.add_argument('--data_path', type=str, default='/datasets3', help='path to data directory')
    path=parser.parse_args().data_path
    dm = ImagenetDataModule(data_dir=path)
    dm.prepare_data()
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch[0].shape)
        break
    for batch in dm.val_dataloader():
        print(batch[0].shape)
        break
    for batch in dm.test_dataloader():
        print(batch[0].shape)
        break
#    print(dm.test_dataloader()[0])
#    print(dm.val_dataloader()[0])
#    print(dm.train_dataloader()[0])

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
