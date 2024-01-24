
from torchvision import transforms
from PIL import Image
import torch     
import os
import zipfile
import tarfile
import json as js
from pySmartDL import SmartDL
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset
from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import time
from pathlib import Path
from transformers import (
  BertTokenizerFast,
  AutoModel,
)
os.environ["TOKENIZERS_PARALLELISM"]='true'

# tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
# model = AutoModel.from_pretrained('ckiplab/gpt2-base-chinese')


prep=Compose([
        Resize(224, interpolation=Image.NEAREST),
        CenterCrop(224),
        #Note: the standard  lambda function here is not supported by pytorch lightning
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
T= transforms.Compose([transforms.Resize((224,224),interpolation=Image.NEAREST),transforms.ToTensor()])




class COCODataset(CocoCaptions):
    def __init__(self, root, annFile, tokenizer, instances=None,*args, **kwargs):
        #print('Loading COCO dataset')
        self.tokenizer=tokenizer
        if os.getenv('ISHEC',False):
            for root, dirs, files in os.walk(root):
                for file in files:
                    Path(os.path.join(root, file)).touch()
            Path(annFile).touch()

        if not os.path.exists(root):
            print("root does not exist {}".format(root))
      
        super().__init__(root, annFile, *args, **kwargs)
        if instances is not None and os.path.exists(instances):
            from pycocotools.coco import COCO
            self.instances=COCO(instances)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx: int):
        try:
            img, target= super().__getitem__(idx)
        except Exception as e:
            print(e)
            print('Error loading image:', idx)
            return None
        id=self.ids[idx]
        ids=self.instances.getAnnIds(imgIds=id)

        instance= self.instances.loadAnns(ids)
        try:
            i=instance[0].get("category_id",-100)
        except:
            i=-100
        if target==[]:
            print("ERROR ON IDX",idx) 
            print("ID",id)
            print("IDS",ids)
            print("INSTANCE",instance)
            return None
        target=torch.cat([self.tokenizer(
                    sent,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    padding = "max_length",
                    truncation=True,
                    return_attention_mask = False,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
        )['input_ids'] for sent in target[:5]],dim=0)
        #find last non-zero token
        #index of first 0 is findable with argmin
        indexes=torch.argmin(target,dim=1)
        EOT=indexes-1
        target[:,EOT]=self.tokenizer._tokenizer.vocab_size
        target[:,0]=self.tokenizer._tokenizer.vocab_size-1
        #We're going to do a manual fix and replace the 102 and 101 tokens with the actual tokens 
        #target=target.masked_fill(target==102,self.tokenizer._tokenizer.vocab_size)
        #target=target.masked_fill(target[:,0],self.tokenizer._tokenizer.vocab_siz-1)
        return img,target,i





class COCOCNDataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.',annotations=".", T=prep, batch_size=256):
        super().__init__()
        self.data_dir = Cache_dir
        if annotations is None:
            annotations=os.path.join(Cache_dir,"annotations")
        else:
            self.ann_dir=annotations
        self.batch_size = batch_size
        self.T=T
        self.splits={"train":["train2014","train2017"],"val":["val2014","val2017"],"test":["test2015"]}

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese',cache_dir=self.data_dir)
        path=self.tokenizer.save_pretrained(os.path.join(self.data_dir,"bert-base-chinese"))
        print("path",path)
        import json
        with open(os.path.join(self.data_dir,"bert-base-chinese","tokenizer.json")) as f:
            token_dict=json.load(f)
            print("token_dict",token_dict.keys())
            token_dict['model']['vocab'].update({'##😂': 102, '##😎': 101,'[CLS]':21126,'[SEP]':21127, '[SOT]':21128,'[EOT]':21129})        
        with open(os.path.join(self.data_dir,"bert-base-chinese","tokenizer.json"),'w') as f:
            json.dump(token_dict,f)
        #in the vocab.txt file, we are going to remove lines 101-103 and then add [CLS] and [SEP] to the end of the file
        # ta/bert-base-chinese/tokenizer_config.json', '/data/bert-base-chinese/special_tokens_map.json' '/data/bert-base-chinese/tokenizer.json')
        #edit the files 
        with open(os.path.join(self.data_dir,"bert-base-chinese","vocab.txt")) as f:
            lines=f.readlines()
            lines=lines[:101]+lines[104:]
            lines.append("[CLS]\n")
            lines.append("[SEP]\n")
        with open(os.path.join(self.data_dir,"bert-base-chinese","vocab.txt"),'w') as f:
            f.writelines(lines)

        with open(os.path.join(self.data_dir,"bert-base-chinese","special_tokens_map.json")) as f:
            token_dict=json.load(f)
            print("token_dict",token_dict)
            token_dict.update({"bos_token":"[SOT]","eos_token":"[EOT]"})
            
        self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(self.data_dir,"bert-base-chinese"),cache_dir=self.data_dir)
        print(self.tokenizer.backend_tokenizer.__dir__())
        print("tokenizer EOT/SEP token", self.tokenizer._tokenizer.token_to_id("[SEP]"))
        print("tokenizer EOT/SEP token", self.tokenizer._tokenizer.token_to_id("[CLS]"))
        print("tokenizer EOT/SEP token", self.tokenizer._tokenizer.token_to_id("[EOT]"))
        print("tokenizer EOT/SEP token", self.tokenizer._tokenizer.token_to_id("[SOT]"))
        print("tokenizer EOT/SEP token", self.tokenizer._tokenizer.id_to_token(21126))
        print("tokenizer EOT/SEP token", self.tokenizer._tokenizer.id_to_token(21127))
        vocab=self.tokenizer.get_vocab()

        assert vocab['[CLS]']==21126
        assert vocab['[SEP]']==21127
        assert vocab['##😂']==102
        assert vocab['##😎']==101
        
    def train_dataloader(self, B=None):
        if B is None:
            B=self.batch_size 
        return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True,drop_last=True)
    def val_dataloader(self, B=None):
        if B is None:
            B=self.batch_size
       
        return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=False, num_workers=2, prefetch_factor=2, pin_memory=True,drop_last=True)
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size


        return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=True, num_workers=1, prefetch_factor=1, pin_memory=True,drop_last=True)
    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir,"coco-cn-version1805v1.1")):
            self.download_data()
    def download_data(self):
        '''called only once and on 1 GPU'''
        # # download data
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir,exist_ok=True)
        if not os.path.exists(self.ann_dir):
            os.makedirs(self.ann_dir,exist_ok=True)
        urls=["http://lixirong.net/data/coco-cn/coco-cn-version1805v1.1.tar.gz"
                ]

        objs=[]
        for url in urls:
            #print("url:",url)
            name=str(url).split('/')[-1]
            location=self.data_dir # if name.startswith("annotations") else self.ann_dir
            #print("Location", location) #/Data/train2014.zip
            #time.sleep(5)
            #print('Downloading',url)
            obj=SmartDL(url,os.path.join(location,name),progress_bar=False, verify=False)
            obj.FileName=name
            if name.endswith(".zip"):
                name=name[:-4]
            if name.startswith("train"):
                self.splits['train'].append(name)
            elif name.startswith("val"):
                self.splits['val'].append(name)
            elif name.startswith("test"):
                self.splits['test'].append(name)
            if not os.path.exists(os.path.join(location,name)):
                print(os.path.join(location,name))
                objs.append(obj)
                obj.start(blocking=True )#There are security problems with Hostename 'images.cocodataset.org' and Certificate 'images.cocodataset.org' so we need to disable the SSL verification
        for obj in objs:
            while not obj.isFinished():
                time.sleep(5)
            if obj.isSuccessful():
                print("Downloaded: %s" % obj.get_dest())
            path = obj.get_dest()
            if path.endswith(".zip"):
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    try:
                        zip_ref.extractall(self.data_dir)
                    except Exception as e:
                        print(e)
                        print("Error extracting zip" ,path)
                        continue        
                    for root, dirs, files in os.walk(zip_ref.namelist()[0]):
                        for file in files:
                            Path(os.path.join(root, file)).touch()
            if path.endswith(".tar.gz"):
                with tarfile.open(path, 'r') as zip_ref:
                    try:
                        zip_ref.extractall(self.data_dir)
                    except Exception as e:
                        print(e)
                        print("Error extracting zip" ,path)
                        continue        
                    # for root, dirs, files in os.walk(self.data_dir):
                    #     for file in files:
                    #         # Path(os.path.join(root, file)).touch()
    def verify(self):
        datasets = ["coco-cn_train", "coco-cn_val","coco-cn_test"]

        imset = [set()
                for _ in range(len(datasets))]
        full = set()

        for i,dataset in enumerate(datasets):
            imsetfile = os.path.join(self.data_dir,"coco-cn-version1805v1.1","{}.txt".format(dataset))
            imset[i] = set(map(str.strip, open(imsetfile).readlines()))
            print("number of images in {} : {}".format(dataset,len(imset[i])))
            full = full.union(imset[i])
        print("number of images in full set: {}".format(len(full)))
        

        for i in range(len(datasets)-1):
            for j in range(i+1, len(datasets)):
                common = imset[i].intersection(imset[j])
                print("number of common images between {} and {} : {}".format(datasets[i],datasets[j],len(common)))
        manual_translated_file = os.path.join(self.data_dir,"coco-cn-version1805v1.1",'imageid.manually-translated-caption.txt')
        subset = [x.split('#')[0] for x in open(manual_translated_file).readlines()]
        subset = set(subset)
        print("number of images in manually-translated-caption : {}".format(len(subset)))
    


        tag_file = os.path.join(self.data_dir,"coco-cn-version1805v1.1",'imageid.human-written-tags.txt')
        subset = [x.split()[0] for x in open(tag_file).readlines()]
        subset = set(subset)
        print("number of images in human-written-tags : {}".format(len(subset)))

        print('verifying sentence files')
        with open(os.path.join(self.data_dir,"coco-cn-version1805v1.1",'imageid.human-written-caption.bosonseg.txt')) as f:
            lines1=f.readlines()
        with open(os.path.join(self.data_dir,"coco-cn-version1805v1.1",'imageid.human-written-caption.txt')) as f:
            lines2=f.readlines()

        imset_from_sent_file = set()
        for x,y in zip(lines1,lines2):
            assert(x.split()[0] == y.split()[0])
            img_id = x.split()[0].split('#')[0]
            assert(img_id in full)
            imset_from_sent_file.add(img_id)

        assert(len(imset_from_sent_file) == len(full))


    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        categorylist=[]
        if stage == 'fit' or stage is None:
            TrainSets=[]
            i=0            
            for version in self.splits['train']:
                #make COCO-style annfile
                read_images=set()

                annfile=os.path.join(self.ann_dir,"{}_{}.json".format('CNcaptions',version))
                cococnfile=os.path.join(self.data_dir,"coco-cn-version1805v1.1","imageid.human-written-caption.txt")
                otherfile=os.path.join(self.data_dir,"coco-cn-version1805v1.1","imageid.human-written-caption.bosonseg.txt")
                original_coco_file=os.path.join(self.ann_dir,"captions_{}.json".format(version))

                coco_json=js.load(open(original_coco_file))

                #read both files, 
                #if each line has the same version as the current version, then add to the annfile
                json=[]
                
                with open(cococnfile) as f:
                    lines1=f.readlines()
                    for line in lines1:
                        if line.split()[0].split('#')[0].split("_")[1]==version:
                            i+=1
                            caption=' '.join(line.split()[1:])
                            image_id=int(line.split()[0].split('#')[0].split('_')[-1])
                            read_images.add(image_id)
                            entry={
                                    'id':i,
                                    'image_id':image_id,
                                    'caption':caption
                                    }
                            # print("idx : {} , caption : {}".format(i,caption))
                            json.append(entry)
                print("i is now {}".format(i))
                with open(otherfile) as f:
                    lines2=f.readlines()
                    for line in lines2:
                        if line.split()[0].split('#')[0].split("_")[1]==version:
                            i+=1
                            caption=' '.join(line.split()[1:])
                            image_id=int(line.split()[0].split('#')[0].split('_')[-1])
                            read_images.add(image_id)
                            entry={
                                    'id':i,
                                    'image_id':int(line.split()[0].split('#')[0].split('_')[-1]),
                                    'caption':caption
                                    }
                            # print("idx : {} , image_id : {}, caption : {}".format(i,int(line.split()[0].split('#')[0].split('_')[-1]), caption))

                            json.append(entry)
                print("i is now {}".format(i))

                #save as annfile
                assert coco_json['annotations'] is not None
                coco_json['annotations']=json
                coco_json['images']=[image for image in coco_json['images'] if image['id'] in read_images]

                with open(annfile, 'w') as outfile:
                    js.dump(coco_json, outfile)
                
                #make COCO-style instancesfile
                original_coco_file=os.path.join(self.ann_dir,"instances_{}.json".format(version))
                coco_json=js.load(open(original_coco_file))
                json=[]
                with open(os.path.join(self.data_dir,"coco-cn-version1805v1.1","imageid.human-written-tags.txt")) as f:
                    lines=f.readlines()
                    for line in lines:
                        if line.split()[0].split('#')[0].split("_")[1]==version:
                            i+=1
                            categories=line.split()[1:]
                            cat_ids=[]
                            #image_id=int(line.split()[0].split('#')[0].split('_')[-1])
                            #read_images.add(image_id)
                            #we don't do this here to ensure we only have images with captions
                            for category in categories:
                                if category not in categorylist:
                                    categorylist.append(category)
                                cat_ids.append(categorylist.index(category))
                            for catid in cat_ids:
                                i+=1
                                entry={
                                        'id':i,
                                        'image_id':int(line.split()[0].split('#')[0].split('_')[-1]),
                                        'category_id':catid
                                        }
                                json.append(entry)
                print("i is now {}".format(i))

                coco_json['annotations']=json
                
                instancesfile=os.path.join(self.ann_dir,"{}_{}.json".format('CNinstances',version))
                coco_json["categories"]=[{'id':i,"name":category,"supercategory":category} for i,category in enumerate(categorylist)]
                coco_json['images']=[image for image in coco_json['images'] if image['id'] in read_images]

                with open(instancesfile, 'w') as outfile:
                    js.dump(coco_json, outfile)
                
                dset=COCODataset(root=os.path.join(self.data_dir,version), annFile=annfile, tokenizer=self.tokenizer,instances=instancesfile, transform=self.T)
                TrainSets.append(dset)
            self.train = ConcatDataset(TrainSets)

            ValSets=[]
            for version in self.splits['val']:
                read_images=set()

                #make COCO-style annfile
                annfile=os.path.join(self.ann_dir,"{}_{}.json".format('CNcaptions',version))
                cococnfile=os.path.join(self.data_dir,"coco-cn-version1805v1.1","imageid.human-written-caption.txt")
                otherfile=os.path.join(self.data_dir,"coco-cn-version1805v1.1","imageid.human-written-caption.bosonseg.txt")
                original_coco_file=os.path.join(self.ann_dir,"captions_{}.json".format(version))
                #read both files, 
                #if each line has the same version as the current version, then add to the annfile
                json=[]
                with open(cococnfile) as f:
                    lines1=f.readlines()
                    for line in lines1:
                        if line.split()[0].split('#')[0].split("_")[1]==version:
                            i+=1
                            image_id=int(line.split()[0].split('#')[0].split('_')[-1])
                            read_images.add(image_id)
                            entry={
                                    'id':i,
                                    'image_id':int(line.split()[0].split('#')[0].split('_')[-1]),
                                    'caption':' '.join(line.split()[1:])
                                    }
                            json.append(entry)
                with open(otherfile) as f:
                    lines2=f.readlines()
                    for line in lines2:
                        if line.split()[0].split('#')[0].split("_")[1]==version:
                            i+=1
                            image_id=int(line.split()[0].split('#')[0].split('_')[-1])
                            read_images.add(image_id)
                            entry={
                                    'id':i,
                                    'image_id':int(line.split()[0].split('#')[0].split('_')[-1]),
                                    'caption':' '.join(line.split()[1:])
                                    }
                            json.append(entry)
                original_json=js.load(open(original_coco_file))
                original_json['annotations']=json
                original_json['images']=[image for image in original_json['images'] if image['id'] in read_images]

                #save as annfile
                with open(annfile, 'w') as outfile:
                    js.dump(original_json, outfile)
                
                #make COCO-style instancesfile
                original_coco_file=os.path.join(self.ann_dir,"instances_{}.json".format(version))
                json=[]
                with open(os.path.join(self.data_dir,"coco-cn-version1805v1.1","imageid.human-written-tags.txt")) as f:
                    lines=f.readlines()
                    for line in lines:
                        if line.split()[0].split('#')[0].split("_")[1]==version:
                            categories=line.split()[1:]
                            cat_ids=[]
                            for category in categories:
                                if category not in categorylist:
                                    categorylist.append(category)
                                cat_ids.append(categorylist.index(category))
                            for catid in cat_ids:
                                i+=1
                                entry={
                                        'id':i,
                                        "image_id":int(line.split()[0].split('#')[0].split('_')[-1]),
                                        "category_id":catid
                                        }
                                json.append(entry)
                original_json=js.load(open(original_coco_file))
                original_json['annotations']=json
                instancesfile=os.path.join(self.ann_dir,"{}_{}.json".format('CNinstances',version))
                original_json['categories']=[{'id':i,'name':category,'supercategory':category} for i,category in enumerate(categorylist)]
                with open(instancesfile, 'w') as outfile:
                    js.dump(original_json, outfile)
                dir=os.path.join(self.data_dir,version)
                ValSets.append(COCODataset(root=dir, annFile=annfile, tokenizer=self.tokenizer,instances=instancesfile, transform=self.T))
            self.val = ConcatDataset(ValSets)


            # torch.save(self.train,"train.pt")
            # torch.save(self.val,"val.pt")    
        if stage == 'test':
            TestSets=[]
            i=0
            read_images=set()

            for version in self.splits['test']:
                #print("BUILDING SPLIT : ",version)
                #make COCO-style annfile
                annfile=os.path.join(self.ann_dir,"{}_{}.json".format('CNcaptions',version))
                cococnfile=os.path.join(self.data_dir,"coco-cn-version1805v1.1","{}.txt".format("imageid.human-written-caption.txt"))
                otherfile=os.path.join(self.data_dir,"coco-cn-version1805v1.1","{}.txt".format("imageid.human-written-caption.bosonseg.txt"))
                original_annfile=os.path.join(self.ann_dir,"captions_{}.json".format(version))
                #read both files, 
                #if each line has the same version as the current version, then add to the annfile
                json=[]
                
                with open(cococnfile) as f:
                    lines1=f.readlines()
                    for line in lines1:
                        if line.split()[0].split('#')[0].split("_")[1]==version:
                            i+=1
                            entry={
                                    'id':i,
                                    'image_id':int(line.split()[0].split('#')[0].split('_')[-1]),
                                    'caption':' '.join(line.split()[1:])
                                    }
                            json.append(entry)
                with open(otherfile) as f:
                    lines2=f.readlines()
                    for line in lines2:
                        if line.split()[0].split('#')[0].split("_")[1]==version:
                            i+=1
                            entry={
                                    'id':i,
                                    'image_id':int(line.split()[0].split('#')[0].split('_')[-1]),
                                    'caption':' '.join(line.split()[1:])
                                    }
                            json.append(entry)
                with open(original_annfile) as f:
                    original_json=js.load(f)
                original_json['annotations']=json


                #save as annfile
                with open(annfile, 'w') as outfile:
                    js.dump(original_json, outfile)
                
                #make COCO-style instancesfile
                original_coco_file=os.path.join(self.ann_dir,"instances_{}.json".format(version))
                json=[]
                with open(os.path.join(self.data_dir,"coco-cn-version1805v1.1","{}.txt".format("imageid.human-written-tags.txt"))) as f:
                    lines=f.readlines()
                    for line in lines:
                        if line.split()[0].split('#')[0].split("_")[1]==version:
                            categories=line.split()[1:]
                            cat_ids=[]
                            for category in categories:
                                if category not in categorylist:
                                    categorylist.append(category)
                                cat_ids.append(categorylist.index(category))
                            for catid in cat_ids:
                                i+=1
                                entry={
                                        'id':i,
                                        "image_id":int(line.split()[0].split('#')[0].split('_')[-1]),
                                        "category_id":catid
                                        }
                                json.append(entry)
            

                with open(original_coco_file) as f:
                    original_json=js.load(f)
                original_json['annotations']=json
                instancesfile=os.path.join(self.ann_dir,"{}_{}.json".format('CNinstances',version))
                original_json["categories"]=[{'id':i,"name":category,"supercategory":category} for i,category in enumerate(categorylist)]

                with open(instancesfile, 'w') as outfile:
                    js.dump(original_json, outfile)

                dir=os.path.join(self.data_dir,version)
                
                #print("annfile:",annfile)
                #print("dir:",dir)
                TestSets.append(COCODataset(root=dir, annFile=annfile,tokenizer=self.tokenizer,instances=instancesfile, transform=self.T))
            self.test = ConcatDataset(TestSets)


    
if __name__=="__main__":

    #do COCODataModule
    # import json
    #show keys in the data/annotations/captions_train2014.json file
    # print(json.load(open("/data/annotations/captions_train2014.json")).keys())
    # print(json.load(open("/data/annotations/captions_train2014.json"))['images'][0])
    # print(json.load(open("/data/annotations/instances_train2014.json"))['categories'])
    '''
        dict_keys(['info', 'images', 'licenses', 'annotations', 'categories'])
        {'segmentation': [[312.29, 562.89, 402.25, 511.49, 400.96, 425.38, 398.39, 372.69, 388.11, 332.85, 318.71, 325.14, 295.58, 305.86, 269.88, 314.86, 258.31, 337.99, 217.19, 321.29, 182.49, 343.13, 141.37, 348.27, 132.37, 358.55, 159.36, 377.83, 116.95, 421.53, 167.07, 499.92, 232.61, 560.32, 300.72, 571.89]], 'area': 54652.9556, 'iscrowd': 0, 'image_id': 480023, 'bbox': [116.95, 305.86, 285.3, 266.03], 'category_id': 58, 'id': 86}

        {'supercategory': 'person', 'id': 1, 'name': 'person'}, {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}, {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}, {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}, {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}, {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}, {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}, {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}, {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}, {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}, {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}, {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}, {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}, {'supercategory': 'animal', 'id': 16, 'name': 'bird'}, {'supercategory': 'animal', 'id': 17, 'name': 'cat'}, {'supercategory': 'animal', 'id': 18, 'name': 'dog'}, {'supercategory': 'animal', 'id': 19, 'name': 'horse'}, {'supercategory': 'animal', 'id': 20, 'name': 'sheep'}, {'supercategory': 'animal', 'id': 21, 'name': 'cow'}, {'supercategory': 'animal', 'id': 22, 'name': 'elephant'}, {'supercategory': 'animal', 'id': 23, 'name': 'bear'}, {'supercategory': 'animal', 'id': 24, 'name': 'zebra'}, {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'}, {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'}, {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'}, {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'}, {'supercategory': 'accessory', 'id': 32, 'name': 'tie'}, {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'}, {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}, {'supercategory': 'sports', 'id': 35, 'name': 'skis'}, {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}, {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}, {'supercategory': 'sports', 'id': 38, 'name': 'kite'}, {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}, {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}, {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}, {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}, {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}, {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'}, {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'}, {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}, {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'}, {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}, {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'}, {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}, {'supercategory': 'food', 'id': 52, 'name': 'banana'}, {'supercategory': 'food', 'id': 53, 'name': 'apple'}, {'supercategory': 'food', 'id': 54, 'name': 'sandwich'}, {'supercategory': 'food', 'id': 55, 'name': 'orange'}, {'supercategory': 'food', 'id': 56, 'name': 'broccoli'}, {'supercategory': 'food', 'id': 57, 'name': 'carrot'}, {'supercategory': 'food', 'id': 58, 'name': 'hot dog'}, {'supercategory': 'food', 'id': 59, 'name': 'pizza'}, {'supercategory': 'food', 'id': 60, 'name': 'donut'}, {'supercategory': 'food', 'id': 61, 'name': 'cake'}, {'supercategory': 'furniture', 'id': 62, 'name': 'chair'}, {'supercategory': 'furniture', 'id': 63, 'name': 'couch'}, {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'}, {'supercategory': 'furniture', 'id': 65, 'name': 'bed'}, {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'}, {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'}, {'supercategory': 'electronic', 'id': 72, 'name': 'tv'}, {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'}, {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'}, {'supercategory': 'electronic', 'id': 75, 'name': 'remote'}, {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'}, {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'}, {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'}, {'supercategory': 'appliance', 'id': 79, 'name': 'oven'}, {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'}, {'supercategory': 'appliance', 'id': 81, 'name': 'sink'}, {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'}, {'supercategory': 'indoor', 'id': 84, 'name': 'book'}, {'supercategory': 'indoor', 'id': 85, 'name': 'clock'}, {'supercategory': 'indoor', 'id': 86, 'name': 'vase'}, {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'}, {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}, {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}, {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}]
    '''
    import argparse
    parser = argparse.ArgumentParser(description='location of data')
    parser.add_argument('--data', type=str, default='/data', help='location of data')
    args = parser.parse_args()
    print("args",args)
    datalocation=args.data
    datamodule=COCOCNDataModule(Cache_dir=datalocation,annotations=os.path.join(datalocation,"annotations"),batch_size=2)

    datamodule.download_data()
    datamodule.setup()
    dl=datamodule.train_dataloader()
    print("Dataloader",dl)
    from itertools import islice

    for i in islice(dl,0,4):
        print(i[1])