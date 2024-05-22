from init import *


def cls_split(data_dir,train_split=0.8):
    """
    将数据集分割为训练集和验证集，并生成相应的文件列表。
    
    Args:
        data_dir (str): 数据集所在的目录路径。
        train_split (float, optional): 训练集所占的比例，默认为0.8。    
    """
    
    labels = []
    
    for names in os.listdir(data_dir):
        path = osp.join(data_dir,names)
        labels.append(path)
        if osp.isdir(path):
            labels.append(names)
    
    with open(osp.join(data_dir,'labels.txt'),'w') as f:
        for label in labels:
            f.write(label+'\n')
    print('labels.txt created {}'.format(labels))
    
    img_list = []
    for idx,label in enumerate(labels):
        for img_name in os.listdir(osp.join(data_dir,label)):
            img_list.append(label+'/'+img_name+" "+str(idx))
    
    random.shuffle(img_list)
    num = len(img_list)
    train_list = img_list[:int(num*train_split)]
    val_list = img_list[int(num*train_split):]
    
    with open(osp.join(data_dir,'train.txt'),'w') as f:
        for img in train_list:
            f.write(f"{img}\n")
    
    with open(osp.join(data_dir,'val.txt'),'w') as f:
        for img in val_list:
            f.write(f"{img}\n")
            
    print('train.txt created {}'.format(train_list))
    print('val.txt created {}'.format(val_list))
            
            

class MyDataset(Dataset):
    def __init__(self,data_dir,file_list,label_list,transforms=None):
        """
        Args:
            data_dir (str): 图片所在目录。
            file_list (str): 文件列表路径。
            label_list (str): 标签列表路径。
            transforms (callable, optional): 图像预处理方法。
        """
        self.transforms = copy.deepcopy(transforms) # 防止transforms被修改
        self.file_list = list()
        self.labels = list()
        
        with open(label_list,encoding=get_encoding(label_list)) as f:
            for line in f:
                self.labels.append(line.strip()) # 标签
        
        with open(file_list,encoding=get_encoding(file_list)) as f:
            for line in tqdm(f):
                items = line.strip().split()
                items[0] = path_normalization(items[0])
                if not is_pic(items[0]):
                    continue
                full_path = osp.join(data_dir,items[0])
                if not osp.exists(full_path):
                    continue
                self.file_list.append({
                    "file":full_path,
                    "label": np.asarray(items[1],dtype=np.int64),
                    "feature": "get_feature",
                })
        self.num = len(self.file_list)
        print(f'{self.num} images loaded from {data_dir}')
        
    def __getitem__(self, idx):
        """
        np.float32
        """
        sample = copy.deepcopy(self.file_list[idx])
        return sample
        
    def __len__(self):
        return len(self.file_list)
    
    
    
    
if __name__ == '__main__':
    """ """
    train_transforms = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(size=(256,256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    train_dataset = MyDataset(
        data_dir='./dataset/train',
        file_list='./dataset/train.txt',
        label_list='./dataset/labels.txt',
        transforms=train_transforms,
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
    )
    
    for i,data in enumerate(train_loader):
        print(i)
    