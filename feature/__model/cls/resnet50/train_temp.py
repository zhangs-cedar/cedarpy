from init import *
from dataset import MyDataset
from network import MutiFeatureNet

# 定义训练和验证时的transforms
train_transforms = T.Compose([T.RandomCrop(crop_size=224), T.RandomHorizontalFlip(), T.Normalize()])

eval_transforms = T.Compose([T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()])

# 定义训练和验证所用的数据集
train_dataset = MyDataset(
    data_dir="vegetables_cls",
    file_list="vegetables_cls/train_list.txt",
    label_list="vegetables_cls/labels.txt",
    transforms=train_transforms,
)

eval_dataset = MyDataset(
    data_dir="vegetables_cls", file_list="vegetables_cls/val_list.txt", label_list="vegetables_cls/labels.txt", transforms=eval_transforms
)

# 初始化模型，并进行训练
model = MutiFeatureNet(num_classes=len(train_dataset.labels))
model.model_train(
    model,
    # 训练轮数
    num_epochs=50,
    # 保存间隔轮数
    save_interval_epochs=5,
    # 训练数据集
    train_loader="DataLoader(train_dataset, batch_size=64, shuffle=True)",
    # 评估数据集
    eval_loader="DataLoader(eval_dataset, batch_size=64)",
    # 优化器
    optimizer="torch.optim.Adam(model.parameters(), lr=0.0001)",
    # 损失函数
    criterion="nn.CrossEntropyLoss()",
    # deivce cuda:0 or cpu ...
    device="torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    # 保存目录
    save_dir="output",
)
