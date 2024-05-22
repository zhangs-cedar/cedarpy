from init import *

class MutiFeatureNet(nn.Module):
    """
    多特征网络，训练和评估模型。
    """
    def __init__(self, num_class=2):
        """
        初始化函数，用于创建一个模型实例。
        
        Args:
            num_class (int, optional): 分类数量，默认为2。
        
        Returns:
            None
        
        """
        super().__init__()
        self.resne50 = models.resnet50(pretrained=False)
        self.resne50.fc = nn.Identity()
        self.f1 = nn.Linear(in_features=2048, out_features=512)
        self.fcfeature = nn.Linear(in_features=72, out_features=512)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_class)
        
    def forward(self, input, feature):
        """
        计算前向传播，将输入和特征进行特定操作并返回结果。
        
        Args:
            input: 输入数据，张量类型，用于通过resnet50网络提取特征。
            feature: 特征数据，张量类型，用于通过全连接层进行特征转换。
        
        Returns:
            cat: 经过特征拼接和全连接层处理后的结果，张量类型。
        
        """
        x = self.resne50(input)
        x = self.f1(x)
        f1 = self.fcfeature(feature)
        cat = self.fc2(torch.cat((x, f1), dim=-1))  # 修正了 axis 为 dim
        return cat  # 移除了不必要的 softmax
    
    @staticmethod
    def train_one_epoch(model, dataloader, criterion, optimizer, device):
        """
        训练一个epoch的函数。
        
        Args:
            dataloader (torch.utils.data.DataLoader): 数据加载器，提供训练数据。
            criterion (torch.nn.Module): 损失函数。
            optimizer (torch.optim.Optimizer): 优化器。
            device (torch.device): 设备信息，指示模型在哪个设备上运行。
        
        Returns:
            Tuple[float, float]: 包含平均损失和学习率的元组。
        """
        model.train()
        loss_sum = 0
        for i, sample in enumerate(dataloader):
            input = sample['image'].to(device)
            label = sample['label'].to(device)
            feature = sample['feature'].to(device)

            optimizer.zero_grad()
            output = model(input, feature)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())  # 简化了 loss 的处理
        lr = optimizer.param_groups[0]['lr']
        return loss_sum / len(dataloader), lr

    @staticmethod
    def eval_one_epoch(model, dataloader, device):
        """
        评估模型一个epoch的函数。
        
        Args:
            model (torch.nn.Module): 待评估的模型。
            dataloader (torch.utils.data.DataLoader): 数据加载器，提供评估数据。
            device (torch.device): 设备信息，指示模型在哪个设备上运行。
            
        Returns:
            float: 评估准确率。
        """
        model.eval()
        eval_preds = []
        eval_trues = []
        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                input = sample['image'].to(device)
                label = sample['label'].to(device)
                feature = sample['feature'].to(device)
                output = model(input, feature)
                _, pred = torch.max(output, dim=1)
                eval_preds.extend(pred.cpu().numpy().tolist())
                eval_trues.extend(label.cpu().numpy().tolist())

        eval_accuracy = accuracy_score(eval_trues, eval_preds)
        return eval_accuracy
    
    @staticmethod
    def model_train(model,
              # 训练轮数
              num_epochs=50,
              # 保存间隔轮数
              save_interval_epochs=5,
              # 训练数据集
              train_loader = "DataLoader(train_dataset, batch_size=64, shuffle=True)",
              # 评估数据集
              eval_loader="DataLoader(eval_dataset, batch_size=64)",
              # 优化器
              optimizer="torch.optim.Adam(model.parameters(), lr=0.0001)",
              # 损失函数
              criterion="nn.CrossEntropyLoss()",
              # deivce cuda:0 or cpu ...
              device="torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
              # 保存目录
              save_dir='output'
              ):
        """ 训练模型 """
        train_log = {
            "loss": [],
            "lr": [],
            "eval_acc": [],
        }
        best_eval_acc = 0
        for epoch in range(num_epochs):
            loss, lr = model.train_one_epoch(model, train_loader, criterion, optimizer, device)
            print('Epoch: {}, loss: {:.4f}, lr: {:.6f}'.format(epoch + 1, loss, lr))
            if (epoch + 1) % save_interval_epochs == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_{}.pth'.format(epoch + 1)))
            
            eval_acc = model.eval_one_epoch(model, eval_loader, device)
            print('Eval Acc: {:.4f}'.format(eval_acc))
            # 保存最好的模型
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                
            train_log["loss"].append(loss)
            train_log["lr"].append(lr)
            train_log["eval_acc"].append(eval_acc)
            
        # 保存训练日志的表
        df = pd.DataFrame(train_log)
        df.to_csv(os.path.join(save_dir, 'train_log.csv'), index=False)
        
        # 绘制训练曲线 loss 和 eval_acc 和 lr
        fig, (ax1, ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16,6))
        sns.lineplot(x=range(1, num_epochs + 1), y=train_log["loss"], ax=ax1)
        ax1.set_title('Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        sns.lineplot(x=range(1, num_epochs + 1), y=train_log["eval_acc"],
                     xticks=range(1, num_epochs + 1), ax=ax2)
        ax2.set_title('Eval Acc Curve')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Eval Acc')
        
        sns.lineplot(x=range(1, num_epochs + 1), y=train_log["lr"],ax = ax3)
        ax3.set_title('Learning Rate Curve')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'train_curve.png'))
        
    @staticmethod
    def model_load(model,model_path,device="torch.device('cuda' if torch.cuda.is_available() else 'cpu')"):
        """ 加载模型 """
        model.load_state_dict(torch.load(model_path,map_location=device))
        model = model.to(device)
        return model
        
    @staticmethod
    def predict(model,sample,device="torch.device('cuda' if torch.cuda.is_available() else 'cpu')"):
        """
        预测函数，用于将模型应用于输入的样本数据，并返回预测结果。
        
        Args:
            model: 待预测的模型。
            sample: 包含图像和特征的样本数据，类型为字典。
            device: 运行模型的设备，默认为'cuda'（如果CUDA可用）或'cpu'。
        Returns:
            pred: 预测结果，类型为整数。
        
        """
        with torch.no_grad():
            model.eval()
            input = sample['image'].to(device)
            feature = sample['feature'].to(device)
            output = model(input, feature)
            _, pred = torch.max(output, dim=1)
            pred = pred.cpu().detach().numpy()[0]
        return  pred



if __name__ == '__main__':
    model = MutiFeatureNet(num_class=2)
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.train()
    input = torch.randn(5, 3, 256, 256)
    feature = torch.randn(5, 72)
    output = model(input, feature)
    print(output,output.shape)
    
    