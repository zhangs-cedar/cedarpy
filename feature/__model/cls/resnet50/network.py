from init import *


class MultiFeatureNet(nn.Module):
    """
    多特征网络，训练和评估模型。
    """

    def __init__(self, num_class=2, device=None):
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
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

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

    def train_one_epoch(self, dataloader, criterion, optimizer):
        """
        训练一个epoch的函数。

        Args:
            dataloader (torch.utils.data.DataLoader): 数据加载器，提供训练数据。
            criterion (torch.nn.Module): 损失函数。
            optimizer (torch.optim.Optimizer): 优化器。
        Returns:
            Tuple[float, float]: 包含平均损失和学习率的元组。
        """

        self.train()
        loss_sum = 0
        for i, sample in enumerate(dataloader):
            input = sample["image"].to(self.device)
            label = sample["label"].to(self.device)
            feature = sample["feature"].to(self.device)
            optimizer.zero_grad()
            output = self.forward(input, feature)
            _, pred = torch.max(output, dim=1)
            loss = criterion(pred.to(dtype=torch.float32), label.squeeze(1).to(dtype=torch.float32))

            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())  # 简化了 loss 的处理
        lr = optimizer.param_groups[0]["lr"]
        return loss_sum / len(dataloader), lr

    def eval_one_epoch(self, dataloader):
        """
        评估模型一个epoch的函数。
        Args:
            dataloader (torch.utils.data.DataLoader): 数据加载器，提供评估数据。
        Returns:
            float: 评估准确率。
        """
        self.eval()
        eval_preds = []
        eval_trues = []
        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                input = sample["image"].to(self.device).requires_grad_(True)
                label = sample["label"].to(self.device).requires_grad_(True)
                feature = sample["feature"].to(self.device).requires_grad_(True)
                output = self(input, feature).requires_grad_(True)
                _, pred = torch.max(output, dim=1)
                eval_preds.extend(pred.cpu().numpy().tolist())
                eval_trues.extend(label.cpu().numpy().tolist())

        eval_accuracy = accuracy_score(eval_trues, eval_preds)
        return eval_accuracy

    def model_train(
        self,
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
        # 保存目录
        save_dir="output",
    ):
        """训练模型"""
        train_log = {
            "loss": [],
            "lr": [],
            "eval_acc": [],
        }
        best_eval_acc = 0
        for epoch in range(num_epochs):
            loss, lr = self.train_one_epoch(train_loader, criterion, optimizer)
            print("Epoch: {}, loss: {:.4f}, lr: {:.6f}".format(epoch + 1, loss, lr))
            if (epoch + 1) % save_interval_epochs == 0:
                torch.save(self.state_dict(), os.path.join(save_dir, "model_{}.pth".format(epoch + 1)))

            eval_acc = self.eval_one_epoch(self, eval_loader)
            print("Eval Acc: {:.4f}".format(eval_acc))
            # 保存最好的模型
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                torch.save(self.state_dict(), os.path.join(save_dir, "best_model.pth"))

            train_log["loss"].append(loss)
            train_log["lr"].append(lr)
            train_log["eval_acc"].append(eval_acc)

        # 保存训练日志的表
        df = pd.DataFrame(train_log)
        df.to_csv(os.path.join(save_dir, "train_log.csv"), index=False)

    def model_load(self, model_path):
        """加载模型"""
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.to(self.device)

    def model_save(self, model_path):
        """保存模型"""
        torch.save(self.state_dict(), model_path)

    @timeit
    def predict(self, input, feature):
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
            self.eval()
            output = self.forward(input, feature)
            _, pred = torch.max(output, dim=1)
            pred = pred.cpu().detach().numpy()
        return pred


@timeit
def test_01():
    print("测试代码 推理代码测试")
    model = MultiFeatureNet(num_class=2)  # 实例化模型
    # 模拟数据
    input = torch.randn(5, 3, 256, 256)
    feature = torch.randn(5, 72)
    output = model.predict(input, feature)
    print(output, output.shape)


def test_02():
    print("测试代码 训练代码测试")
    model = MultiFeatureNet(num_class=2)  # 实例化模型
    # 模拟数据

    class MyDataset(Dataset):

        def __getitem__(self, idx):
            sample_image = np.random.randn(3, 256, 256).astype(np.float32)
            sample_feature = np.random.randn(72).astype(np.float32)
            sample_label = np.array([1], dtype=np.float32)

            sample = {"image": copy.deepcopy(sample_image), "feature": copy.deepcopy(sample_feature), "label": copy.deepcopy(sample_label)}
            return sample

        def __len__(self):
            return 100

    md = MyDataset()

    model.model_train(
        num_epochs=5,
        save_interval_epochs=1,
        train_loader=DataLoader(md, batch_size=2, shuffle=True),
        eval_loader=DataLoader(md, batch_size=2),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
        criterion=nn.CrossEntropyLoss(),
        save_dir="output",
    )


if __name__ == "__main__":
    # test_01()
    test_02()
