# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# 设置 matplotlib 支持中文显示 (确保你的环境安装了支持中文的字体, 如 SimHei)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows/Linux
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac?
    plt.rcParams['axes.unicode_minus'] = False # 解决负号'-'显示为方块的问题
except Exception as font_e:
    print(f"警告：设置中文字体失败 ({font_e})。图表中的中文可能显示为乱码。")
    print("请确保安装了中文字体并修改 plt.rcParams['font.sans-serif'] 的设置。")


# ================================
# 1. 模型定义: 简化基线 LeNet-5
# ================================
class BaselineLeNet5(nn.Module):
    """
    简化的 LeNet-5 基线模型 (使用 Tanh 和 Average Pooling)。
    直接处理 28x28 输入。
    """
    def __init__(self):
        super(BaselineLeNet5, self).__init__()
        # 第一个卷积块: Conv -> Tanh -> AvgPool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 第二个卷积块: Conv -> Tanh -> AvgPool
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 全连接层块
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.act4 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=84, out_features=10) # 输出层

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x) # 输出 logits
        return x
# 2. 数据加载函数
def load_data(batch_size=64):
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 数据集的均值和标准差
    ])

    try:
        train_dataset = torchvision.datasets.MNIST(
            root='MNIST_data',
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='MNIST_data',
            train=False,
            download=True,
            transform=transform
        )
    except Exception as e:
        print(f"数据加载或下载失败: {e}")
        print("请检查网络连接或 './data' 目录权限。")
        return None, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 # 根据你的CPU核心数调整
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 # 根据你的CPU核心数调整
    )
    return train_loader, test_loader

# 3. 训练函数 (单个 Epoch)
def train_epoch(model, device, train_loader, optimizer, criterion):
    """训练一个epoch"""
    model.train() # 设置模型为训练模式
    train_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device) # 数据移动到设备

        optimizer.zero_grad()      # 清除梯度
        output = model(data)       # 前向传播
        loss = criterion(output, target) # 计算损失
        loss.backward()            # 反向传播
        optimizer.step()           # 更新权重

        train_loss += loss.item() # 累加损失值
        logits = torch.argmax(output, dim=-1) # 获取预测结果 (概率最高的类别索引)
        total += target.size(0)      # 累加样本总数
        correct += (logits == target).sum().item() # 累加预测正确的数量

    avg_loss = train_loss / len(train_loader) # 计算平均损失
    accuracy = 100. * correct / total         # 计算准确率 (百分比)
    return avg_loss, accuracy

# ================================
# 4. 测试函数 (计算指标)
# ================================
def test_model(model, device, test_loader, criterion):
    """在测试集上评估模型，返回损失和四个核心指标"""
    model.eval() # 设置模型为评估模式
    test_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad(): # 禁用梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            logits = torch.argmax(output, dim=-1)
            total += target.size(0)
            correct += (logits == target).sum().item()

            all_predictions.extend(logits.cpu().numpy()) # 收集预测
            all_targets.extend(target.cpu().numpy())       # 收集真实标签

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    # 计算精确度, 召回率, F1分数 (宏平均)
    # 使用 .cpu().numpy() 转换后才能用于 sklearn
    precision_val = precision_score(all_targets, all_predictions, average='macro', zero_division=0) * 100
    recall_val = recall_score(all_targets, all_predictions, average='macro', zero_division=0) * 100
    f1_val = f1_score(all_targets, all_predictions, average='macro', zero_division=0) * 100

    return avg_loss, accuracy, precision_val, recall_val, f1_val

# 5. 完整训练流程函数
def train_model(model, device, train_loader, test_loader, optimizer,
                criterion, epochs, model_name):
    """完整训练流程，记录所有评价指标，并在最后打印最优测试结果"""
    print(f"开始训练: {model_name}")

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'test_precision': [], 'test_recall': [], 'test_f1': []
    }

    # 用于跟踪最优结果的变量
    best_epoch = 0
    best_test_acc = 0.0
    best_test_loss = float('inf') # 损失越小越好，初始化为无穷大
    best_test_precision = 0.0
    best_test_recall = 0.0
    best_test_f1 = 0.0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(model, device, train_loader,
                                            optimizer, criterion)

        # 测试（获取所有指标）
        test_loss, test_acc, test_precision, test_recall, test_f1 = test_model(model, device, test_loader, criterion)

        # 记录所有历史数据 (用于后续绘图)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_precision'].append(test_precision)
        history['test_recall'].append(test_recall)
        history['test_f1'].append(test_f1)

        # --- 检查是否是当前最优结果 (基于测试准确率) ---
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_test_loss = test_loss
            best_test_precision = test_precision
            best_test_recall = test_recall
            best_test_f1 = test_f1

        print(f'Epoch {epoch}/{epochs}: 当前损失: {test_loss:.4f} 当前测试准确率: {test_acc:.2f}% 当前精确率: {test_precision:.2f}% 当前召回率: {test_recall:.2f}% 当前F1分数: {test_f1:.2f}%')

    training_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {training_time:.2f} 秒")

    # --- 只打印最优结果 ---
    print(f"\n--- {model_name} 的最优测试结果 (出现在第 {best_epoch} 轮) ---")
    print(f"  损失 (Loss):       {best_test_loss:.4f}")
    print(f"  准确率 (Accuracy):  {best_test_acc:.2f}%")
    print(f"  精确率 (Precision): {best_test_precision:.2f}%")
    print(f"  召回率 (Recall):    {best_test_recall:.2f}%")
    print(f"  F1分数 (F1-Score):  {best_test_f1:.2f}%")

    return history, training_time # 仍然返回完整的 history 用于绘图

# ================================
# 6. 主函数
# ================================
def main():
    print("简化基线 LeNet-5 模型 MNIST 识别实验")

    # --- 设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    batch_size = 128 # 增大 batch size 可能加速训练
    epochs = 15     # 训练轮数
    learning_rate = 0.01 # SGD 的学习率

    # --- 加载数据 ---
    print("\n正在加载 MNIST 数据集...")
    train_loader, test_loader = load_data(batch_size)
   
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    # --- 初始化模型、损失函数、优化器 ---
    print("\n初始化基线模型...")
    model = BaselineLeNet5().to(device)
    criterion = nn.CrossEntropyLoss() # 交叉熵损失，自带 Softmax
    # 使用 SGD 作为基线优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # --- 训练和评估模型 ---
    history, training_time = train_model(
        model, device, train_loader, test_loader,
        optimizer, criterion, epochs,
        "基线模型 (Tanh + AvgPool + SGD)"
    )

    # --- (可选) 绘制结果曲线 ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 绘制损失曲线
        axes[0].plot(range(1, epochs + 1), history['train_loss'], label='训练损失', marker='o')
        axes[0].plot(range(1, epochs + 1), history['test_loss'], label='测试损失', marker='x')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失 (Loss)')
        axes[0].set_title('训练和测试损失曲线')
        axes[0].legend()
        axes[0].grid(True)

        # 绘制准确率曲线
        axes[1].plot(range(1, epochs + 1), history['train_acc'], label='训练准确率', marker='o')
        axes[1].plot(range(1, epochs + 1), history['test_acc'], label='测试准确率', marker='x')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('准确率 (%)')
        axes[1].set_title('训练和测试准确率曲线')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_ylim(min(min(history['test_acc']), min(history['train_acc'])) - 1 , 101) # y轴范围

        plt.suptitle("基线模型训练结果", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('baseline_training_curves.png', dpi=150)
        print("\n训练结果曲线图已保存: baseline_training_curves.png")
        plt.show()
    except Exception as plot_e:
        print(f"\n绘制图表时出错: {plot_e}")


    print("\n基线模型实验完成!")
    print("="*80)


if __name__ == '__main__':
    main()