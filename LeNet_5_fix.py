# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import time
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows/Linux
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac
plt.rcParams['axes.unicode_minus'] = False

#    (使用 ReLU, MaxPool, Dropout, 且更深)
class DeepMNISTCNN(nn.Module):
    """
    一个更深的卷积神经网络模型，用于MNIST识别。
    固定使用 ReLU 激活函数, Max Pooling, 并包含 Dropout。
    """
    def __init__(self, dropout_rate=0.5):
        super(DeepMNISTCNN, self).__init__()

        # --- 卷积层 ---
        # 卷积块 1: (B, 1, 28, 28) -> (B, 32, 14, 14)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 卷积块 2: (B, 32, 14, 14) -> (B, 64, 7, 7)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 卷积块 3 (增加的层): (B, 64, 7, 7) -> (B, 128, 3, 3)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # 使用 3x3 卷积核
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 输出尺寸 (7-2)/2+1 = 3 (向下取整)
        )

        # --- 全连接层 ---
        self.flatten = nn.Flatten() # 展平层

        # 全连接块: (128 * 3 * 3 = 1152) -> 128 -> 10
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=128 * 3 * 3, out_features=128),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate), # 在第一个 FC 层后加入 Dropout
            nn.Linear(in_features=128, out_features=10) # 输出层 (logits)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x) # 通过新增的卷积块
        x = self.flatten(x)    # 展平
        x = self.fc_block(x)   # 通过全连接块 (包含 ReLU, Dropout, 输出层)
        return x


# 数据加载函数 (与之前相同)
def load_data(batch_size=64):
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        train_dataset = torchvision.datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./MNIST_data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"数据加载或下载失败: {e}")
        return None, None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# 训练函数 (单个 Epoch) (与之前相同)
def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# 测试函数 (计算指标) (与之前相同)
def test_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    precision_val = precision_score(all_targets, all_predictions, average='macro', zero_division=0) * 100
    recall_val = recall_score(all_targets, all_predictions, average='macro', zero_division=0) * 100
    f1_val = f1_score(all_targets, all_predictions, average='macro', zero_division=0) * 100
    return avg_loss, accuracy, precision_val, recall_val, f1_val

# 完整训练流程函数 (打印最优结果版, 与之前相同)
def train_model(model, device, train_loader, test_loader, optimizer,
                criterion, epochs, model_name):
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"{'='*60}")
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
               'test_precision': [], 'test_recall': [], 'test_f1': []}
    best_epoch, best_test_acc, best_test_loss = 0, 0.0, float('inf')
    best_test_precision, best_test_recall, best_test_f1 = 0.0, 0.0, 0.0
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc, test_precision, test_recall, test_f1 = \
            test_model(model, device, test_loader, criterion)
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss); history['test_acc'].append(test_acc)
        history['test_precision'].append(test_precision); history['test_recall'].append(test_recall)
        history['test_f1'].append(test_f1)
        if test_acc > best_test_acc:
            best_test_acc, best_epoch, best_test_loss = test_acc, epoch, test_loss
            best_test_precision, best_test_recall, best_test_f1 = test_precision, test_recall, test_f1
        print(f'Epoch {epoch}/{epochs}: 当前损失: {test_loss:.4f} 当前测试准确率: {test_acc:.2f}% 当前精确率: {test_precision:.2f}% 当前召回率: {test_recall:.2f}% 当前F1分数: {test_f1:.2f}%')

    training_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {training_time:.2f} 秒")
    print(f"\n--- {model_name} 的最优测试结果 (出现在第 {best_epoch} 轮) ---")
    print(f"  损失 (Loss):       {best_test_loss:.4f}")
    print(f"  准确率 (Accuracy):  {best_test_acc:.2f}%")
    print(f"  精确率 (Precision): {best_test_precision:.2f}%")
    print(f"  召回率 (Recall):    {best_test_recall:.2f}%")
    print(f"  F1分数 (F1-Score):  {best_test_f1:.2f}%")
    print("-"*(len(model_name) + 40))
    return history, training_time

def generate_classification_report(model, device, test_loader, model_name):
    """生成详细的分类报告和混淆矩阵"""
    print(f"\n{'='*80}")
    print(f"{model_name} - 详细分类报告与混淆矩阵")
    print(f"{'='*80}")
    model.eval()
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 分类报告
    print("\n分类报告 (各类别详细指标):")
    print("-" * 80)
    class_report = classification_report(all_targets, all_predictions, target_names=[str(i) for i in range(10)], digits=4)
    print(class_report)

    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    print("\n混淆矩阵:")
    print("-" * 80)
    print("预测 →")
    print("真实 ↓  ", end="")
    for i in range(10): print(f"{i:>6}", end="")
    print("\n" + "-" * 80)
    for i in range(10):
        print(f"    {i}    ", end="")
        for j in range(10): print(f"{cm[i, j]:>6}", end="")
        print()
    print("-" * 80)

    # 可视化混淆矩阵
    try:
        fig, ax = plt.subplots(figsize=(10, 8)) # 使用单数 ax
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=range(10), yticklabels=range(10),
               title=f'{model_name} - 混淆矩阵',
               ylabel='真实标签',
               xlabel='预测标签')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(f'{model_name}_confusion_matrix.png', dpi=150)
        print(f"\n混淆矩阵图已保存: {model_name}_confusion_matrix.png")
        plt.show()
    except Exception as plot_cm_e:
        print(f"\n绘制混淆矩阵图时出错: {plot_cm_e}")

    # 最容易混淆的类别对
    print("\n最容易混淆的类别对 (Top 5):")
    print("-" * 80)
    confusion_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((i, j, cm[i, j]))
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    for rank, (true, pred, count) in enumerate(confusion_pairs[:5], 1):
        print(f"{rank}. 真实={true}, 预测={pred}: {count}次混淆")
    print("-" * 80)
    return cm # 返回混淆矩阵


# 主函数
def main():
    print("更深的 CNN 模型 MNIST 识别实验")
    print("(使用 ReLU, MaxPool, Dropout, 增加卷积层)")

    # --- 设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    batch_size = 128
    epochs = 15 # 对于更深的模型，可以适当增加训练轮数
    learning_rate = 0.001 # Adam 通常使用较小的学习率
    dropout_rate = 0.3

    print("\n正在加载 MNIST 数据集...")
    train_loader, test_loader = load_data(batch_size)
    if train_loader is None or test_loader is None: return
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

    print("\n初始化最终模型...")
    model = DeepMNISTCNN(dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    # 使用 Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history, training_time = train_model(
        model, device, train_loader, test_loader,
        optimizer, criterion, epochs,
        "最终模型 (Deep + Tanh + MaxPool + Dropout)"
    )

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(range(1, epochs + 1), history['train_loss'], label='训练损失', marker='o')
        axes[0].plot(range(1, epochs + 1), history['test_loss'], label='测试损失', marker='x')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('损失'); axes[0].set_title('损失曲线'); axes[0].legend(); axes[0].grid(True)
        axes[1].plot(range(1, epochs + 1), history['train_acc'], label='训练准确率', marker='o')
        axes[1].plot(range(1, epochs + 1), history['test_acc'], label='测试准确率', marker='x')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('准确率 (%)'); axes[1].set_title('准确率曲线'); axes[1].legend(); axes[1].grid(True)
        axes[1].set_ylim(max(0, min(min(history['test_acc']), min(history['train_acc'])) - 5) , 101)
        plt.suptitle("最终模型训练结果", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('final_model_training_curves.png', dpi=150)
        print("\n训练结果曲线图已保存: final_model_training_curves.png")
        plt.show()
    except Exception as plot_e:
        print(f"\n绘制图表时出错: {plot_e}")
    try:
        generate_classification_report(model, device, test_loader, "最终模型")
    except NameError:
        print("\n提示: generate_classification_report 函数未定义，跳过详细报告生成。")
    except Exception as report_e:
        print(f"\n生成详细报告时出错: {report_e}")


    print("\n最终模型实验完成!")
    print("="*80)


if __name__ == '__main__':
    main()