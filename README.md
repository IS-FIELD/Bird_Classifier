# 训练流程解析

数据集包含两部分
1.各种鸟的名称和对应的音频路径
2.各类鸟类子文件夹下的音频文件


## 1. 数据载入
首先，程序从CSV文件中读取音频数据的相关元数据：

```python
df = pd.read_csv(f"{BASE_PATH}/try2_csv.csv", encoding="ISO-8859-1", header=None)
```

该CSV文件包含音频文件的路径、鸟名称等信息。数据被分为训练集和验证集：

```python
train_df, valid_df = train_test_split(df, test_size=0.2)
```

在这一步，数据按照8:2的比例随机划分为训练集和验证集。

## 2. 数据集与数据加载器

为了将音频数据输入到模型中，需要将其转换为网络可用的格式。程序定义了`MyData_Set`类，继承自`torch.utils.data.Dataset`，用来处理和加载音频数据。

```python
class MyData_Set(Dataset):
    def __init__(self, df, transform):
        self.df = pd.read_csv(
            f"{BASE_PATH}/try2_csv.csv", encoding="ISO-8859-1", header=None
        )
        self.df.columns = [chr(i) for i in range(ord("A"), ord("L") + 1)]
        self.audiopath = BASE_PATH + "/train_audio/" + df["A"]
        self.transform = transform
        self.name2label = get_label()
```

### 2.1 音频路径与标签

在`__getitem__`方法中，程序根据索引获取音频文件路径，并加载音频文件：

```python
    def __getitem__(self, idx):
        try:
            filename = self.df.iloc[idx]["L"]
            if pd.isna(filename) or filename == "nan":
                filename = self.df.iloc[idx]["K"]
            audiopath = os.path.join(BASE_PATH, "train_audio", str(filename))
            if not os.path.exists(audiopath):
                raise ValueError(f"File does not exist: {audiopath}")
            spec = decode(audiopath)

            label_name = self.df.iloc[idx]["A"]
            target = self.name2label[label_name]
            target = F.one_hot(torch.tensor(target), num_classes=len(self.name2label))
            return spec, target
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
```

## 3. 音频预处理

在`decode()`函数中，音频被加载并转换为网络可用的格式。

```python
def decode(audiopath):
    target_len = 480000
    waveform = torchaudio.load(audiopath, normalize=True)[0]
    waveform = waveform.mean(0, keepdim=True)  # 立体声转换为单声道

    # 裁剪或填充到目标长度
    waveform_len = waveform.shape[1]
    diff_len = abs(target_len - waveform_len)
    if waveform_len < target_len:
        pad1 = torch.randint(0, diff_len, (1,)).item()
        pad2 = diff_len - pad1
        waveform = torch.nn.functional.pad(waveform, (pad1, pad2))
    elif waveform_len > target_len:
        waveform = waveform[:, :target_len]

    # 转换为梅尔频谱图
    spec = T.MelSpectrogram(
        sample_rate=32000, n_fft=2028, n_mels=256, hop_length=512, f_min=20, f_max=16000
    )(waveform)

    # 标准化和归一化
    mean = torch.mean(spec)
    std = torch.std(spec)
    spec = torch.where(std == 0, spec - mean, (spec - mean) / std)
    min_val = torch.min(spec)
    max_val = torch.max(spec)
    spec = torch.where(
        max_val - min_val == 0, spec - min_val, (spec - min_val) / (max_val - min_val)
    )

    # 调整为3通道并改变尺寸
    spec = spec.repeat(3, 1, 1)
    spec = F.interpolate(
        spec.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)
    return spec
```

## 4. 数据增强

定义了`augment`类，用于对频谱图进行数据增强，包括频率掩码和时间掩码：

```python
class augment(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_masking = T.FrequencyMasking(freq_mask_param=30)
        self.time_masking = T.TimeMasking(time_mask_param=80)

    def forward(self, spec):
        if torch.rand(1).item() < 0.5:
            spec = self.freq_masking(spec)
            if torch.rand(1).item() < 0.5:
                spec = self.time_masking(spec)
        return spec
```

## 5. 模型定义

使用`torchvision.models`中的预训练模型`VGG16`作为基础模型，并修改了最后一层以适应当前分类任务：

```python
class My_model(nn.Module):
    def __init__(self, num_classes):
        super(My_model, self).__init__()
        self.backbone = torchvision.models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1).to('cuda')
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x
```

在这里，`VGG16`的预训练权重用于加速训练，同时仅训练最后的分类器层来适应具体任务需求。

## 6. 训练过程

在训练过程中，使用交叉熵损失函数以及Adam优化器。训练循环如下：

```python
if __name__ == "__main__":
    num_classes = len(train_DataSet.name2label)
    model = My_model(num_classes=num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    num_epochs = 3
    
    device = torch.device("cuda")
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, (spec, target) in enumerate(train_dataloader):
            if spec is None or target is None:
                continue
            
            spec = augmentor(spec).to(device)
            target = target.to(device)
            target = torch.argmax(target, dim=1)

            optimizer.zero_grad()
            outputs = model(spec)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader)}")

    torch.save(model.state_dict(), "model.pth")
    print("MODEL_PARAM has saved in model.pth")
```

### 6.1 训练细节
- **数据增强**：在每个批次中，使用`augmentor`对输入的频谱图进行数据增强，增加模型的鲁棒性。
- **优化器与学习率调度**：使用Adam优化器，并使用余弦退火学习率调度器在训练过程中动态调整学习率。
- **损失函数**：使用交叉熵损失函数来计算预测与实际标签之间的误差。

## 7. 模型保存

训练完成后，模型参数被保存到文件`model.pth`中，以便后续进行推理或再训练。

```python
torch.save(model.state_dict(), "model.pth")
```

---

通过以上步骤，从载入音频数据、数据预处理、数据增强、使用预训练模型模型，完整地实现了一个基于鸟类音频数据的分类模型的训练流程。
```
