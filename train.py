import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
import torchaudio
import os
import pandas as pd
import torchaudio.transforms as T
from torch import t
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torchvision.models
from torchvision.models import vgg16, VGG16_BN_Weights

BASE_PATH = "E:/birdclef-2024/TRY_FILE"
df = pd.read_csv(
    f"{BASE_PATH}/try2_csv.csv",
    encoding="ISO-8859-1",
    encoding_errors="ignore",
    header=None,
)
df.columns = [chr(i) for i in range(ord("A"), ord("L") + 1)]

train_df, valid_df = train_test_split(df, test_size=0.2)


class MyData_Set(Dataset):

    def __init__(self, df, transform):
        self.df = pd.read_csv(
            f"{BASE_PATH}/try2_csv.csv", encoding="ISO-8859-1", header=None
        )
        self.df.columns = [chr(i) for i in range(ord("A"), ord("L") + 1)]
        self.audiopath = BASE_PATH + "/train_audio/" + df["A"]
        self.transform = transform
        self.name2label = get_label()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            filename = self.df.iloc[idx]["L"]
            if pd.isna(filename) or filename == "nan":
                filename = self.df.iloc[idx]["K"]
            audiopath = os.path.join(BASE_PATH, "train_audio", str(filename))
            if not os.path.exists(audiopath):
                raise ValueError(f"File does not exist: {audiopath}")
            spec = decode(audiopath)
            if spec == None:
                raise ValueError(f"Failed to decode audio file: {audiopath}")

            label_name = self.df.iloc[idx]["A"]
            target = self.name2label[label_name]
            target = F.one_hot(torch.tensor(target), num_classes=len(self.name2label))
            if target == None:
                raise ValueError(f"Failed to get target for index {idx}")
            # print(
            #     f"audiopath: {audiopath}, target: {target.tolist()}, spec shape: {spec.shape}"
            # )
            return spec, target
        except Exception as e:
            print(f"Error processing index {idx}: {e}")


def custom_collate_fn(batch):
    # Delete None items
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()

    # pack the batch
    specs, targets = zip(*batch)

    # list -> tensor
    specs = torch.stack(specs)
    targets = torch.stack(targets)

    return specs, targets


def decode(audiopath):
    target_len = 480000
    waveform = torchaudio.load(audiopath, normalize=True)[0]
    # stero to mono
    waveform = waveform.mean(0, keepdim=True)

    # crop or pad
    waveform_len = waveform.shape[1]
    diff_len = abs(target_len - waveform_len)
    if waveform_len < target_len:
        pad1 = torch.randint(0, diff_len, (1,)).item()
        pad2 = diff_len - pad1
        waveform = torch.nn.functional.pad(waveform, (pad1, pad2))
    elif waveform_len > target_len:
        idx = torch.randint(0, diff_len, (1,)).item()
        waveform = waveform[:, :target_len]

    # mel_spectrogram
    spec = T.MelSpectrogram(
        sample_rate=32000, n_fft=2028, n_mels=256, hop_length=512, f_min=20, f_max=16000
    )(waveform)
    mean = torch.mean(spec)
    std = torch.std(spec)
    # Standardize
    spec = torch.where(std == 0, spec - mean, (spec - mean) / std)
    min_val = torch.min(spec)
    max_val = torch.max(spec)
    # Normalize using Min-Max
    spec = torch.where(
        max_val - min_val == 0, spec - min_val, (spec - min_val) / (max_val - min_val)
    )
    # 3channel
    if len(spec.shape) == 2:
        spec = spec.unsqueeze(0)
    spec = spec.repeat(3, 1, 1)
    # Resize the spectrogram to 224x224
    spec = F.interpolate(
        spec.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    )
    spec = spec.squeeze(0)  # Remove the extra batch dimension

    if spec is None:
        print(f"Failed to decode audio file: {audiopath}")
    return spec


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


def get_label():
    class_names = sorted(os.listdir("E:/birdclef-2024/TRY_FILE/train_audio/"))
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    name2label = dict(zip(class_names, class_labels))
    return name2label


augmentor = augment()
train_DataSet = MyData_Set(train_df, transform=augmentor)
valid_DataSet = MyData_Set(valid_df, transform=augmentor)


train_dataloader = DataLoader(
    train_DataSet,
    batch_size=32,
    shuffle=True,
    sampler=None,
    batch_sampler=None,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    collate_fn=custom_collate_fn,
)

valid_dataloader = DataLoader(
    valid_DataSet,
    batch_size=32,
    shuffle=True,
    sampler=None,
    batch_sampler=None,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    collate_fn=custom_collate_fn,
)


class My_model(nn.Module):
    def __init__(self, num_classes):
        super(My_model, self).__init__()
        self.backbone = torchvision.models.vgg16_bn(
            weights=VGG16_BN_Weights.IMAGENET1K_V1
        ).to("cuda")
        for i, param in enumerate(self.backbone.parameters()):
            param.requires_grad = False
            self.backbone.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == "__main__":
    num_classes = len(train_DataSet.name2label)
    model = My_model(num_classes=num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    num_epochs = 3
    # start trainning
    device = torch.device("cuda")

    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, (spec, target) in enumerate(train_dataloader):
            if spec is None or target is None:
                continue

            print(f"Processing step {step}")

            try:
                spec = augmentor(spec).to(device)
                target = target.to(device)
            except Exception as e:
                print(f"Error cuda processing step {step}")
                continue
            target = torch.argmax(target, dim=1)
            optimizer.zero_grad()
            spec.requires_grad_(True)
            outputs = model(spec)
            print(f"{step}\n,outputs:{outputs}\n,target:{target}")
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader)}"
        )
    torch.save(model.state_dict(), "model.pth")
    print("MODEL_PARAM has saved in model.pth")
