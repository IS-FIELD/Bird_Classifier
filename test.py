from train import *

num_classes = len(valid_DataSet.name2label)
model = My_model(num_classes=num_classes).cuda()
correct = 0
total = 0
model.load_state_dict(torch.load("model.pth"))
device = torch.device("cuda")
model.to(device)
model.eval()


if __name__ == "__main__":
    with torch.no_grad():
        for step, (spec, target) in enumerate(valid_dataloader):
            if spec is None or target is None:
                continue
            spec = spec.cuda()
            target = target.cuda()
            outputs = model(spec)
            print(f"outputs:,{outputs}")
            predicted = torch.argmax(outputs, 1)
            print(f"predicted:,{predicted}")
            total += target.size(0)
            predicted = predicted.cuda()
            target = torch.argmax(target, dim=1).cuda()
            correct += (predicted == target).sum().item()
            print("t", target, step)
            print("p", predicted, step)
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")
