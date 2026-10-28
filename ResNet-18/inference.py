from dataset import val_loader
from model import ResNet_LameAhh
import torch

def main():
    device = 'cpu'
    model = ResNet_LameAhh().to(device)
    model.load_state_dict(torch.load("ResNet-18/ResNet_LameAhh.pt", map_location=device))
    model.eval()

    all_probs = []
    all_labels = []
    cm = torch.zeros(10, 10)

    with torch.no_grad():
        for image, label in val_loader:
            output_logits = model(image)
            y_preds = torch.argmax(output_logits, dim=1)
            probs = torch.softmax(output_logits, dim=1)

            all_probs.append(probs) # (256, 10) per batch
            all_labels.append(y_preds) # (256,) per batch

            for true, pred in zip(label, y_preds):
                cm[true, pred] += 1

    # (256, ...) + (256, ...) + ... = (3925, 10)
    all_probs = torch.cat(all_probs, dim=0)
    # (B_1, ...) + (B_2, ...) + ... = (3925,)
    all_labels = torch.cat(all_labels, dim=0)

    true_positive = cm.diag()
    false_postive = cm.sum(dim=0) - cm.diag()
    false_negative = cm.sum(dim=1) - cm.diag()

    precision = true_positive / (true_positive + false_postive).clamp(min=1)
    recall = true_positive / (true_positive + false_negative).clamp(min=1)
    accuracy = cm.diag().sum() / cm.sum()
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)


    torch.save({"probs": all_probs, "labels": all_labels}, "ResNet-18/val_probs.pt")

    print("Precision", precision.mean().item())
    print("Recall", recall.mean().item())
    print("F1", f1.mean().item())
    print("Accuracy", accuracy.item())

if __name__ == '__main__':
    main()