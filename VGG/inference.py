from dataset import val_loader
from model import VGG_D_LameAhh
import torch


def main():
    device = 'cpu'
    model = VGG_D_LameAhh().to(device)
    model.load_state_dict(torch.load('VGG/VGG_D_LameAhh.pt', map_location=device))
    model.eval()

    all_probs = []
    all_labels = []

    cm = torch.zeros(10, 10)
    with torch.no_grad():
        for images, labels in val_loader:
            outputs_logits = model(images)
            y_preds = torch.argmax(outputs_logits, dim=1)
            probs = torch.softmax(outputs_logits, dim=1)

            all_probs.append(probs) # (256, 10) each batch
            all_labels.append(labels) # (256,) each batch
            
            for true, pred in zip(labels, y_preds):
                cm[true, pred] += 1

    all_probs = torch.cat(all_probs, dim=0) # (3925, 10)
    all_labels = torch.cat(all_labels, dim=0) # (3925,)

    true_positive = cm.diag()
    # dim = x : the x-th index is the one that gets added up
    # and disappear; while others stay as the result
    false_positive = cm.sum(dim=0) - cm.diag()
    false_negative = cm.sum(dim=1) - cm.diag()

    precision = true_positive/(true_positive + false_positive).clamp(min=1)
    recall = true_positive/(true_positive + false_negative).clamp(min=1)
    f1 = 2 * ((precision * recall) / (precision + recall).clamp(min=1e-12))

    torch.save({"probs": all_probs, "labels": all_labels}, "VGG/val_probs.pt")

    print("Precision:", precision.mean().item())
    print("Recall:", recall.mean().item())
    print("F1:", f1.mean().item())

if __name__ == '__main__':
    main()