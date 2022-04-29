import torch
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def render(data, is_full_scale=False):
    if is_full_scale:
        img = Image.fromarray(data.reshape(28, 28))  # Assuming already scaled
    else:
        img = Image.fromarray((data * 255).reshape(28, 28))  # Scale all 1's to 255
    img.show()  # Show the image


def validation(model, device, num_examples, x, y):
    correct_count_val = 0

    predictions, y_trues, correct_count_train_per_class = [0] * 10, [0] * 10, [0] * 10

    preds = torch.zeros_like(y, device=device)

    x.to(device)
    for i in range(num_examples):
        pred = model.predict(x[i, :])
        preds[i] = pred

        if pred == y[i]:
            correct_count_val += 1
            correct_count_train_per_class[y[i]] += 1

        predictions[pred] += 1
        y_trues[y[i]] += 1

    acc_score = precision_score(y, preds)
    p_score = precision_score(y, preds, average='macro')
    r_score = precision_score(y, preds, average='macro')
    f_score = precision_score(y, preds, average='macro')

    return correct_count_val, predictions, y_trues, correct_count_train_per_class, (acc_score, p_score, r_score, f_score)
