from PIL import Image


def render(data, is_full_scale=False):
    if is_full_scale:
        img = Image.fromarray(data.reshape(28, 28))  # Assuming already scaled
    else:
        img = Image.fromarray((data * 255).reshape(28, 28))  # Scale all 1's to 255
    img.show()  # Show the image


def validation(model, device, num_examples, x, y):
    correct_count_val = 0

    predictions, y_trues, correct_count_train_per_class = [0] * 10, [0] * 10, [0] * 10

    for i in range(num_examples):
        pred = model.predict(x[i, :].to(device))

        if pred == y[i]:
            correct_count_val += 1
            correct_count_train_per_class[y[i]] += 1

        predictions[pred] += 1
        y_trues[y[i]] += 1

    return correct_count_val, predictions, y_trues, correct_count_train_per_class
