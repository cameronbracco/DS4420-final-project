from PIL import Image


def render(data, is_full_scale=False):
    if is_full_scale:
        img = Image.fromarray(data.reshape(28, 28))  # Assuming already scaled
    else:
        img = Image.fromarray((data * 255).reshape(28, 28))  # Scale all 1's to 255
    img.show()  # Show the image


def validation(model, device, num_examples, x, y):
    correct_count_val = 0
    for i in range(num_examples):
        pred, spikeCounts = model.forward(x[i, :].to(device))

        if pred == y[i]:
            correct_count_val += 1

    return correct_count_val
