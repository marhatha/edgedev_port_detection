import torch

def run_inference(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        predictions = torch.sigmoid(outputs) > 0.5
    return predictions.numpy()

def run_inference_raw_output(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    return torch.sigmoid(outputs).numpy()
