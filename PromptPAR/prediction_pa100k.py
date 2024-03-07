### python prediction_pa100k.py PA100k
## --use_textprompt --use_div --use_vismask --use_GL --use_mm_former (dont need now)




import torch
from torchvision import transforms
from PIL import Image

from models.base_block import TransformerClassifier
from tools.utils import set_seed
from clip import clip
from clip.model import *
set_seed(605)
device = "cuda"

# Define the path to the image
image_path = '/media/dev/HDD_2TB/newopenpar/OpenPAR/PromptPAR/dataset/PA100/Pad_datasets/018911.jpg'

# Load the image
image = Image.open(image_path)

# Define the transformation function to preprocess the image
def get_transform(height, width):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    valid_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        normalize
    ])
    return valid_transform

def main():
    attr_num = 26
    attributes = [
        'A female pedestrian', 'A pedestrian over the age of 60', 'A pedestrian between the ages of 18 and 60',
        'A pedestrian under the age of 18', 'A pedestrian seen from the front', 'A pedestrian seen from the side',
        'A pedestrian seen from the back', 'A pedestrian wearing a hat', 'A pedestrian wearing glasses',
        'A pedestrian with a handbag', 'A pedestrian with a shoulder bag', 'A pedestrian with a backpack',
        'A pedestrian holding objects in front', 'A pedestrian in short-sleeved upper wear',
        'A pedestrian in long-sleeved upper wear', 'A pedestrian in stride upper wear',
        'A pedestrian in upper wear with a logo', 'A pedestrian in plaid upper wear',
        'A pedestrian in splice upper wear', 'A pedestrian in striped lower wear', 'A pedestrian in patterned lower wear',
        'A pedestrian in a long coat', 'A pedestrian in trousers', 'A pedestrian in shorts',
        'A pedestrian in skirts and dresses', 'A pedestrian wearing boots'
    ]
   
    # Load the CLIP model
    clip_model, _ = clip.load("ViT-L/14", device=device, download_root='/media/dev/HDD_2TB/newopenpar/OpenPAR/PromptPAR/data/weights')

    # Define the model architecture
    model = TransformerClassifier(clip_model, attr_num, attributes)

    # Load the trained weights
    model_path = '/media/dev/HDD_2TB/newopenpar/OpenPAR/PromptPAR/logs/PA100k/2024-03-07_13_13_03/epoch1.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Define the transformation for the input image
    height, width = 224, 224  # Define the desired height and width for the input image
    transform = get_transform(height, width)
    input_image = transform(image).unsqueeze(0).to(device)  # Add a batch dimension and move to device
    model.eval()
    # Perform inference
    with torch.no_grad():
        output, _ = model(input_image, clip_model)

    # Apply sigmoid activation to convert logits to probabilities
    probabilities = torch.sigmoid(output)

    # Define a threshold for making binary predictions
    threshold = 0.5
    binary_predictions = (probabilities > threshold).int()

    # Print or use the predictions as needed
    for i in range(len(attributes)):
        if binary_predictions[0][i] == 1:
            print(f"Attribute '{attributes[i]}' is present.")

if __name__ == '__main__':
    main()
