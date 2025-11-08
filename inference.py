import argparse
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from models.unet import UNET
from models.resnet34_unet import RESNET_34_UNET
import torch
from utils import dice_score
from tqdm import tqdm


def infer(args):
    # Define the parameters
    model_path = "saved_models\\" + args.model + ".pth"
    data_path = args.data_path
    batch_size = args.batch_size
    model_choice = args.model

    # Define the dataloader
    test_dataset = load_dataset(data_path, mode="test")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_choice=="unet":
        model = UNET(in_channel=3).to(device)
    else:
        model = RESNET_34_UNET(in_channel=3).to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to("cpu")

    #Compute the dice score
    avg_dice_score = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            images, masks = batch["image"].to(device, dtype=torch.float32), batch["mask"].to(device, dtype=torch.float32)
            outputs = model(images)  # Forward pass
            outputs = outputs.squeeze(1)  # Remove channel dim if needed
            masks = masks.squeeze(1)  # Remove channel dim if needed
            masks = masks.long()  # Convert float labels to long so that it is supported by the loss function
            dice = dice_score(outputs, masks, model_choice)
            avg_dice_score += dice
        avg_dice_score = avg_dice_score/len(test_dataloader)
        print ("Average dice score: ", avg_dice_score)



def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='resnet or unet')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    infer(args)