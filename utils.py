import torch
import numpy as np
import matplotlib.pyplot as plt
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from models.unet import UNET
from models.resnet34_unet import RESNET_34_UNET
from tqdm import tqdm

#Compute the dice score
def dice_score(pred_mask, gt_mask, model_choice):
    if model_choice=="unet":
        pred_mask = torch.argmax(pred_mask, dim=1)  # Convert logits to class indices
    else:
        pred_mask = (pred_mask > 0.5).float()

    # Ensure both tensors have the same shape
    pred_mask = pred_mask.view(-1)
    gt_mask = gt_mask.view(-1)

    intersection = torch.sum((pred_mask == gt_mask) * (gt_mask > 0))  # Count correct foreground pixels
    union = torch.sum(pred_mask > 0) + torch.sum(gt_mask > 0)

    if union == 0:
        return 1.0  # If no foreground in both, Dice is 1 (perfect match)
    
    dice = 2.0 * intersection / union
    return dice.item()  # Convert to scalar for readability

#Save the plots to /plot directory
def show_plot(x, x_label, y, y_label, title):
    import os
    plt.plot(x,y, color="maroon")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    file_path = os.path.join('plots', title+'png')
    plt.savefig(file_path)
    plt.close()

#Di
def show_evolution(model_choice):
    data_path = "dataset/oxford-iiit-pet"
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = load_dataset(data_path, mode="test")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Get a batch of images and masks
    batch = next(iter(test_dataloader))
    images, masks = batch["image"].to(device, dtype=torch.float32), batch["mask"].to(device, dtype=torch.float32)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))  # Adjust height for better spacing

    d = {0:"5", 1:"10", 2:"20"}

    for i in tqdm(range(3)):
        model_path = "saved_models/"+model_choice+"_epoch_no"+d[i]+".pth"
        if model_choice=="unet":
            model = UNET(in_channel=3).to(device)
        else:
            model = RESNET_34_UNET(in_channel=3).to(device)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.to("cpu")
        model.eval()

        with torch.no_grad():
            outputs = model(images)  # Forward pass
        outputs = outputs.squeeze(1)  # Remove channel dim if needed
        masks = masks.squeeze(1)  # Remove channel dim if needed
        masks = masks.long()  # Convert float labels to long so that it is supported by the loss function
        if model_choice == "unet":
            predicted_masks = torch.argmax(outputs, dim=1)  # Convert logits to class indices
        else:
            predicted_masks = (outputs > 0.5).float()

        image_np = images[0].cpu().numpy().transpose(1, 2, 0)  # Change shape from (C, H, W) to (H, W, C)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to 0-1
        true_mask_np = masks[0].cpu().numpy()
        pred_mask_np = predicted_masks[0].cpu().numpy()

        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true_mask_np, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask_np, cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

        # Add epoch label to the left of each row
        axes[i, 0].set_ylabel(f"Epoch {i}", fontsize=14, labelpad=20, rotation=90, weight="bold")

    plt.subplots_adjust(left=0.1)  # Adjust space to show y-axis labels
    plt.tight_layout()
    plt.show()



def visualize_predictions(model_choice):
    data_path = "dataset/oxford-iiit-pet"
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = load_dataset(data_path, mode="test")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Get a batch of images and masks
    batch = next(iter(test_dataloader))
    images, masks = batch["image"].to(device, dtype=torch.float32), batch["mask"].to(device, dtype=torch.float32)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))  # Adjust height for better spacing

    model_path = f"saved_models/"+model_choice+".pth"
    if model_choice=="unet":
        model = UNET(in_channel=3).to(device)
    else:
        model = RESNET_34_UNET(in_channel=3).to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to("cpu")
    model.eval()

    for i in tqdm(range(3)):
        with torch.no_grad():
            outputs = model(images)  # Forward pass
        outputs = outputs.squeeze(1)  # Remove channel dim if needed
        masks = masks.squeeze(1)  # Remove channel dim if needed
        masks = masks.long()  # Convert float labels to long so that it is supported by the loss function
        if model_choice=="unet":
            predicted_masks = torch.argmax(outputs, dim=1)  # Convert logits to class indices
        else:
            predicted_masks = (outputs > 0.5).float()


        image_np = images[i].cpu().numpy().transpose(1, 2, 0)  # Change shape from (C, H, W) to (H, W, C)
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to 0-1
        true_mask_np = masks[i].cpu().numpy()
        pred_mask_np = predicted_masks[i].cpu().numpy()

        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true_mask_np, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask_np, cmap="gray")
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis("off")

        # Add epoch label to the left of each row
        axes[i, 0].set_ylabel(f"Epoch {i}", fontsize=14, labelpad=20, rotation=90, weight="bold")

    plt.subplots_adjust(left=0.1)  # Adjust space to show y-axis labels
    plt.tight_layout()
    plt.show()
