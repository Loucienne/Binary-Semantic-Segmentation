import argparse
import torch
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from models.unet import UNET
from models.resnet34_unet import RESNET_34_UNET
from tqdm import tqdm
from utils import dice_score, show_plot

def train(args):
    # Define the parameters
    data_path = args.data_path
    EPOCH = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    model_choice = args.model

    #Define the dataloaders
    train_dataset = load_dataset(data_path, mode="train")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = load_dataset(data_path, mode="valid")
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    #Create the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_choice=="unet":
        model = UNET(in_channel=3).to(device)
    else:
        model = RESNET_34_UNET(in_channel=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initalise for plotting
    abscisse_pour_plot=[]
    learning_curve_y=[]
    dice_score_curve_y=[]

    #train loop
    for epoch in range(EPOCH):
        model.train()
        epoch_loss = 0
        print("Epoch: ", epoch+1, "/", EPOCH)

        for batch in tqdm(train_dataloader):
            images, masks = batch["image"].to(device, dtype=torch.float32), batch["mask"].to(device, dtype=torch.float32)
            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass

            # Ensure masks and outputs have the same shape for loss calculation
            outputs = outputs.squeeze(1)  # Remove channel dim if needed
            masks = masks.squeeze(1)  # Remove channel dim if needed
            if model_choice=="unet":
                masks = masks.long()  # Convert float labels to long so that it is supported by the loss function

            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(outputs, masks)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            epoch_loss += loss.item()

        #Show the performance on the training data
        avg_loss = epoch_loss / len(train_dataloader)  # Compute epoch loss
        print("Training loss: ", avg_loss,)

        #Calculate the validation loss for this epoch and the dice score
        epoch_validation_loss = 0
        epoch_dice_score = 0
        print("validation computation")
        model.eval()
        with torch.no_grad():
            for batch in tqdm(validation_dataloader):
                # Get the model output
                images, masks = batch["image"].to(device, dtype=torch.float32), batch["mask"].to(device, dtype=torch.float32)
                outputs = model(images)  # Forward pass
                outputs = outputs.squeeze(1)  # Remove channel dim if needed
                masks = masks.squeeze(1)  # Remove channel dim if needed
                if model_choice=="unet":
                    masks = masks.long()  # Convert float labels to long so that it is supported by the loss function
                # Calculate the loss
                loss = loss_function(outputs, masks)  # Compute loss
                epoch_validation_loss += loss.item()
                #Calculate the dice score
                dice = dice_score(outputs, masks, model_choice)
                epoch_dice_score += dice
            avg_validation_loss = epoch_validation_loss / len(validation_dataloader)
            avg_dice_score = epoch_dice_score / len(validation_dataloader)

            # Fill the informations on dice score and loss to plot them later
            abscisse_pour_plot.append(epoch)
            learning_curve_y.append(avg_validation_loss)
            dice_score_curve_y.append(avg_dice_score)

            #Show the performance on the validation data
            print("Validation loss: ", avg_validation_loss, "   ||   Dice score: ", avg_dice_score)

        # Save intermediate model
        model_save_path = "saved_models//"+str(model_choice)+"_epoch_no"+str(epoch)+".pth"
        torch.save(model.state_dict(), model_save_path)


    #Save the model
    model_save_path = "saved_models//"+str(model_choice)+".pth"
    torch.save(model.state_dict(), model_save_path)

    # Save the plot
    show_plot(abscisse_pour_plot, "Number of epoch", learning_curve_y, "Loss", "Learning curve(resnet)")
    show_plot(abscisse_pour_plot, "Number of epoch", dice_score_curve_y, "Dice score", "Evolution of the dice score(resnet)")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model', '-m', type=str, help='resnet or unet')


    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)

