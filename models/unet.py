import torch
import torch.nn as nn

#This block implement the 2 blue arrows that we can see on the explainative schema
class Horizontal_block(nn.Module):
    def __init__(self, nb_input_channel, nb_output_channel):
        super().__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(nb_input_channel, nb_output_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nb_output_channel, nb_output_channel,  kernel_size=3, padding=1),
            nn.ReLU())
        
    def forward(self, input):
        return self.operation(input)

   
class Down_block(nn.Module):
    def __init__(self, nb_input_channel, nb_output_channel):
        super().__init__()
        self.horizontal = Horizontal_block(nb_input_channel, nb_output_channel)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        intermediate_value = self.horizontal.forward(input)
        final_value = self.down(intermediate_value)
        return intermediate_value, final_value
    

class Up_block(nn.Module):
    def __init__(self, nb_input_channel, nb_output_channel):
        super().__init__()
        self.up = nn.ConvTranspose2d(nb_input_channel, nb_input_channel//2, kernel_size=2, stride=2)
        self.horizontal = Horizontal_block(nb_input_channel, nb_output_channel)

    def forward(self, input, intermediate_input):
        input = self.up(input)
        output = torch.cat([input, intermediate_input], 1)
        output = self.horizontal.forward(output)
        return output
        
class UNET(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.down1 = Down_block(in_channel, 64)
        self.down2 = Down_block(64, 128)
        self.down3 = Down_block(128, 256)
        self.down4 = Down_block(256, 512)
        self.bottom = Horizontal_block(512, 1024)
        self.up1 = Up_block(1024, 512)
        self.up2 = Up_block(512, 256)
        self.up3 = Up_block(256, 128)
        self.up4 = Up_block(128,64)
        self.last = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, input):
        inter1, input = self.down1.forward(input)
        inter2, input = self.down2.forward(input)
        inter3, input = self.down3.forward(input)
        inter4, input = self.down4.forward(input)
        input = self.bottom(input)
        input = self.up1.forward(input, inter4)
        input = self.up2.forward(input, inter3)
        input = self.up3.forward(input, inter2)
        input = self.up4.forward(input, inter1)
        output = self.last(input)
        return output