import subprocess
from utils import visualize_predictions, show_evolution


def evaluate(net, data, device):
    # implement the evaluation function here

    assert False, "Not implemented yet!"


if  __name__=="__main__":
    print("Please choose your model to evaluate (unet or resnet34_unet)")
    model_choice = input()
    print ("\n What do you wish to see ?")
    print("Type 1 to see the average dice score on the test data")
    print("Type 2 to visualise the predicition on 3 unseen examples")
    print("Type 3 to see the evolution of the predicition along the epochs")
    choice = int(input())
    if choice==1:
        command = ["python", "src/inference.py", "--model", model_choice, "--data_path", "dataset/oxford-iiit-pet", "--batch_size", "32"]
        subprocess.run(command)
    elif choice==2:
        visualize_predictions(model_choice)
    elif choice==3:
        show_evolution(model_choice)
    
