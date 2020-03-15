import os
import torch
import torch.nn as nn  # package class based neural networks
import torch.optim as optim  # package of optimization algorithms
import torchvision  # package for computer vision
from torchvision.transforms import ToTensor  # convert PIL or numpy to tensor
import torchvision.transforms as transforms  # common image transformations
import torchvision.models as models  # contains definitions of computer vision models
import torchvision.datasets as datasets  # datasets mostly image based


def train_model(model, data_loaders, criterion, optimizer, epochs=50):
    """Our training function
    
    Args:
        model ([type]): The model we want to use
        data_loaders ([type]): our dataloader
        criterion ([type]): our loss
        optimizer ([type]): our optimizer
        epochs (int, optional): number of epochs to run. Defaults to 50.
    """

    # loop over the number of epochs
    for epoch in range(epochs):

        # print some debug information
        print(f"{'-'* 30}\nEpoch {epoch+1} / {epochs}")

        # iterate over the training and validation phases
        for phase in ["train", "val"]:

            # if phase is train, we tell the model to train
            if phase == "train":
                model.train()

            # else we evaluate the model
            else:
                model.eval()

            # some variables to keep track of the running loss and the total number of correct guess
            running_loss = 0.0
            correct = 0

            # iterate over the inputs and labels in our data loaders
            for inputs, labels in data_loaders[phase]:

                # send our inputs and labels to our GPU device
                inputs = inputs.cuda()
                labels = labels.cuda()

                # every time we enter a training loop, we need to zero out our optimizer
                optimizer.zero_grad()

                # when phase is set to train, enable gradient descent
                with torch.set_grad_enabled(phase == "train"):

                    # get the feed forwared for our model
                    outputs = model(inputs)

                    # calculate our loss
                    loss = criterion(outputs, labels)

                    # get the prediction classes for our outputs
                    _, predictions = torch.max(outputs, 1)

                    # if we are training, back proprogate and step our optimizer
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # keep track of our running loss
                running_loss += loss.item() * inputs.size(
                    0
                )  # de-reference the tensor with .item()

                # calculate the sum of where the predictions equals the labels.data will give us the
                # total number of correct prodictions
                correct += torch.sum(predictions == labels.data)

            # at the end of each epoch we want to keep track of the total epoch loss.
            epoch_loss = running_loss / len(data_loaders[phase].dataset)

            # as well as the total epoch accuracy
            epoch_acc = correct.double() / len(
                data_loaders[phase].dataset
            )  # double() converts the tensor to float64

            # define some print statements for losses and accuracies
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


if __name__ == "__main__":

    # define a root directory for our data
    root_dir = os.path.join("input", "hymenoptera_data")

    # define a series of image transforms for train and validate. The purpose of this is to
    # convert our images into tensors and prep the images into a form the the model can work with
    image_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation((-270, 270)),
                transforms.Resize((224, 224)),  # resnet expects this size as input
                transforms.ToTensor(),
                transforms.Normalize(
                    # what the model expects for mean and standard deviation as input
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.RandomRotation((-270, 270)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    # a generator object that maps the transforms to the images we are going to load from disk
    data_generator = {
        k: datasets.ImageFolder(os.path.join(root_dir, k), image_transforms[k])
        for k in ["train", "val"]
    }

    # every generator needs a loader.
    data_loader = {
        k: torch.utils.data.DataLoader(
            data_generator[k], batch_size=8, shuffle=True, num_workers=8
        )
        for k in ["train", "val"]
    }

    # handle instantiating our device. defaults to gpu else cpu.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # our model is resnet18 and use the pretrained version
    model = models.resnet18(pretrained=True)

    # freeze all the layers by turning off gradient descent
    for param in model.parameters():
        param.requires_grad = False

    # we need to know how many in features were going into the fully connected
    # layer for our new fully connected layer
    num_in_features = model.fc.in_features

    # override the fully connected layer with our own that has 2 outputs with
    # a linear classifier
    model.fc = nn.Linear(num_in_features, 2)

    # send it back to our defice
    model.to(device)

    # We use cross entropy loss for multi class classification
    criterion = nn.CrossEntropyLoss()

    # using adan optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # sanity check to make sure we are updating the parameters we think we are
    params_to_update = []

    # we loop over the parameters that require training and print them to the terminal
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            params_to_update.append(param)
            print(f"Training: {name}")

    # finally we train our model
    train_model(model, data_loader, criterion, optimizer)

    print("\nFinished.\n")

