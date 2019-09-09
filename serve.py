import torch


# Function that takes loads in our pickled word processor
# and defines a function for using it. This makes it easy
# to do these steps together when serving our model.
def model_loader():
    
    # read in pickled word processor. You could also load in
    # other models as this step.
    model = torch.load(("Model/model_plant.pt"))
    model.eval()
    return model
    
    
    # Function to apply our model & extract keywords from a 
   
