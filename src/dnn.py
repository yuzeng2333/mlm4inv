import torch
import torch.nn as nn
import torch.nn.functional as F
from config import input_size, num_heads, num_layers, dim_feedforward, max_seq_len, model_file, num_epochs, MASK_IDX, AUG_DATA
from dataloader import GenDataloader, canonicalize
import torch.optim as optim
import numpy as np
import json

# Define the model
class DivisionModel(nn.Module):
    def __init__(self, var_num, input_dim, hidden_dim, output_dim):
        super(DivisionModel, self).__init__()
        self.var_num = var_num 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.feature_size = 16
        
        # Define the first layer (input to first hidden layer)
        self.fc1 = nn.Linear(self.input_dim, self.feature_size)
        
        # Define the second, third, and fourth layers (hidden layers)
        self.fc2 = nn.Linear(self.feature_size * self.var_num, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Define the final layer
        self.fc5 = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x):
        # Pass through the first layer and apply ReLU activation
        x = F.relu(self.fc1(x))
        x = x.view(-1, self.feature_size * self.var_num)
        # Pass through the second, third, and fourth layers with ReLU activation
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # Pass through the final layer
        x = self.fc5(x)
        return x


def group_lasso_regularization(model, weight_decay):
    reg_loss = 0.0
    # Iterate over the model parameters to find the fc2 layer weights
    for name, param in model.named_parameters():
        if not "fc1" in name and "weight" in name:
            tmp = torch.sum(param**2, dim=0)
            tmp2 = torch.sum(tmp)
            reg_loss += torch.sum(tmp2)
    return weight_decay * reg_loss


class ActivationRecorder:
    def __init__(self, start_epoch):
        self.activations_sum = {}
        self.counts = {}
        self.current_epoch = 0
        self.start_epoch = start_epoch

    def save_activation(self, name):
        def hook(model, input, output):
            # Only record activations if the current epoch is >= start_epoch
            if self.current_epoch >= self.start_epoch:
                # if the layer is not fc5, apply relu to the output
                if "fc5" not in name:
                    output = F.relu(output)
                if name not in self.activations_sum:
                    sum_shape = output.shape[1:]
                    self.activations_sum[name] = torch.zeros(*sum_shape, device=output.device)
                    self.counts[name] = 0  
                self.activations_sum[name] += output.detach().sum(dim=0)  # Sum across the batch dimension
                self.counts[name] += output.size(0)  # Increment the count by the batch size
        return hook
    
    def increment_epoch(self):
        self.current_epoch += 1

    def calculate_average_activations(self):
        # return a dictionary of average activations for each layer
        return {name: self.activations_sum[name] / self.counts[name] for name in self.activations_sum}


def calculate_importance_pytorch(model, activations):
    """
    Calculate the importance of neurons in a PyTorch model based on weights and activations.
    
    :param model: A PyTorch model from which to extract weights.
    :param activations: A list of activation tensors for each layer of the model.
    :return: A list of importance scores for neurons in each layer.
    """
    # Extract weights for each layer from the model
    weights = [param.data for name, param in model.named_parameters() if "weight" in name]
    weight_name = [name for name, param in model.named_parameters() if "weight" in name]

    # Reverse the weights and activations lists to start calculations from the output layer
    weights.reverse()
    
    # Extract activations for each layer from the dictionary of activations
    activation_values = list(activations.values())
    activation_name = list(activations.keys())
    activation_values.reverse()
    
    # The importance of the output layer is initially set to its activations
    importance = [torch.tensor(1)]

    # calculate the shape of each weight
    weight_shapes = [weight.shape for weight in weights]
    print ("weight_shapes: ", weight_shapes)
    # get the shape of each activation
    activation_shapes = [activation.shape for activation in activation_values]
    print ("activation_shapes: ", activation_shapes)

    #importance = importance * weight[0] # [1] * [1, 64] = [1, 64]
    #importantce = importance * activation[1] # [1, 64] * [64] = [1, 64]
    #importance = importance * weight[1] # [1, 64] * [64, 64] = [1, 64]
    #importance = importance * activation[2] # [1, 64] * [64] = [1, 64]
    #importance = importance * weight[2] # [1, 64] * [64, 64] = [1, 64]
    #importance = importance * activation[3] # [1, 64] * [64] = [1, 64]
    #importance = importance * weight[3] # [1, 64] * [64, 48] = [1, 48]
    ## flattern activation[4]
    #activation[4] = activation[4].view(-1)
    #importance = importance * activation[4] # [1, 48] * [48] = [1, 48]

    # Calculate importance scores for each layer, propagating back from the output
    for i in range(len(weights) - 1):
        # Ensure weights and activations are on the same device and are compatible for multiplication
        weight = weights[i].detach().cpu().numpy()
        activation = activation_values[i+1].detach().cpu().numpy()  # +1 skips input layer activation
        
        # Calculate the importance for the current layer
        # flattern activation
        activation = activation.flatten()
        # print the shape of weight, importance, and activation
        print ("--- weight: ", weight.shape)
        print ("--- importance: ", importance[-1].shape)
        print ("--- activation: ", activation.shape)
        tmp = np.dot(importance[-1], weight)
        layer_importance = activation * tmp
        importance.append(layer_importance)
     
    return importance[-1]


def pick_scores(scores):
    """
    Picks scores from a list according to the specified rule:
    Starts with the largest score, then keeps picking the next largest score
    until the next largest score is less than 0.2 times the last picked score.

    Parameters:
    scores (list): A list of scores (floats or integers).
    
    Returns:
    dict: A dictionary where the key is the original position of the score in the list,
          and the value is the score itself.
    """
    # Sort the scores along with their original indices
    sorted_scores_with_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    
    # Initialize the dictionary to store selected scores and their positions
    selected_scores = {}
    
    # Keep track of the last picked score
    last_picked_score = sorted_scores_with_indices[0][1]  # Start with the highest score
    
    for index, score in sorted_scores_with_indices:
        if score < last_picked_score * 0.2:
            break  # Stop if the next score is less than 0.2 times the last picked score
        selected_scores[index] = score
        last_picked_score = score  # Update the last picked score
    
    return selected_scores


# the labels contain the set of used variables
# read it with json
def load_used_var_set(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    all_used_var_set = []
    for used_var_list in data["used_var"]:
        used_var_set = set(used_var_list)
        all_used_var_set.append(used_var_set)
    
    return all_used_var_set

    
def get_dependent_var(all_used_var_set, masked_var):
    dependent_var = set()
    for used_var_set in all_used_var_set:
        if masked_var in used_var_set:
            dependent_var = dependent_var.union(used_var_set)
    # remove the masked_var from the dependent_var
    dependent_var.remove(masked_var)
    return dependent_var

def get_var_indices(var_dict, var_set):
    var_indices = []
    for var in var_set:
        var_indices.append(var_dict[var])
    # sort
    var_indices.sort()
    return var_indices

# Example usage (assuming you have a model and activations list)
# importance_scores = calculate_importance_pytorch(model, activations)

# Note: This is a conceptual example. In a real use case, ensure that 'activations' are collected properly using hooks,
# and consider the structure of your specific model when implementing the logic to traverse layers and calculate importance.

def dnn_train(args, file_path):
    COMP_ALL = 0
    USE_TRANSFORMER = 1
    PRINT_PARAMS = 1
    USE_MASK = 1
    USE_RAND = 0
    USE_EXTRA_TOKEN = False
    if args.all:
        RUN_ALL_BATCH = 1
    else:
        RUN_ALL_BATCH = 0
    # check the visibility of the cuda
    print('cuda is available: ', torch.cuda.is_available())
    print('cuda device count: ', torch.cuda.device_count())
  
    var_num = 5
    input_dim = 5
    model = DivisionModel(var_num, input_dim=input_dim, hidden_dim=64, output_dim=1)
    model = torch.nn.DataParallel(model)
    device = args.device
    batch_size = args.batch_size
    model.to(device)
    # if file_path is xxx/data/1.csv, then label_file_path is xxx/label/1.json
    label_file_path = file_path.replace("data", "label").replace("csv", "json")
    used_var_set = load_used_var_set(label_file_path)
    dataloader, var_dict = GenDataloader(file_path, batch_size, device, aug_data=AUG_DATA, shuffle=False)
    masked_var = var_dict[MASK_IDX]
    dependent_var = get_dependent_var(used_var_set, masked_var)
    dependent_var_indices = get_var_indices(var_dict, dependent_var)
    if AUG_DATA:
        criterion = nn.MSELoss(reduction='sum').to(device)
    else:
        criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate is 0.001 by default

    if args.load:
        state_dict = torch.load(model_file)
        model.load_state_dict(state_dict)

    activation_recorder = ActivationRecorder(start_epoch=args.epoch - 10)
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            layer.register_forward_hook(activation_recorder.save_activation(name))

    # Training Loop
    for epoch in range(args.epoch):
        model.train()  # Set the model to training mode
        activation_recorder.increment_epoch()
        total_loss = 0
        total_weight_loss = 0

        if RUN_ALL_BATCH:
            max_batch_idx = len(dataloader)
        else: 
            max_batch_idx = 96
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_idx > max_batch_idx:
                break
            batch_data = batch_data[0]
            #batch_data = sort_tensor(batch_data, MASK_IDX)
            if not AUG_DATA:
                batch_data = canonicalize(batch_data, 2)
            # Masking a random element in each sequence of the batch
            if USE_RAND:
                mask_indices = np.random.randint(max_seq_len-1, size=batch_size)
            else:
                mask_indices = np.full(batch_size, MASK_IDX)
            mask_indices = torch.tensor(mask_indices)
            read_data = batch_data.clone()
            masked_data = batch_data.clone()

            # Create a tensor of batch indices
            batch_indices = torch.arange(masked_data.size(0)).to(device)

            # Mask the data
            # This will set masked_data[i, idx, :] to random values for each i and corresponding idx
            #masked_data = masked_data[:, MASK_IDX+1:MASK_IDX+4, :]
            masked_data = torch.cat((masked_data[:, :MASK_IDX, :], masked_data[:, MASK_IDX+1:, :]), dim=1)
            if var_num != masked_data.size(1):
                print("The number of variables is not correct:")
                print(masked_data.size(1))
                return

            # Forward pass
            ret = model(masked_data)
            # flatten ret
            output = ret.squeeze()

            # weight loss
            weight_loss = group_lasso_regularization(model, 0.0005)

            masked_data = read_data[batch_indices, mask_indices, 2]
            # Calculate loss only for the masked elements
            loss = criterion(output, masked_data) + weight_loss
  
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
  
            total_loss += loss.item()
            total_weight_loss += weight_loss.item()
  
        avg_loss = total_loss / len(dataloader)
        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{args.epoch}, Average Loss: {avg_loss}")
            print(f"Epoch {epoch+1}/{args.epoch}, Average Weight Loss: {total_weight_loss / len(dataloader)}")
  
        # Add validation step here if you have a validation dataset
        if epoch % 100 == 0:
        #  #if PRINT_PARAMS:
        #  #  for name, param in model.named_parameters():
        #  #    print(f"Name: {name}")
        #  # print attn weights
        #  print("attn_weights0: ", avg_batch_weights[0])
        #  #print("====================")
        #  #print("attn_weights1: ", attn_weights[0][1])
        #  #print("attn_weights1: ", attn_weights[1][1])
        #  #print("attn_weights1: ", attn_weights[2][1])
        #  #print("====================")
        #  #print("attn_weights2: ", attn_weights[0][2])
        #  #print("attn_weights2: ", attn_weights[1][2])
        #  #print("attn_weights2: ", attn_weights[2][2])
        #  # Save the trained model
            torch.save(model.state_dict(), model_file)
            print(f"Random Sample at Epoch {epoch+1}:")
            print("Masked Output:", output.detach().cpu().numpy())
            print("Masked Data:", masked_data.cpu().numpy())

    average_activations = activation_recorder.calculate_average_activations()
    importance_scores = calculate_importance_pytorch(model, average_activations)
    # convert the importance scores to a tensor
    importance_scores = torch.tensor(importance_scores[0])
    # divide the scores into 3 groups
    importance_scores = importance_scores.view(var_num, -1)
    # sum the importance scores for each group
    importance_scores = importance_scores.sum(dim=1)
    print(importance_scores)
    # pick the scores
    selected_scores = pick_scores(importance_scores)
    # adjust the indices of the selected scores
    # if the index is smaller than MASK_IDX, then the index is the same
    # if the index is larger than or equal to MASK_IDX, then the index should be increased by 1
    selected_scores = {i: selected_scores[i] for i in selected_scores if i < MASK_IDX}
    selected_scores.update({i+1: selected_scores[i] for i in selected_scores if i >= MASK_IDX})
    # collect the indices
    selected_indices = list(selected_scores.keys())
    # sort the indices
    selected_indices.sort()
    if dependent_var_indices == selected_indices:
        print("The selected indices are the same as the dependent var indices")
    else:
        print("The selected indices are not the same as the dependent var indices")
        print("The selected indices : ", selected_indices)
        print("The dependent indices: ", dependent_var_indices)