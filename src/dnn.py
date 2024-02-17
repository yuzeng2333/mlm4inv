import torch
import torch.nn as nn
import torch.nn.functional as F
from config import input_size, num_heads, num_layers, dim_feedforward, max_seq_len, model_file, num_epochs, MASK_IDX, AUG_DATA
from dataloader import GenDataloader, canonicalize
import torch.optim as optim
import numpy as np

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
    importance = [activation_values[0].detach().cpu().numpy()]
    
    # Calculate importance scores for each layer, propagating back from the output
    for i in range(len(weights) - 1):
        # Ensure weights and activations are on the same device and are compatible for multiplication
        weight = weights[i].detach().cpu().numpy()
        activation = activation_values[i + 1].detach().cpu().numpy()  # +1 skips input layer activation
        
        # Calculate the importance for the current layer
        layer_importance = np.dot(weight.T, importance[-1] * activation)
        importance.append(layer_importance)
    
    # Reverse the importance list to match the original layer order
    importance.reverse()
    
    return importance


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
  
    var_num = 3
    input_dim = 5
    model = DivisionModel(var_num, input_dim=input_dim, hidden_dim=64, output_dim=1)
    model = torch.nn.DataParallel(model)
    device = args.device
    batch_size = args.batch_size
    model.to(device)
    dataloader = GenDataloader(file_path, batch_size, device, aug_data=AUG_DATA, shuffle=False)
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
            masked_data = masked_data[:, MASK_IDX+1:MASK_IDX+4, :]
            assert var_num == masked_data.size(1)

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
    print(importance_scores)