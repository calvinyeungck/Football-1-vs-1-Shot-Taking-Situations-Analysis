
import torch


def epoch(model, dataloader, optimizer, device):
    # Set the model to train mode
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    # Initialize variables to keep track of loss and accuracy
    total_loss = 0
    total_examples = 0
    total_loss_noreg = 0

    # Loop over the batches of data
    for batch in dataloader:
        # Get the inputs and targets for the batch
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the model outputs
        outputs = model(inputs)
        # Compute the loss
        loss = model.compute_loss(outputs, targets) 
        loss_noreg = loss.detach().clone()
        # Add L1 and L2 regularization terms to the loss
        # l1_loss = model.l1_loss()
        # l2_loss = model.l2_loss()
        # loss += l1_loss + l2_loss
        # pdb.set_trace()
        # Compute the number of correct predictions in this batch


        # Update the loss and accuracy totals
        total_loss += loss.item() * inputs.size(0)
        total_loss_noreg += loss_noreg.item() * inputs.size(0)
        total_examples += inputs.size(0)

        # Zero the gradients, backpropagate the loss, and update the weights
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Compute the average loss and accuracy over the epoch
    average_loss = total_loss / total_examples
    avg_loss_noreg = total_loss_noreg / total_examples
    return average_loss, avg_loss_noreg

def train_loop(model, train_dataloader, valid_dataloader, optimizer, device, num_epochs=10, verbose=True):
    # Initialize lists to store the training and validation metrics for each epoch
    train_loss_history = []
    train_loss_nr_history = []
    valid_loss_history = []
    valid_loss_nr_history = []
    early_stop_threshold = 999
    early_stop_count = 0
    # Loop over the specified number of epochs
    for epoch_idx in range(num_epochs):
        # Train the model on the training data for one epoch
        train_loss, train_loss_nr = epoch(model, train_dataloader, optimizer, device)

        # Evaluate the model on the validation data for one epoch
        with torch.no_grad():
            valid_loss, valid_loss_nr = epoch(model, valid_dataloader, None, device)

        # Store the training and validation metrics for this epoch
        train_loss_history.append(train_loss)
        train_loss_nr_history.append(train_loss_nr)
        valid_loss_history.append(valid_loss)
        valid_loss_nr_history.append(valid_loss_nr)

        if valid_loss_nr < early_stop_threshold:
            early_stop_threshold = valid_loss_nr
            model_params = model.state_dict()

        #early stopping
        # if valid_loss_nr > early_stop_threshold:
        #     early_stop_count += 1
        #     if early_stop_count > 10:
        #         print("early stopping")
        #         break

        # Print the training progress
        if verbose:
            print(f"Epoch {epoch_idx+1}/{num_epochs}: train_loss={train_loss_nr:.4f}, valid_loss={valid_loss_nr:.4f}")

    return train_loss_history, valid_loss_history, train_loss_nr_history, valid_loss_nr_history, model_params