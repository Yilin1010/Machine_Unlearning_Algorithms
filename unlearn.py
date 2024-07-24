import torch
from train import test
import utils
device = utils.device_config()
utils.set_seed()


# unleaning algorithm use selective ascend or descent gradients the trained model

def default_select_grads(retain_grads, forget_grads):
    selected_grads = []
    print("no select_grad function provided")
    return selected_grads

def select_grads_fn(retain_grads, forget_grads, diff_threshold_ratio=0.8, magnitude_threshold_ratio=0.5):
    selected_grads = []

    for rg, fg in zip(retain_grads, forget_grads):
        # Initialize selected_grads to zeros
        selected_grad = torch.zeros_like(rg)

        # Compute the quantile thresholds
        diff_threshold = (rg - fg).quantile(diff_threshold_ratio)
        magnitude_threshold = fg.abs().quantile(magnitude_threshold_ratio)

        # Condition 1: Different signs, select retain_grad
        sign_diff = (rg * fg < 0)
        selected_grad = torch.where(sign_diff, rg, selected_grad)

        # Condition 2: Same sign
        sign_same = (rg * fg >= 0)

        # Sub-condition a: Retain_grad more large than forget_grad
        larger_retain = (rg.abs() - fg.abs() > diff_threshold)
        selected_grad = torch.where(sign_same & larger_retain, rg, selected_grad)

        # Sub-condition b: Forget_grad is small
        small_forget = (fg.abs() < magnitude_threshold)
        selected_grad = torch.where(sign_same & small_forget, -fg, selected_grad)

        # Any other condition results in zero, which is already set by initialization

        selected_grads.append(selected_grad)

    return selected_grads

# Function to filter parameters and their corresponding gradients
def filter_parameters_and_grads(model, filter_fn, grads):
    filtered_params = []
    filtered_grads = []
    for (name, param), grad in zip(model.named_parameters(), grads):
        if filter_fn(name, param):
            filtered_params.append(param)
            filtered_grads.append(grad)
    return filtered_params, filtered_grads

def unlearn_selectiveGrad(model, retain_loader, forget_loader, test_loader,
                       criterion, num_epochs,  learning_rate=0.01,
                       select_grads_fn=default_select_grads,
                       filter_param_fn=None,
                       ft_acc_threshold = 0.1):
    """
    Perform gradient ascent on the forget set to 'unlearn' or reduce model performance on these samples.
    Select the gradients by comparing the relationship and magnitude between retain and forget data
    
    Args:
    forget_loader (DataLoader): DataLoader containing the data to forget.
    """

    for epoch in range(num_epochs):
        model.train()

        passed_samples_num = 0

        for (retain_inputs, retain_labels), (forget_inputs, forget_labels) in zip(retain_loader, forget_loader):
            # Forward pass for retain_loader
            retain_inputs, retain_labels = retain_inputs.to(device), retain_labels.to(device)
            forget_inputs, forget_labels = forget_inputs.to(device), forget_labels.to(device)

            retain_outputs = model(retain_inputs)

            # hugging face ImageClassifierOutput
            if hasattr(retain_outputs, 'logits'):retain_outputs= retain_outputs.logits

            retain_loss = criterion(retain_outputs, retain_labels)
            retain_loss.backward()
            retain_grads = [param.grad.clone() for param in model.parameters()]

            # Clear gradients before the next forward pass
            model.zero_grad()

            # Forward pass for forget_loader
            forget_outputs = model(forget_inputs)

            if hasattr(forget_outputs, 'logits'):forget_outputs= forget_outputs.logits

            forget_loss = criterion(forget_outputs, forget_labels)
            forget_loss.backward()
            forget_grads = [param.grad.clone() for param in model.parameters()]

            # Select gradients
            selected_grads = select_grads_fn(retain_grads, forget_grads)

            # print(selected_grads[0][0][0])

            # Filter parameters and gradients if filter_fn is provided
            if filter_param_fn:
                parameters_to_update, selected_grads = filter_parameters_and_grads(model, filter_param_fn, selected_grads)
            else:
                parameters_to_update = list(model.parameters())

            # Manually update model parameters with selected gradients
            with torch.no_grad():
                for param, selected_grad in zip(parameters_to_update, selected_grads):
                        param -= learning_rate * selected_grad

            # Clear gradients after parameter update
            model.zero_grad()


            # Evaluate on the test set after each epoch
            passed_samples_num = passed_samples_num + forget_inputs.shape[0]
            print(f"unlearn model test Acc | forget Acc | Epoch {epoch + 1} | Passed Samples: {passed_samples_num}")
            acc_test = test(model, test_loader)
            acc_forget = test(model, forget_loader)

            if acc_forget < ft_acc_threshold:break
        if acc_forget < ft_acc_threshold:break 

    res = {'Unlearned Achieved Epoch':epoch+1,'Passed Samples':passed_samples_num}
    print(res)
    return model, res


