import torch
import utils
device = utils.device_config()
utils.set_seed()

# Training function
def train(model, train_loader, test_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            if inputs is None: continue
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # hugging face ImageClassifierOutput
            if hasattr(outputs,"logits"):
                outputs = outputs.logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        test(model, test_loader)

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs is None: continue

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # pytorch nn model 
            if hasattr(outputs, 'data'):
                outputs = outputs.data
            # hugging face ImageClassifierOutput
            elif hasattr(outputs,'logits'):
                outputs = outputs.logits

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy/100