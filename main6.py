import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from torch.nn import functional as F
import time # Import time for checking epoch duration
import math # Import math for calculating H_out, W_out

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


data_root = 'data'
try:
    if not os.path.isdir(data_root):
         raise FileNotFoundError(f"Dataset directory '{data_root}' not found or is not a directory.")
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    if not dataset.classes:
        raise ValueError(f"No classes found in '{data_root}'. Check directory structure.")
    train_size = int(0.7 * len(dataset))
    val_size = max(1, len(dataset) - train_size)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Calculated training set size is zero or negative.")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    num_workers = min(4, os.cpu_count() if os.cpu_count() else 0)
    # print(f"Using {num_workers} workers for DataLoader.")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)
    num_classes = len(dataset.classes)

except FileNotFoundError as e:
    print(f"Error: {e}"); exit()
except ValueError as e:
    print(f"Error: {e}"); exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the dataset: {e}"); exit()


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, H * W)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        out = self.gamma * out + x
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            input_features = torch.cat(features, dim=1)
            new_features = layer(input_features)
            features.append(new_features)
        return torch.cat(features, dim=1)

class Capsule(nn.Module):
    def __init__(self, in_channels, out_channels, num_capsules, num_iterations=3):

        super(Capsule, self).__init__()
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.out_channels = out_channels
        self.in_channels = in_channels # Store for assertion


        self.W = nn.Parameter(torch.randn(in_channels, num_capsules * out_channels))


    def squash(self, tensor, dim=-1):
        """Squashes capsule vectors."""
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm + 1e-8) # Add epsilon for stability
        return scale * tensor / torch.sqrt(squared_norm + 1e-8) # Add epsilon here too

    def forward(self, x):

        batch_size, C_in, H_in, W_in = x.size()
        # Ensure input channels match the dimension expected by W
        assert C_in == self.W.shape[0], f"Input channels {C_in} mismatch W input dimension {self.W.shape[0]}"


        x_reshaped = x.permute(0, 2, 3, 1).contiguous()
        x_reshaped = x_reshaped.view(batch_size * H_in * W_in, C_in)


        u_hat = torch.einsum('bi,io->bo', x_reshaped, self.W)


        u_hat = u_hat.view(batch_size, H_in, W_in, self.num_capsules, self.out_channels)

        u_hat = u_hat.permute(0, 3, 1, 2, 4).contiguous()


        b = torch.zeros((batch_size, self.num_capsules, H_in, W_in, 1), device=x.device, dtype=x.dtype)

        for i in range(self.num_iterations):

            c = F.softmax(b, dim=1) # Shape: (B, num_caps, H, W, 1)


            s = (c * u_hat).sum(dim=1, keepdim=True) # Shape: (B, 1, H, W, out_ch)


            v = self.squash(s, dim=-1) # Squash along the capsule dimension. Shape: (B, 1, H, W, out_ch)


            delta_b = (u_hat * v).sum(dim=-1, keepdim=True) # Shape: (B, num_caps, H, W, 1)
            b = b + delta_b


        final_output = v.squeeze(1) # Shape: (B, H_in, W_in, out_channels)


        final_output = final_output.view(batch_size, -1) # Shape: (B, H_in * W_in * out_channels)

        return final_output


class MainNet(nn.Module):
    def __init__(self, num_classes, input_size=(128, 128)):
        super(MainNet, self).__init__()
        self.input_size = input_size

        # Initial Convolution Block
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=2, bias=False) # H/2, W/2
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        # Self Attention Block
        self.attention = SelfAttention(128) # Input channels = output channels of conv1

        # Dense Block 1
        self.dense_block1 = DenseBlock(in_channels=128, growth_rate=32, num_layers=6)
        current_channels = 128 + 6 * 32 # 320

        # Transition Layer 1
        self.transition1_bn = nn.BatchNorm2d(current_channels)
        self.transition1_relu = nn.ReLU(inplace=True)
        self.transition1_conv = nn.Conv2d(current_channels, 128, kernel_size=1, bias=False)
        self.transition1_pool = nn.AvgPool2d(kernel_size=2, stride=2) # H/4, W/4
        current_channels = 128

        # Dense Block 2
        self.dense_block2 = DenseBlock(in_channels=current_channels, growth_rate=32, num_layers=6)
        current_channels = 128 + 6 * 32 # 320

        # Final Layers before Capsules
        self.final_bn = nn.BatchNorm2d(current_channels)
        self.final_relu = nn.ReLU(inplace=True)

        # Capsule Layer (Using the nn.Parameter W + Dynamic Routing version)
        capsule_out_dim = 16 # Dimension of each capsule vector
        self.capsules = Capsule(in_channels=current_channels,
                                out_channels=capsule_out_dim,
                                num_capsules=num_classes, # NOTE: Usually num_capsules might differ from num_classes initially
                                num_iterations=3)


        final_H, final_W = input_size[0] // 4, input_size[1] // 4

        capsule_output_flat_size = final_H * final_W * capsule_out_dim

        self.final_fc = nn.Linear(capsule_output_flat_size, num_classes)


    def forward(self, x):
        # Initial Convolution Block
        identity = self.relu(self.bn1(self.conv1(x))) # Shape: [B, 128, H/2, W/2]

        # Self Attention Block
        attention_out = self.attention(identity) # Shape: [B, 128, H/2, W/2]

        # Skip Connection
        x = identity + attention_out

        # Feed the sum into Dense Block 1
        x = self.dense_block1(x) # Shape: [B, 320, H/2, W/2]

        # Transition Layer 1
        x = self.transition1_bn(x)
        x = self.transition1_relu(x)
        x = self.transition1_conv(x) # Shape: [B, 128, H/2, W/2]
        x = self.transition1_pool(x) # Shape: [B, 128, H/4, W/4]

        # Dense Block 2
        x = self.dense_block2(x) # Shape: [B, 320, H/4, W/4]

        # Final Layers before Capsules
        x = self.final_bn(x)
        x = self.final_relu(x) # Shape: [B, 320, H/4, W/4]

        # Capsule Layer
        x = self.capsules(x) # Shape: [B, (H/4 * W/4 * capsule_out_dim)]

        # --- Added Final Fully Connected Layer ---
        # Map the flattened capsule output to class logits
        x = self.final_fc(x) # Shape: [B, num_classes]

        return x

# --- End of MainNet ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Pass input_size to MainNet constructor for dynamic calculation
model = MainNet(num_classes=num_classes, input_size=(128, 128)).to(device)
criterion = nn.CrossEntropyLoss()

fixed_lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=fixed_lr, weight_decay=1e-5)
# print(f"Optimizer: Adam with fixed learning rate = {fixed_lr}")

# Training and validation functions (Keep the improved versions)
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0
    for inputs, labels in train_loader:
        batch_count += 1
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0
    accuracy = (100. * correct / total) if total > 0 else 0
    return epoch_loss, accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total if total > 0 else 0
    accuracy = (100. * correct / total) if total > 0 else 0
    return avg_loss, accuracy

def save_best_model(model, save_path):
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    try:
        torch.save(model.state_dict(), save_path)
    except Exception as e:
        print(f"Error saving model: {e}")


if __name__ == "__main__":
    if 'train_loader' not in locals() or 'val_loader' not in locals():
        print("Exiting due to dataset loading failure.")
        exit()

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # Training loop
    num_epochs = 1000
    best_val_acc = 0.0
    best_model_path = "best_model_dynamic_capsule.pth" # Changed save name

    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Duration: {epoch_duration:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            print(f"Validation accuracy improved ({best_val_acc:.2f}% -> {val_acc:.2f}%). Saving model to {best_model_path} ...")
            best_val_acc = val_acc
            save_best_model(model, best_model_path)
            print("-" * 30)

    print("\n" + "=" * 40)
    print(f"Training complete after {num_epochs} epochs.")
    print(f"Best validation accuracy achieved: {best_val_acc:.2f}%")
    print(f"Best model saved at: {best_model_path}")
    print("=" * 40)