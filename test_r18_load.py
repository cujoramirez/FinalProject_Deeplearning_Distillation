import torch
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn

r18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
r18.fc = nn.Linear(r18.fc.in_features, 200)

try:
    state = torch.load('./checkpoints_aktp/teacher_r18_tiny.pth', weights_only=True)
except TypeError:
    state = torch.load('./checkpoints_aktp/teacher_r18_tiny.pth')

r18.load_state_dict(state)
print('✓ ResNet18 loaded successfully')
print(f'Total params: {sum(p.numel() for p in r18.parameters()):,}')

# Test forward pass
x = torch.randn(2, 3, 64, 64)
r18.eval()
with torch.no_grad():
    out = r18(x)
print(f'Output shape: {out.shape}')
print(f'Expected: torch.Size([2, 200])')
