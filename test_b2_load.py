import torch
from torchvision import models
import torch.nn as nn

b2 = models.efficientnet_b2(weights=None)
b2.classifier[1] = nn.Linear(b2.classifier[1].in_features, 200)

try:
    state = torch.load('./checkpoints_aktp/teacher_b2_tiny.pth', weights_only=True)
except TypeError:
    state = torch.load('./checkpoints_aktp/teacher_b2_tiny.pth')

b2.load_state_dict(state)
print('✓ EfficientNet-B2 loaded successfully')
print(f'Total params: {sum(p.numel() for p in b2.parameters()):,}')

# Test forward pass
x = torch.randn(2, 3, 64, 64)
b2.eval()
with torch.no_grad():
    out = b2(x)
print(f'Output shape: {out.shape}')
print(f'Expected: torch.Size([2, 200])')
