import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return out * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class ResBlock_CBAM(nn.Module):
    """Residual block with CBAM attention module."""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        try:
            # support ultralytics 8+ standard modules
            from ultralytics.nn.modules.conv import Conv
        except ImportError:
            # fallback
            from ultralytics.nn.modules import Conv
        
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.cbam = CBAM(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # Handle different flavors of the unpickled module
        if 'bottleneck' in self._modules:
            bottleneck = self._modules.get('bottleneck')
            cbam_mod = self._modules.get('cbam')
            relu_mod = self._modules.get('relu')
            
            ans = cbam_mod(bottleneck(x))
            should_add = getattr(self, 'add', False)
            if not hasattr(self, 'add'):
                should_add = (ans.shape == x.shape)
            
            # The structure suggests ReLU is applied after addition
            out = x + ans if should_add else ans
            if relu_mod is not None:
                out = relu_mod(out)
            return out
        else:
            # Fallback for the custom structure assuming cv1/cv2
            cv1 = getattr(self, 'cv1', self._modules.get('cv1'))
            cv2 = getattr(self, 'cv2', self._modules.get('cv2'))
            cbam_mod = getattr(self, 'cbam', self._modules.get('cbam'))
            
            ans = cbam_mod(cv2(cv1(x)))
            should_add = getattr(self, 'add', False)
            if not hasattr(self, 'add'):
                should_add = (ans.shape == x.shape)
                
            return x + ans if should_add else ans

def patch_ultralytics():
    """Injects ResBlock_CBAM into ultralytics.nn.modules to allow model loading."""
    try:
        import ultralytics.nn.modules.conv as conv_modules
        import ultralytics.nn.modules as core_modules
        
        # Patch both places it might be looked for
        if not hasattr(conv_modules, 'ResBlock_CBAM'):
            setattr(conv_modules, 'ResBlock_CBAM', ResBlock_CBAM)
        if not hasattr(core_modules, 'ResBlock_CBAM'):
            setattr(core_modules, 'ResBlock_CBAM', ResBlock_CBAM)
            
        return True
    except ImportError as e:
        print(f"Failed to patch ultralytics: {e}")
        return False
