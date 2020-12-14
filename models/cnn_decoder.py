import torch.nn as nn

class CNNDecoder(nn.Module):
    def __init__(self,
                 latent_dim: int = 512,
                 norm_type: str = 'batch_norm',
        ):
        super(CNNDecoder, self).__init__()
#         self.save_hyperparameters()
    
        n_ch = [256, 256, 128, 128, 64, 64, 3]
        conv_blocks = []
        self.linear1 = nn.Linear(latent_dim, 4096)
        self.linear2 = nn.Linear(4096, 8*8*256)
        self.act = nn.LeakyReLU(0.2)

        for i in range(len(n_ch) - 1):
            conv_blocks.append(self.act)
            if n_ch[i+1] == 3:
                conv = nn.ConvTranspose2d(n_ch[i],n_ch[i+1], kernel_size=1, stride=1, padding=0)
            else:
                conv = nn.ConvTranspose2d(n_ch[i],n_ch[i+1], kernel_size=4, stride=2, padding=1)
                
            if norm_type == 'spectral_norm' and i in [1,2,3,4,5]:
                conv_sn = nn.utils.spectral_norm(conv)
                conv_blocks.append(conv_sn)
            elif norm_type == 'batch_norm' and i in [4,5]:
                bn = nn.BatchNorm2d(n_ch[i+1])
                conv_blocks.extend([conv, bn])
            else:
                conv_blocks.append(conv)
                
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.out = nn.Tanh()
        
    def forward(self, latent):
        x = self.linear1(latent)
        x = self.act(x)
        x = self.linear2(x).view(-1,256,8,8)
        
        for layer in self.conv_blocks:
            x = layer(x)
        
        x = self.out(x)
        return x