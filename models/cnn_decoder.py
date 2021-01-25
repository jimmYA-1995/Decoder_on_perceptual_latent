import torch.nn as nn

class CNNDecoder(nn.Module):
    def __init__(self,
                 train_size,
                 latent_dim,
                 latents=None,
                 norm_type: str = 'batch_norm',
        ):
        super(CNNDecoder, self).__init__()
        
        if latents is not None:
            self.replace = True
            self.embed = nn.Embedding.from_pretrained(latents,
                                                      freeze=False,
                                                      max_norm=0.38) ## manually setting accroding to latents stats
        else:
            # TODO: set max_norm and scale_gard_by_freq
            self.embed = nn.Embedding(train_size, latent_dim, max_norm=0.038)
        self.embed_In = nn.LayerNorm(self.embed.weight.size()[1:], elementwise_affine=False)
        
        n_ch = [256, 256, 128, 128, 64, 64, 3]
        conv_blocks = []
        self.linear1 = nn.Linear(latent_dim, 4096)
        self.linear1_In = nn.LayerNorm(4096, elementwise_affine=False)
        self.linear2 = nn.Linear(4096, 8*8*256)
        self.linear2_In = nn.LayerNorm(8*8*256, elementwise_affine=False)
        self.act = nn.LeakyReLU(0.2)
        print(f"using norm: {norm_type}")

        for i in range(len(n_ch) - 1):
            conv_blocks.append(self.act)
            if n_ch[i+1] == 3:
                conv = nn.ConvTranspose2d(n_ch[i],n_ch[i+1], kernel_size=1, stride=1, padding=0)
            else:
                conv = nn.ConvTranspose2d(n_ch[i],n_ch[i+1], kernel_size=4, stride=2, padding=1)
                
            if norm_type == 'spectral_norm' and i in [0,1,2,3,4,5]:
                conv_sn = nn.utils.spectral_norm(conv)
                conv_blocks.append(conv_sn)
            elif norm_type == 'batch_norm' and i in [4,5]:
                bn = nn.BatchNorm2d(n_ch[i+1])
                conv_blocks.extend([conv, bn])
            elif norm_type == 'instance_norm':
                In = nn.InstanceNorm2d(n_ch[i+1])
                conv_blocks.extend([conv, In])
            else:
                conv_blocks.append(conv)
                
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.out = nn.Tanh()
        
    def forward(self, latent, indices=None, replace=False, get_latent=False):
        if indices is not None:
            embed_latent = self.embed(indices)
            embed_latent = self.embed_In(latent)
            latent = embed_latent if replace \
                     else embed_latent + latent
        
        if get_latent:
            return latent
        
        x = self.linear1(latent)
        x = self.linear1_In(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.linear2_In(x).view(-1,256,8,8)
        
        for layer in self.conv_blocks:
            x = layer(x)
        
        x = self.out(x)
        return x