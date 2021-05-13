""" Parts of the SK-UNet model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSelector(nn.Module):
    def __init__(self, channel, gumbel_temp, reduction=16):
        super(GumbelSelector, self).__init__()

        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(self.channel, max(16, self.channel // reduction), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(max(16, self.channel // reduction), 4, bias=True),
        )
        self.gumbel_temp = gumbel_temp

    def forward(self, x):
        b, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, -1)
        y = self.fc(y)

        y1 = y[:,0:2] # [B, 2]
        # TODO 1 do we need to change this to hard=T?
        y1 = F.gumbel_softmax(y1, tau=self.gumbel_temp, hard=True)[:,:1] # This denotes the usage of spade. [B, 1]
        y2 = y[:,2:4]
        y2 = F.gumbel_softmax(y2, tau=self.gumbel_temp, hard=True)[:,:1] # This denotes the usage of adain. [B, 1]

        # Note that each module's attentions is calculated based on the below formula
        # y1*(IN+AdaIN+SPADE) + (1-y1)*(y2*(IN+AdaIN)+(1-y2)*(IN))
        # = IN + (y1+(1-y1)*y2)*AdaIN + y1*SPADE
        att_adain = y1+(1-y1)*y2
        att_spade = y1

        return att_adain, att_spade

class HierarchicalInstanceNorm(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels,
                 first=False, opt=None):
        super(HierarchicalInstanceNorm, self).__init__()
        assert (opt.use_spade_conv and opt.use_out_conv) == False, f'At least one of them must be False. use_spade_conv={opt.use_spade_conv}, use_in_conv={opt.use_in_conv}'

        self.first = first

        self.use_residual = opt.use_residual
        if self.use_residual:
            if opt.use_residual_ks_5 is None:
                self.residual = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True) # bias must be true for residual branch
            elif opt.use_residual_ks_5=='5':
                self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=True)
            elif opt.use_residual_ks_5 == '33':
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=padding, bias=True),
                    nn.ReLU(),
                    nn.Conv2d(in_channels, out_channels, 3, padding=padding, bias=True)
                )
            else:
                assert False, "use_residual_ks_5 type error"

        self.use_skip = opt.use_skip
        self.use_in = opt.use_in
        self.use_spade = opt.use_spade
        self.use_adain_output_for_spade = opt.use_adain_output_for_spade
        self.use_adain = opt.use_adain

        self.front_norm_type = opt.front_norm_type

        if self.front_norm_type == 'in':
            self.norm = nn.InstanceNorm2d(in_channels, affine=False)
        elif self.front_norm_type == 'bn':
            self.norm = nn.BatchNorm2d(in_channels, affine=False)
        else:
            self.norm = nn.Identity()

        if self.use_in:
            self.IN_gamma = nn.Parameter(torch.ones(in_channels))
            self.IN_beta = nn.Parameter(torch.zeros(in_channels))

        self.relu = nn.LeakyReLU(inplace=False)

        self.use_front_conv = opt.use_front_conv
        self.use_in_conv = opt.use_in_conv
        self.use_spade_conv = opt.use_spade_conv
        self.use_adain_conv = opt.use_adain_conv
        self.use_out_conv = opt.use_out_conv

        self.use_selector = opt.use_selector
        self.use_gumbel_selector = opt.use_gumbel_selector

        self.use_zero_mean_spade = opt.use_zero_mean_spade

        if self.use_front_conv:
            self.front_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        if self.use_in_conv:
            self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        # assert out_channels%4==0
        if self.use_spade:
            if self.use_adain_output_for_spade:
                self.adain2spade = nn.Sequential(
                    nn.Linear(2*in_channels, in_channels)
                )

                self.spade = Spade(2*in_channels, in_channels, kernel_size=1, opt=opt)
            else:
                self.spade = Spade(in_channels, in_channels, kernel_size=1, opt=opt)
            # self.spade = Spade(in_channels, kernel_size=kernel_size, opt=opt)

            if self.use_spade_conv:
                self.spade_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

            if self.use_zero_mean_spade:
                self.spade_norm = nn.InstanceNorm2d(in_channels, affine=False)
                self.learnable_spade_gamma_var = nn.Parameter(torch.ones(in_channels))
                self.learnable_spade_beta_var = nn.Parameter(torch.ones(in_channels))


        if self.use_adain:
            self.adain = AdaIN(in_channels, reduction, opt=opt)

            if self.use_adain_conv:
                self.adain_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        if self.use_out_conv:
            self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        if self.use_selector:
            self.selector = nn.Sequential(
                nn.Linear(in_channels, in_channels//reduction),
                nn.ReLU(),
                nn.Linear(in_channels, 3)
            )

        if self.use_gumbel_selector:
            self.selector = GumbelSelector(in_channels, opt.gumbel_temp)


    def forward(self, x):

        if self.first:
            # TODO 1: self.first's meaning??????
            # TODO 1: should we use skip(=residual) connection here?
            out = x

        else:
            x_origin = x

            if self.use_front_conv:
                x = F.relu(x)
                x = self.front_conv(x)

            normed_x = self.norm(x)
            # mu_x = x.mean(dim=(2,3), keepdim=True) # [B, C, 1, 1]
            # sigma_x = x.std(dim=(2,3), keepdim=True) # [B, C, 1, 1]

            out = 0

            if self.use_skip:
                out_skip = x
                out += out_skip

            if self.use_in:
                out_in = normed_x*self.IN_gamma[None, :, None, None] + self.IN_beta[None, :, None, None]

                if self.use_in_conv:
                    out_in = F.relu(out_in)
                    out_in = self.in_conv(out_in)

                out += out_in

            if self.use_adain:
                adain_gamma, adain_beta = self.adain(x)
                out_adain = normed_x*adain_gamma[:,:,None,None] + adain_beta[:,:,None,None]

                if self.use_adain_conv:
                    out_adain = F.relu(out_adain)
                    out_adain = self.adain_conv(out_adain)

                out += out_adain

            # out = torch.mul(out1, att[0]) + torch.mul(out2, att[1])

            if self.use_spade:
                if self.use_adain_output_for_spade:
                    adain_beta_, adain_gamma_ = adain_beta.detach(), adain_gamma.detach()
                    ada_info = self.adain2spade(torch.cat([adain_beta_,adain_gamma_], dim=1))
                    ada_info = ada_info[:,:,None,None].expand(-1,-1,x.shape[2],x.shape[3])
                    spade_gamma, spade_beta = self.spade(torch.cat([x, ada_info], dim=1))
                else:
                    spade_gamma, spade_beta = self.spade(x)

                if self.use_zero_mean_spade:
                    spade_gamma = self.spade_norm(spade_gamma)
                    spade_beta = self.spade_norm(spade_beta)
                    spade_gamma *= self.learnable_spade_gamma_var[None, :, None, None]
                    spade_beta *= self.learnable_spade_beta_var[None, :, None, None]

                out_spade = normed_x*spade_gamma + spade_beta

                if self.use_spade_conv:
                    out_spade = F.relu(out_spade)
                    out_spade = self.spade_conv(out_spade)

                out += out_spade

            if self.use_selector:
                select = F.sigmoid(self.selector(out.mean(dim=(2, 3)))) # [B, C]
                out = out_in*select[:,0, None, None, None]+out_adain*select[:,1, None, None, None]+out_spade*select[:,2, None, None, None]


            if self.use_gumbel_selector:
                # TODO 0 in the test time, we need to change the position of gumbel_selector
                # in the training time because of batch operation, hard selection cannot be processed.
                att_adain, att_spade=self.selector(x) # each [B, 1]
                out = out_in + out_adain * att_adain[:,None,None]+ out_spade * att_spade[:, None, None]
            else:
                att_adain = x.new_ones((x.shape[0], 1)) #[B,1]
                att_spade = att_adain
            out = F.leaky_relu(out)

        if self.use_out_conv:
            # TODO should we add relu here?
            # I guess not.
            # I think out_conv is must. If not, there is no non-linearity and position of relu gets weird.
            out = self.out_conv(out)

        if self.use_residual:
            out = out + self.residual(F.relu(x_origin))

        # use self.IN_gamma just for dimension matching
        adain_regul_loss = torch.zeros_like(self.IN_gamma)
        spade_regul_loss = torch.zeros_like(self.IN_gamma)
        adain_var_max_loss = torch.zeros_like(self.IN_gamma)
        spade_var_max_loss = torch.zeros_like(self.IN_gamma)

        if not self.first:
            # regul_loss = torch.abs(self.IN_gamma)+torch.abs(self.IN_beta)

            if self.use_adain:
                # adain_regul_loss = torch.abs(adain_gamma).mean(0)+torch.abs(adain_beta).mean(0)
                adain_regul_loss = torch.abs(adain_gamma.mean(0)) + torch.abs(adain_beta.mean(0)) # [B,C] -> [C]
                adain_var_max_loss = torch.var(adain_gamma, dim=0) + torch.var(adain_beta, dim=0)

            if self.use_spade:
                # spade_regul_loss = torch.abs(spade_gamma).mean((0,2,3))+torch.abs(spade_beta).mean((0,2,3)) # [C]
                # TODO should we include B dimension calculating spade_gamma.mean?
                # spade_regul_loss = torch.mean(torch.abs(spade_gamma.mean((2,3))) + torch.abs(spade_beta.mean((2,3))), dim=0) # [B, C, H, W] -> [C]
                spade_regul_loss = torch.abs(spade_gamma.mean((2, 3))) + torch.abs(
                    spade_beta.mean((2, 3)))  # [B, C, H, W] -> [B, C]
                spade_var_max_loss = torch.var(spade_gamma, dim=(2, 3)) + torch.var(spade_beta, dim=(2, 3))

        out_layer_info = {"adain_regul_loss": adain_regul_loss,
                          "spade_regul_loss": spade_regul_loss,
                          "adain_var_max_loss": adain_var_max_loss,
                          "spade_var_max_loss": spade_var_max_loss,
                          "att_adain": att_adain,
                          "att_spade": att_spade
                          }

        # return out, out_layer_info

        return out

class AdaIN(nn.Module):
    def __init__(self, in_channels, reduction, opt = None):
        super(AdaIN, self).__init__()
        self.add_spade_adain_front_conv = opt.add_spade_adain_front_conv
        self.use_delta_input = opt.use_delta_input

        if opt.add_spade_adain_front_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.fc = nn.Linear(in_channels, int(in_channels // reduction))
        self.gamma = nn.Linear(int(in_channels // reduction), in_channels)
        self.beta = nn.Linear(int(in_channels // reduction), in_channels)

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        B, C, _, _ = x.shape

        if self.add_spade_adain_front_conv:
            x= self.conv(x)
            x= F.relu(x)

        squeezed_x = self.gap(x).view(B, C)

        # TODO 0: fix this code right after experiment
        if self.use_delta_input:
        # if False:
            adain_mean = squeezed_x.mean(dim=0, keepdim=True) # [1, C]
            squeezed_x = squeezed_x-adain_mean # [B, C] - [1, C] = [B, C]

        squeezed_x = F.relu(self.fc(squeezed_x))

        gamma_x = self.gamma(squeezed_x)
        beta_x = self.beta(squeezed_x)

        return gamma_x, beta_x

class Spade(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, opt = None):
        super(Spade, self).__init__()

        self.use_delta_input = opt.use_delta_input
        self.add_spade_adain_front_conv = opt.add_spade_adain_front_conv

        if opt.add_spade_adain_front_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.fc = nn.Conv2d(in_channels, int(in_channels // opt.reduction), kernel_size=kernel_size, padding=kernel_size//2)
        ## TODO should it share the same conv1 at firxt???
        self.gamma_net = nn.Conv2d(int(in_channels // opt.reduction), out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.beta_net = nn.Conv2d(int(in_channels // opt.reduction), out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        #
        # self.spade_norm = opt.spade_norm
        #
        # assert opt.spade_norm == 'in'

        # if opt.spade_norm == 'in':
        #     self.param_free_norm = nn.InstanceNorm2d(in_channels, affine=False)
        # elif opt.spade_norm == 'bn':
        #     self.param_free_norm = nn.BatchNorm2d(in_channels, affine=False)
        # elif opt.spade_norm == 'nonorm':
        #     pass
        # else:
        #     raise ValueError('%s is not a recognized param-free norm type in SPADE'
        #                      % opt.spade_norm)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.add_spade_adain_front_conv:
            x= self.conv(x)
            x= F.relu(x)

        if self.use_delta_input:
            spade_mean = x.mean(dim=(2,3), keepdim=True) # [B, C, 1, 1]
            x = x-spade_mean

        f_input = self.fc(x)
        # TODO 2: maybe leaky relu?
        f_input = F.relu(f_input)

        gamma = self.gamma_net(f_input)
        beta = self.beta_net(f_input)

        # assert (B, C, H, W) == gamma.shape and (B, C, H, W) == beta.shape

        # if self.spade_norm != 'nonorm':
        #     norm_x = self.param_free_norm(x)
        # else:
        #     norm_x = x
        #
        # out = norm_x * gamma + beta
        # out = group_conv(norm_x, weights, groups=self.groups)

        return gamma, beta

if __name__=='__main__':

    print()
