from torch import nn
import torch
from models import FFA
from option import opt,model_name,log_dir

def model_summary(opt, model, input):
    if len(input.shape)==4:
        input = input[0]
    summary_(model, input.shape, batch_size=1)



if __name__ =='__main__':
    from torchsummary import summary as summary_

    input = torch.randn(1, 3, 240, 240).cuda()
    model = FFA(gps=opt.gps,blocks=opt.blocks, opt=opt).cuda()

    #=============================== model summary (optional)
    model_summary(opt, model, input)

    #=============================== based on the above summary,
    from thop import profile
    from ptflops import get_model_complexity_info

    # Change here
    # ###############
    # in_channels = 64
    # out_channels = 16
    # height = 227
    # width = 227
    # ##############

    macs, params = profile(model, inputs=(input,))
    macs2, params2 = get_model_complexity_info(model, tuple(input.shape) if len(input.shape)==3 else tuple(input[0].shape), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)

    print(f'FFA macs: {macs}, params: {params}')
    print(f'FFA macs: {macs2}, params: {params2}')