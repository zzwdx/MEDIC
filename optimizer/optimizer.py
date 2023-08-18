from torch import optim

def get_optimizer(net, instr="SGD", **options):    
    optimizer_map = {
        "SGD": (optim.SGD, {
            "params": net.parameters(),
            "lr": None, # <require value>
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "nesterov": False
        })
    }
    
    optimizer, args = optimizer_map[instr]
    args.update(options)

    return optimizer(**args)

def get_scheduler(optimizer, instr="StepLR", **options):
    scheduler_map = {
        "StepLR": (optim.lr_scheduler.StepLR, {
            "optimizer": optimizer,
            "step_size": None, # <require value>
            "gamma": 0.1
        })
    }

    scheduler, args = scheduler_map[instr]
    args.update(options)

    return scheduler(**args)