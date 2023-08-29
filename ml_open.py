import argparse
import torch
import pickle
import os, copy
from dataset.dataloader import get_dataloader, get_transform
from dataset.dataset import SingleDomainData, SingleClassData
from model.model import MutiClassifier, MutiClassifier_, resnet18_fast, resnet50_fast, ConvNet
from optimizer.optimizer import get_optimizer, get_scheduler
from loss.OVALoss import OVALoss
from train.test import eval
from util.log import log, save_data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from util.ROC import generate_OSCR
from util.util import ForeverDataIterator, ConnectedDataIterator, split_classes
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument('--source-domain', nargs='+', default=['photo', 'cartoon', 'art_painting'])
    parser.add_argument('--target-domain', nargs='+', default=['sketch'])
    parser.add_argument('--known-classes', nargs='+', default=['dog', 'elephant', 'giraffe', 'horse', 'guitar', 'house',])
    parser.add_argument('--unknown-classes', nargs='+', default=['person'])
    
    # parser.add_argument('--dataset', default='OfficeHome')
    # parser.add_argument('--source-domain', nargs='+', default=['Art', 'Clipart', 'Product'])
    # parser.add_argument('--target-domain', nargs='+', default=['RealWorld'])
    # parser.add_argument('--known-classes', nargs='+', default=['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 
    #     'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 
    #     'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 
    #     'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 
    #     'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 
    #     'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard',  
    #     'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 
    #     'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven',
        
    #     ])
    # parser.add_argument('--unknown-classes', nargs='+', default=[      
    #     'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 
    #     'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 
    #     'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 
    #     'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 
    #     'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam'
    #     ])

    # parser.add_argument('--dataset', default='DigitsDG')
    # parser.add_argument('--source-domain', nargs='+', default=['mnist', 'mnist_m', 'svhn'])
    # parser.add_argument('--target-domain', nargs='+', default=['syn'])
    # parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4', '5'])
    # parser.add_argument('--unknown-classes', nargs='+', default=['6', '7', '8', '9'])

    parser.add_argument('--no-crossval', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--net-name', default='resnet50')
    parser.add_argument('--optimize-method', default="SGD")
    parser.add_argument('--schedule-method', default='StepLR')
    parser.add_argument('--num-epoch', type=int, default=10000)
    parser.add_argument('--eval-step', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--meta-lr', type=float, default=0.01)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--without-bcls', action='store_true')
    parser.add_argument('--share-param', action='store_true')

    parser.add_argument('--save-dir', default='save')
    parser.add_argument('--save-name', default='demo')
    parser.add_argument('--save-best-test', action='store_true')
    parser.add_argument('--save-later', action='store_true')

    parser.add_argument('--num-epoch-before', type=int, default=0)
    
    args = parser.parse_args()

    # It can be used to replace the following code, but the editor may take it as an error.
    # locals().update(vars(args))

    # It can be replaced by the preceding code.
    dataset = args.dataset
    source_domain = sorted(args.source_domain)
    target_domain = sorted(args.target_domain)
    known_classes = sorted(args.known_classes)
    unknown_classes = sorted(args.unknown_classes)
    crossval = not args.no_crossval   
    gpu = args.gpu
    batch_size = args.batch_size
    net_name = args.net_name
    optimize_method = args.optimize_method
    schedule_method = args.schedule_method
    num_epoch = args.num_epoch
    eval_step = args.eval_step
    lr = args.lr
    meta_lr = args.meta_lr
    nesterov = args.nesterov
    without_bcls = args.without_bcls
    share_param = args.share_param
    save_dir = args.save_dir
    save_name = args.save_name   
    save_later = args.save_later
    save_best_test = args.save_best_test
    num_epoch_before = args.num_epoch_before

    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == 'PACS':
        train_dir = '/data/datasets/PACS_train'
        val_dir = '/data/datasets/PACS_crossval'
        test_dir = ['/data/datasets/PACS_train', '/data/datasets/PACS_crossval']
        sub_batch_size = batch_size // 2
        small_img = False
    elif dataset == 'OfficeHome':
        train_dir = ''
        val_dir = ''
        test_dir = ''
        sub_batch_size = batch_size // 4
        small_img = False
    elif dataset == "DigitsDG":
        train_dir = ''
        val_dir = ''
        test_dir = ''
        sub_batch_size = batch_size // 2
        small_img = True
        

    log_path = os.path.join(save_dir, 'log', save_name + '_train.txt')
    param_path = os.path.join(save_dir, 'param', save_name + '.pkl')
    model_val_path = os.path.join(save_dir, 'model', 'val', save_name + '.tar')
    model_test_path = os.path.join(save_dir, 'model', 'test', save_name + '.tar')
    renovate_step = int(num_epoch*0.6) if save_later else 0

    log('GPU: {}'.format(gpu), log_path)

    log('Loading path...', log_path)

    log('Save name: {}'.format(save_name), log_path)
    log('Save best test: {}'.format(save_best_test), log_path)
    log('Save later: {}'.format(save_later), log_path)

    with open(param_path, 'wb') as f: 
        pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)

    log('Loading dataset...', log_path)

    num_domain = len(source_domain)
    num_classes = len(known_classes)

    class_index = [i for i in range(num_classes)]
    group_length = (num_classes-1) // 10 + 1

    if dataset == "OfficeHome" and len(unknown_classes) == 0:
        group_length = 6

    log('Group length: {}'.format(group_length), log_path)
    
    group_index = [i for i in range((num_classes-1)//group_length + 1)]
    num_group = len(group_index)

    domain_specific_loader = []
    for domain in source_domain:       
        dataloader_list = []
        if num_classes <= 10:
            for i, classes in enumerate(known_classes):
                scd = SingleClassData(root_dir=train_dir, domain=domain, classes=classes, domain_label=-1, classes_label=i, transform=get_transform("train", small_img=small_img))
                loader = DataLoader(dataset=scd, batch_size=sub_batch_size, shuffle=True, drop_last=True, num_workers=1)
                dataloader_list.append(loader)
        else:
            classes_partition = split_classes(classes_list=known_classes, index_list=class_index, n=group_length)
            for classes, class_to_idx in classes_partition:
                sdd = SingleDomainData(root_dir=train_dir, domain=domain, classes=classes, domain_label=-1, get_classes_label=True, class_to_idx=class_to_idx, transform=get_transform("train", small_img=small_img))
                loader = DataLoader(dataset=sdd, batch_size=sub_batch_size, shuffle=True, drop_last=True, num_workers=1)
                dataloader_list.append(loader)

        domain_specific_loader.append(ConnectedDataIterator(dataloader_list=dataloader_list, batch_size=batch_size))
    
    if crossval:
        val_k = get_dataloader(root_dir=val_dir, domain=source_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="val", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
    else:
        val_k = None
    
    test_k = get_dataloader(root_dir=test_dir, domain=target_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
    if len(unknown_classes) > 0:
        test_u = get_dataloader(root_dir=test_dir, domain=target_domain, classes=unknown_classes, batch_size=batch_size, get_domain_label=False, get_class_label=False, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)   
    else:
        test_u = None

    log('DataSet: {}'.format(dataset), log_path)
    log('Source domain: {}'.format(source_domain), log_path)
    log('Target domain: {}'.format(target_domain), log_path)
    log('Known classes: {}'.format(known_classes), log_path)
    log('Unknown classes: {}'.format(unknown_classes), log_path)
    log('Batch size: {}'.format(batch_size), log_path)
    log('CrossVal: {}'.format(crossval), log_path)
    log('Loading models...', log_path)

    if share_param:
        muticlassifier = MutiClassifier_
    else:
        muticlassifier = MutiClassifier

    if net_name == 'resnet18':
        net = muticlassifier(net=resnet18_fast(), num_classes=num_classes)
    elif net_name == 'resnet50':
        net = muticlassifier(net=resnet50_fast(), num_classes=num_classes, feature_dim=2048)
    elif net_name == "convnet":
        net = muticlassifier(net=ConvNet(), num_classes=num_classes, feature_dim=256)


    net = net.to(device)
    
    optimizer = get_optimizer(net=net, instr=optimize_method, lr=lr, nesterov=nesterov)
    scheduler = get_scheduler(optimizer=optimizer, instr=schedule_method, step_size=int(num_epoch*0.8), gamma=0.1)

    log('Network: {}'.format(net_name), log_path)
    log('Number of epoch: {}'.format(num_epoch), log_path)
    log('Learning rate: {}'.format(lr), log_path)
    log('Meta learning rate: {}'.format(meta_lr), log_path)

    if num_epoch_before != 0:
        log('Loading state dict...', log_path)  
        if save_best_test == False:
            net.load_state_dict(torch.load(model_val_path))
        else:
            net.load_state_dict(torch.load(model_test_path))
        for epoch in range(num_epoch_before):
            scheduler.step()
        log('Number of epoch-before: {}'.format(num_epoch_before), log_path)

    log('Without binary classifier: {}'.format(without_bcls), log_path)
    log('Share Parameter: {}'.format(share_param), log_path)

    log('Start training...', log_path)  

    if crossval:
        best_val_acc = eval(net=net, loader=val_k, log_path=log_path, epoch=-1, device=device, mark="Val") 
    else:
        best_val_acc = 0
    best_val_test_acc = []
    best_test_acc = best_test_acc_ = eval(net=net, loader=test_k, log_path=log_path, epoch=-1, device=device, mark="Test") 
    best_test_test_acc = []
    criterion = torch.nn.CrossEntropyLoss()
    ovaloss = OVALoss()
    if without_bcls:
        ovaloss = lambda *args: 0
    exp_domain_index = 0   
    exp_group_num = (num_group-1) // 3 + 1
    exp_group_index = random.sample(group_index, exp_group_num)

    domain_index_list = [i for i in range(num_domain)]

    fast_parameters = list(net.parameters())
    for weight in net.parameters():
        weight.fast = None
    net.zero_grad()
    
    for epoch in range(num_epoch_before, num_epoch):

#################################################################### meta train open

        net.train()
        meta_train_loss = meta_val_loss = 0

        domain_index_set = set(domain_index_list) - {exp_domain_index}
        i, j = random.sample(list(domain_index_set), 2)

        domain_specific_loader[i].remove(exp_group_index)
        input, label = next(domain_specific_loader[i])      
        domain_specific_loader[i].reset()  

        input = input.to(device)
        label = label.to(device)
        out, output = net.c_forward(x=input)
        meta_train_loss += criterion(out, label)
        output = output.view(output.size(0), 2, -1)
        meta_train_loss += ovaloss(output, label)

        domain_specific_loader[j].remove(exp_group_index)
        input, label = next(domain_specific_loader[j])
        domain_specific_loader[j].reset()

        input = input.to(device)
        label = label.to(device)
        out, output = net.c_forward(x=input)
        meta_train_loss += criterion(out, label)
        output = output.view(output.size(0), 2, -1)
        meta_train_loss += ovaloss(output, label)

        domain_specific_loader[exp_domain_index].keep(exp_group_index)
        input, label = next(domain_specific_loader[exp_domain_index])
        domain_specific_loader[exp_domain_index].reset()

        input = input.to(device)
        label = label.to(device)
        out, output = net.c_forward(x=input)
        meta_train_loss += criterion(out, label)
        output = output.view(output.size(0), 2, -1)
        meta_train_loss += ovaloss(output, label)

########################################################################## meta val open

        grad = torch.autograd.grad(meta_train_loss, fast_parameters,
                                create_graph=True, allow_unused=True)

        for k, weight in enumerate(net.parameters()):
            if grad[k] is not None:
                if weight.fast is None:
                    weight.fast = weight - meta_lr * grad[k]
                else:
                    weight.fast = weight.fast - meta_lr * grad[
                        k]

        domain_specific_loader[i].keep(exp_group_index)
        input_1, label_1 = domain_specific_loader[i].next(batch_size=batch_size//2)      
        domain_specific_loader[i].reset() 

        domain_specific_loader[j].keep(exp_group_index)
        input_2, label_2 = domain_specific_loader[j].next(batch_size=batch_size//2)      
        domain_specific_loader[j].reset() 
        
        input = torch.cat([input_1, input_2], dim=0)
        label = torch.cat([label_1, label_2], dim=0)

        input = input.to(device)
        label = label.to(device)
        out, output = net.c_forward(x=input)
        meta_val_loss += criterion(out, label)
        output = output.view(output.size(0), 2, -1)
        meta_val_loss += ovaloss(output, label)

        for i in range(2):

            domain_specific_loader[exp_domain_index].remove(exp_group_index)
            input, label = next(domain_specific_loader[exp_domain_index])
            domain_specific_loader[exp_domain_index].reset()

            input = input.to(device)
            label = label.to(device)
            out, output = net.c_forward(x=input)
            meta_val_loss += criterion(out, label)
            output = output.view(output.size(0), 2, -1)
            meta_val_loss += ovaloss(output, label)

##################################################################### 

        total_loss = meta_train_loss + meta_val_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        fast_parameters = list(net.parameters())
        for weight in net.parameters():
            weight.fast = None
        net.zero_grad()


####################################################################

        exp_domain_index = (exp_domain_index+1)%num_domain
        exp_group_index = random.sample(group_index, exp_group_num)

        if (epoch+1) % eval_step == 0:      
       
            net.eval()         

            if test_u != None:
                output_k_sum = []
                b_output_k_sum = []
                label_k_sum = []  
                with torch.no_grad():  
                    for input, label, *_ in test_k:
                        input = input.to(device)
                        label = label.to(device)

                        output = net(x=input)
                        output = F.softmax(output, 1)
                        b_output = net.b_forward(x=input)
                        b_output = b_output.view(output.size(0), 2, -1)
                        b_output = F.softmax(b_output, 1)

                        output_k_sum.append(output)
                        b_output_k_sum.append(b_output)
                        label_k_sum.append(label)

                output_k_sum = torch.cat(output_k_sum, dim=0)
                b_output_k_sum = torch.cat(b_output_k_sum, dim=0)
                label_k_sum = torch.cat(label_k_sum)

                output_u_sum = []
                b_output_u_sum = []
    
                with torch.no_grad():
                    for input, *_ in test_u:
                        input = input.to(device)
                        label = label.to(device)

                        output = net(x=input)
                        output = F.softmax(output, 1)
                        b_output = net.b_forward(x=input)
                        b_output = b_output.view(output.size(0), 2, -1)
                        b_output = F.softmax(b_output, 1)

                        output_u_sum.append(output)
                        b_output_u_sum.append(b_output)

                output_u_sum = torch.cat(output_u_sum, dim=0)
                b_output_u_sum = torch.cat(b_output_u_sum, dim=0)

    #################################################################################
                log('C classifier:', log_path)

                conf_k, argmax_k = torch.max(output_k_sum, axis=1)
                conf_u, _ = torch.max(output_u_sum, axis=1)

                OSCR_C = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)

        
                log('OSCR_C: {:.4f}'.format(OSCR_C), log_path) 
            
    ###################################################################################################################

                log('B classifier:', log_path)

                _, argmax_k = torch.max(output_k_sum, axis=1)
                _, argmax_u = torch.max(output_u_sum, axis=1)

                argmax_k_vertical = argmax_k.view(-1, 1)
                conf_k = torch.gather(b_output_k_sum[:, 1, :], dim=1, index=argmax_k_vertical).view(-1)

                argmax_u_vertical = argmax_u.view(-1, 1)
                conf_u = torch.gather(b_output_u_sum[:, 1, :], dim=1, index=argmax_u_vertical).view(-1)

                OSCR_B = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)

                log('OSCR_B: {:.4f}'.format(OSCR_B), log_path) 

            else:
                OSCR_C = OSCR_B = 0 
                log("", log_path)

            
            if val_k != None:
                acc = eval(net=net, loader=val_k, log_path=log_path, epoch=epoch, device=device, mark="Val") 
            
            acc_ = eval(net=net, loader=test_k, log_path=log_path, epoch=epoch, device=device, mark="Test")     
            
            if val_k != None:           
                if acc > best_val_acc:
                    best_val_acc = acc
                    best_test_acc_ = acc_
                    best_val_test_acc = [{
                        "test_acc": "%.4f" % acc_.item(),
                        "OSCR_C": "%.4f" % OSCR_C,
                        "OSCR_B": "%.4f" % OSCR_B,
                    }]
                    best_val_model = copy.deepcopy(net.state_dict())
                    torch.save(best_val_model, model_val_path)
                elif acc == best_val_acc:
                    best_val_test_acc.append({
                        "test_acc": "%.4f" % acc_.item(),
                        "OSCR_C": "%.4f" % OSCR_C,
                        "OSCR_B": "%.4f" % OSCR_B,
                    })
                    if acc_ > best_test_acc_:
                        best_test_acc_ = acc_
                        best_val_model = copy.deepcopy(net.state_dict())
                        torch.save(best_val_model, model_val_path)
                log("Current best val accuracy is {:.4f} (Test: {})".format(best_val_acc, best_val_test_acc), log_path)
                
            if acc_ > best_test_acc:
                best_test_acc = acc_    
                best_test_test_acc = [{
                    "OSCR_C": "%.4f" % OSCR_C,
                    "OSCR_B": "%.4f" % OSCR_B,
                }]    
                if save_best_test:
                    best_test_model = copy.deepcopy(net.state_dict())
                    torch.save(best_test_model, model_test_path)
            log("Current best test accuracy is {:.4f} ({})".format(best_test_acc, best_test_test_acc), log_path)

        if epoch+1 == renovate_step:
                log("Reset accuracy history...", log_path)

                best_val_acc = 0
                best_val_test_acc = []
                best_test_acc = 0
                best_test_test_acc = []

        scheduler.step()