import argparse
import torch
import pickle
import os
import sys
from dataset.dataloader import get_dataloader
from model.model import MutiClassifier, MutiClassifier_, resnet18_fast, resnet50_fast, ConvNet
from util.log import log
from torch.nn import functional as F
from util.ROC import generate_OSCR

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dataset', default=None)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hits', type=int, default=10)

    parser.add_argument('--save-dir', default='save')
    parser.add_argument('--save-name', default='demo')
    
    args = parser.parse_args()

    # It can be used to replace the following code, but the editor may take it as an error.
    # locals().update(vars(args))

    # It can be replaced by the preceding code.

    gpu = args.gpu
    dataset = args.dataset
    batch_size = args.batch_size
    hits = args.hits
    save_dir = args.save_dir
    save_name = args.save_name

    param_path = os.path.join(save_dir, 'param', save_name + '.pkl')
    
    with open(param_path, 'rb') as f: 
        param = pickle.load(f)

    print(param)
        
    if dataset is None:
        dataset = param['dataset']    
    if dataset == 'PACS':
        root_dir = '/data/datasets/PACS'
        small_img = False
    elif dataset == 'OfficeHome':
        root_dir = ''
        small_img = False
    elif dataset == "DigitsDG":
        root_dir = ''
        small_img = True

    source_domain = sorted(param['source_domain'])
    target_domain = sorted(param['target_domain'])
    known_classes = sorted(param['known_classes'])
    unknown_classes = sorted(param['unknown_classes'])
     
    net_name = param['net_name']
    if "share_param" in param:
        share_param = param['share_param']
    else:
        share_param = False


    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    num_classes = len(known_classes)
    log_path = os.path.join('./', save_dir, 'log', save_name + '_test.txt')
    model_path = os.path.join('./', save_dir, 'model', 'val', save_name + '.tar')

    log('Save name: {}'.format(save_name), log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log('Loading dataset...', log_path)

    test_k = get_dataloader(root_dir=root_dir, domain=target_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
    if len(unknown_classes) > 0:
        test_u = get_dataloader(root_dir=root_dir, domain=target_domain, classes=unknown_classes, batch_size=batch_size, get_domain_label=False, get_class_label=False, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)    
    else:
        test_u = None

    log('Source domain: {}'.format(source_domain), log_path)
    log('Target domain: {}'.format(target_domain), log_path)

    log('Known classes: {}'.format(known_classes), log_path)
    log('Unknown classes: {}'.format(unknown_classes), log_path)
    log('Loading models...', log_path)

    if share_param:
        muticlassifier = MutiClassifier_
    else:
        muticlassifier = MutiClassifier

    if net_name == 'resnet18':
        net = muticlassifier(net=resnet18_fast(), num_classes=num_classes)
    elif net_name == 'resnet50':
        net = muticlassifier(net=resnet50_fast(), num_classes=num_classes, feature_dim=2048)
    elif net_name == 'convnet':
        net = muticlassifier(net=ConvNet(), num_classes=num_classes, feature_dim=256)

    net.load_state_dict(torch.load(model_path))

    net = net.to(device)

    log('Network: {}'.format(net_name), log_path)
    log('Share Parameter: {}'.format(share_param), log_path)

    log('Start Testing...', log_path)  

    net.eval()

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

    argmax = torch.argmax(output_k_sum, axis=1)
    num_correct = (argmax == label_k_sum).sum()
    test_acc = num_correct / len(output_k_sum)

    log("Close-set acc: {:.4f}".format(test_acc), log_path)

    if test_u is None:
        sys.exit()

    output_u_sum = []
    b_output_u_sum = []
    label_u_sum = [] 
    with torch.no_grad():
        for input, *_ in test_u:
            input = input.to(device)

            output = net(x=input)
            output = F.softmax(output, 1)
            b_output = net.b_forward(x=input)
            b_output = b_output.view(output.size(0), 2, -1)
            b_output = F.softmax(b_output, 1)

            output_u_sum.append(output)
            b_output_u_sum.append(b_output)
            label_u_sum.append(label)


    output_u_sum = torch.cat(output_u_sum, dim=0)
    b_output_u_sum = torch.cat(b_output_u_sum, dim=0)
    label_u_sum = torch.cat(label_u_sum)

#################################################################################
    log('C classifier:', log_path)

    max_prob_k, _ = torch.max(output_k_sum, 1)
    max_prob_u, _ = torch.max(output_u_sum, 1)
    thd_min = min(torch.min(max_prob_k).item(), torch.min(max_prob_u).item())
    thd_max = max(torch.max(max_prob_k).item(), torch.max(max_prob_u).item())

    outlier_range = [thd_min + (thd_max - thd_min) * i / (hits-1) for i in range(hits)]

    best_overall_acc = 0.0
    best_thred_acc = 0.0
    best_overall_Hscore = 0.0
    best_thred_Hscore = 0.0

    for threshold in outlier_range:
        num_correct_k = num_correct_u = 0
        num_total_k = num_total_u = 0

        argmax_k = torch.argmax(output_k_sum, axis=1)
        for i in range(len(argmax_k)):
            if argmax_k[i] == label_k_sum[i] and output_k_sum[i][argmax_k[i]] >= threshold:
                num_correct_k +=1
        num_total_k += len(output_k_sum)


        argmax_u = torch.argmax(output_u_sum, axis=1)
        for i in range(len(argmax_u)):
            if output_u_sum[i][argmax_u[i]] < threshold:
                num_correct_u +=1
        num_total_u += len(output_u_sum)


        acc_k = num_correct_k / num_total_k
        acc_u = num_correct_u / num_total_u
        acc = (num_correct_k + num_correct_u) / (num_total_k + num_total_u)
        hs = 2*acc_k*acc_u/(acc_k + acc_u)

        if acc > best_overall_acc:
            best_overall_acc = acc
            best_thred_acc = threshold
        if hs > best_overall_Hscore:
            best_overall_Hscore = hs
            best_thred_Hscore = threshold

       
        log('Acc_k: {:.4f} Acc_u: {:.4f}, Acc: {:.4f}, H-Score: {:.4f} ({})'.format(acc_k, acc_u, acc, hs, threshold), log_path) 
        
    print('Best OverallAcc: %.4f' % (best_overall_acc), 'Best OverallHscore: %.4f' % (best_overall_Hscore))

    conf_k, argmax_k = torch.max(output_k_sum, axis=1)
    conf_u, _ = torch.max(output_u_sum, axis=1)

    OSCR = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)

       
    log('OSCR: {:.4f}'.format(OSCR), log_path) 

###################################################################################################################

    log('B classifier:', log_path)
    max_prob_k, _ = torch.max(b_output_k_sum[:, 1, :], 1)
    max_prob_u, _ = torch.max(b_output_u_sum[:, 1, :], 1)
    thd_min = min(torch.min(max_prob_k).item(), torch.min(max_prob_u).item())
    thd_max = max(torch.max(max_prob_k).item(), torch.max(max_prob_u).item())

    outlier_range = [thd_min + (thd_max - thd_min) * i / (hits-1) for i in range(hits)]

    best_overall_acc = 0.0
    best_thred_acc = 0.0
    best_overall_Hscore = 0.0
    best_thred_Hscore = 0.0

    for threshold in outlier_range:
        num_correct_k = num_correct_u = 0
        num_total_k = num_total_u = 0

        argmax_k = torch.argmax(output_k_sum, axis=1)
        for i in range(len(argmax_k)):
            if argmax_k[i] == label_k_sum[i] and b_output_k_sum[i][1][argmax_k[i]] >= threshold:
                num_correct_k +=1
        num_total_k += len(output_k_sum)


        argmax_u = torch.argmax(output_u_sum, axis=1)
        for i in range(len(argmax_u)):
            if b_output_u_sum[i][1][argmax_u[i]] < threshold:
                num_correct_u +=1
        num_total_u += len(output_u_sum)


        acc_k = num_correct_k / num_total_k
        acc_u = num_correct_u / num_total_u
        acc = (num_correct_k + num_correct_u) / (num_total_k + num_total_u)
        hs = 2*acc_k*acc_u/(acc_k + acc_u)

        if acc > best_overall_acc:
            best_overall_acc = acc
            best_thred_acc = threshold
        if hs > best_overall_Hscore:
            best_overall_Hscore = hs
            best_thred_Hscore = threshold

        
        log('Acc_k: {:.4f} Acc_u: {:.4f}, Acc: {:.4f}, H-Score: {:.4f} ({})'.format(acc_k, acc_u, acc, hs, threshold), log_path) 
        
    print('Best OverallAcc: %.4f' % (best_overall_acc), 'Best OverallHscore: %.4f' % (best_overall_Hscore))

    _, argmax_k = torch.max(output_k_sum, axis=1)
    _, argmax_u = torch.max(output_u_sum, axis=1)

    argmax_k_vertical = argmax_k.view(-1, 1)
    conf_k = torch.gather(b_output_k_sum[:, 1, :], dim=1, index=argmax_k_vertical).view(-1)

    argmax_u_vertical = argmax_u.view(-1, 1)
    conf_u = torch.gather(b_output_u_sum[:, 1, :], dim=1, index=argmax_u_vertical).view(-1)

    OSCR = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)

       
    log('OSCR: {:.4f}'.format(OSCR), log_path) 