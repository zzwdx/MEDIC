import torch
import math

def calculate_AUC(curve_x, curve_y):
    AUC = 0
    for i in range(len(curve_x)-1):
        w = math.fabs(curve_x[i] - curve_x[i+1])
        h = (curve_y[i] + curve_y[i+1])/2
        AUC += h*w

    return AUC

def generate_OSCR(argmax_k, conf_k, label, conf_u):
    num_samples = len(conf_k) + len(conf_u)

    confidence = torch.cat((conf_k, conf_u))
    conf_index_desc = confidence.argsort(descending=True)

    score_k = torch.cat((torch.where(argmax_k==label, 1, 0), torch.zeros_like(conf_u)))
    score_k_desc = score_k[conf_index_desc]
    score_u = torch.cat((torch.zeros_like(conf_k), torch.ones_like(conf_u)))
    score_u_desc = score_u[conf_index_desc]

    curve_x = torch.zeros(num_samples + 1)
    curve_y = torch.zeros(num_samples + 1)
    
    for i in range(num_samples):
        curve_y[i+1] = curve_y[i] + score_k_desc[i] 
        curve_x[i+1] = curve_x[i] + score_u_desc[i] 

    curve_y /= len(conf_k)
    curve_x /= len(conf_u)

    AUC = calculate_AUC(curve_x=curve_x, curve_y=curve_y)
    return AUC