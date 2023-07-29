'''
The provided code demonstrates an attack on the ImageNet dataset targeting four popular classifiers. 
It can be readily converted for attacks against a specific classifier on different datasets.
'''


import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as torch_models
import os

from utils import    get_label
from utils import valid_bounds, clip_image_values
from PIL import Image
from torch.autograd import Variable
import time
from proposed_attack import Proposed_attack

###############################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
np.random.seed(992)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
###############################################################


model_arc = 'resnet50'
attack_methods = ['CGBA_H', 'CGBA']
for attack_method in attack_methods:
    dim_reduc_factor=4
    pair_num = 1000
    iteration = 93
    
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
        
    
    # Models ##############################################################
    
    if model_arc == 'resnet50':
        net = torch_models.resnet50(pretrained=True)
    if model_arc == 'resnet101':
        net = torch_models.resnet101(pretrained=True)
    if model_arc == 'vgg16':
        net = torch_models.vgg16(pretrained=True)
    if model_arc == 'ViT':
        import timm
        net = timm.create_model('vit_base_patch16_224', pretrained=True)            
    net = net.to(device)
    net.eval()
    
    
    
    
    
    all_norms = []
    all_queries = []
    image_iter =0
    
    for i in range(1, 1000): 
        
        idxs = np.random.choice(range(1,5000), 2)      #  Randomly picked indices of of two images
        
        if image_iter>=pair_num:
            break
        
        image_iter1= idxs[0]
        if len(str(image_iter1))==1:
            temp = "000"+ str(image_iter1)
        if len(str(image_iter1))==2:
            temp = "00"+ str(image_iter1)
        if len(str(image_iter1))==3:
            temp = "0"+ str(image_iter1) 
        if len(str(image_iter1))==4:
            temp =  str(image_iter1)
        img_name = "ILSVRC2012_val_0000" + temp + ".JPEG"
        # print('figure_num', temp)
        img_path = "Image_path/ImageNet/val"
    
        t11 = time.time()
        
        im_orig = Image.open(os.path.join(img_path, img_name))
        im_sz = 224
        im_orig = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig)
           
        # Bounds for Validity and Perceptibility
        delta = 255
        lb, ub = valid_bounds(im_orig, delta)
            
        # Transform data
        
        im = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean,
                                 std = std)])(im_orig)
        
        
        lb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(lb)
        ub = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(ub)
        
        lb = lb[None, :, :, :].to(device)
        ub = ub[None, :, :, :].to(device)
        
        x_0 = im[None, :, :, :].to(device)
        x_0_np = x_0.cpu().numpy()
        
        orig_label = torch.argmax(net.forward(Variable(x_0, requires_grad=True)).data).item()
        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
        str_label_orig = get_label(labels[np.int32(orig_label)].split(',')[0])
        
        ground_truth  = open(os.path.join('val.txt'), 'r').read().split('\n')        
        ground_name_label = ground_truth[image_iter1-1]   
        ground_label_split_all =  ground_name_label.split       
        ground_label_split =  ground_name_label.split()       
        ground_label =  ground_name_label.split()[1]
        ground_label_int = int(ground_label)
            
        
        str_label_ground = get_label(labels[np.int32(ground_label)].split(',')[0])
        
    ##############################################################################
        image_iter2= idxs[1]
        if len(str(image_iter2))==1:
            temp_t = "000"+ str(image_iter2)
        if len(str(image_iter2))==2:
            temp_t = "00"+ str(image_iter2)
        if len(str(image_iter2))==3:
            temp_t = "0"+ str(image_iter2) 
        if len(str(image_iter2))==4:
            temp_t =  str(image_iter2)
        img_name_t = "ILSVRC2012_val_0000" + temp_t + ".JPEG"
        #inp = "./data/ILSVRC2012_val_0000" + '0064' + ".JPEG" 
        # print('figure_num', temp_t) 
        
        im_orig_t = Image.open(os.path.join(img_path, img_name_t))
        im_sz = 224
        im_orig_t = transforms.Compose([transforms.Resize((im_sz, im_sz))])(im_orig_t)
        
           
        # Bounds for Validity and Perceptibility
        delta = 255
        lb, ub = valid_bounds(im_orig, delta)
            
            # Transform data
        
        im_t = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean,
                                 std = std)])(im_orig_t)
        
        
        lb = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(lb)
        ub = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])(ub)
        
        im_deepfool = im.to(device)
        lb = lb[None, :, :, :].to(device)
        ub = ub[None, :, :, :].to(device)
        
        x_0_t = im_t[None, :, :, :].to(device)
        x_0_t_np = x_0.cpu().numpy()
        # print('x_0_t', x_0_t)
        
        tar_label = torch.argmax(net.forward(Variable(x_0_t, requires_grad=True)).data).item()
        labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')
        str_label_tar = get_label(labels[np.int32(tar_label)].split(',')[0])
               
        ground_name_label_t = ground_truth[image_iter2-1]        
        ground_label_t =  ground_name_label_t.split()[1]
        
        print(f'\nSource image {temp}:  Class ID: {ground_label}   Class Name: {str_label_ground}')
        print(f'Target image {temp_t}:  Class ID: {ground_label_t}   Class Name: {str_label_tar}')
    
    
    ##############################################################################        
                  
        if ground_label_int != int(orig_label) or int(orig_label)==int(tar_label):
            print('Already missclassified ... Lets try another one!')
            
        else:    
            
        
            image_iter = image_iter + 1
            print('Image number good to go: ', image_iter)
    
        
        
            print('###############################################################################')
            print(f'Start: {attack_method} targeted will be run for {iteration} iterations with dim_reduc_facot: {dim_reduc_factor}')
            print('###############################################################################')
        
        
            t3 = time.time()
            attack = Proposed_attack(net, x_0, mean, std, lb, ub, 
                                      dim_reduc_factor=dim_reduc_factor, 
                                      tar_img=x_0_t, 
                                      attack_method = attack_method, 
                                      iteration=iteration)
            x_adv, n_query, norms = attack.Attack()
            t4 = time.time()
            print(f'##################### End Itetations:  took {t4-t3:.3f} sec #########################')
            
            all_norms.append(norms)
            all_queries.append(n_query)
    
    norm_array = np.array(all_norms)
    query_array = np.array(all_queries)
    norm_median = np.median(norm_array, 0)
    query_median = np.median(query_array, 0)
    
    
    if not os.path.exists('Targeted_results'):
        os.makedirs('Targeted_results')
        
    np.savez(f'Targeted_results/{attack_method}_Tar_{model_arc}_dimReducFac_{dim_reduc_factor}_imgNum_{pair_num}_iteration_{iteration}',
              norm = norm_median,
              quer = query_median,
              all_norms = norm_array,
              all_queries = query_array)
    
    
    
