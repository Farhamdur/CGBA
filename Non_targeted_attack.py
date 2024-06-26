'''
The provided code demonstrates an attack on the ImageNet dataset targeting four popular classifiers. 
It can be readily converted for attacks against a specific classifier on different datasets.
'''
import torchvision.transforms as transforms
import torchvision.models as torch_models
import numpy as np
import torch
import os
from utils import    get_label
from utils import valid_bounds
from PIL import Image
from torch.autograd import Variable
import time
from proposed_attack import Proposed_attack



    
##############################################################################
torch.manual_seed(992)
torch.cuda.manual_seed(992)
np.random.seed(992)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##############################################################################



num_img = 50
iteration =93  
model_arc = 'resnet50'             # Enter 'resnet50' or 'resnet101' or 'vgg16' or 'ViT
attack_methods = ['CGBA_H', 'CGBA']             # Attacking methods: 'CGBA' or 'CGBA_H'  
dim_reduc_factor = 4               # dim_reduc_factor=1 for full dimensional image space

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225] 

for attack_method in attack_methods:
    # Models 
    
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

    for image_iter1 in range(1, 10000): 
        if image_iter>=num_img:
            break
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
    
        print(f'\nSource image {temp}:  Class ID: {ground_label}   Class Name: {str_label_ground}')
    
    ##############################################################################        
        
        
                  
        if ground_label_int != int(orig_label) :
            print('Already missclassified ... Lets try another one!')
            
        else:    
            
        
            image_iter = image_iter + 1
            print('Image number good to go: ', image_iter)
    
        
        
            print('#################################################################################')
            print(f'Start: {attack_method} non-targeted will be run for {iteration} iterations with dim_reduc_facot: {dim_reduc_factor}')
            print('#################################################################################')
        
        
            t3 = time.time()
            attack = Proposed_attack(net, x_0, mean, std, lb, ub, 
                                     dim_reduc_factor=dim_reduc_factor,  
                                     attack_method = attack_method, 
                                     iteration=iteration)
            x_adv, n_query, norms= attack.Attack()
            t4 = time.time()
            print(f'##################### End Itetations:  took {t4-t3:.3f} sec #########################')
            
            all_norms.append(norms)
            all_queries.append(n_query)
    
    norm_array = np.array(all_norms)
    query_array = np.array(all_queries)
    norm_median = np.median(norm_array, 0)
    query_median = np.median(query_array, 0)
    
    
    if not os.path.exists('Non_targeted_results'):
        os.makedirs('Non_targeted_results')
        
    np.savez(f'Non_targeted_results/{attack_method}_nonTar_{model_arc}_dimReducFac_{dim_reduc_factor}_imgNum_{num_img}_iteration_{iteration}',
              norm = norm_median,
              query = query_median,
              all_norms = norm_array,
              all_queries = query_array)
    

