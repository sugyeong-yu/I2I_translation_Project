from model import Generator
from model import Discriminator
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os,time,datetime

class Solver(object):

    def __init__(self,celeba_loader,rafd_loader,config):
        self.celeba_loader=celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # 시작!
        self.build_model()
        if self.use_tensorboard ==True:
            self.build_tensorboard()

    def build_model(self):
        # 모델정의
        if self.dataset in ['CelebA','RaFD']:
            self.G=Generator(self.g_conv_dim,self.c_dim,self.g_repeat_num)
            self.D=Discriminator(self.image_size,self.d_conv_dim,self.c_dim,self.d_repeat_num)
        elif self.dataset in ['Both']:
            self.G=Generator(self.g_conv_dim,self.c_dim+self.c2_dim+2,self.g_repeat_num) #2 for mask vector
            self.D=Discriminator(self.image_size,self.d_conv_dim,self.c_dim+self.c2_dim,self.d_repeat_num)

        # optimizer설정
        self.g_optimizer=torch.optim.Adam(self.G.parameters(),self.g_lr,[self.beta1,self.beta2])
        self.d_optimizer=torch.optim.Adam(self.D.parameters(),self.d_lr,[self.beta1,self.beta2])
        self.print_network(self.G,'G')
        self.print_network(self.D,'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self,model,name):
        # 네트워크 출력
        num_params=0
        for p in model.parameters():
            num_params+=p.numel() #tensor.numel()> input텐서의 총 요소수를 반환
        print(model)
        print(name)
        print("the number of parameter: {}".format(num_params))

    def restore_model(self,resume_iters):
        # trained model restore (g&d)
        print('loading the trained models from step{}'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir,'{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir,'{}-D.ckpt'.format(resume_iters))
        # 모델 가중치 불러오기
        self.G.load_state_dict(torch.load(G_path,map_location=lambda storage,loc:storage))
        self.D.load_state_dict(torch.load(D_path,map_location=lambda storage,loc:storage))


    def train(self):
        if self.dataset == 'CelebA':
            data_loader=self.celeb_loader
        elif self.dataset == 'RaFD':
            data_loader=self.rafd_loader

        # 디버깅을 위해 고정 입력을 받아옴
        data_iter=iter(data_loader)
        x_fixed, c_org= next(data_iter)
        x_fixed=x_fixed.to(self.device) # data to device
        c_fixed_list=self.create_labels(c_org,self.c_dim,self.dataset,self.selected_attrs)



