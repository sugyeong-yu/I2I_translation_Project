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

    def create_labels(self,c_org,c_dim=5,dataset='CelebA',selected_attrs=None):
        if dataset=='CelebA':
            hair_color_indices=[]
            for i,attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair']:
                    hair_color_indices.append(i)
        c_trg_list=[]
        for i in range(c_dim):
            if dataset=='CelebA':
                c_trg=c_org.clone()

    def label2onehot(self,labels,dim):
        batch_size=labels.size(0)
        out=torch.zeros(batch_size,dim)
        out[np.arange(batch_size),labels.long()]=1
        return out
    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)
    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def train(self):
        if self.dataset == 'CelebA':
            data_loader=self.celeb_loader
        elif self.dataset == 'RaFD':
            data_loader=self.rafd_loader

        # 디버깅을 위해 고정 입력을 받아옴
        data_iter=iter(data_loader)
        x_fixed, c_org= next(data_iter) # img & label
        x_fixed=x_fixed.to(self.device) # data to device
        c_fixed_list=self.create_labels(c_org,self.c_dim,self.dataset,self.selected_attrs)

        g_lr=self.g_lr
        d_lr=self.d_lr

        start_iters=0
        # 학습을 이어서 하는경우
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # training 시작!
        print("===========Start training================")
        start_time=time.time()
        for i in range(start_iters,self.num_iters):
            # ================================================================
            #                     1. preprocess input data
            #  ===============================================================
            try:
                x_real, label_org =next(data_iter)# batchsize만큼 가져옴 데이터 > domain수 7일때 label_org 예시 : [2,2,5,6,4,3]
            except:
                data_iter = iter(data_loader)
                x_real ,label_org=next(data_iter)
            # 생성할 타겟도메인 랜덤으로 뽑기
            rand_idx=torch.randperm(label_org.size(0))
            label_trg=label_org[rand_idx] # ex) label_trg=2

            if self.dataset=='CelebA':
                c_org=label_org.clone()
                c_trg=label_trg.clone()
            elif self.dataset=='RaFD':
                c_org=self.label2onehot(label_org,self.c_dim)
                c_trg=self.label2onehot(label_trg,self.c_dim)

            x_real=x_real.to(self.device) # Input images
            c_org=c_org.to(self.device) # Original domain labels
            c_trg=c_trg.to(self.device) # target domain labels
            label_org=label_org.to(self.device) # classification loss 계산을 위한 labels
            label_trg=label_trg.to(self.device)# classification loss 계산을 위한 labels

            #====================================================================
            #                   2. Train the discriminator
            #===================================================================

            # real image 분류 에러
            out_src,out_cls=self.D(x_real)
            d_loss_real=-torch.mean(out_src) # 원본이미지에 대한 Real/Fake 판별값의 평균
            d_loss_cls = self.classification_loss(out_cls,label_org,self.dataset)

            # fake image 분류에러
            x_fake=self.G(x_real,c_trg)
            out_src,out_cls=self.D(x_fake.detach())
            d_loss_fake=torch.mean(out_src)#합성이미지에 대한 Real/Fake 판별값의 평균

            # loss gradient penalty 계산
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # backward and optimize
            d_loss=d_loss_real+d_loss_fake+self.lambda_cls*d_loss_cls+self.lambda_gp*d_loss_gp
            self.g_optimizer.zero_grad()
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # loging ( loss monitoring을 위함 )
            loss = {}
            loss['D/loss_real']=d_loss_real.item()
            loss['D/loss_fake']=d_loss_fake.item()
            loss['D/loss_cls']=d_loss_cls.item()
            loss['D/loss_gp']=d_loss_gp.item