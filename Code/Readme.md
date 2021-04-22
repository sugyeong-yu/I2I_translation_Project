# code discription

## main.py
- train 수행시 main.py실행
- parameter config 
  - 1. cmd에서 함수 호출 시 -- 로 train 인자를 넘겨줌 (인자config)\
  ![image](https://user-images.githubusercontent.com/70633080/115144805-a38fb500-a089-11eb-9e96-798c6ef22bc7.png)
  - 2. command에서 호출\
  ![image](https://user-images.githubusercontent.com/70633080/115144847-d33ebd00-a089-11eb-83a1-a5e89315b505.png)
- parameter 설명
  - c_dim : dataset에서 사용한 특성(attribute) ->  default 5 (기본적으로 CelebA에서 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young' 을 사용)
  - image size : 이미지 크기 -> default 128 * 128
  - g_conv_dim : Generator에서 첫번째 layer의 filter 수 -> default 64\
  ![image](https://user-images.githubusercontent.com/70633080/115144999-814a6700-a08a-11eb-8df4-f8619a4fec45.png)
  - g_repeat_num : Generator에서 Residual Block수 -> default 6
  - d_conv_dim : Discriminator에서 첫번째 layer의 filter수 -> default 64\
  ![image](https://user-images.githubusercontent.com/70633080/115145056-beaef480-a08a-11eb-8fa4-d34eced9d873.png)
  - d_repeat_num : Discriminator에서 Output layer를 제외한 conv layer의 수 -> default 6
  - lambda_gp : adversarial loss를 구하는데 사용되는 gradient penalty -> default 10
  - num_iters : 학습과정에서 몇번의 iteration -> default 200000
  - n_critic : Discriminator가 몇번 update되었을때 Generator를 한번 update?\
  - selected_attrs : CelebA 에서 사용할 특성들 -> default  'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young' 
  - test_iters : 모델테스트를 위해 학습된 모델을 몇번쨰 step에서 가져올것인지. 
  - num_workers : 몇개의 CPU코어를 할당? -> default 1
### def main(config)
1. data_load\
![image](https://user-images.githubusercontent.com/70633080/115144882-008b6b00-a08a-11eb-9699-67345dae83c8.png)
2. mode select
- train인지 test인지 모드설정
- solver.py에 정의된 train()또는 test()를 실행하게됨.
```
if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()
```

## model.py
### Residual Block
![image](https://user-images.githubusercontent.com/70633080/115145220-747a4300-a08b-11eb-932b-89120a7a737f.png)
```
def __init__(self,dim_in,dim_out):
  self.main = nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
              nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
              nn.ReLU(inplace=True),
              nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
              nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
def forward(self, x):
        return x + self.main(x)
```
- nn.Module을 상속받는다.
- dim_in/dim_out : 입력/출력 dimention
  - 논문에서는 dim_in과 out이 256으로 설정.
- 처음의 정보를 더해 정보를 보존하도록 하는게 핵심.
- forward()는 클래스객체명(forward의 매개변수) 형태로 호출하면 자동으로 forward()가 호출된다.

### Generator
#### init()
```
def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
```
- conv_dim : 첫번째 Conv layer의 output dimention
- c_dim : domain_label수 
- repeat_num : Residual Block의 수 
```
layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
# Down-sampling layers.
      curr_dim = conv_dim
      for i in range(2):
          layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
          layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
          layers.append(nn.ReLU(inplace=True))
          curr_dim = curr_dim * 2
```
![image](https://user-images.githubusercontent.com/70633080/115145414-5bbe5d00-a08c-11eb-9896-6338169848b0.png)
- **DownSampling** 
  - conv2d()를 사용하며 dimention은 점점 커짐.
  - 첫번째 Conv layer의 입력 dimention은 3+c_dim  > forward()에서 이유설명
  - Instance Normalization과 ReLu를 거친다.
  - 첫번째 downsampling layer가 끝났으면 conv_dim대신 curr_dim을 사용
  - 두번째, 세번째 downsampling layer를 거친다. -> output dimention은 input dimention의 2배가 된다.
```
for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
```
- **Bottleneck 구조**
  - Residual Block을 repeat_num만큼 만들어 layers에 append
  - dim_in과 dim_out이 같다. (curr_dim > 256)
  - 논문에서는 6개의 block이 사용됨.
```
# Up-sampling layers.
for i in range(2):
    layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
    layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
    layers.append(nn.ReLU(inplace=True))
    curr_dim = curr_dim // 2
layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
layers.append(nn.Tanh())
```
- **Upsampling**
  - Deconvolution을 위해 ConvTranspose2d()를 사용 , dimention이 작아진다.
  - Deconv,IN,ReLU 과정을 2  번 반복한다.
  - for 문 탈출 후 Conv와 Tanh를 거친다.
```
self.main = nn.Sequential(*layers)
```
- layer들을 sequential로 묶는다.
#### forward()
```
def forward(self, x, c):
    c = c.view(c.size(0), c.size(1), 1, 1)
    c = c.repeat(1, 1, x.size(2), x.size(3))
    x = torch.cat([x, c], dim=1)
    return self.main(x)
```
- forward 매개변수에 real image x와 target domain c가 들어옴. (solver.py에서 전달)
- img_size : 128 * 128 * 3 이 16개 (CelebA 기준)
- c : domain값들로 아래그림과 같은 형태를 가짐
  - ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']\
![image](https://user-images.githubusercontent.com/70633080/115145795-5a8e2f80-a08e-11eb-8d78-fe69d04bab5d.png)
- c.view & c.repeat 
  - c.size는 16,7,128,128 이 됨\
![image](https://user-images.githubusercontent.com/70633080/115145826-8f9a8200-a08e-11eb-8f08-9e5bba29b93f.png)
![image](https://user-images.githubusercontent.com/70633080/115145837-9aedad80-a08e-11eb-9ce0-6a1af46e1294.png)
- torch.cat([x,c],dim=1)
  - x.size() : [16,10,128,128]\
![image](https://user-images.githubusercontent.com/70633080/115145890-f6b83680-a08e-11eb-8c0b-d4209b81c8b8.png)
- 이후 self.main에 x를 입력으로 넣어 호출한다. 
- Generator의 첫번째 layer입력 dimention이 3+c_dim인 이유 : 초기 x의 dimention은 3이지만 torch.cat과정으로인해 3+c_dim이 되기 때문.
- self.main(x)에서 return되는 image는 초기 x와 같은 size [16,3,128,128]
  - 논문에서의 Generator의 마지막 layer의 shape과 동일하다.

### Discriminaor
- 논문에서의 Discriminaor\
![image](https://user-images.githubusercontent.com/70633080/115385663-58a1a900-a213-11eb-8241-1d2e7668012f.png)\
![image](https://user-images.githubusercontent.com/70633080/115385683-5f302080-a213-11eb-98fc-0ebdee714c81.png)
#### init()
```
 def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
```
- Discriminator에는 입력으로 평범한 RGB이미지가 들어옴
  - 따라서 input dimention이 3
- 입력으로 들어온 image는 Conv layer를 먼저 거친다.
```
curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
```
- 이는 Hidden layer부분 이다.
- repeat num이 defalut로 6으로 설정되어 있으므로 5번 반복한다. (1~5)
- dimention이 layer를 진행할 수록 2배가 된다.
```
kernel_size = int(image_size / np.power(2, repeat_num))
self.main = nn.Sequential(*layers)
self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
```
- Output layer부분이다.
  - D_src : real/fake -> conv1
  - D_cls : 입력이미지의 Domain label  -> conv2
- Discriminator는 입력 img의 real/fake 를 구분한다.
- kernel_size = (img한변길이/2^repeat_num) 을 두번째 Conv layer에 kernel size로 할당한다. (h/64)
- conv1 : real/fake여부를 출력해야하므로 output dimention=1
- conv2 : Domain의 label을 출력해야하므로 output dimention=c_dim

#### forward()
```
 def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
```
- forward(self,x) 
  - x : 진짜인지 가짜인지 판별할 img
- self.main(x) : Hidden layer까지 모두 거친 output이 return된다.
- self.conv1과 self.conv2에 인자를 전달해 각 결과를 return한다. 
- out_cls 의 size를 조정해 out_src와 함께 return된다.

## solver.py
### init()
```
class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader
```
- Solver class는 nn.Module을 상속받지않는다. 
- Solver 객체 호출시 celeba_loader, rafd_loader, config를 넘겨준다.
  - 즉 각 DB에 대한 dataloader와 파라미터 설정값인 config를 넘겨주는 것
- line 23~65 : config를 넘겨주는 과정
### build_model()
- build_model()은 Generator와 Discriminator를 만든다.
```
if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)
```
- self.G, self.D에 생성자와 판별자를 할당.
```
self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
self.G.to(self.device)
self.D.to(self.device)
```
- StarGan에서의 모든 모델은 학습과정에서 Adam optimizer를 사용한다.
- 필요한 파라미터들은 main.py의 parsing과정에서 불러온다.
- 모델.to(device) : 사용중인 장치에 최적화된 형태로 모델을 변환하는 작업
### print_network()
```
self.print_network(self.G, 'G')
self.print_network(self.D, 'D')
def print_network(self, model, name):
      """Print out the network information."""
      num_params = 0
      for p in model.parameters():
          num_params += p.numel()
      print(model)
      print(name)
      print("The number of parameters: {}".format(num_params))
```
- print_network()는 인자로 모델과 모델의 이름을 전달받음
- 모델의 네트워크 정보를 출력하는 함수.
- numel(): 파라미터의 원소 수를 구함
- num_params : numel()로 구한 원소수를 반복하며 더함.

### restore_model()
```
def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
```
- restore_model()은 이전에 학습해 저장된 모델을 불러오는 역할
- 정해진 iteration('num_iters')만큼 학습이 완료되지 못한 경우 'resume_iters'인자에 학습을 이어할 수 있는 iteration수를 지정해주면 해당 iteration부터 학습을 이어할 수 있다.
- 이때, 그 iteration에 해당하는 저장된 모델이 있어야한다. 
- resume_iters : main.py에서 parsing > default가 None
- load_state_dict : 학습된 G와 D모델에서 state_dict(학습된 가중치)를 불러와 각각 self.G와 D에 저장한다. 
### build_tensorboard()
```
def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)
```
- logger.py에 정의된 Logger class 객체를 생성한다.
- logger.py는 뒤에서 설명
### update_lr()
```
def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
```
- g_optimizer와 d_optimizer에서의 lr을 업데이트하는 함수
### reset_grad()
```
def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
```
- g_optimizer와 d_optimizer의 가중치를 0으로 리셋
### denorm()
```
def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
```
- out의 모든원소를 0,1 범위로 만들어 반환
###
```
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
```
![image](https://user-images.githubusercontent.com/70633080/115658658-278cba00-a374-11eb-8f7d-332b53c5d3b4.png)
- 람다gp : gradiemt panalty이다.
- l2 norm은 각 원소를 제곱해 모두 더한것에 루트를 씌워 구한다.
