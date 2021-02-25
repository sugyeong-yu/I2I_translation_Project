# starGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation (CVPR 2018)

**starGAN**: 하나의 단일 뉴럴 네트워크만 학습해서 사용해도 다중 도메인을 가진 이미지 변환을 할 수 있는 방법을 제시  

```domain```: 동일한 attribute-value를 가지는 이미지 셋  


### 배경
- 기존의 image-to-image translation 연구(pix2pix, CycleGAN, DiscoGAN 등)는 3개 이상의 도메인에서는 작동하지 않음
- 도메인 수에 따라 Generator수가 증가하게 되는 단점이 있음 (K가 도메인의 갯수이면 k(k-1)개의 Generator 학습)

### starGAN Contributes
- 단일 모델을 사용하여 여러가지의 domain들에 대해 image-to-image translation이 가능(한 이미지에서 나이, 성별 등 여러개의 attribute를 한번에 바꾸기 가능)
- 동시에 다른 domain을 가진 dataset들을 동시에 학습 (mask vector method) / No pair between each domain


### 모델 비교(Cross-domain Models & StarGAN)
![image](https://user-images.githubusercontent.com/72767245/109122244-a6f27880-778b-11eb-9fee-cb16d6874019.png)


**```Cross-domain Models```**  
- 기존 연구[CycleGAN, DiscoGAN, pix2pix, cGAN 등]에서 한 개의 특징만을 학습해서 변환하는 방법을 제시
- Attributes가 K개면 K(K-1)개의 모델 생성
- G_ij는 도메인 i에서 도메인 j로 변환하는 하나의 신경망을 나타냄

**```StarGAN```**  
- 하나의 Generator
- 모든 데이터의 정보를 이용하여 Train 가능
- 단점: image의 사이즈가 다 같아야됨(CycleGAN은 Generator가 다 달라서 Flexibility하다)


### StarGAN의 단일 모델(Unified Model)

![image](https://user-images.githubusercontent.com/72767245/109122462-fcc72080-778b-11eb-8ff0-10dadc1791a7.png)

- 기존의 GAN은 잠재변수 z를 입력값으로 받는 반면, starGAN에서는 변환하고자 하는 ```도메인 정보(c)```와 ```원본 이미지(x)```를 입력값으로 받음
- 원본 이미지를 입력값으로 받는 건 VAE가 사용된 UNIT 모델에서 아이디오 차용
- 판별기는 원본 이미지의 Real/Fake 여부에 더해서 특정 도메인 정보까지 맞추는걸 목표로 함

#### StarGAN Architecture for short
- Generator: G(x,c) -> y
- Discriminator: D:x -> {D_src(x), D_cls(x)}
- real/fake, Attributions

#### StarGAN Architecture
- (a) D:x -> {Dsrc(x), Dcls(x)}  
  - D는 real image 나 fake image가 들어오면 real인지 fake인지 동시에 구분함
  - real image 일때 해당 domain으로 분류해내는 것을 학습
  - D는 source labels에 대한 확률 분포
- (b) G의 input으로 {input image, target domain} 이 들어갑니다
  - 여기서 target domain은 label(binary or one-hot vector 형태)로 들어감
  - output은 fake 이미지를 생성
- (c) G는 original domain label(원래 내가 가지고 있던 image의 one-hot vector)을 가지고 fake image를 다시 original image로 reconstruction하려고 함
  - output이 원래의 input image와 유사하게끔 만들어진 이미지라해서 Reconstrcuted image라고 부른다
- (d) D를 속이는 과정
  - G는 real image와 구분 불가능하고 D에 의해 Target domain으로 분류될 수 있는 이미지를 생성
```Generator가 2개인 것처럼 보이지만 한개의 Generator를 다른 용도로 2번 사용된 것```  


### StarGAN Loss

![image](https://user-images.githubusercontent.com/72767245/109124990-e5d5fd80-778e-11eb-8484-db1985211428.png)

#### 1. Adversarial Loss
![image](https://user-images.githubusercontent.com/72767245/109125076-fb4b2780-778e-11eb-9431-73c322ee78a7.png)

- G는 x와 target domain label G(x, c)라는 이미지를 생성하고, D는 real and fake image들을 구분하려하는 loss
- real image인 x에 대해서 Dsrc(x)는 Dsrc(x)는 1을 가지도록, Dsrc(G(x,c))에서는 G가 real인 것처럼 학습을 진행해야 하니 역시 1을 가지도록 학습
  - -log함수는 1에 가까우면 0으로 수렴, 0에 가까우면 ∞로 발산

#### 2. Domain Classification Loss
- input image x와 target domain label c가 주어질때 x가 output image y로 변환되어 그것이 target domain c로 분류되는 것이 목적
- 그렇기 때문에 D와 G를 optimize할 때 domain classification loss를 부과
- Domain Classification Loss 두개의 term으로 나눌 수 있음


##### 2-1. domain classification loss of real images used to optimize D
![image](https://user-images.githubusercontent.com/72767245/109127571-ec19a900-7791-11eb-9ba1-62be86cef799.png)  


real image가 들어오면 real image 에 대한 original 도메인 값으로 분류되게 하는 loss 임
adversarial loss와 같고, 결국엔 이 loss를 최소화시켜야함

##### 2-2. domain classification loss of fake image used to optimize G



