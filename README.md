# 1. starGAN
starGAN의 개념과 특징, 구조에 대해 살펴본다.

## 1. 등장배경
- 기존의 image-to-image translation 연구( Pix2Pix, CycleGAN, DiscoGAN 등)는 3개이상의 domain에서 안정적이지 않다.
- domain수에 따라 Generator의 수가 증가하게 되는 단점이 있다.
  - domain수가 k일때 generator는 k(k-1)개
## 2. 특징
- 단일 모델을 사용하여 여러가지의 domain에 대해 img-to-img translation이 가능하다.
  - 한 이미지에서 나이, 성별 등 여러 attribute를 한번에 바꾸기 가능
  - attribute : 이미지에 내재된 의미있는 특징 ex)성별, 나이, 머리색
  - attribute value : attribute의 특정한 값 ex) 10/29, female / male, yellow/black
  - domain : 동일한 attribute value를 가지는 image set이다. ex) gender가 male인 image들이 하나의 domain이다.
- mask ventor를 이용해 다른 domain을 가진 dataset들을 동시에 학습시킬 수 있다.
  - A DB: 나이, 성별, 머리색 /  B DB : 표정, 옷 >> A DB에서 B DB의 특징을 이용해 변환가능\
![image](https://user-images.githubusercontent.com/70633080/109119331-e15a1680-7787-11eb-9cf5-53c2dd6c4101.png) 

## 3. Overview of StarGAN 
![image](https://user-images.githubusercontent.com/70633080/109119763-765d0f80-7788-11eb-8488-4bef923d50d9.png)
- (a) D : x ->  {Dsrc(x), Dcls(x)} D는 real과 fake를 구분함과 동시에 real image일때 해당 domain으로 분류해내는 것을 학습한다. 
  - 즉, D는 source와 domain labels에 대한 확률분포를 만든다.
- (b) G의 input으로 {input img, target domain}이 들어간다.
  - 여기서 target domain은 label(binary or one-hot vector)형태로 들어간다. 
  - output으로 fake이미지를 생성한다.
- (c) G는 original domain label(원래 가지고있던 image의 one-hot vector)를 가지고 fake image를 다시 origin image로 reconstruction하려한다. 
  - 따라서 output은 reconstructed image이다. 
- (d) D를 속이는 과정. G는 real과 구분 불가능하고 D에 의해 target domain으로 분류될 수 있는 이미지를 생성한다.

- **한개의 Generator를 다른 용도로 2번 사용된다.**

## 4. Loss
- 전체 Loss
![image](https://user-images.githubusercontent.com/70633080/109122147-82969c00-778b-11eb-84c7-6a4e00668b81.png)

### 1. Adversarial Loss
![image](https://user-images.githubusercontent.com/70633080/109122231-a0640100-778b-11eb-9678-848403c2e89f.png)
- G(x,c) : x와 target domain label을 가지고 G(x,c)라는 이미지를 생성한다. 
- D는 real과 fake를 구분하려는 loss
- D가 real로 분류할 경우 : 1에 가까운 값으로 출력된다. 
- D가 fake로 분류할 경우 : 0에 가까운 값으로 출력된다.
- real img인 x에 대해선 Dsrc(x)는 1을 가지도록, Dsrc(G(x,c))에서는 G가 real인거처럼 학습하므로 1을 가지도록 학습
- 따라서 LOSS가 최소가 됨 (-log는 1에가까우면 0으로, 0에 가까우면 무한대로 발산)

### 2. Domain Classification Loss
- input img x와 target domain label c가 주어졌을 때, x가 output img y로 변환되어 이것이 target domain c로 분류되는 것이 목적이다. 
- 따라서, D와 G를 optimize할때 domain classification loss를 부과한다.

#### 2-1. domain classification loss of real images used to optimize D
![image](https://user-images.githubusercontent.com/70633080/109123102-cccc4d00-778c-11eb-9134-d5f048c524af.png)
- real image가 들어오면 real image에 대한 original domain값으로 분류되게 하는 loss이다.
- D를 위해 사용되는 loss ( 얼마나 잘 분류했는가)

#### 2-2. domain classification loss of fake images used to optimize G
![image](https://user-images.githubusercontent.com/70633080/109123339-10bf5200-778d-11eb-9285-a1f28e234344.png)
- 생성된 fake img(target domain으로 변환된 이미지)가 target domain으로 분류될수 있도록 LOSS를 최소화한다.
- G를 위해 사용되는 loss ( 얼마나 잘 속였는가)

### 3. Reconstruction Loss
![image](https://user-images.githubusercontent.com/70633080/109124255-0c476900-778e-11eb-92f2-d7546d9aa5f5.png)
- G가 생성한 fake image와 바뀐 target domain(origin label)을 다시 G의 input으로 넣는다.
- fake image를 origin image의 형태로 다시 복원한 image가 출력됨.
- 따라서 **target domain부분은 변화시키되 input image의 형태를 유지하게끔 복원하기 위해 cycle consistence loss**를 이용한다.

### 4. Total loss
![image](https://user-images.githubusercontent.com/70633080/109122231-a0640100-778b-11eb-9678-848403c2e89f.png)
- λcls, λrec : 하이퍼파라미터임. domain분류와 reconstruction loss들의 상대적인 중요도를 컨트롤함.
- D는 adversarial loss를 maximize하길 원하기 때문에 
