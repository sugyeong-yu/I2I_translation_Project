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
  - 논문에서 λcls = 1로, λrec=10으로 설정
- D는 adversarial loss를 maximize하길 원하기 때문에 마이너스가 붙은 것이다.
- G는 adversarial loss를 minimize하길 원하므로 마이너스가 붙지 않은 것이다.  

## 5. Mask vector
- 논문에서 사용된 dataset은 ficial attribute만 가지고 있는 CelebA와 facial expression만 가지고 있는 RaFD를 사용한다.
- 이 둘은 서로 다른 도메인이기 때문에 저자는 mask vector , m 이라는 개념을 도입했다.
- m은 one-hot vector로 나타내지고 두 dataset을 합칠때는 concate하면 된다.
``` ~c = [c1,c2,...,cn,m]```
- ci : i번째 dataset의 label들의 vector , ci는 binary attribute를 가진 binary vector 또는 categorical attribute를 가진 one-hot vector이다.
- CelebA와 RaFD를 교차시킴으로써 D는 두 dataset에서 차이를 구분짓는 모든 feature들을 학습하게 된다.
- G는 모든 label을 컨트롤하는 것을 학습하게 된다.

## 6. 전체적인 구조
![image](https://user-images.githubusercontent.com/70633080/109127529-e328d780-7791-11eb-9830-0901b0b10ca4.png)

## 7. Result
![image](https://user-images.githubusercontent.com/70633080/109127607-f5a31100-7791-11eb-8429-099c88090ba2.png)\
![image](https://user-images.githubusercontent.com/70633080/109127658-0489c380-7792-11eb-9ad6-9a42c09e4a2e.png)
- Celeb A와 RaFD를 128 * 128로 동일하게 맞춰준 후 모델에 입력한다.
- Celeb A에서는 40개의 attribute 중 7개만 뽑아 사용했고 RaFD는 작은 dataset이기 때문에 모두 사용한다.
- 여러개의 dataset을 이용한 StarGAN joint는 dataset을 모두 사용해 좋은 성능을 보인다.
- multiple domains + multiple datasets 학습가능ㅇㅇㄴㄹ

# 2. starGAN v2
## 1. starGAN v1 -> v2
- starGAN
  - starGAN v1은 각 domain에 대한 결정을 한번에 하나씩 직접해야한다.
  - 데이터 분포에 대한 다양한 특성을 반영하지 못함
- starGAN v2
  - 어떤 domain의 image한개를 target domain의 여러 다양한 image로 변경할 수 있다.
  - 특정 도메인에 대한 다양한 style들을 표현할 수 있다.
## 2. key point
- Mapping Network :  임의의 가우스 노이즈를 스타일 코드로 변환하는 것을 학습
- Style Encoder : 주어진 소스 이미지에서 스타일 코드를 추출하는 것을 학습

## 3. FrameWork
![image](https://user-images.githubusercontent.com/70633080/109932836-0c0b1880-7d0e-11eb-9916-db3a25623aa6.png)
- X를 이미지의 집합 그리고 Y를 가능한 도메인의 집합이라고 가정
- X에 속하는 이미지 x와 Y에 속하는 도메인 y가 주어졌을때, StarGAN v2의 목표는 하나의 generator만으로 이미지 x를 도메인 y의 이미지로 변형하되, 다양한 스타일로 변형할 수 있도록 학습하는 것이다. 

- (a) Generator : G의 역할은 input image가 들어오면 output으로 G(x,s)가 나온다.
  - s는 style vector로 AdalN(Adaptive instance normalization)을 통해 주입된다.
  - s는 도메인 y의 style을 대표하도록 mapping network F나 style encoder E에 의해 생성된다.
  
- (b) Mapping network : random latent vector z와 domain y가 주어졌을때 Mapping network인 F는 style vector s=Fy(z)를 만든다.
  - 즉, domain y를 대표하는 latent vector z를 style vector s로 mapping해준다. 
  - F는 다중출력 MLP로 구성된다.

- (c) Style Encoder : image x와 domain y가 주어지면 E는 image x에서 style information을 추출하는 역할을 한다. s=Ey(x)

- (d) Discriminator : D는 다중출력 Discriminator이다. D의 각 branch는 이미지 x가 real인지 fake인지 이진분류할 수 있도록 학습한다.

## 4. Training objectives
![image](https://user-images.githubusercontent.com/70633080/109937884-bb49ee80-7d12-11eb-877d-9e205221c45a.png)
1. Adversarial objective
    - StarGAN에서 봤던것과 동일. 
    - 특징은 latent vector z와 타깃도메인 ~y를 랜덤하게 샘플링해, target style vector ~s를 input으로 넣었다는 것이다.

2. Style reconstruction
    - style에 맞게 잘 변화시키기 위한 것이다.
    - ~ s = F ~ y(z), fake image를 만드는데 사용한 style code와 만들어진 fake image를 단일인코더 E에 넣어 얻은 style code를 비교하는 것이다.
    - fake image에 우리가 원하는 스타일코드 ~ s가 많이 적용되었을 수록 인코더를 통과한 fake image ~ s랑 비슷해질 것이다.
    - 


# 참고문헌
- <https://velog.io/@tobigs-gm1/Multidomain-ImageTranslation>

