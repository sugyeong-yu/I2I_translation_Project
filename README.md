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
- 

