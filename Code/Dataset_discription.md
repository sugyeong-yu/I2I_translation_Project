# celebA DB
- 대표적인 얼굴 데이터 셋 중 하나.
- 약 20000개 정도의 얼굴 이미지로 구성되어 있음.
- 약 10000명 가량의 사람이 포함되어 있음 (약 한명당 20장)
- 이미지는 178 * 218 
- 각 얼굴에 대해 40개의 이진레이블이 있다. (0 또는 1) 
- 얼굴마다 5개의 랜드마크 위치를 가지고 있음. 
- young, male, bald(대머리) 등...
- Kaggle 사이트에서 다운로드 가능 <www.kaggle.com/jessicali9530/celeba-dataset>
```
FeaturesDict({
    'attributes': FeaturesDict({
        '5_o_Clock_Shadow': tf.bool,
        'Arched_Eyebrows': tf.bool,
        'Attractive': tf.bool,
        'Bags_Under_Eyes': tf.bool,
        'Bald': tf.bool,
        'Bangs': tf.bool,
        'Big_Lips': tf.bool,
        'Big_Nose': tf.bool,
        'Black_Hair': tf.bool,
        'Blond_Hair': tf.bool,
        'Blurry': tf.bool,
        'Brown_Hair': tf.bool,
        'Bushy_Eyebrows': tf.bool,
        'Chubby': tf.bool,
        'Double_Chin': tf.bool,
        'Eyeglasses': tf.bool,
        'Goatee': tf.bool,
        'Gray_Hair': tf.bool,
        'Heavy_Makeup': tf.bool,
        'High_Cheekbones': tf.bool,
        'Male': tf.bool,
        'Mouth_Slightly_Open': tf.bool,
        'Mustache': tf.bool,
        'Narrow_Eyes': tf.bool,
        'No_Beard': tf.bool,
        'Oval_Face': tf.bool,
        'Pale_Skin': tf.bool,
        'Pointy_Nose': tf.bool,
        'Receding_Hairline': tf.bool,
        'Rosy_Cheeks': tf.bool,
        'Sideburns': tf.bool,
        'Smiling': tf.bool,
        'Straight_Hair': tf.bool,
        'Wavy_Hair': tf.bool,
        'Wearing_Earrings': tf.bool,
        'Wearing_Hat': tf.bool,
        'Wearing_Lipstick': tf.bool,
        'Wearing_Necklace': tf.bool,
        'Wearing_Necktie': tf.bool,
        'Young': tf.bool,
    }),
    'image': Image(shape=(218, 178, 3), dtype=tf.uint8),
    'landmarks': FeaturesDict({
        'lefteye_x': tf.int64,
        'lefteye_y': tf.int64,
        'leftmouth_x': tf.int64,
        'leftmouth_y': tf.int64,
        'nose_x': tf.int64,
        'nose_y': tf.int64,
        'righteye_x': tf.int64,
        'righteye_y': tf.int64,
        'rightmouth_x': tf.int64,
        'rightmouth_y': tf.int64,
    }),
})
```
