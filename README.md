# Resnet 을 사용한 이미지 분류 학습

CNN은 이미지나 영상처리 분야에서 우수한 성능을 보여주는 모델입니다. CNN은 모델의 layer층을 쌓아 깊은 네트워크를 구현하여 성능을 향상 시킬 수 있지만, 실제로 모델의 layer가 너무 깊어지면 gradient vanishing/exploding 문제로 인해 오히려 성능이 떨어지는 현상이 생긴다는 것을 발견했습니다. gradient vanishing 이란 layer가 깊어질수록 전달되는 오차가 크게 줄어들어 training data에도 원활한 학습이 되지 않는 현상을 의미합니다. 이러한 현상을 degradation problem 이라 부르며 이 문제를 극복하기 위해 ResNet이 고안되었습니다.
ResNet은 2015년 ILSVRC 대회에서 우승한 마이크로소프트에서 개발한 알고리즘으로 VGG-19의 구조를 뼈대로 152개의 층이 존재하며 Block단위로 Parameter을 전달하기 전 이전의 값을 더해, 즉 입력값이 출력으로 그대로 더해지는 shortcut 구조를 취하게 되면서 입력값이 출력값에 들어감에 따라 위에서 언급되었던 degradation problem가 해결된 향상된 성능을 갖는 모델입니다.

이러한 ResNet 구조를 사용하여 전이학습으로 2022년 대선 후보들의 얼굴을 인식/분류하는 학습을 테스트 해보겠습니다.

# 대선후보 얼굴 사진 크롤링
본격적인 학습을 하기 전, 우선 대선후보들의 사진을 학습데이터 자료로 가지고 있어야 합니다. 사진을 모으기 위해 웹 크롤러는 깃허브의 오픈소스 프로젝트인 Auto Crawler를 사용했습니다. (https://github.com/YoongiKim/AutoCrawler)

크롤링 방법
  1. 연결된 링크의 파이썬 소스를 다운로드 https://github.com/YoongiKim/AutoCrawler
  2. keywords.txt 파일에 크롤링하고 싶은 검색어를 한 줄씩 수정 후 저장
  3. 터미널 창에 python main.py 실행
  4. 자동으로 크롬이 실행되며, selenium으로 이미지를 폴더별로 다운로드
  5. 프로그램이 종료 될 때까지 기다리기

학습의 정확도를 위해 여러 명이 나온 사진들과 같은 부적합한 이미지는 제거해줍니다.

# Resnet 을 이용한 이미지 분류 전이학습 및 테스트
학습데이터 자료를 크롤링 해왔으니, pyTorch에서 Resnet을 사용하여 전이학습을 해보겠습니다. 아래 코드는 다음 사이트를 참고 하여 작성했습니다. (https://www.kaggle.com/pmigdal/transfer-learning-with-resnet-50-in-pytorch)

- pyTorch 딥러닝 학습에 필요한 라이브러리를 import합니다.  
- 특히 Resnet50 모델을 사용하기 위해 torchvision의 models을 import 해야합니다.

```python
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import troch
from torchvision import datasets, models, transforms
import torch.nn as nn
form torch.nn import functional as F
import torch.optim as optim
```
- GPU가 있다면 GPU로 학습하도록 설정합니다.
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- 모델을 가져오기 전, 크롤링한 이미지의 전처리 작업부터 진행하겠습니다.  
- 크롤링한 대선 후보의 사진들을 Resnet 입력에 적합하도록 변형하는 함수를 만듭니다. Resnet의 이미지 size 는 224 * 224 입니다.
   * transforms.Normalize(mean, std, inplace=False) -> 이미지 정규화
   * transforms.RandomAffine(degrees) -> 랜덤으로 affine 변형
   * transforms.RandomHorizontalFlip() -> 이미지를 랜덤하게 수평으로 뒤집기
   * transforms.ToTensor() -> 이미지 데이터를 tensor로 변환

```python
normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                std=[0.229,0.224,0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
}
```
- 이미지가 저장되어 있는 폴더의 경로 불러오기
```python
data_path = 'C:/kotorch/AutoCrawler-master/download/'
```
- DataLoader를 사용하여 저장한 이미지들을 읽어온 뒤 train 데이터와 validation 데이터로 분류합니다.
```python
image_datasets = {
    'train':
    datasets.ImageFolder(data_path + 'train', data_transforms['train']),
    'validation':
    datasets.ImageFolder(data_path + 'validation', data_transforms['validation'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                               batch_size = 128,
                               shuffle=True,
                               num_workers=0),
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                               batch_size = 128,
                               shuffle=False,
                               num_workers=0)
}
```
- 학습데이터가 준비되었으니 Resnet50 모델을 가져옵니다. pretrained는 ImageNet으로 사전 학습된 모델을 가져올 지를 결정하는 parameter로, True로 설정합니다.
- 또한 사전에 학습된 모델을 finetuning 하는 것이므로 requires_grad = False로 설정해주어야 학습이 되지 않도록 고정시킬 수 있습니다. 불러온 모델의 마지막 fully connected layer를 수정하여 fc layer를 원하는 layer로 변경합니다. 이 코드는 출력이 5명으로 분류되는 모델을 만들 것이므로 nn.Linear(128, 5)를 사용합니다.

```python
model = models.resnet50(pretrained=True).to(device)
    
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 5)).to(device)
```
-손실 함수는 CrossEntropyLoss, 옵티마이저는 Adam 을 사용하도록 설정합니다.
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())
```
- 학습할 이미지와 모델이 준비되었으니 이미지를 학습하는 함수를 작성합니다. 일반적인 pyTorch 학습 코드와 동일합니다.
```python
def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
    return model
```
- 모델을 학습시킵니다.
```python
model_trained = train_model(model, criterion, optimizer, num_epochs=3)
```
- 학습이 완료된 모델을 저장합니다.
```python
PATH = 'C:/kotorch/AutoCrawler-master/weights.h5'
torch.save(model_trained.state_dict(), PATH)
```
- 매번 학습을 반복할 수 없기 때문에 모델을 다시 만듭니다.
- 이번에는 학습을 하지 않고 위에서 저장한 모델의 weight 만을 load 할 것이기 떄문에 이번엔 pretrained를 False로 설정합니다. 
```python
model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 5)).to(device)
model.load_state_dict(torch.load(PATH))
```
- 테스트할 이미지를 준비합니다.
```python
validation_img_paths = ["validation/심상정/google_0048.jpg",
                        "validation/추미애/google_0013.jpg",
                        "validation/이낙연/naver_0362.jpg",
                       "validation/이낙연/naver_0399.jpg",
                       "validation/홍준표/naver_0222.jpg"]
img_list = [Image.open(data_path + img_path) for img_path in validation_img_paths]
```
- 불러온 이미지를 Resnet50 에 적합한 입력으로 만들기 위해 transform 합니다.
```python
validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                for img in img_list])
```
- 학습된 모델로 테스트 이미지를 예측합니다. 예측된 결과는 softmax를 사용하여 확률을 보여줍니다.
```python
pred_logits_tensor = model(validation_batch)
pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
```
- 확률 값으로 출력된 결과를 이름으로 변환하기 위해 이미지 크롤링을 진행할 때 저장했던 keywords파일을 읽어와 labels값을 붙여줍니다.
```python
with open ('C:/kotorch/AutoCrawler-master/keywords.txt', 'rt', encoding = 'UTF8' ) as f:
    labels = [line.strip() for line in f.readlines()]
```
- matplotlib 에서 한글이 깨지는 것을 방지하기 위한 처리를 합니다.
```python
from matplotlib import font_manager, rc

font_path = 'C:/kotorch/font/NanumBarunGothic.ttf'
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)
```
- 테스트 이미지에 해당하는 인물의 사진을 화면에 출력하며, argmax 함수를 통해 가장 확률이 높게 나온 사람의 labels를 이미지와 함께 출력합니다.
```python
fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.set_title((labels[np.argmax(pred_probs[i])]))
    ax.imshow(img)
```
- 테스트 이미지를 합습된 모델로 분류한 결과입니다. 약간의 오차가 있긴 하지만 잘 작동합니다.
![ex_capture](./캡처.PNG)