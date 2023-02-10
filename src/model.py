import torch
import torch.nn as nn
from torch.nn import functional as F

# 以下を埋めてみよう
# 今回の研修では
# モデルとして入力から出力チャネル数6, kernel_size5の畳み込み層→Maxpooling(2×2)→出力チャネル数12, kernel_size3の畳み込み層
# → MaxPooling(2×2)→1次元にする→Linearで10次元出力
# というモデルを作成してください(strideなどは考えないでください)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 出力チャンネル数6, kernel size 5のCNNを定義する
        # 畳み込みの定義はPytorchの場合torch.nn.Conv2dで行います。ヒント:白黒画像とはチャネル数いくつかは自分で考えよう
        # 公式documentで使い方を確認する力をつけてほしいので、自分でconv2dなどの使い方は調べよう
        #一層目の畳み込み層
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 出力チャネル数12, kernel_size 3のCNNを定義する 上記と同様に今度は自分で書いてみよう
        self.conv2 = nn.Conv2d(6, 12, 3)
        
        # Maxpoolingの定義(fowardでするのでもどっちでも)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Linearの定義
        # 線形変換を行う層を定義してあげます: y = Wx + b
        # self.conv1, conv2のあと，maxpoolingを通すことで，
        # self.fc1に入力されるTensorの次元は何になっているか計算してみよう！
        # これを10クラス分類なので，10次元に変換するようなLinear層を定義します
        
        self.fc1 = nn.Linear(12*5*5, 10)

    
    def forward(self, x):
        batch_size = x.shape[0]
        # forward関数の中では，，入力 x を順番にレイヤーに通していきます．みていきましょう．    
        # まずは，画像をCNNに通します
        x = self.conv1(x)

        # 活性化関数としてreluを使います
        x = F.relu(x)
        
        # 次に，MaxPoolingをかけます．
        x = self.maxpool(x)
        
        # 2つ目のConv層に通します
        x = self.conv2(x)
        
        # MaxPoolingをかけます
        x = self.maxpool(x)
        
         # 少しトリッキーなことが起きます．
        # CNNの出力結果を fully-connected layer に入力するために
        # 1次元のベクトルにしてやる必要があります
        # 正確には，　(batch_size, channel, height, width) --> (batch_size, channel * height * width)
        x = x.view(batch_size, -1)
        
        # linearと活性化関数に通します
        x = self.fc1(x)
        # x = F.relu(x)
        return x