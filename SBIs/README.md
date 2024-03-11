# Self-Blended Images(SBIs)によるCNNベースの手法の学習と評価

## 動作環境＆学習済みモデル，結果
<details>
<summary>ライブラリのバージョン</summary>

* Python  3.6.9
* CUDA  9.1
* efficientnet-pytorch  0.6.3
* torch 1.8.1+cu111
* torchaudio  0.8.1
* torchinfo 1.5.4
* torchmetrics  0.8.2
* torchsummary  1.5.1
* torchvision 0.9.1+cu111
* scikit-image  0.17.2
* scikit-learn  0.24.2
* retinaface  1.1.0
* retinaface-pytorch  0.0.8
* timm  0.5.4
* tlt  0.1.0
* numpy  1.19.5
* Pillow  8.4.0
* tqdm  4.64.0
* opencv-python  4.5.1.48
* opencv-python-headless  4.6.0.66
* scipy  1.5.4
* matplotlib  3.3.4
* mmcv  1.7.1  
</details>


## ファイル一覧
＊がついているコードは基本的に触らないでいいはずです．

<details>
<summary>学習用コード等</summary>
 
|ファイル名|説明|備考|
|----|----|----|
|make_graph.py|学習曲線を可視化するコード．|*|
|model.py|EfficientNet-B4のモデル構造を定義したコード．|*|
|model2.py|ResNet-50のモデル構造を定義したコード．|*|
|model3.py|Xceptionのモデル構造を定義したコード．|*|
|train_sbi.py|モデルをSBIsで学習するコード．|学習したいモデルによってimportするモデルを変更．(EfficientNet-B4はmodel.py，ResNet-50はmodel2.py，Xceptionはmodel3.py)|
</details>

<details>
<summary>configs</summary>

 <details>
 <summary>sbi</summary>
  
 |ファイル名|説明|備考|
 |----|----|----|
 |base.json|EfficientNet-B4を学習する際のパラメータ．|パラメータの変更|
 |base2.json|ResNet-50を学習する際のパラメータ．|パラメータの変更|
 |base3.json|Xceptionを学習する際のパラメータ．|パラメータの変更|
 </details>
 </details>

<details>
<summary>inference</summary>
 
|ファイル名|説明|備考|
|----|----|----|
|datasets.py|評価時のデータセットのパスを設定するコード．|FF++データセット動画ファイルとラベルファイルのパスであるinit_ffの「original_path」，「fake_path」，「list_dict」，DFDCPデータセット動画ファイルとラベルファイルのパスであるinit_dfdcpの「fol_lab_list_all」，「folder_list」，「with open」，FFIWデータセット動画ファイルとラベルファイルのパスであるinit_ffiwの「path」，「folder_list」，CDFデータセット動画ファイルとラベルファイルのパスであるinit_cdfの「video_list_txt」，「folder_list」の変更．|
|inference_dataset.py|評価を行うコード．|AUCのグラフを保存するファルダのパス「save_path」の変更と評価したいモデルによってimportするモデルを変更．(EfficientNet-B4はmodel.py，ResNet-50はmodel2.py，Xceptionはmodel3.py)|
|inference_dataset_filter.py|評価を行うコード(フィルタあり)．|AUCのグラフを保存するファルダのパス「save_path」と評価したいモデルによってimportするモデルを変更．(EfficientNet-B4はmodel.py，ResNet-50はmodel2.py，Xceptionはmodel3.py)|
|model.py|EfficientNet-B4のモデル構造を定義したコード．|*|
|model2.py|ResNet-50のモデル構造を定義したコード．|*|
|model3.py|Xceptionのモデル構造を定義したコード．|*|
|preprocess.py|評価時にフレームを切り出すコード．|モデルの入力画像サイズに合わせて「image_size」を変更．(EfficientNet-B4は380，ResNet-50は224，Xceptionは299)|
</details>

<details>
<summary>preprocess</summary>
 
|ファイル名|説明|備考|
|----|----|----|
|crop_dlib_ff.py|フレーム，ランドマーク，Bboxを抽出するコード．|各データセットのパス「dataset_path」を変更．|
|crop_retina_ff.py|フレーム，ランドマーク，Bboxを抽出するコード．|各データセットのパス「dataset_path」と保存先のパス「save_path」を変更．|
</details>

<details>
<summary>utils</summary>
 
|ファイル名|説明|備考|
|----|----|----|
|blend.py|学習時に顔を合成するコード．|*|
|funcs.py|学習時にBboxに基づいて顔を切り取る設定しコード．|*|
|initialize.py|学習時のデータセットのパスを設定するコード．|FF++データセット動画ファイルとラベルファイルのパスであるinit_ffの「dataset_pat」，「list_dict」の変更|
|logs.py|学習時のログを録るコード．|*|
|sam.py|OptimizerであるSAMを定義したコード(従来手法)．|*|
|sbi.py|SBIsで擬似的なDeepfake画像を生成するコード．|*|
|scheduler.py|学習時のスケジューラーを設定しているコード(従来手法)．|*|
</details>

## 実行手順

### 0，環境設定

<details>
<summary>環境設定</summary>

[先述の環境](https://github.com/cu-milab/pro-CNN-base-Deepfake-detection/tree/main/SBIs#%E5%8B%95%E4%BD%9C%E7%92%B0%E5%A2%83%E5%AD%A6%E7%BF%92%E6%B8%88%E3%81%BF%E3%83%A2%E3%83%87%E3%83%AB%E7%B5%90%E6%9E%9C)を整えてください．
</details>

### 1，データの用意

<details>
<summary>データのダウンロード</summary>

 Face Forensics＋＋データセットをダウンロードする．[FF++](https://github.com/ondyari/FaceForensics)
下記のようなフォルダ構成にするとスムーズに行える．

```
.
└── data
    └── FaceForensics++
        ├── original_sequences
        │   └── youtube
        │       └── raw
        │           └── videos
        │               └── *.mp4
        ├── train.json
        ├── val.json
        └── test.json
```
</details>

<details>
<summary>ランドマーク検出器のダウンロード</summary>

 ランドマーク検出器のファイル(shape_predictor_81_face_landmarks.dat) をダウンロードする． [LINK](https://github.com/codeniko/shape_predictor_81_face_landmarks)  
ダウンロードしたファイルを`./src/preprocess/`に配置する．   
</details>

<details>
<summary>顔の検出・クロップ</summary>
 
2つのコードを実行し，顔画像を抽出する．
```
python3 src/preprocess/crop_dlib_ff.py -d Original
CUDA_VISIBLE_DEVICES=* python3 src/preprocess/crop_retina_ff.py -d Original
```
</details>

<details>
<summary>拡張用ファイルのダウンロード</summary>

ランドマーク拡張用に[Face X-ray](https://github.com/AlgoHunt/Face-Xray) のコードを利用するため下記のコマンドでダウンロードする．
```
mkdir src/utils/library
git clone https://github.com/AlgoHunt/Face-Xray.git src/utils/library
```
</details>

### 2，学習

バッチサイズやエポック数は`base.json`を編集し行う．

<details>
<summary>EfficientNet-B4の場合</summary>
 
```
python3 src/train_sbi.py src/configs/sbi/base.json -n sbi
```
</details>

<details>
<summary>ResNet-50の場合</summary>
 
```
python3 src/train_sbi.py src/configs/sbi/base2.json -n sbi
```
</details>

<details>
<summary>Xceptionの場合</summary>
 
```
python3 src/train_sbi.py src/configs/sbi/base3.json -n sbi
```
</details>

### 3，評価

#### FF＋＋で評価を行う場合は下記になる．

Deepfake_typeは'all','Deepfakes','Face2Face','FaceSwap','NeuralTextures'から選択
```
python3 src/inference/inference_dataset.py -w WWW/XXX/YYY/ZZZ.tar -d DatasetNames -ft MethodNames
```

#### CDF，DFDCP，FFIWで評価を行う場合は下記になる．

DatasetNamesは'FF','CDF','DFDCP','FFIW'から選択
```
python3 src/inference/inference_dataset.py -w WWW/XXX/YYY/ZZZ.tar -d DatasetNames 
```

#### FF＋＋のフィルタありで評価を行う場合は下記になる．

Deepfake_typeは'all','Deepfakes','Face2Face','FaceSwap','NeuralTextures'から選択，FilterTypeは'high','low'から選択
```
python3 src/inference/inference_dataset_filter.py -w WWW/XXX/YYY/ZZZ.tar -d DatasetNames -ft MethodNames -type FilterType
```
