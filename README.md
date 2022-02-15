# RL_segmentation

這是“應用深度強化學習方法於醫學影像分割”的程式

整個模型架構由兩個神經網路組成。第一個網路是FirstP-Net，其目標是找到目標物件中的第一個邊緣點並生成可能屬於邊緣點位置的機率圖。整個模型架構由兩個神經網路組成。第一個網路是FirstP-Net，其目標是找到目標物件中的第一個邊緣點並生成可能屬於邊緣點位置的機率圖。第二個網路是NextP-Net，它負責根據之前的邊緣點和圖像資訊來定位下一個點。這支程式的強化學習模型透過逐步尋找邊緣點對圖像進行分割，最終得到一個封閉且準確的分割結果。



1. 系統的總體流程：FirstP-Net負責先找到第一個邊緣點並生成邊緣點位置的機率圖。NextP-Net根據前一個邊緣點和圖像資訊定位下一個點。



2. 預設情況下Ground Truth(GT)邊界以藍色繪製，洋紅色點是NextP-Net找到的點。紅色五角星代表FirstP-Net找到的第一個邊緣點。


## 軟體/套件需求
* Python2.7
* torch 0.4.0
* torchvision 0.2.1
* matplotlib 2.2.3
* numpy 1.16.4
* opencv-python 4.1.0.25
* scikit-image 0.14.3
* scikit-learn 0.20.4
* shapely 1.6.4.post2
* cffi
* scipy


## 程式安裝
1. Clone this repository.

        git clone https://github.com/Shengyw1711/RL_segmentation.git

2. 由於系統使用類似 Fast R-CNN (https://github.com/longcw/RoIAlign.pytorch) 中的裁剪和調整大小功能來修復狀態的大小，因此需要在訓練之前使用正確的 -arch 選項來讓 Cuda 套件能夠支援。 (https://github.com/multimodallearning/pytorch-mask-rcnn)

    | GPU | arch |
    | --- | --- |
    | TitanX | sm_52 |
    | GTX 960M | sm_50 |
    | GTX 1070 | sm_61 |
    | GTX 1080 (Ti) | sm_61 |

        cd nms/src/cuda/
        nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
        cd ../../
        python build.py
        cd ../

        cd roialign/roi_align/src/cuda/
        nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
        cd ../../
        python build.py
        cd ../../
        
3. 執行train.py為經過前處理的影像數據集訓練分割用的深度Q網路(DQN)agent，執行val.py測試訓練好的模型。

### 消融研究(可自行測試的控制變量分析)
* State: 

實驗 0: grayscale layer, Sobel layer, cropped probability map, global probability map and past points map.

實驗 1: grayscale layer, Sobel layer and past points map layer. 

實驗 2: grayscale layer, Sobel layer, cropped probability map, global probability map.

* 獎勵(Reward):

實驗 3: 使用不同的IoU獎勵作為最終的即時獎勵。

