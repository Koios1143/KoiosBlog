---
date: 2024-05-18
category:
  - Note
tag:
  - Paper Read
  - Domain Adaptation
  - Computer Vision
  - ACM Multimedia
---

# PiPa: Pixel- and Patch-wise Self-supervised Learning for Domain Adaptative Semantic Segmentation

## Basic Information

- Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua
- 2022 ACM Multimedia

## 問題描述

這一篇與過去看過的 [DACS](https://koios1143.github.io/PaperBlog/posts/DACS.html), [ProDA](https://koios1143.github.io/PaperBlog/posts/ProDA.html), [DAFormer](https://koios1143.github.io/PaperBlog/posts/DAFormer.html), [HRDA](https://koios1143.github.io/PaperBlog/posts/HRDA.html) 同樣都是以 Unsupervised 的方式解決 Semantic Segmentationb 的 Domain Adaptation問題。

也就是說，我們會在一個 Source Domain 上具有標記過的資料，但是 Target Domain 上則缺乏標記。我們的目標是透過這些資料去學習，讓這個模型有辦法對 Target Domain 上的資料順利地給予正確的 Label。

過去的這些 Works 普遍關注於如何在不同 Domain 之間築起橋梁，讓他們的 Domain Gaps 減少。包含了在 pixel level, feature level 以及 prediction level 之間的 Domain Gaps。

然而，這卻忽略了在同一個 Domain 當中的特徵。

:::info
這就好比我們學會對應 Source Domain 當中的"汽車"如何對應到 Target Domain，卻不太了解同樣 Source Domain 當中同樣是"汽車"的部份有怎樣的關聯。
:::

作者提出 PiPa，一個 Pixel-wise 以及 Patch-wise 的 self-supervised 的架構，能夠應用在過去的各種 UDA for semantic segmentation 問題上，並超越過去的 SOTA。

## Related Works

- Unsupervised Domain Adaptation (UDA)
- Contrastive Learning

## Methodology

大致分成了三個主軸
1. 基本的 UDA Loss 設定
2. Pixel-wise Contrastive Learning
3. Patch-wise Contrastive Learning

最後再將這三個部份結合起來。

### 基本的 UDA Loss 設定

與過往我們讀過的幾篇論文一樣，我們會先設定好最基本的兩組 Loss。分別會希望我們的模型在 **Source Domain** 以及 **Target Domain** 的預測結果要與對應的 Label 相同。

對於 Target Domain，因為缺少了 Label 標記，因此會使用 Pseudo Label。

$$
\begin{align*}
\mathcal{L}_{ce}^{S} &= \mathbb{E}{\left[ -p_u^S \log{h_{cls}(g_{\theta}(x_u^S))} \right]} \\
\mathcal{L}_{ce}^{T} &= \mathbb{E}{\left[ -\bar{p_v}^T \log{h_{cls}(g_{\theta}(x_v^T))} \right]}
\end{align*}
$$

也與過去的做法相同，會加上知識蒸餾(Knowledge Distillation, KD)，因此會包含了 Student Network 以及 Teacher Network。

其中的一些 Notations：

- $S$ 表示 Source Domain，包含了 $U$ 個資料。
- $T$ 表示 Target Domain，包含了 $V$ 個資料。
- $x_u^S$ 表示在 Source Domain 的第 $u$ 個資料。而 $y_u^S$ 是對應的 Label。
- $x_v^T$ 表示在 Target Domain 的第 $v$ 個資料。
- $p_u^S$ 是 $y_u^S$ 轉換成 one-hot 的形式。
- $\bar{p_v}^T$ 是預測的 pseudo label $\bar{y_v}^T$ 轉換成 one-hot 的形式。
- $g_{\theta}$ 是我們的模型 backbone。對應的 teacher network 為 $g_{\bar{\theta}}$。
- $h_{cls}$ 是最終給出 label 的 network。

針對 Target Domain 的 Loss，作者提及因為 Domain Gap 的存在，導致 pseudo label 必然會存在 noise。也就意味著並不是所有的 pseudo label 都值得信任。

因此 $\mathcal{L}_{ce}^T$ 計算上只會考慮 $\bar{y_v}^T$ 大於某個 threshold 的部份。意即只有那些足以信任的預測結果才會被考慮進去。

此外，也如同 DAFormer 與 DACS，他們會對 Source Domain 與 Target Domain 的圖片去做混合，得到相對應的圖片 $x_v^{Mix}$ 與 Pseudo Label $\bar{y_v}^{Mix}$，當然也有對應的 one-hot vector $\bar{p_v}^{Mix}$。因此 $\mathcal{L}_{ce}^T$ 被改寫如下。

$$
\mathcal{L}_{ce}^{T} = \mathbb{E}{\left[ -\bar{p_v}^{Mix} \log{h_{cls}(g_{\theta}(x_v^{Mix}))} \right]}
$$

### Pixel-wise Contrastive Learning

<center>
<img src="/PiPa/r1VL40SXR.png" width=400>
</center>
</br>

> Image from [Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua (2022)](https://arxiv.org/pdf/2211.07609)

Pixel-wise 的想法就是希望具有相同 Class 的 Pixel 要有類似的 Feature，反之則要有較為不同的 Feature。如上圖，同樣是圓圈的部份會被拉進，不同的則被推遠。

實際上的做法是把 $g_{\theta}$ 產出的 feature 經過 Projection Head $h_{pixel}$ 得到對應的 Feature Map $e$。接著透過 Contrastive Learning 去把應該要相近的 feature 拉近，反之推遠。

定義 Loss $\mathcal{L}_{pixel}$ 如下。

$$
\mathcal{L}_{pixel} = - \sum_{C(i) = C(j)}{ \log{\frac{r(e_i, e_j)}{\sum_{k=1}^{N_{pixel}}{(r(e_i, e_j))}}} }
$$

其中

- $e_i$ 表示 Feature Map $e$ 的第 $i$ 個特徵。也就是 pixel $i$ 對應的特徵。
- $C(i)$ 表示 pixel $i$ 對應的 class。
- $r(e_i, e_j)$ 用來描述 $e_i$ 與 $e_j$ 的相似性。採用的是 Exponential Cosine Similarity。
    $$
    r(e_i, e_j) = \exp(s(e_i, e_j) / \tau)
    $$
    其中 $s$ 表示 cosine similarity。

由於這裡需要 class 的資訊，因此只會使用到 Source Domain 的資料。

### Patch-wise Contrastive Learning

<center>
<img src="/PiPa/BkdzpCSmC.png" width=400>
</center>
</br>

> Image from [Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua (2022)](https://arxiv.org/pdf/2211.07609)

Patch-wise 的想法是今天同一個 crop 無論是出現在 patch $M_1$ 的右上角還是 patch $M_2$ 的左下角，因為同樣都是對應到相同的 crop，所以特徵也必須要相同。

作法上會把 Mixed Image 經過 Projection Head $h_{patch}$ 後，從當中切出兩個大小相同的 Patch $M_1, M_2$，並且他們之間有一個重疊的區塊 $O_1, O_2$。目標是要讓同樣都在 $O_1, O_2$ 的特徵拉近，否則推遠。

定義 Loss $\mathcal{L}_{patch}$ 如下。

$$
\mathcal{L}_{patch} = - \sum_{O_1(i) = O_2(j)}{ \log{\frac{r(f_i, f_j)}{\sum_{k=1}^{N_{patch}}{r(f_i, f_k)}}} }
$$

其中

- $f_i$ 表示 pixel $i$ 經過 $h_{patch}$ 得到對應的特徵。
- $r$ 與 Pixel-wise 一樣表示 Exponential Cosine Similarity。

### 結合

最後結合上面三個部分，可以得到最終的 Loss 如下。

$$
\mathcal{L}_{total} = \mathcal{L}_{ce}^S + \mathcal{L}_{ce}^T + \alpha \mathcal{L}_{pixel} + \beta \mathcal{L}_{patch}
$$

整體的架構可以用底下這張圖來簡單了解。

![image](/PiPa/SyNS-JUQC.png)

> Image from [Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua (2022)](https://arxiv.org/pdf/2211.07609)

由於 Pixel-wise Consistency 與 Patch-wise Consistency 都是在幫助訓練，因此在測試階段的時候這兩個部分是不會參與的。上圖當中的藍色區塊都是只在訓練階段包含的架構。


## Results

### 實驗設定

與過去相同，我們一樣會有 GTA5, Cityscapes, SYNTHIA 這三個 datasets，測試 (1) GTA5 $\rightarrow$ Cityscapes 以及 (2) SYNTHIA $\rightarrow$ Cityscapes 的結果。

實作上採用了常見的 [mmsegmentation framework](https://github.com/open-mmlab/mmsegmentation)，Network 的架構由於 PiPa 的通用性，作者有在 DAFormer 以及 HRDA 分別搭配 PiPa 去做實驗。

### Quantitative Comparison

首先看到 GTA5 $\rightarrow$ Cityscapes 的結果。

![image](/PiPa/BktIGJImR.png)

> Image from [Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua (2022)](https://arxiv.org/pdf/2211.07609)

可以看到無論是把 PiPa 搭配 DAFormer 或是 HRDA 都可以進一步得到更好的最終結果。而對於細部每一個預測的 Class 則可以看到在絕大多數的類別都有了提升。

接下來看到 SYNTHIA $\rightarrow$ Cityscapes 的結果。

![image](/PiPa/rk4SQJIXC.png)

> Image from [Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua (2022)](https://arxiv.org/pdf/2211.07609)

同樣也可以看到與 GTA5 $\rightarrow$ Cityscapes 相同的結果。

:::tip
與 HRDA + MIC 相較之下，HRDA + PiPa 在 GTA5 $\rightarrow$ Cityscapes 小輸 0.3 mIoU，而 SYNTHIA $\rightarrow$ Cityscapes 則大贏 7.5 mIoU。
:::

:::info
SYNTHIA 這個 Dataset 因為部分 paper 採用 16 個 classes，部分則是 13 個 classes 的資料去訓練，因此在數據上 mIoU 有兩列分別表示 16 個平均跟 13 個的平均。
:::

### Qualitative Results

![image](/PiPa/rJ9SrkIXC.png)

> Image from [Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua (2022)](https://arxiv.org/pdf/2211.07609)

從圖片上的結果可以看到結果有了一些提升，主要是在細節的呈現上更精準了。

### Ablation Studies

<center>
<img src="/PiPa/Hkc2HyIQR.png" width=400>
</center>
</br>

> Image from [Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua (2022)](https://arxiv.org/pdf/2211.07609)

針對 GTA5 $\rightarrow$ Cisyscapes 的部份作者嘗試了解 Patch Contrast 與 Pixel Contrast 分別帶來的效益。可以看到兩者分別使最後結果提升了 1.4 mIoU 與 2.3 mIoU，並且兩者結合後可以再帶來更高的 3.3 mIoU 的提升。

<center>
<img src="/PiPa/rk3nL1LQR.png" width=400>
</center>
</br>

> Image from [Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua (2022)](https://arxiv.org/pdf/2211.07609)

針對 $\alpha$ 與 $\beta$ 的選擇上，作者認為 PiPa 比較不會對 hyperparameter 敏感。在實驗的幾組 $\alpha, \beta$ 都會帶來相近的結果，並且都比 baseline DAFormer 的 68.4 來得高。

<center>
<img src="/PiPa/SkU4PyUXC.png" width=200>
</center>
</br>

> Image from [Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua (2022)](https://arxiv.org/pdf/2211.07609)

最後，針對 Patch-wise 得 Crop Size，作者認為普遍來說 Crop Size 是越高越好，不過實驗中 720x720 的大小會是最恰當的。當 Crop Size 過大時，會時常導致 Overlapped 的區域過大，因此也不太適合將 Crop Size 社得太大。

## Contribution

- 針對 intra-domain knowledge 去改善 UDA 的成果
- 設計出一個通用的架構
- 在 GTA5 $\rightarrow$ Cityscapes 與 SYNTHIA $\rightarrow$ Cityscapes 與 HRDA 結合後分別得到 75.6 與 68.2 mIoU
