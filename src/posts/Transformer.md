---
date: 2024-05-22
category:
  - Note
tag:
  - Paper Read
  - NLP
  - Computer Vision
  - NeurIPS
---

# Attention is all you need

## Basic Information
- NIPS 2017 (former NeuralPS)
- Ashish Vaswani, Noam Shazeer, Niki Parmar et al. from Google Brain and Google Research

## 問題描述

### RNN

近年來自然語言處理(Natural Language Processing, NLP)與機器翻譯等任務上時常使用 Recurrent Neural Network(RNN), Long Short-Term Memory(LSTM), Gated Recurrent Neural Network 等模型架構，我們也看到使用 Recurrent 模型以及 Encoder-Decoder 架構蔚為流行。

Recurrent Model 雖然強大，卻有兩個很大的缺點。

1. 由於每個 state $h_t$ 依賴於上一個 state $h_{t-1}$，使 RNN 平行度極差
2. 前面序列的資訊隨著長度越長會逐漸被稀釋

雖然後續有一些研究試圖在 RNN 的基礎上去改善上述兩個缺點，**但是這些問題仍然存在**。

### CNN

也有一些相異於 RNN 的做法，使基於 Convolution Neural Network(CNN) 處理，如 `Extended Neural GPU`, `ByteNet` 以及 `ConvS2S`。透過 CNN 這種單純的矩陣運算可以提高運算的平行度，解決 RNN 平行度差的問題，不過也會使得 input sequence 當中相距較遠的元素，需要花費更多次運算得到彼此的關係。

### Self-Attention

Self-Attention 這個技術能夠將一個 sequence 中每個位置的元素依據與其他元素之間的關係，得出一個對應的 representation。這樣的機制在許多的任務當中都看到了不錯的結果。

基於上述的問題以及過去的研究，作者提出了 Transformer 架構，單純依賴 Attention 機制，並且將 Encoder-Decoder 架構替換成 Multi-headed Self-Attention。除了能夠有高度的平行度，也能夠將過去的資訊好好地保留。在翻譯的任務上也打破 `WMT 2014 English-to-German` 與 `WMT 2014
English-to-French` 的紀錄成為新的 SOTA。

## Related Works

- Recurrent Neural Network(RNN)
- Seq2Seq
- Attention

### Recurrent Neural Network(RNN)

<center>
<img src="/Transformer/ByfeC6_mR.gif">
</center>
</br>

> Image from [LeeMeng - 進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E6%9C%89%E8%A8%98%E6%86%B6%E7%9A%84%E5%BE%AA%E7%92%B0%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF_1)

像是文意理解或是語言翻譯這種任務，每個詞語的意義都會與前面的內容相關。RNN 就像是模擬了人在閱讀文章的狀態，由左至右地逐漸理解文意。

RNN 會將每個序列中的每個元素 $x_t$ 依序放入一個網路中，除了得到一個對應的 Representation $h_t$ 以外，也會將這個 $h_t$ 會與 $x_{t+1}$ 作為下一個網路的輸入。如此一來就可以將前面的資訊傳遞下去。

<center>
<img src="/Transformer/SJk21Cu7A.png">
</center>
</br>

> Image from [LeeMeng - 進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E6%9C%89%E8%A8%98%E6%86%B6%E7%9A%84%E5%BE%AA%E7%92%B0%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF_1)

不過正因為這樣的架構設計，使得 RNN 的每個狀態 $h_t$ 都依賴於前一個狀態 $h_{t-1}$，難以透過平行化加速運算。同時，隨著 sequence 長度越長，前面的狀態在傳遞過程中也會不斷被稀釋。

### Seq2Seq

Seq2Seq 是 Sequence to Sequence 的意思，如同字面意義能夠將一個 sequence 轉換成另一個 sequence，常見在翻譯任務上。

<center>
<img src="/Transformer/B1EnQROX0.gif">
</center>
</br>

> Image from [Frederick Lee - Attention in Text：注意力機制](https://medium.com/@fredericklee_73485/attention-in-text-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%A9%9F%E5%88%B6-bc12e88f6c26)

Seq2Seq 將 RNN 分別製作成 Encoder 與 Decoder。Encoder 負責將原本的文字變成一個 context vector，交給 Decoder 產生出對應的 sequence。

前面所提及的 Encoder-Decoder 架構就是指這個部分。

### Attention

在 Seq2Seq 當中我們會透過 Encoder 將 input sequence 壓成單一的 context vector，不過 RNN 在傳遞過程當中可能導致訊息的流失，且單一 vector 的表達能力也有所侷限。Attention 機制是其中一個解方。

Attention 簡單來說，就是能夠告訴我們需要注意 sequence 當中的哪些元素。

:::tip
這裡十分推薦可以去看看 [3Blue1Brown 的解說影片](https://youtu.be/eMlx5fFNoYc?si=2TgN_dwFDMew_aJT)，相信可以給你更多的啟發，這裡我會簡單地說明 Attention 的機制。
:::

對於每個 sequence 中的元素，Attention 會先將他們各自透過一個矩陣得到對應的 **Query Vector** $\vec{Q_i}$ 以及 **Key Vector** $\vec{K_i}$。

> 值得一提的是，除了元素本身之外，元素所在的位置(position)也會被納入考量產生 $\vec{Q_i}, \vec{K_i}$。

當 Query Vector $\vec{Q_i}$ 與 Key Vector $\vec{K_j}$ 很像的時候，我們會說元素 $i$ 需要多注意元素 $j$。

<center>
<img src="/Transformer/B14em-cXR.png" height=400>
</center>
</br>

> Image from [3Blue1Brown - Attention in transformers, visually explained | Chapter 6, Deep Learning](https://www.youtube.com/watch?v=eMlx5fFNoYc)

對於每個 Query 與 Key 都去計算他們的相似程度，我們就可以得到一個 Attention Matrix，告訴我們每個元素需要關注其他元素多少程度。

<center>
<img src="/Transformer/r1cVLZq70.png" height=400>
</center>
</br>

> Image from [3Blue1Brown - Attention in transformers, visually explained | Chapter 6, Deep Learning](https://www.youtube.com/watch?v=eMlx5fFNoYc)

例如從上面可以知道 `creature` 需要多注意 `fluffy` 和 `blue`。

不過這樣的數值範圍可以是 $(-\infty, \infty)$，我們其實更想知道的只是數值之間的大小差異，讓我們知道要著重於哪些部分，因此會再進一步加上 SoftMax function，讓他看起來就像是機率一樣，數值範圍介於 $[0, 1]$。

<center>
<img src="/Transformer/H1GWvbq7C.png" height=400>
</center>
</br>

> Image from [3Blue1Brown - Attention in transformers, visually explained | Chapter 6, Deep Learning](https://www.youtube.com/watch?v=eMlx5fFNoYc)

:::info
計算相似度的方法有許多種，上面呈現的是使用 dot product 的版本。透過 dot product，你可以知道兩個向量在方向上的相似性。
:::

最後我們還有一個向量 $\vec{v_i}$，這個向量就像是在描述一個元素的特性。以下圖為例，`creature` 根據剛剛計算出來的結果，我們需要多注意 `fluffy` 以及 `blue`，也就是說，根據前後文，`creature` 實際上是指 `fluffy blue creature`。

我們期待代表 `creature` 的向量 $\vec{E_4}$ 加上了一些些的 $\vec{v_2}$ (`fluffy`) 與一些些的 $\vec{v_3}$ (`blue`) 後，可以變成我們期待的 `fluffy blue creature`。

<center>
<img src="/Transformer/rkhZYWcQC.png" height=400>
</center>
</br>

> Image from [3Blue1Brown - Attention in transformers, visually explained | Chapter 6, Deep Learning](https://www.youtube.com/watch?v=eMlx5fFNoYc)

綜合上面的幾個操作，將許多的向量以矩陣改寫後，我們得到簡單的 Attention 數學描述。

$$
\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V
$$

## Methodology

<center>
<img src="/Transformer/rJpNKN9QC.png" height=400>
</center>
</br>

> Image from [Ashish Vaswani, Noam Shazeer, Niki Parmar et al. (2017)](https://arxiv.org/pdf/1706.03762)

Transformer 的架構圖如上所示，其中左邊是 Encoder，右邊則是 Decoder。接下來我們就仔細看看 Transformer 每個部分的設計。

### Scaled Dot-Product Attention

原本的 Attention 如同上面的描述如下。

$$
\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V
$$

把 Dimention 也標記後如下圖所示。
- $d_k$: Sequence 大小
- $n$: Sequence 數量
- $m$: Key value 數量
- $d_v$: Output Dimension

<center>
<img src="/Transformer/SJ1BX45mA.png" width=500>
</center>
</br>

在語言模型當中 sequence 大小 $d_k$ 可能會很大。當兩個很大的向量座內積的時候可能會得到過大或是過小的數值。這會導致 Softmax 出來的結果可能極端地靠近 $1$ 或是 $0$。

當 Softmax 結果有些趨近於 $1$ 有些趨近於 $0$ 時，意味著我們認為我們已經成功地讓模型知道 Query 與 Key 之間的關聯，換言之，我們認為模型已經訓練得差不多了，也就使梯度收斂。

因此，前面提到 $d_k$ 過大就可能使我們誤判現在已經訓練差不多，梯度就收斂了。

為了避免這個情況，他們將原本的 Attention 除上 $\sqrt{d_k}$，得到了 Scaled 的版本。

$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

### Masking

在 Attention 當中，我們會把每個詞跟前後文之間的關係都直接計算出來。不過實際上我們在輸出的過程當中，是不會知道未來才會輸出的內容的。

具體來說，如果今天要將 `Hello World` 翻譯成 `你好，世界`，那麼在處理輸出 `好` 的當下，你是不會知道你現在要輸出 `好`，也不知道後面會輸出 `，世界` 的。

以數學來描述，也就是說在時間 $t$，你只能考慮時間 $[0, t-1]$ 之間的資料。

但 Attention 顯然會將未來的資料也考量進去。為了避免這個問題，我們會在 Attention 的計算中間加上 Masking，把時間 $[t, n]$ 之間的部分都乘上一個很大的負數，就能使 softmax 的計算結果趨近於 $0$，而達到忽略未來的效果。

<center>
<img src="/Transformer/r1VLmzqX0.png" height=300>
</center>
</br>

> Image from [Ashish Vaswani, Noam Shazeer, Niki Parmar et al. (2017)](https://arxiv.org/pdf/1706.03762)

### Multi-head Attention

到目前為止我們看到的 Attention 都可以被稱為 **Single-head Attention**。仔細看 Attention 當中的 learnable parameters 就只有投影到 $Q, K, V$ 的矩陣 $W^Q, W^K, W^V$。或許在表達能力上有所不足。

在 Attention 當中，一個單詞決定要去看哪些 sequence 中的元素，取決於對應到的投影矩陣 $W^Q, W^K, W^V$。

如果我們把 Single-head Attention 看作是一種解讀事物的視角，那麼有更多的的視角去理解一個事情，直覺上能帶來對事物更加深刻與全面的理解。

換言之，如果我們有多組的 $W^Q, W^K, W^V$，那麼不同 Head 能代表不同的表達，綜合起來可以達到更加全面的描述及理解。這就是 Multi-head Attention 想做到的事情。

<center>
<img src="/Transformer/rJgGZ4cQ0.png" height=300>
</center>
</br>

> Image from [Ashish Vaswani, Noam Shazeer, Niki Parmar et al. (2017)](https://arxiv.org/pdf/1706.03762)

我們設定有 $h$ 種不同的視角，將 $V, K, Q$ 分別經過 $h$ 個 Linear Layer，再接回前面的 Scaled Dot-Product Attention。最後再 Concat 再一起，經過一個 Linear Layer 作為最後的輸出，如上圖所示。

於是我們的 Multi-head Attention 可以寫成底下的數學表達。

$$
\begin{align*}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \\
\text{head}_i &= \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)
\end{align*}
$$

這裡的 $\text{Concat}$ 就是單純把結果串接在一起而已。其中各個矩陣的大小如下

- $W_i^Q \in \mathbb{R}^{d_{\text{m}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{m}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{m}} \times d_v}$
- $W_i^O \in \mathbb{R}^{hd_v \times d_{\text{m}}}$

也就是說，最後得到的結果會是一個大小為 $\mathbb{R}^{d_m \times hd_v} \times \mathbb{R}^{hd_v \times d_m} = \mathbb{R}^{d_m \times d_m}$ 的結果。

再 Transformer 當中設定的參數如下。
- $h = 8$
- $d_k = d_v = d_{m} / h = 64$
- $d_m = 512$

原本 Single-head 的大小是 $d_{m}$，現在在 Multi-head 我們希望產出相同大小的結果，因此會讓每個 head 的大小平分，變成了上面的 $d_{m} / h$。

如此一來，在 Multi-head 的設定上每個 head 的大小會比 Single-head 小，而總共需要的計算量與 Single-head 會相似。

### Encoder and Decoder Stacks

接下來可以詳細看 Transformer 的架構了！

***Encoder***

<center>
<img src="/Transformer/HkPcKVqmR.png" height=300>
</center>
</br>

> Image from [Ashish Vaswani, Noam Shazeer, Niki Parmar et al. (2017)](https://arxiv.org/pdf/1706.03762)

在 Encoder 的部份，首先將 Input Embedding 加上 Position 的資訊。

> 如同前面 Attention 的部份所提及。
> 除了元素本身之外，元素所在的位置(position)也會被納入考量產生 $\vec{Q_i}, \vec{K_i}$。
> 這就是在這邊加上的。

接下來可以看到連接了一個 **Multi-Head Attention**，而串接的 $V, K, Q$ 都是同樣的輸入，也就是 Self-Attention。

將輸入與 Attention 的結果相加，再經過 Layer Normalization，接著會經過 Feed Forward Layer，然後再次相加、經過 Layer Normalization。

這樣的 block 在 Transformer 的 Encoder 當中會重複 $N = 6$ 次。

:::tip
Multi-Head Attention 可以把 Query 跟 Key 去比較相似度。當 Query 與 Key 來源相同，也就意味著擷取**自己跟大家的相似度有多少**。

可以簡單理解成 Encoder 會將 Input Sequence 的特徵擷取起來，變成一個 Embedding。
:::

***Decoder***

<center>
<img src="/Transformer/Hkh4j4qmR.png" height=400>
</center>
</br>

> Image from [Ashish Vaswani, Noam Shazeer, Niki Parmar et al. (2017)](https://arxiv.org/pdf/1706.03762)

Decoder 的部分與 Encoder 設計部分類似，不過在 Output Embedding 會先經過 Masked Multi-Head Attention。原因與前述相同，是為了把未來的資訊 Mask 掉，避免去考慮到後面的內容。

接下來，Encoder 的輸出作為 Multi-Head Attention 的 Value 與 Key 輸入，而 Output 的部份作為 Query 輸入。後續部分與 Encoder 相同。

這樣的 block 在 Transformer 的設計中會重複 $N = 6$ 次。

:::tip
這意味著輸出的內容會根據 Input 擷取出來的 Embedding 來決定輸出特徵。
:::

最後經過 Linear Layer 與 SoftMax，就得到 Transformer 的最後輸出了！

### Position-wise Feed-Forward Networks

在 Encoder 與 Decoder 都有 FFN，會將 Attention 擷取特徵後的結果再做一些加工。實際上 FFN 的設計如下。

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

這裡的兩個矩陣大小分別是 $W_1 \in \mathbb{R}^{d_m \times f_{ff}}, W_2 \in \mathbb{R}^{d_{ff} \times d_m}$。其中 $d_m = 512, f_{ff} = 2048$。

也就是說，經過 FFN 之後他的大小仍然會跟原本是相同的。

### Embeddings and Softmax

無論是 Input 或是 Output 原本都是一個單純的詞語，會需要轉換成向量才能計算，這也就是所謂的 Embedding。

論文當中並沒有特別說明 Embedding 的設計方式，不過有特別提到每一個需要 Embedding 的地方都會是用相同的權重，並且這個權重會依照現在選用的 $d_m$ 去乘上 $\sqrt{d_m}$ 的大小。

可以理解成當 Embedding 的維度越大時，學出來的權重就可能越小，因此乘上一個 $\sqrt{d_m}$ 會讓他比較好學。

### Positional Encoding

如同前面我們提到，實際上每個元素都會再加上位置的資訊。這是因為原本的 Attention 無論現在的 sequence 順序為何，都不會影響到最後的結果。然而很直覺地，一個句子當中的內容如果調換，對於語句的理解也會有不同。

因此在這裡我們會再加上位置的資訊，就可以避免這個問題。

:::info
這也是 Attention 跟 RNN 不同的地方。RNN 會依序接收上一個時間點的訊息，因此本身就已經包含順序的特徵。但 Attention 需要我們自己告訴他。
:::

在 Transformer 當中使用的 Positional Encoding 如下。

$$
\begin{align*}
PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_m}) \\
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_m})
\end{align*}
$$

## Results

### Why Self-Attention?

為什麼我們要選擇 Transformer 這種 Self-Attention 的架構，而不要用 RNN 與 CNN 的做法呢？

在計算的複雜度上比較一下。

<center>
<img src="/Transformer/HyLzDdhXA.png" width=500>
</center>
</br>

> Image from [Ashish Vaswani, Noam Shazeer, Niki Parmar et al. (2017)](https://arxiv.org/pdf/1706.03762)

給定輸入序列數量 $n$，序列長度 $d$，比較 Self-Attention 與 RNN。

首先在 Complexity per Layer 的部份一個是 $O(n^2 \cdot d)$，一個是 $O(n \cdot d^2)$。在現在我們的序列數量基本上跟序列長度的數量級是差不多的狀況下，兩者沒有太大的差異。

> 當然，實際上你還是可以依照狀況決定要哪一種架構，但 Sequential Operations 所需的時間仍然是 Attention 較佳。

而 Sequential Operations 的部份由於 RNN 需要等待前面的輸入，因此會是 $O(n)$，而 Self-Attention 只需要 $O(1)$。

最後的 Maximum Path Length 可以看成是資訊流失的程度。在 Attention 當中我們可以直接獲得訊息，不需要等待傳遞，也不需要擔心中間的流失。而 RNN 就會需要擔心了。

:::info
CNN 的部份雖然我並沒有深入去理解，不過大致上他的做法是會經過幾個 kernel 運算，因此 Complexity 的部份會多上一個 $k$。

而訊息流失則是看 kernel 需要跑過幾次，因此會是 $O(\log_k(n))$。
:::

可以看到 Attention 確實能夠避免最初提及 RNN 並行度差的狀況，以及資訊流失的狀況。

### 實驗設定

***Datasets***

在實驗上使用的 Dataset 為 WMT 2014 English-German dataset。當中包含了 4.5 million sentence pairs。

這些 sentence 會透過 byte-pair encoding 被 encode 成一個 token。他們的 token 總共有 37000 種。

對於英文轉法文的部分使用的是 WMT 2014 English-French dataset。當中包含了 36 million sentence pairs。而 token 則有 32000 種。

***Hardware***

硬體上使用的是 8 張 NVIDIA P100 GPU。

Base Model 大約訓練了 12 小時，而大的模型則訓練約 3.5 天。

***Optimizer***

使用了 Adam Optimizer，細節設定如下。

- $\beta_1 = 0.9$
- $\beta_2 = 0.98$
- $\epsilon = 10^{-9}$
- $warmup\_steps = 4000$

$$
\text{lrate} = d^{-0.5}_m \cdot \min (step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
$$

***Regularization***

首先是對每個 sub-layer 在經過 add&normalize 之前會經過 **Dropout**。Base Model 使用 $P_{drop} = 0.1$。

訓練期間他們採用了 **Label Smoothing**。輸出是會經過一個 Softmax 函數去決定每個 token 被輸出的機率。一般來說我們會期待正確的輸出要越接近 $1$ 越好，但是這需要結果趨近於無限大才可能發生。

據說一般會將目標改成越接近 $0.9$ 越好，讓模型比較好學，不過 Transformer 這邊選擇只要接近 $0.1$ 就好。

雖然這樣的做法會讓模型的困惑度(perplexity)增加，但他們發現這樣會在最後得到更好的準確度與 BLEU score。

### Model Variations 實驗結果

<center>
<img src="/Transformer/rkeJyFhmA.png" width=500>
</center>
</br>

> Image from [Ashish Vaswani, Noam Shazeer, Niki Parmar et al. (2017)](https://arxiv.org/pdf/1706.03762)

BLEU 是一種評價機器翻譯品質的方法，數字是越大越好。而 PPL 是困惑度，也是評量一個語言模型優劣的方法，數字是越小越好。

#### English Constituency Parsing 實驗結果

<center>
<img src="/Transformer/ryeDXthm0.png" width=500>
</center>
</br>

> Image from [Ashish Vaswani, Noam Shazeer, Niki Parmar et al. (2017)](https://arxiv.org/pdf/1706.03762)

在同樣的 Training Data 下，Transformer 都可以得到更好的結果。

## Contribution

總結來說，Transformer 是一個跨時代的傑作，順利解決掉 RNN 的兩個重大缺點：**極低的平行度**以及**資訊的流失**。

雖然 Transformer 起初只針對機器翻譯領域做研究，不過他的靈活性在當前許多的領域都可以看到 Transformer 的身影。包含了 LLM、Diffusion Model、Object Detection、Domain Adaptation 等，都可以看到替換成 Transformer base 後帶來的優勢。是一篇相當值得深讀的論文。

## 值得一看的文章們

- [Mu Li - Transformer论文逐段精读](https://www.youtube.com/watch?v=nzqlFIcCSWQ)
- [LeeMeng - 進入 NLP 世界的最佳橋樑：寫給所有人的自然語言處理與深度學習入門指南](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html#%E6%9C%89%E8%A8%98%E6%86%B6%E7%9A%84%E5%BE%AA%E7%92%B0%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF_1)
- [Frederick Lee - Attention in Text：注意力機制](https://medium.com/@fredericklee_73485/attention-in-text-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%A9%9F%E5%88%B6-bc12e88f6c26)
- [3Blue1Brown - Attention in transformers, visually explained | Chapter 6, Deep Learning](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [Sharon Peng - BLEU評估方法](https://mycollegenotebook.medium.com/bleu%E8%A9%95%E4%BC%B0%E6%96%B9%E6%B3%95-2509c2149387)
- [CHEN TSU PEI - Perplexity（困惑度）是什麼？](https://medium.com/nlp-tsupei/perplexity%E6%98%AF%E4%BB%80%E9%BA%BC-426f52897513)