---
date: 2024-10-15
category:
  - Note
  - Translate
tag:
  - Translate
  - Mixture-of-Experts
---

# A Visual Guide to Mixture of Experts (MoE)

:::success
這篇文章是翻譯自 [Maarten Grootendorst 的文章：A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)，此篇文章僅作為翻譯，版權歸該作者所有。
:::

在許多 Large Language Models (LLMs) 的最新發布版本當中，你也許時常會發現到 **MoE** 出現在標題當中。這裡的 **MoE** 究竟是指什麼，又為什麼有那麼多的 LLMs 在使用他呢？

在這篇文章當中我們將透過**超過 50 張的視覺化圖像**來探索這個重要的模型架構。

<圖一>

在這一篇文章當中，我們將說明兩個在 MoE 當中相當重要的元素，他們時常被應用在 LLM-based 架構當中，也就是 **Exeprts** 以及 **Router**。

如果想要看更多視覺化 LLMs 相關內容和支持 newsletter，您可以參考我撰寫的書籍

<圖二：[Official website](https://www.llm-book.com/) of the book. You can order the book on [Amazon](https://www.amazon.com/Hands-Large-Language-Models-Understanding/dp/1098150961). All code is uploaded to [GitHub](https://github.com/handsOnLLM/Hands-On-Large-Language-Models).>

## 什麼是 Mixtrure of Experts？

Mixture of Experts (MoE) 是一個透過運用多個不同的 sub-models (*"experts"*)來提升 LLMs 品質的技術。

我們可以用兩個主要的元素來定義 MoE：

- Experts - 現在在每個 Feed Forward Neural Network (FFNN) layer 當中都會有一群 *"experts"* 可以選用。而這些 *"experts"* 本質上就是許多 FFNNs。
- Router (Gate Network) - 決定哪些 tokens 要經過哪些 experts。

在每一個 MoE-based LLM 當中的 layer，我們都可以看到(專精於某些領域的) experts：

<圖三>

事實上這邊所稱的 *"experts"* 並不是指專精於心理學、生物學等等的專業領域。它最多只能學習到單字層級的語法資訊：

<圖四>

更準確地說，它們是擅長於在特定前後文當中的特定 tokens。而 router (gate network) 對於給定的輸入會選擇最適合的 expert(s) 來處理特定的 token(s)：

<圖五>

每一個 expert 並不是一個獨立的 LLM，而是在 LLM 架構當中的其中一部分。

## The Experts

為了釐清 experts 的深層意義以及實際上如何運作，我們首先來看 MoE 架構想要取代的部分 - *dense layers*。

### Dense Layers

Mixture of Experts (MoE) 都源自於 LLMs 當中最基本的機能，*Feed Forward Neural Network (FFNN)*。

在一個標準的 decoder-only Transformer 架構當中，在 layer normalization 後面都會接上一個 FFNN：

<圖六>

FFNN 可以運用 attention mechanism 給出的前後文訊息，進一步擷取出其中更複雜的關係。

然而，FFNN 隨著 size 的提升，參數量的成長速度也相當地快。而為了理解複雜的關係，尤其在輸入大小的擴張尤其顯著。

<圖七>

### Sparse Layers

在傳統的 Transformer 架構當中，FFNN 又被稱為 dense model，因為其中的所有參數 (包含 weights 以及 biases) 都是被啟動著的。沒有任何一個資訊被落下，所有的資訊都會被用來計算出最後的輸出結果。

如果我們更仔細看 dense model，注意輸入是如何在某種程度上影響所有參數的：

<圖八>

在另一方面，sparse models 則只啟動部分的參數，而這個相當接近於 Mixture of Experts 的架構。

為了說明這一點，我們可以把 dense model 切分成多個部分 (這些部分又被稱為 experts)
