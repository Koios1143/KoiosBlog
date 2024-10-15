<template><div><h1 id="a-visual-guide-to-mixture-of-experts-moe" tabindex="-1"><a class="header-anchor" href="#a-visual-guide-to-mixture-of-experts-moe" aria-hidden="true">#</a> A Visual Guide to Mixture of Experts (MoE)</h1>
<p>:::success
這篇文章是翻譯自 <a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts" target="_blank" rel="noopener noreferrer">Maarten Grootendorst 的文章：A Visual Guide to Mixture of Experts (MoE)<ExternalLinkIcon/></a>，此篇文章僅作為翻譯，版權歸該作者所有。
:::</p>
<p>在許多 Large Language Models (LLMs) 的最新發布版本當中，你也許時常會發現到 <strong>MoE</strong> 出現在標題當中。這裡的 <strong>MoE</strong> 究竟是指什麼，又為什麼有那麼多的 LLMs 在使用他呢？</p>
<p>在這篇文章當中我們將透過<strong>超過 50 張的視覺化圖像</strong>來探索這個重要的模型架構。</p>
<p>&lt;圖一&gt;</p>
<p>在這一篇文章當中，我們將說明兩個在 MoE 當中相當重要的元素，他們時常被應用在 LLM-based 架構當中，也就是 <strong>Exeprts</strong> 以及 <strong>Router</strong>。</p>
<p>如果想要看更多視覺化 LLMs 相關內容和支持 newsletter，您可以參考我撰寫的書籍</p>
<p>&lt;圖二：<a href="https://www.llm-book.com/" target="_blank" rel="noopener noreferrer">Official website<ExternalLinkIcon/></a> of the book. You can order the book on <a href="https://www.amazon.com/Hands-Large-Language-Models-Understanding/dp/1098150961" target="_blank" rel="noopener noreferrer">Amazon<ExternalLinkIcon/></a>. All code is uploaded to <a href="https://github.com/handsOnLLM/Hands-On-Large-Language-Models" target="_blank" rel="noopener noreferrer">GitHub<ExternalLinkIcon/></a>.&gt;</p>
<h2 id="什麼是-mixtrure-of-experts" tabindex="-1"><a class="header-anchor" href="#什麼是-mixtrure-of-experts" aria-hidden="true">#</a> 什麼是 Mixtrure of Experts？</h2>
<p>Mixture of Experts (MoE) 是一個透過運用多個不同的 sub-models (<em>&quot;experts&quot;</em>)來提升 LLMs 品質的技術。</p>
<p>我們可以用兩個主要的元素來定義 MoE：</p>
<ul>
<li>Experts - 現在在每個 Feed Forward Neural Network (FFNN) layer 當中都會有一群 <em>&quot;experts&quot;</em> 可以選用。而這些 <em>&quot;experts&quot;</em> 本質上就是許多 FFNNs。</li>
<li>Router (Gate Network) - 決定哪些 tokens 要經過哪些 experts。</li>
</ul>
<p>在每一個 MoE-based LLM 當中的 layer，我們都可以看到(專精於某些領域的) experts：</p>
<p>&lt;圖三&gt;</p>
<p>事實上這邊所稱的 <em>&quot;experts&quot;</em> 並不是指專精於心理學、生物學等等的專業領域。它最多只能學習到單字層級的語法資訊：</p>
<p>&lt;圖四&gt;</p>
<p>更準確地說，它們是擅長於在特定前後文當中的特定 tokens。而 router (gate network) 對於給定的輸入會選擇最適合的 expert(s) 來處理特定的 token(s)：</p>
<p>&lt;圖五&gt;</p>
<p>每一個 expert 並不是一個獨立的 LLM，而是在 LLM 架構當中的其中一部分。</p>
<h2 id="the-experts" tabindex="-1"><a class="header-anchor" href="#the-experts" aria-hidden="true">#</a> The Experts</h2>
<p>為了釐清 experts 的深層意義以及實際上如何運作，我們首先來看 MoE 架構想要取代的部分 - <em>dense layers</em>。</p>
<h3 id="dense-layers" tabindex="-1"><a class="header-anchor" href="#dense-layers" aria-hidden="true">#</a> Dense Layers</h3>
<p>Mixture of Experts (MoE) 都源自於 LLMs 當中最基本的機能，<em>Feed Forward Neural Network (FFNN)</em>。</p>
<p>在一個標準的 decoder-only Transformer 架構當中，在 layer normalization 後面都會接上一個 FFNN：</p>
<p>&lt;圖六&gt;</p>
<p>FFNN 可以運用 attention mechanism 給出的前後文訊息，進一步擷取出其中更複雜的關係。</p>
<p>然而，FFNN 隨著 size 的提升，參數量的成長速度也相當地快。而為了理解複雜的關係，尤其在輸入大小的擴張尤其顯著。</p>
<p>&lt;圖七&gt;</p>
<h3 id="sparse-layers" tabindex="-1"><a class="header-anchor" href="#sparse-layers" aria-hidden="true">#</a> Sparse Layers</h3>
<p>在傳統的 Transformer 架構當中，FFNN 又被稱為 dense model，因為其中的所有參數 (包含 weights 以及 biases) 都是被啟動著的。沒有任何一個資訊被落下，所有的資訊都會被用來計算出最後的輸出結果。</p>
<p>如果我們更仔細看 dense model，注意輸入是如何在某種程度上影響所有參數的：</p>
<p>&lt;圖八&gt;</p>
<p>在另一方面，sparse models 則只啟動部分的參數，而這個相當接近於 Mixture of Experts 的架構。</p>
<p>為了說明這一點，我們可以把 dense model 切分成多個部分 (這些部分又被稱為 experts)</p>
</div></template>


