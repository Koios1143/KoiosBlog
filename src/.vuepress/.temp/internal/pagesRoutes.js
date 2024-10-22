export const pagesRoutes = [
  ["v-8daa1a0e","/",{"y":"h","t":"Blog Home","i":"home"},["/README.md"]],
  ["v-184f4da6","/intro.html",{"d":1704067200000,"l":"January 1, 2024","e":"<h1> About Me</h1>\n<p>本名林禾堃，一個喜愛資訊領域的人。目前就讀於清華大學資訊工程學系，過去曾擔任臺南一中資訊社社長，也是 SCIST 的共同創辦人之一。</p>\n<p>高中接觸了演算法、資訊安全、網路管理等領域，目前正在朝向 Deep Learning 領域發展，關注的主題包含 Computer Vision、Reinforcement Learning 以及 Large Language Model。</p>\n<p>希望透過這個 blog 紀錄學習的點滴，也歡迎一起來討論 ML 領域的各種知識！</p>\n<h2> Skills</h2>\n<ul>\n<li>\n<p>Programming Languagues</p>\n<p>C/C++, Python, JavaScripts</p>\n</li>\n<li>\n<p>Frameworks</p>\n<p>React, Hexo, LINE BOT, PyTorch</p>\n</li>\n<li>\n<p>Machine Learning</p>\n<p>Computer Vision, Reinforcement Learning</p>\n</li>\n<li>\n<p>Miscellaneous</p>\n<p>UNIX Programming, Cryptography, Reverse engineering, Git, Markdown, Vim</p>\n</li>\n<li>\n<p>Languages</p>\n<p>Mandarin (Native), English (TOEFL 81), Japanese (JLPT N1)</p>\n</li>\n</ul>","y":"a","t":"About Me","i":"circle-info"},[":md"]],
  ["v-620a6165","/posts/2024TSMC_CareerHack.html",{"d":1706400000000,"l":"January 28, 2024","c":["Feedbacks"],"g":["TSMC","CareerHack"],"e":"<h1> 2024 TSMC CareerHack 心得</h1>\n<p>前幾天去參加了 2024 台積電的黑客松，大概是人生第一次走進台積辦公室。</p>\n<p>這場比賽是一組四人的比賽，前面有一個預賽，需要解出一些簡單的演算法題目。每個人題目會不太相同，但基本上都不會太難，簡單的 Sort、Greedy、Graph、DP。</p>\n<h2> 比賽題目</h2>\n<p>我們這一組拿到的是 <strong>AI 看圖說故事</strong> 的題目，基本上就是會有一些工地的照片，希望我們可以去找到</p>\n<ul>\n<li>照片中有多少人</li>\n<li>有多少人有戴安全帽</li>\n<li>有多少人沒戴安全帽</li>\n<li>安全帽是甚麼顏色</li>\n</ul>","y":"a","t":"2024 TSMC CareerHack 心得"},[":md"]],
  ["v-3caeec67","/posts/Agent57.html",{"d":1708560000000,"l":"February 22, 2024","c":["Note"],"g":["Paper Read","Reinforcement Learning","ICML"],"e":"<h1> Agent57: Outperforming the Atari Human Benchmark</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>Adrià Puigdomènech Badia, Bilal Piot, Steven Kapturowski, et al. @ Google DeepMind</li>\n<li>2020 ICML</li>\n</ul>\n<h2> 問題描述</h2>\n<p>在 RL 當中，Atari games 是一個相當重要的 benchmark。過去的 RL 模型已經能夠在大多的 atari games 當中獲得相當不錯的 performance，例如 MuZero、R2D2，分別在 57 個遊戲當中有 51 和 52 個遊戲是 outperform 人類的。不過可惜的是，在剩下的遊戲當中這些 SoTA 就通常完全沒辦法學習。</p>","y":"a","t":"Agent57: Outperforming the Atari Human Benchmark"},[":md"]],
  ["v-c0336012","/posts/DACS.html",{"d":1705708800000,"l":"January 20, 2024","c":["Note"],"g":["Paper Read","Domain Adaptation","Computer Vision","WACV"],"e":"<h1> DACS: Domain Adaptation via Cross-domain Mixed Sampling</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>2020 Release</li>\n<li>2021 WACV(Winter Conference on Applications of Computer Vision)</li>\n<li>Chalmers University of Technology(查爾摩斯理工大學)與 Volvo Cars 共同發表</li>\n</ul>\n<h2> What is Domain Adaption</h2>","y":"a","t":"DACS: Domain Adaptation via Cross-domain Mixed Sampling"},[":md"]],
  ["v-6fdb6976","/posts/DAFormer.html",{"d":1710115200000,"l":"March 11, 2024","c":["Note"],"g":["Paper Read","Domain Adaptation","Computer Vision","CVPR"],"e":"<h1> DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>Lukas Hoyer, Dengxin Dai, Luc Van Gool @ ETH Zurich &amp; MPI for Informatics</li>\n<li>2022 CVPR</li>\n</ul>\n\n<br>\n<blockquote>\n<p>Image from <a href=\"https://arxiv.org/abs/2111.14887\" target=\"_blank\" rel=\"noopener noreferrer\">Lukas Hoyer, Dengxin Dai, Luc Van Gool (2022)</a></p>\n</blockquote>","y":"a","t":"DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation"},[":md"]],
  ["v-32d63a0d","/posts/DQN.html",{"d":1707436800000,"l":"February 9, 2024","c":["Note"],"g":["Paper Read","Reinforcement Learning","NeurIPS"],"e":"<h1> Playing Atari with Deep Reinforcement Learning</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>2013 NeurIPS</li>\n<li>Volodymyr Mnih, Koray Kavukcuoglu David Silver et al.</li>\n<li>這個論文提出的做法稱為 DQN(Deep Q-Networks)</li>\n</ul>\n<h2> 問題描述</h2>\n<p>過去在 RL 領域當中把一些 high-dimensional 的感官資料（如：視覺影像、語音資料等）作為 agent 的輸入去學習一直是一個很大的挑戰。然而我們也看到近幾年 Deep Learning 已經能夠在這種資料上去擷取特徵，進而去完成許多複雜的任務。</p>","y":"a","t":"Playing Atari with Deep Reinforcement Learning"},[":md"]],
  ["v-073f61cf","/posts/Dropout.html",{"d":1710633600000,"l":"March 17, 2024","c":["Note"],"g":["Paper Read","Regularization","JMLR"],"e":"<h1> Dropout: A Simple Way to Prevent Neural Networks from Overfitting</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov @ Toronto University</li>\n<li>2014 JMLR</li>\n</ul>\n<h2> 問題描述</h2>\n<p>在近年來發現到 Neural Network 參數越多就有越強大的表達能力，並且通常會有更好的表現。不過隨著參數量的上升，我們也發現到模型越來越會傾向於 Overfitting。</p>","y":"a","t":"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"},[":md"]],
  ["v-25c9f246","/posts/HRDA.html",{"d":1710547200000,"l":"March 16, 2024","c":["Note"],"g":["Paper Read","Domain Adaptation","Computer Vision","ECCV"],"e":"<h1> HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>Lukas Hoyer, Dengxin Dai, Luc Van Gool @ ETH Zurich &amp; MPI for Informatics</li>\n<li>2022 ECCV</li>\n</ul>\n<h2> 問題描述</h2>\n<p>這篇 paper 如同 DAFormer 關注在 UDA for semantic segmentation 。</p>","y":"a","t":"HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation"},[":md"]],
  ["v-5b18c8c4","/posts/Noisy%20Networks%20for%20Exploration.html",{"d":1706918400000,"l":"February 3, 2024","c":["Note"],"g":["Paper Read","Reinforcement Learning","ICLR"],"e":"<h1> Noisy Networks for Exploration</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>2018 ICLR</li>\n<li>Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, et al. @ Google Deepmind</li>\n</ul>\n<h2> 問題描述</h2>\n<p>在過去的 RL 當中我們往往仰賴對 agent 的 policy 增加 randomness 去增加 exploration，例如 <code>ϵ-greedy</code> 和 <code>entropy regularization</code> 等。不過這樣的做法往往只能在較於簡單的環境當中有比較有效率的探索，然而在現實狀況下往往並不會如此簡單，而這種探索的困難度甚至是指數性地成長。</p>","y":"a","t":"Noisy Networks for Exploration"},["/posts/Noisy Networks for Exploration.html","/posts/Noisy Networks for Exploration.md",":md"]],
  ["v-d4413c4c","/posts/PiPa.html",{"d":1715990400000,"l":"May 18, 2024","c":["Note"],"g":["Paper Read","Domain Adaptation","Computer Vision","ACM Multimedia"],"e":"<h1> PiPa: Pixel- and Patch-wise Self-supervised Learning for Domain Adaptative Semantic Segmentation</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua</li>\n<li>2022 ACM Multimedia</li>\n</ul>\n<h2> 問題描述</h2>\n<p>這一篇與過去看過的 <a href=\"https://koios1143.github.io/PaperBlog/posts/DACS.html\" target=\"_blank\" rel=\"noopener noreferrer\">DACS</a>, <a href=\"https://koios1143.github.io/PaperBlog/posts/ProDA.html\" target=\"_blank\" rel=\"noopener noreferrer\">ProDA</a>, <a href=\"https://koios1143.github.io/PaperBlog/posts/DAFormer.html\" target=\"_blank\" rel=\"noopener noreferrer\">DAFormer</a>, <a href=\"https://koios1143.github.io/PaperBlog/posts/HRDA.html\" target=\"_blank\" rel=\"noopener noreferrer\">HRDA</a> 同樣都是以 Unsupervised 的方式解決 Semantic Segmentationb 的 Domain Adaptation問題。</p>","y":"a","t":"PiPa: Pixel- and Patch-wise Self-supervised Learning for Domain Adaptative Semantic Segmentation"},[":md"]],
  ["v-0fd9e004","/posts/ProDA.html",{"d":1710115200000,"l":"March 11, 2024","c":["Note"],"g":["Paper Read","Domain Adaptation","Computer Vision","CVPR"],"e":"<h1> Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>Pan Zhang1, Bo Zhang, Ting Zhang, Dong Chen, Yong Wang, Fang Wen @ University of Science and Technology of China, Microsoft Research Asia</li>\n<li>2021 CVPR</li>\n</ul>","y":"a","t":"Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation"},[":md"]],
  ["v-ed4def16","/posts/Transformer.html",{"d":1716336000000,"l":"May 22, 2024","c":["Note"],"g":["Paper Read","NLP","Computer Vision","NeurIPS"],"e":"<h1> Attention is all you need</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>NIPS 2017 (former NeuralPS)</li>\n<li>Ashish Vaswani, Noam Shazeer, Niki Parmar et al. from Google Brain and Google Research</li>\n</ul>\n<h2> 問題描述</h2>\n<h3> RNN</h3>\n<p>近年來自然語言處理(Natural Language Processing, NLP)與機器翻譯等任務上時常使用 Recurrent Neural Network(RNN), Long Short-Term Memory(LSTM), Gated Recurrent Neural Network 等模型架構，我們也看到使用 Recurrent 模型以及 Encoder-Decoder 架構蔚為流行。</p>","y":"a","t":"Attention is all you need"},[":md"]],
  ["v-3501ffcb","/posts/Tsukuba%20week01.html",{"d":1727308800000,"l":"September 26, 2024","c":["Diary"],"g":["Diary","University of Tsukuba","Exchange Program"],"e":"<h1> 筑波大學交換週記 Week 01</h1>\n<div style=\"text-align:center\">\n<p><em><strong>感謝一路上教導我日語的阿普魯老師，陪伴我持續練習日文的冠霆、致越、陳曦、6uc，日文課程當中不厭其煩地協助我的川越老師、勇氣、小川、北之間、關口、雅之、鉦洋、冠文、阿郡助教們，大力推薦我的濬屹教授，支持我的家人們，以及我的摯愛心瑤</strong></em></p>\n</div>\n<h2> 契機</h2>\n<p>大一剛入學的我因為好奇而選了一門日文課，沒想到因此開啟了前往筑波大學交換的契機。</p>\n<p>當時遇到的阿普魯老師用他的熱情感染了每一個人。在學習日文以外，包含我在內的不少人也開始萌生想要用學習到的日語能力到日本交換的念頭。</p>","y":"a","t":"筑波大學交換週記 Week 01"},["/posts/Tsukuba week01.html","/posts/Tsukuba week01.md",":md"]],
  ["v-36b6d86a","/posts/Tsukuba%20week02.html",{"d":1727740800000,"l":"October 1, 2024","c":["Diary"],"g":["Diary","University of Tsukuba","Exchange Program"],"e":"<h1> 筑波大學交換週記 Week 02</h1>\n<h2> 選課</h2>\n<p>就快要到開學的日子了，系上的 Orientation 也發了文件讓我們準備選課。</p>\n<p>在這之前 Tutor 已經告訴過我們學校的各個系統，所以基本上都已經了解過了，甚至也分享了筑波大學學生自己製作的 APP，相當方便。</p>\n<p>目前挑了幾堂系上的專業課程還有日語課程，到時候要跑每間教室去問能不能收留。希望想修的課都能順利修到。</p>\n<p>專業課程也許會想選</p>\n<ul>\n<li>分散式系統</li>\n<li>電腦視覺</li>\n<li>人工智慧</li>\n<li>平行處理</li>\n<li>資訊檢索概論</li>\n</ul>","y":"a","t":"筑波大學交換週記 Week 02"},["/posts/Tsukuba week02.html","/posts/Tsukuba week02.md",":md"]],
  ["v-386bb109","/posts/Tsukuba%20week03.html",{"d":1728345600000,"l":"October 8, 2024","c":["Diary"],"g":["Diary","University of Tsukuba","Exchange Program"],"e":"<h1> 筑波大學交換週記 Week 03</h1>\n<h2> 上課 ouo</h2>\n<p>開學過了一週的時間，課表大致上已經確定好了，也幾乎都上過第一堂課了。</p>\n<figure><img src=\"/Tsukuba_week03/01.png\" alt=\"\" width=\"300\" tabindex=\"0\" loading=\"lazy\"><figcaption>大致已經抵定的的課表</figcaption></figure>\n<p>這次選了一些系上的專業課程還有日本語課程，希望可以在回去之前順利把學分都集滿，也順利拿到成績單。</p>\n<p>筑波的學制跟臺灣比較不一樣，一個學期裡面會切分成 A, B, C 三個 modules，每個 module 大概就是 5 週左右的時間，不包含期末考。</p>","y":"a","t":"筑波大學交換週記 Week 03"},["/posts/Tsukuba week03.html","/posts/Tsukuba week03.md",":md"]],
  ["v-3a2089a8","/posts/Tsukuba%20week04.html",{"d":1728950400000,"l":"October 15, 2024","c":["Diary"],"g":["Diary","University of Tsukuba","Exchange Program"],"e":"<h1> 筑波大學交換週記 Week 04</h1>\n<h2> 21.0975</h2>\n<p>進入到來到日本的第一個月了。在這一個月當中有不少的時間都在為了推甄擔心，難以暫時放下這些負擔去看看身邊的風景。</p>\n<p>投了中央、成大、清大、台大、交大的許多系所，不斷在這個過程當中去回放過去自己做的事情、思索將來想要做什麼。在這之外，更多的煩惱也許是完成每個系所要求的文件。</p>\n<p>不過這一切終於告一段落，在這一週順利告一個段落了。</p>\n<p>接下來就是一邊準備面試一邊靜待佳音。</p>\n<h2> 一張一弛</h2>\n<p>每年 10 月的第二個星期一是日本的スポーツの日(運動之日)，據說是為了紀念 1964 年在日本舉辦的奧運所以設立的節日。由於在周一的關係，會有一個三天的連假，而我們安排了兩天的水戶・大洗的旅程。</p>","y":"a","t":"筑波大學交換週記 Week 04"},["/posts/Tsukuba week04.html","/posts/Tsukuba week04.md",":md"]],
  ["v-3bd56247","/posts/Tsukuba%20week05.html",{"d":1729555200000,"l":"October 22, 2024","c":["Diary"],"g":["Diary","University of Tsukuba","Exchange Program"],"e":"<h1> 筑波大學交換週記 Week 05</h1>\n<h2> 自己煮ㄉ生活 ouo</h2>\n<p>大概從上禮拜開始有嘗試煮之後就幾乎每天都是自己煮的生活，有時候不想花時間就隨意煮，有點時間就可以嘗試嘗試其他不一樣的做法。</p>\n<p>不過一開始我大概都是圍繞在麵、雞肉、香菇、豆腐、蛋、菜(高麗菜、洋蔥、豆芽菜之類的)。</p>\n<p>尤其如果不想花太多時間，湯麵尤其輕鬆。只要把材料依序丟進去等他們在鍋子裡面熟了就好。</p>\n<figure><img src=\"/Tsukuba_week05/01.jpg\" alt=\"\" width=\"300\" tabindex=\"0\" loading=\"lazy\"><figcaption>把所有材料依序放進去，湯麵就完成了！有雞肉的話還會有雞湯的味道還不錯w</figcaption></figure>","y":"a","t":"筑波大學交換週記 Week 05"},["/posts/Tsukuba week05.html","/posts/Tsukuba week05.md",":md"]],
  ["v-ad1b5b16","/posts/WKM.html",{"d":1717372800000,"l":"June 3, 2024","c":["Note"],"g":["Paper Read","Reinforcement Learning","LLM"],"e":"<h1> Agent Planning with World Knowledge Model</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>2024/05/13 發布 (尚未正式於 Conf. 發表)</li>\n<li>Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. @ Zhejiang University, National University of Singapore, Alibaba Group</li>\n</ul>\n<h2> 問題描述</h2>\n<p>近年來大型語言模型(LLM)在許多自然語言處理的問題有很快速的成長，而近期開始出現一些使用 LLM 作為 agent model 來處理物理環境中的規劃問題。然而由於當前 SOTA 的 LLM 幾乎都是 autoregressive model，模型實際上會做的事情是去預測下一個 output token 要是什麼，實際上他們對於物理環境是沒有任何理解的。</p>","y":"a","t":"Agent Planning with World Knowledge Model"},[":md"]],
  ["v-3706649a","/404.html",{"y":"p","t":""},[]],
  ["v-e1e3da16","/posts/",{"y":"p","t":"Posts"},[]],
  ["v-5bc93818","/category/",{"y":"p","t":"Category","I":false},[]],
  ["v-744d024e","/tag/",{"y":"p","t":"Tag","I":false},[]],
  ["v-e52c881c","/article/",{"y":"p","t":"Articles","I":false},[]],
  ["v-154dc4c4","/star/",{"y":"p","t":"Star","I":false},[]],
  ["v-01560935","/timeline/",{"y":"p","t":"Timeline","I":false},[]],
  ["v-17dd2679","/category/feedbacks/",{"y":"p","t":"Feedbacks Category","I":false},[]],
  ["v-2936d0ec","/tag/tsmc/",{"y":"p","t":"Tag: TSMC","I":false},[]],
  ["v-58706565","/category/note/",{"y":"p","t":"Note Category","I":false},[]],
  ["v-52d83d62","/tag/careerhack/",{"y":"p","t":"Tag: CareerHack","I":false},[]],
  ["v-b7a24a38","/category/diary/",{"y":"p","t":"Diary Category","I":false},[]],
  ["v-1724726c","/tag/paper-read/",{"y":"p","t":"Tag: Paper Read","I":false},[]],
  ["v-7f7459b2","/tag/reinforcement-learning/",{"y":"p","t":"Tag: Reinforcement Learning","I":false},[]],
  ["v-28948988","/tag/icml/",{"y":"p","t":"Tag: ICML","I":false},[]],
  ["v-4a51c4c1","/tag/domain-adaptation/",{"y":"p","t":"Tag: Domain Adaptation","I":false},[]],
  ["v-cc78423a","/tag/computer-vision/",{"y":"p","t":"Tag: Computer Vision","I":false},[]],
  ["v-2958c584","/tag/wacv/",{"y":"p","t":"Tag: WACV","I":false},[]],
  ["v-2848ab8c","/tag/cvpr/",{"y":"p","t":"Tag: CVPR","I":false},[]],
  ["v-468b3b36","/tag/neurips/",{"y":"p","t":"Tag: NeurIPS","I":false},[]],
  ["v-2554e3d9","/tag/regularization/",{"y":"p","t":"Tag: Regularization","I":false},[]],
  ["v-28a729b8","/tag/jmlr/",{"y":"p","t":"Tag: JMLR","I":false},[]],
  ["v-285c0730","/tag/eccv/",{"y":"p","t":"Tag: ECCV","I":false},[]],
  ["v-28948681","/tag/iclr/",{"y":"p","t":"Tag: ICLR","I":false},[]],
  ["v-0f313cf4","/tag/acm-multimedia/",{"y":"p","t":"Tag: ACM Multimedia","I":false},[]],
  ["v-b30a616a","/tag/nlp/",{"y":"p","t":"Tag: NLP","I":false},[]],
  ["v-3c78b6cc","/tag/diary/",{"y":"p","t":"Tag: Diary","I":false},[]],
  ["v-26af5756","/tag/university-of-tsukuba/",{"y":"p","t":"Tag: University of Tsukuba","I":false},[]],
  ["v-1791fbf2","/tag/exchange-program/",{"y":"p","t":"Tag: Exchange Program","I":false},[]],
  ["v-b30c33a0","/tag/llm/",{"y":"p","t":"Tag: LLM","I":false},[]],
]
