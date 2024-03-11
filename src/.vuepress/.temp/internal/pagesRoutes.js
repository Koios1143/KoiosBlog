export const pagesRoutes = [
  ["v-8daa1a0e","/",{"y":"h","t":"Blog Home","i":"home"},["/README.md"]],
  ["v-184f4da6","/intro.html",{"d":1704067200000,"l":"January 1, 2024","e":"<h1> About Me</h1>\n<p>本名林禾堃，一個喜愛資訊領域的人。目前就讀於清華大學資訊工程學系，過去曾擔任臺南一中資訊社社長，也是 SCIST 的共同創辦人之一。</p>\n<p>高中接觸了演算法、資訊安全、網路管理等領域，目前正在朝向 Deep Learning 領域發展，關注的主題包含 Computer Vision、Reinforcement Learning 以及 Large Language Model。</p>\n<p>希望透過這個 blog 紀錄學習的點滴，也歡迎一起來討論 ML 領域的各種知識！</p>\n<h2> Skills</h2>\n<ul>\n<li>\n<p>Programming Languagues</p>\n<p>C/C++, Python, JavaScripts</p>\n</li>\n<li>\n<p>Frameworks</p>\n<p>React, Hexo, LINE BOT, PyTorch</p>\n</li>\n<li>\n<p>Machine Learning</p>\n<p>Computer Vision, Reinforcement Learning</p>\n</li>\n<li>\n<p>Miscellaneous</p>\n<p>UNIX Programming, Cryptography, Reverse engineering, Git, Markdown, Vim</p>\n</li>\n<li>\n<p>Languages</p>\n<p>Mandarin (Native), English (TOEFL 81), Japanese (JLPT N1)</p>\n</li>\n</ul>","y":"a","t":"About Me","i":"circle-info"},[":md"]],
  ["v-620a6165","/posts/2024TSMC_CareerHack.html",{"d":1706400000000,"l":"January 28, 2024","c":["Feedbacks"],"g":["TSMC","CareerHack"],"e":"<h1> 2024 TSMC CareerHack 心得</h1>\n<p>前幾天去參加了 2024 台積電的黑客松，大概是人生第一次走進台積辦公室。</p>\n<p>這場比賽是一組四人的比賽，前面有一個預賽，需要解出一些簡單的演算法題目。每個人題目會不太相同，但基本上都不會太難，簡單的 Sort、Greedy、Graph、DP。</p>\n<h2> 比賽題目</h2>\n<p>我們這一組拿到的是 <strong>AI 看圖說故事</strong> 的題目，基本上就是會有一些工地的照片，希望我們可以去找到</p>\n<ul>\n<li>照片中有多少人</li>\n<li>有多少人有戴安全帽</li>\n<li>有多少人沒戴安全帽</li>\n<li>安全帽是甚麼顏色</li>\n</ul>","y":"a","t":"2024 TSMC CareerHack 心得"},[":md"]],
  ["v-3caeec67","/posts/Agent57.html",{"d":1708560000000,"l":"February 22, 2024","c":["Note"],"g":["Paper Read","Reinforcement Learning","ICML"],"e":"<h1> Agent57: Outperforming the Atari Human Benchmark</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>Adrià Puigdomènech Badia, Bilal Piot, Steven Kapturowski, et al. @ Google DeepMind</li>\n<li>2020 ICML</li>\n</ul>\n<h2> 問題描述</h2>\n<p>在 RL 當中，Atari games 是一個相當重要的 benchmark。過去的 RL 模型已經能夠在大多的 atari games 當中獲得相當不錯的 performance，例如 MuZero、R2D2，分別在 57 個遊戲當中有 51 和 52 個遊戲是 outperform 人類的。不過可惜的是，在剩下的遊戲當中這些 SoTA 就通常完全沒辦法學習。</p>","y":"a","t":"Agent57: Outperforming the Atari Human Benchmark"},[":md"]],
  ["v-c0336012","/posts/DACS.html",{"d":1705708800000,"l":"January 20, 2024","c":["Note"],"g":["Paper Read","Domain Adaption","Computer Vision","WACV"],"e":"<h1> DACS: Domain Adaptation via Cross-domain Mixed Sampling</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>2020 Release</li>\n<li>2021 WACV(Winter Conference on Applications of Computer Vision)</li>\n<li>Chalmers University of Technology(查爾摩斯理工大學)與 Volvo Cars 共同發表</li>\n</ul>\n<h2> What is Domain Adaption</h2>","y":"a","t":"DACS: Domain Adaptation via Cross-domain Mixed Sampling"},[":md"]],
  ["v-6fdb6976","/posts/DAFormer.html",{"a":"Koios","d":1710115200000,"l":"March 11, 2024","c":["Note"],"g":["Paper Read","Domain Adaption","Computer Vision","CVPR"],"e":"<h1> DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>Lukas Hoyer, Dengxin Dai, Luc Van Gool @ ETH Zurich &amp; MPI for Informatics</li>\n<li>2022 CVPR</li>\n</ul>\n\n<br>\n<blockquote>\n<p>Image from <a href=\"https://arxiv.org/abs/2111.14887\" target=\"_blank\" rel=\"noopener noreferrer\">Lukas Hoyer, Dengxin Dai, Luc Van Gool (2022)</a></p>\n</blockquote>","y":"a","t":"DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation"},[":md"]],
  ["v-32d63a0d","/posts/DQN.html",{"d":1707436800000,"l":"February 9, 2024","c":["Note"],"g":["Paper Read","Reinforcement Learning","NeurIPS"],"e":"<h1> Playing Atari with Deep Reinforcement Learning</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>2013 NeurIPS</li>\n<li>Volodymyr Mnih, Koray Kavukcuoglu David Silver et al.</li>\n<li>這個論文提出的做法稱為 DQN(Deep Q-Networks)</li>\n</ul>\n<h2> 問題描述</h2>\n<p>過去在 RL 領域當中把一些 high-dimensional 的感官資料（如：視覺影像、語音資料等）作為 agent 的輸入去學習一直是一個很大的挑戰。然而我們也看到近幾年 Deep Learning 已經能夠在這種資料上去擷取特徵，進而去完成許多複雜的任務。</p>","y":"a","t":"Playing Atari with Deep Reinforcement Learning"},[":md"]],
  ["v-5b18c8c4","/posts/Noisy%20Networks%20for%20Exploration.html",{"d":1706918400000,"l":"February 3, 2024","c":["Note"],"g":["Paper Read","Reinforcement Learning","ICLR"],"e":"<h1> Noisy Networks for Exploration</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>2018 ICLR</li>\n<li>Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, et al. @ Google Deepmind</li>\n</ul>\n<h2> 問題描述</h2>\n<p>在過去的 RL 當中我們往往仰賴對 agent 的 policy 增加 randomness 去增加 exploration，例如 <code>ϵ-greedy</code> 和 <code>entropy regularization</code> 等。不過這樣的做法往往只能在較於簡單的環境當中有比較有效率的探索，然而在現實狀況下往往並不會如此簡單，而這種探索的困難度甚至是指數性地成長。</p>","y":"a","t":"Noisy Networks for Exploration"},["/posts/Noisy Networks for Exploration.html","/posts/Noisy Networks for Exploration.md",":md"]],
  ["v-0fd9e004","/posts/ProDA.html",{"a":"Koios","d":1710115200000,"l":"March 11, 2024","c":["Note"],"g":["Paper Read","Domain Adaption","Computer Vision","CVPR"],"e":"<h1> Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation</h1>\n<h2> Basic Information</h2>\n<ul>\n<li>Pan Zhang1, Bo Zhang, Ting Zhang, Dong Chen, Yong Wang, Fang Wen @ University of Science and Technology of China, Microsoft Research Asia</li>\n<li>2021 CVPR</li>\n</ul>","y":"a","t":"Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation"},[":md"]],
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
  ["v-1724726c","/tag/paper-read/",{"y":"p","t":"Tag: Paper Read","I":false},[]],
  ["v-7f7459b2","/tag/reinforcement-learning/",{"y":"p","t":"Tag: Reinforcement Learning","I":false},[]],
  ["v-28948988","/tag/icml/",{"y":"p","t":"Tag: ICML","I":false},[]],
  ["v-65da17ce","/tag/domain-adaption/",{"y":"p","t":"Tag: Domain Adaption","I":false},[]],
  ["v-cc78423a","/tag/computer-vision/",{"y":"p","t":"Tag: Computer Vision","I":false},[]],
  ["v-2958c584","/tag/wacv/",{"y":"p","t":"Tag: WACV","I":false},[]],
  ["v-2848ab8c","/tag/cvpr/",{"y":"p","t":"Tag: CVPR","I":false},[]],
  ["v-468b3b36","/tag/neurips/",{"y":"p","t":"Tag: NeurIPS","I":false},[]],
  ["v-28948681","/tag/iclr/",{"y":"p","t":"Tag: ICLR","I":false},[]],
]
