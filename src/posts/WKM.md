---
date: 2024-06-03
category:
  - Note
tag:
  - Paper Read
  - Reinforcement Learning
  - LLM
---

# Agent Planning with World Knowledge Model

## Basic Information

- 2024/05/13 發布 (尚未正式於 Conf. 發表)
- Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. @ Zhejiang University, National University of Singapore, Alibaba Group

## 問題描述

近年來大型語言模型(LLM)在許多自然語言處理的問題有很快速的成長，而近期開始出現一些使用 LLM 作為 agent model 來處理物理環境中的規劃問題。然而由於當前 SOTA 的 LLM 幾乎都是 autoregressive model，模型實際上會做的事情是去預測下一個 output token 要是什麼，實際上他們對於物理環境是沒有任何理解的。

這導致過去使用 LLM 作為 agent model 時常出現兩個問題。
1. 時常會產生怪異的行為
2. 出現一些無腦的 trial-and-error

如果跟人類的決策方式去比較，中間的差異在於人類會具有跟真實世界相關的知識，因此**會在心中先預想解決問題的方式**。而實際在執行的過程當中，人類也會**依據現有的資訊持續解決當前任務**。

我們稱呼這個**事先預想**的知識為**global task knowledge**。
而在執行 task 過程中的**現有資訊**則被稱為**local state knowledge**。

:::info
舉一個簡單的例子來說明。

想像你現在在廚房當中，你的目標是要把一個乾淨的雞蛋放進微波爐當中。

- ***global task knowledge***
    首先，雞蛋很可能出現在冰箱裡面，而要洗乾淨最常去洗手檯。這些預先知道的知識讓你決定接下來解決的步驟大概會像

    1. 去找到冰箱，並找到雞蛋
    2. 去洗手檯把雞蛋洗乾淨
    3. 拿到微波爐裡

- ***local state knowledge***
    當規劃出整體的步驟後，接著就要去實踐。實踐的過程當中我們會需要理解當前所處的狀態。我要完成什麼步驟、完成的狀況如何、下一步該做什麼。

    例如現在你要完成第一個步驟：找到冰箱、並找到雞蛋。那麼你會需要知道
    - 我現在找到冰箱了嗎
    - 如果找到冰箱了，那我找到雞蛋了嗎
    
    諸如此類的問題

透過 ***global task knowledge*** 與 ***local state knowledge***，我們可以事先理解許多事情，避免執行怪異的行為以及無腦的嘗試。像是去烤箱裡面找雞蛋，或是嘗試在書房完成整個任務等等。
:::

因此作者依照上面的想法，提出了 ***World Knowledge Model(WKM)*** 去提供 agent **global task knowledge** 與 **local state knowledge**。

透過與當前 SOTA LLM model 如 Mistral-7B, Gemma-7B, Llama-3-8B 結合，在 ALFWorld, WebShop, ScienceWorld 獲得了極佳的成果。

## Related Works

### Knowledge Augumented Agent Planning

在過去也有一些透過 LLM 輔助 agent 做決策的研究。底下以發表於 ICML 2022 的 [Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents](https://arxiv.org/abs/2201.07207) 為例來說明。

他們讓 LLM 先輸出一個步驟，接下來為了要讓輸出的動作變成環境能夠接收的詞句，會再去找到所有 agent 能執行的步驟當中語意最接近的 action 輸出。

<center>
<img src="/WKM/ryq-BjFV0.gif">
</center>
</br>

> Image from [Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents](https://wenlong.page/language-planner/)

底下是 *Browse Internet* 任務的結果。

<center>
<img src="/WKM/SynBIsYN0.gif">
</center>
</br>

> Image from [Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents](https://wenlong.page/language-planner/)

雖然在部分的例子當中可以看到這種相關的做法還不錯，但他們往往需要手動設定 prompt template 以及 task procedures。因此這類的結果很難直接應用到其他的環境當中。

其他的研究方向則是單純考慮到整體的工作流程，或是只考慮到局部的 action，並沒有兩方面都顧慮到。作者提出的 WKM 則有辦法解決上述的三個痛點。

## Methodology

### Preliminaries

他們將整個環境以 Partially Observable Markov Decision Process(POMDP): $(\mathcal{U}, \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T})$ 來描述。

- $\mathcal{U}$ 表示 insturction space，定義了有哪些 task 以及各自的規範
- $\mathcal{S}$ 表示 state space
- $\mathcal{A}$ 表示 action space
- $\mathcal{O}$ 表示 observation space
- $\mathcal{T} : \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$ 表示 transition function

此外，我們也定義 historical trajectory $h_t$ 包含了這次目標的 task $u \in \mathcal{U}$、執行的 actions $a \in \mathcal{A}$ 以及接收到的 observations $o \in \mathcal{O}$。

$$
h_t = (u, a_0, o_0, a_1, o_1, \dots, a_t, o_t)
$$

我們的 policy model $\pi_{\theta}$ 在這邊是以 LLM 來扮演，也就是說 LLM 會去決定下一個 action。這裡的 $\theta$ 描述 LLM 的參數。

$$
a_{t+1} \sim \pi_{\theta}(\cdot | h_t)
$$

不過其中比較特別的是最初的 action $a_0$ 是根據 task 去決定的。

$$
a_0 \sim \pi_{\theta}(\cdot | u)
$$

一個 trajectory $\tau$ 結束的時機分成 *已經完成 task* 以及 *超過執行時間上限* 兩個狀況。

那麼一個 task 會執行 trajectory $\tau$ 的機率也就可以描述成

$$
\pi(\tau | u) = \prod_{t=0}^{n}{\pi_{\theta}(a_{t+1}|h_t)} \pi_{\theta}(a_0 | u)
$$

最終，一個 trajectory 得到的 reward $r(i, \tau)$ 會是一個介於 $[0, 1]$ 之間的數值，用來描述 task $u$ 完成率有多少。

作者提出的 WKM 當中的 **world** 意指模擬 task 所使用的環境。而 **world knowledge** 則包含了 **真實世界的知識(prior global knowledge)** 以及 **模擬環境的知識(dynamic local knowledge)**。

### 整體流程

<center>
<img src="/WKM/SJFrShtNA.png">
</center>
</br>

> Image from [Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. (2024)](https://arxiv.org/pdf/2405.14205)

### Task Knowledge Synthesis

這個步驟的目標是產出 **global task knowledge**。藉此讓 agent 可以學習規劃整體的解決步驟，也避免不必要的 trial-and-error。這裡分成了兩個步驟。

***Experienced Agent Exploration***

要訓練模型產出 global task knowledge，我們會需要知道正確的步驟會是什麼，也就是知道 Expert Trajectory。

不過作者認為如果我們只看 Expert Trajectory，或許會使模型缺乏一些比較高層次的認知與知識。我們雖然知道跟隨一個步驟可以完成任務，但卻不知道每個步驟安排的意義是什麼。

因此作者希望產出另一組 Trajectory，讓模型可以參考兩個做法去推論。不過如果這個產出的 Trajectory 太差，那也沒有太大的參考價值。

因此作者首先透過 Expert Trajectory 訓練了一個 Experienced Agent。接下來再讓這個 agent 去產出 Rejected Trajectory。也就是說，我們會預設產出來的 Trajectory 比起 Expert Trajectory 是包含一些需要避免的操作。

***Self Knowledge Synthesis***

有了 Expert Trajectory 以及 Rejected Trajectory，接下來就可以把它們作為作為參考資訊，去產出 **global task knowledge** $\kappa$。

$$
\kappa \sim \pi_{\theta}(\cdot | \rho_{\text{TaskKnow}}, u, \tau_w, \tau_l)
$$

- $\kappa \in \mathcal{K}$ 表示 task knowledge
- $\rho_{\text{TaskKnow}}$ 是讓 LLM model 產出 task knowledge 的 prompt
- $u$ 表示 task
- $\tau_w$ 表示 Expert Trajectory
- $\tau_l$ 表示 Rejected Trajectory

### State Knowledge Summarization

這個步驟的目標是要產出 **local state knowledge**。藉此讓 agent 可以釐清當前的狀況，並給出接下來目標方向。

做法是製作 prompt 讓模型去產出當前步驟的總結。為了確保最終輸出結果的品質，這裡會只考慮 Expert Trajectory。

> 假如我們在這裡也考慮了 Rejected Trajectory，那麼也許會出現前後動作缺乏連貫性，或是與最終目標不符等問題。這會導致無法好好判斷當前的狀態。

$$
s_t \sim \pi_{\theta}(\cdot | \rho_{\text{StateKnow}}, h_t)
$$

- $s_t \in \mathcal{S}$ 表示 state knowledge，是 state space 的一部分
- $\rho_{\text{StateKnow}}$ 表示讓模型產出總結的 prompt
- $h_t$ 表示 historical trajectory

在 inference 階段，直覺上直接把產出的 state knowledge 接在 agent 的輸入當中就可以作為參考輸出下一個 action。不過作者發現過多的資訊反倒會導致模型會感到困惑。

:::tip
在 **inference 階段**，我們在意的是**現在執行 action $a_{\text{pre}}$ 並總結出 state knowledge $s$ 後，接下來我該做什麼**。
:::

Summarize 後的 state knowledge $s$ 沒必要再丟給 agent，我們只要知道 $(s, a_{\text{pre}})$ 會對應哪個 $a_{\text{next}}$ 即可。

因此作者製作了一個 state knowledge base $\mathcal{B}$ 去紀錄 $(s, a_{\text{pre}}, a_{\text{next}})$。

$$
\mathcal{B} = \{ (s, a_{\text{pre}}, a_{\text{next}})^{(i)} \}_{i=1}^{|\mathcal{B}|}
$$

- $s$ 表示 state knowledge
- $a_{\text{pre}} = a_{t}$
- $a_{\text{next}} = a_{t+1}$
- $|\mathcal{B}|$ 表示 state knowledge base 的大小

如此一來，我們就可以更好地運用 state knowledge 了！

### Model Training

我們總共會有兩個模型需要訓練，分別是用來決策的 **Agent Model** 以及用來提供 global task knowledge 與 local state knowledge 的 **World Knowledge Model**。接著會詳細說明兩個模型的訓練。

不過值得一提的是，作者在訓練兩個模型都是用相同的 Backbone，並且都是使用 LoRA 而非 Full Training。

***Agent Model Training***

給定 Expert Trajectories Dataset $\mathcal{D}$

$$
\mathcal{D} = \{(u, \kappa, \tau_w)^{(i)}\}_{i=1}^{|\mathcal{D}|}
$$

- $u$ 表示 task
- $\kappa$ 表示前面產出的 global task knowledge
- $\tau_w$ 表示 Expert Trajectory

我們可以定義 Loss $\mathcal{L}_{\text{agent}}(\pi_{\theta})$ 如下

$$
\mathcal{L}_{\text{agent}}(\pi_{\theta}) = -\mathbb{E}_{\tau_w \sim \mathcal{D}}{\left[ \pi_{\theta}(\tau_w | u, \kappa) \right]}
$$

:::tip
也就是說，我們期待根據 task $u$ 與 global task knowledge $\kappa$ 去決策出來的結果，要越接近目標的 Expert Trajectory $\tau_w$ 越好。
:::

每個 Expert Trajectory 實質上都是一連串的 token，描述 **Agent 的行為**、**觀察到的結果**、**得到的 Reward**。令 $\mathcal{X} = (x_1, x_2. \dots, x_{|\mathcal{X}|})$ 表示 $\tau_{w}$ 的 token sequence。那麼我們定義 $\pi_{\theta}(\tau_w | u, \kappa)$ 如下。

$$
\pi_{\theta}(\tau_w | u, \kappa) = - \sum_{j=1}^{|\mathcal{X}|}{\left( \mathbb{1}(x_j \in \mathcal{A}) \times \log{\pi_{\theta}(x_j | u, \kappa, x_{<j})} \right)}
$$

- $\mathbb{1}(x_j \in \mathcal{A})$ 會把跟 action 無關的 token 篩除

:::tip
也就是說，如果 token $x_j$ 是合法的 token，就去看現在他被輸出的機率是多少。

而整體而言，$\pi_{\theta}(\tau_w | u, \kappa)$ 就在描述在當前的 task $u$ 與 global task knowledge $\kappa$ 底下，得到輸出 $\tau_w$ 的機率有多高。
:::

***World Knowledge Model Training***

與 Agent Model 不一樣的地方是，World Knowledge Model 還需要考慮 **local state knowledge**。

因此定義新的 Dataset $\mathcal{D}'$ 如下

$$
\mathcal{D}' = \{ (u, \kappa, \tau_w')^{(i)} \}_{i=1}^{\mathcal{|D'|}}
$$

- $\tau_w' = (a_0, o_0, s_0, \dots, a_n, o_n, s_n)$
    > 也就是多考慮了 local state knowledge $s$

那麼定義 World Knowledge Model $\pi_{\phi}$ 對應的 Loss $\mathcal{L}_{\text{know}}$

$$
\mathcal{L}_{\text{know}} = - \mathbb{E}_{\kappa, \tau_w' \sim \mathcal{D}'} \left[ \pi_{\phi}(\kappa | u) \pi_{\phi}(\tau_w' | u, \kappa) \right]
$$

與前面相同，Trajectories 實際上都是 token 組成。

- 令 $\mathcal{X}' = (x_1', x_2', \dots, x_{|\mathcal{X}'|})$ 表示 $\tau_w'$ 的 token sequence
- 令 $\mathcal{Y} = (y_1, y_2, \dots, y_{|\mathcal{Y}|})$ 表示 global task knowledge $\kappa$ 的 token sequence

我們可以定義$\pi_{\phi}(\kappa | u)$ 與 $\pi_{\phi}(\tau_w' | u, \kappa)$ 如下。

$$
\begin{align*}
\pi_{\phi}(\kappa | u) &= - \sum_{i=1}^{|\mathcal{Y}|}{\log{\pi_{\phi}(y_i | u, y_{<i})}} \\
\pi_{\phi}(\tau_w' | u, \kappa) &= - \sum_{j=1}^{|\mathcal{X}'|}{\left( \mathbb{1}(x_j' \in \mathcal{S}) \times \log{\pi_{\phi}(x_j' | u, \kappa, x'_{<j})} \right)}
\end{align*}
$$

- $\mathbb{1}(x'_j \in \mathcal{S})$ 會把跟 state 無關的 token 篩除

:::tip
也就是說，我們期待給定 task $u$ 時產出 global task knowledge $\kappa$ 的機率要越大越好。同時也要讓這個狀況下產出的 trajectory 跟目標 $\tau_w'$ 越像越好。
:::

### Agent Planning with World Knowledge Model

在 Inference 階段，agent 會搭配 World Knowledge Model(WKM) 去做決策。整體流程可以簡單描述成 7 個步驟。

1. 給定目標 task $u$
2. World Knowledge Model(WKM) 產出 global task knowledge $\kappa$
3. World Knowledge Model(WKM) 產出 local state knowledge $s_t$
4. 根據 $s_t$ 從 state knowledge base $\mathcal{B}$ 決定出 next action 的機率分布
5. Agent 也產出 next action 的機率分布
6. 考慮兩者的機率分布給出最終的 next action
7. 重複步驟 3~7 直到結束

***Step 1, 2：給定 $u$, WKM 產出 $\kappa$***

給定一個 task $u$，WKM 首先會產出 global task knowledge $\kappa$。

$$
\kappa \sim \pi_{\phi}(\cdot | u)
$$

***Step 3：WKM 產出 $s_t$***

接下來 agent 就會開始規劃。令 task $u$ 當中所有合法的 action $\mathcal{A}_u \subseteq \mathcal{A}$ 為 $\left( \alpha_u^{(1)}, \alpha_u^{(2)}, \dots, \alpha_u^{(|\mathcal{A}_u|)} \right)$。在時間 $t \geq 0$ 時考慮下一個 action 之前我們需要去考慮 local state knowledge $s_t$。

$$
\begin{align*}
s_t &\sim \pi_{\phi}(\cdot | h_t) \\
h_t &= (u, \kappa, a_0, o_0, a_1, o_1, \dots, a_t, o_t)
\end{align*}
$$

> 注意，這邊的 $h_t$ 定義有加上了 global task knowledge $\kappa$。

***Step 4：根據 $s_t$ 從 $\mathcal{B}$ 決定出第一個 next action 機率分布***

當取得 local state knowledge $s_t$ 後，如同前面 **State Knowledge Summarization** 當中所提及，我們會去 state knowledge base $\mathcal{B}$ 當中尋找下一個 action $a_{\text{next}}$。

這裡的作法是以 $s_t$ 作為 key，找到 $a_{\text{pre}} = a_t$ 當中語意最接近的前 $\mathcal{N}$ 個 next actions。依據這 $\mathcal{N}$ 個 next actions 去決定最終的機率分布。如果一個 next action 被決定的次數越高，我們就給他更高的機率。

$$
\begin{align*}
P_{\text{know}}(\mathcal{A}_u) &= \left( p_{\text{know}}(\alpha_u^{(1)}), p_{\text{know}}(\alpha_u^{(2)}), \dots, p_{\text{know}}(\alpha_u^{(|\mathcal{A}_u|)}) \right), \sum_{i=1}^{|\mathcal{A}_u|}{p_{\text{know}}(\alpha_u^{(i)})} = 1 \\
p_{\text{know}}(\alpha_u^{(i)}) &= \frac{\mathcal{N}_i}{\mathcal{N}}
\end{align*}
$$

- $\mathcal{N}_i$ 表示 action $i$ 被 sample 的次數
- $p_{\text{know}}(\alpha_u^{(i)})$ 表示 action $i$ 被 sample 到的機率
- $P_{\text{know}}(\mathcal{A}_u)$ 表示 task $u$ 下，選擇 next action 的機率分布

***Step 5：Agent 決定出第二個 next action 機率分布***

Agent 訓練過程中不會考慮到 local state knowledge，給出 next action 的機率分布。底下列出的數學式只是在說明模型給出的機率分布總和是 $1$。方法是經過一個 SoftMax。

$$
P_{\text{model}}(\mathcal{A}_u) = \left( p_{\text{agent}}(\alpha_u^{(1)}), p_{\text{agent}}(\alpha_u^{(2)}), \dots, p_{\text{agent}}(\alpha_u^{(|\mathcal{A}_u|)}) \right), \sum_{i=1}^{|\mathcal{A}_u|}{p_{\text{agent}}(\alpha_u^{(i)}) = 1}
$$

***Step 6：考慮兩個機率分布決定最終 next action***

最後，我們會把兩者都考慮進來，並選擇其中機率最高的作為最後的 next action。

$$
a_{t+1} = \underset{\alpha_u^{(i)} \in \mathcal{A}_u, 1 \leq i \leq |\mathcal{A}_u|}{\mathrm{argmax}}{\left( \gamma \cdot p_{\text{agent}}(\alpha_u^{(i)}) + (1 - \gamma) \cdot p_{\text{know}}(\alpha_u^{(i)}) \right)}
$$

- $\gamma$ 是一個調整兩者重要性的 hyperparameter

## Results

### 實驗設定

#### Dataset 與環境

實驗做在三個模擬真實世界環境的 datasets 上，分別是 **ALFWorld**, **WebShop** 以及 **ScienceWorld**。

***ALFWorld***

一個在房子當中的環境，大致的目標是要在房子當中移動，並且與各式物件互動。

<center>
<img src="/WKM/rJTc275ER.gif" width=200>
</center>
</br>

> Image from [ALFWorld: Aligning Text and Embodied Environments for Interactive Learning](https://alfworld.github.io/)

以上面的圖片為例，prompt 內容如下。

:::details 開啟 prompt 內容
```
You are in the middle of a room. Looking quickly around you, you see a drawer 2, a shelf 5, a drawer 1, a shelf 4, a sidetable 1, a drawer 5, a shelf 6, a shelf 1, a shelf 9, a cabinet 2, a sofa 1, a cabinet 1, a shelf 3, a cabinet 3, a drawer 3, a shelf 11, a shelf 2, a shelf 10, a dresser 1, a shelf 12, a garbagecan 1, a armchair 1, a cabinet 4, a shelf 7, a shelf 8, a safe 1, and a drawer 4. 

Your task is to: put some vase in safe. 

> go to shelf 6
You arrive at loc 4. On the shelf 6, you see a vase 2.

> take vase 2 from shelf 6
You pick up the vase 2 from the shelf 6.

> go to safe 1
You arrive at loc 3. The safe 1 is closed.

> open safe 1
You open the safe 1. The safe 1 is open. In it, you see a keychain 3.

> put vase 2 in/on safe 1
You won!
```
:::

ALFWorld 當中包含了 unseen tasks，可以用來測試模型的一般性。此外，他的 Reward 設定上只包含 0 和 1，表示是否完成 task。

***WebShop***

一個模擬在網購的網頁虛擬環境。任務大致是要在網頁當中購買特定的物品，而 agent 需要透過查詢、點擊按鈕等操作去達成任務。

<center>
<img src="/WKM/ry7L6ksNC.gif" width=600>
</center>
</br>

> Image from [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://webshop-pnlp.github.io/)

Reward 設定上會是 $[0, 1]$ 之間的數值，表示任務完成率。

***ScienceWorld***

一個 2D 的模擬環境，包含了許多的空間。目標是要在這個空間當中達成指定的任務。例如要燒開水、找到動物、種植水果、觀察物體熔點等等。

<center>
<img src="/WKM/SkHZkxiNC.png" width=600>
</center>
</br>

> Image from [ScienceWorld: Is your Agent Smarter than a 5th Grader?](https://sciworld.apps.allenai.org/)

ScienceWorld 當中包含了 unseen tasks，用來測試模型的一般性。而在 Reward 的設定上會是 $[0, 1]$ 之間的數字表示完成率。

#### 模型架構與 Baseline

LLM 採用的模型有三種，分別是 **Mistral-7B-Instruct-v0.2**、**Gemma-1.1-7B-it** 以及 **Llama-3-8B-Instruct**。

Baseline 比較上分成三個部分，分別是 **prompt-based**、**with rejected-trajectories** 以及 **knowledge-augumented planning method**。

***prompt-based***

這個部分的做法是透過 prompt engineering 去提升 LLM 輸出結果的品質，包含了 **REACT** 與 **Reflection**。

:::info
**REACT** 是將 Chain-of-Thought(CoT) 引入 prompt 當中，讓 agent 在規劃的過程當中會歷經 **思考(Thought)**、**行動(Action)**、**觀察(Observation)** 三個步驟。此篇 paper 為了比較的公平性採用 one-shot prompting。

**Reflection** 則是會讓 agent 規劃後當中根據過去的經驗給予 feedback，並且依據 feedback 重新規劃出更好的方案。此篇 paper 為了比較的公平性採用 one-shot prompting。
:::

由於這部分是針對 prompt 去做設計，因此也延伸出如 REACT-style prompt format。此篇 paper 採用的設計也幾乎基於這個。

***with rejected-trajectories***

訓練中包含 rejected trajectories 並不是這篇論文的首創，其他如 **NAT** 以及 **ETO** 也都有這樣的設計。

:::info
**NAT** 是在 fine-tune 的過程當中引入不同的 prompt 設計，使模型產出 rejected-trajectories。此篇 paper 當中使用 LoRA 方式去 fine-tune。

**ETO** 會在訓練過程中先訓練在 expert trajectory，接下來把訓練中失敗的 trajectories 當作 rejected-trajectories 進一步訓練。訓練方式採用 DPO。
:::

***knowledge-augumented planning method***

這種類型的模型也同樣會在模型決策時提供其他的知識。這裡選擇了 **KnowAgent**。

:::info
**KnowAgent** 在決策的過程當中會先將 action 的知識提取出來，包含 action 的定義與使用方式等，接下來構建出 planning path。執行的過程當中會透過 prompt 等方式指引 agent 持續待在正確的 planning path 上。
:::

#### Training and Inference Setups

訓練當中使用 LlamaFactory 去訓練模型，都是採用 LoRA。訓練使用了 8 張 NVIDIA V100 32G GPUs 訓練 12 小時。更詳細的參數設定請參考論文。

:::details 部分參數細節
- Learning Rate: `1e-4`
- Sequence Length: `2048`
- Training Epoch: `3`
- Batch Size: `32`
- Optimizer: `AnamW with consine learning scheduler`
- $\mathcal{N}$: `3000`
- $\gamma = \{0.4, 0.5, 0.7\}$
:::

### 實驗結果

<center>
<img src="/WKM/rk-NseiN0.png" height=400>
</center>
</br>

> Image from [Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. (2024)](https://arxiv.org/pdf/2405.14205)

首先可以看到無論採用的 Backbone 為何，WKM 都可以得到更出色的結果。這樣的結果無論是在哪個 dataset 當中，或是有沒有看過的 task 都是更好的。

此外，除了 GPT4 以外，普遍而言單純用 prompt-based 方法都會得到與其他方法相距甚遠的結果。

同樣是給予額外知識的 KnowAgent，可以觀察到 KnowAgent 面對沒看過的 task 上得到的結果比起看過的 task 都會相差蠻多。從這裡可以看出 WKM 是具有足夠的一般性。

<center>
<img src="/WKM/BJbryWsVA.png" width=600>
</center>
</br>

> Image from [Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. (2024)](https://arxiv.org/pdf/2405.14205)

接著是 Ablation Study，比較 Global Task Knowledge 與 Local State Knowledge 分別帶來的影響。這裡以 Mistral-7B 作為 Backbone 實驗。

可以觀察到無論是加上 state knowledge 或是 task knowledge 都可以對結果帶來好的影響，而兩者結合也可以帶來更好的結果。其中又以 task knowledge 帶來的正面效益為最大。無論在哪個 dataset 當中都可以觀察到相同的趨勢。

如果更進一步觀察可以發現到 state knowledge 在 seen data 上面會有更多的正面影響。而 unseen data 則比較沒有顯著的差異。作者認為這樣的結果是因為 state knowledge 是從 training 階段製作的 knowledge base 中擷取的，這也許導致了 state knowledge 減少了一般性。

<center>
<img src="/WKM/BkofEZo4C.png" width=400>
</center>
</br>

> Image from [Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. (2024)](https://arxiv.org/pdf/2405.14205)

接下來作者想確認 WKM 是否真的能避免無謂的 trial-and-error，因此計算了每個方法在各個環境當中執行的 step 數量。其中 ALFWorld 的 step 上限為 40，WebShop 為 10，而 ScienceWorld 根據 task 不同有不同上限，但平均落在 40 上下。

可以看到無論在哪個 dataset 底下，無論是不是有看過的 task，WKM 都可以用更少的平均 steps 達成。可以認為 WKM 確實可以避免無謂的 trail-and-error。

<center>
<img src="/WKM/SkRXwZoN0.png" width=300>
</center>
</br>

> Image from [Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. (2024)](https://arxiv.org/pdf/2405.14205)

而由於 ALFWorld 可以接收錯誤的 action，因此這裡也統計了一個 trajectory 平均出現怪異 action 的頻率。

可以發現到無論是哪個環境當中 WKM 都有更低的 Hallucinatory Action Rates。尤其在 Unseen task 當中可以降低許多。可以推斷 WKM 可以降低模型產生怪異 action 的發生機率，並且可以很好適應到 unseen data 上。

<center>
<img src="/WKM/r1bacWjVC.png" width=300>
</center>
</br>

> Image from [Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. (2024)](https://arxiv.org/pdf/2405.14205)

在前面的實驗當中，agent 跟 world knowledge model 都是使用相同的 Backbone，作者嘗試使用不同 Backbone 去觀察。這裡固定 knowledge model 使用 Mistral-7B，得到的結果如上。

可以觀察到當 agent model 比 knowledge model 弱時，結果會很差。但是反過來當 agent 比起 knowledge model 還強時，得到的結果有大幅度的成長。

作者猜想這是因為比較弱的模型包含了強的模型當中缺乏的資訊所帶來的結果。

<center>
<img src="/WKM/BJCp2bsEA.png" width=300>
</center>
</br>

> Image from [Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. (2024)](https://arxiv.org/pdf/2405.14205)

接著，作者嘗試混合各個 datasets 的資訊，然後只訓練一個 WKM，搭配各個 agent model。作者發現到這樣的作法除了能大幅超越其他的方法外，甚至可以比沒有混合的 WKM 還要更亮眼。

這個結果展現了單一模型可以帶來相當強大的廣泛性。現在我們的實驗只做在單一的 WKM 上，作者也大膽預測，如果連 agent model 也是單一的，可以帶來更強大的效益，這也許會是前往 AGI 的關鍵。不過此篇 paper 中並沒有呈現這樣的結果。

<center>
<img src="/WKM/BygdRZjVC.png" width=300>
</center>
</br>

> Image from [Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. (2024)](https://arxiv.org/pdf/2405.14205)

最後，在前面我們設計 local state knowledge 時決定製作一個 knowledge base 而非直接加入 prompt 的理由是因為會使模型感到困惑。這裡做了相關的實驗。

如果我們直接把 state knowledge 加入 prompt 當中，也就是圖中的 **Explicit State**，可以發現到結果甚至會比起沒有 local state knowledge 更差。也因此確認了上面的結論。以較隱晦的方式傳遞 local state knowledge 會是更好的做法。

## Contribution

- 提出 WKM，結合了 LLM 與 agent model planning
- 降低無腦 trial-and-error 的發生
- 避免錯誤的 action
- 面對 unseen task 有更好的一般性
- 發現弱模型可以引導強模型有更好的效益
- 發現混合 dataset 製作的 WKM 模型搭配其他 agent 可以達到更加效益

不過同時也存在幾個 limitation。

- 他們仍然無法說明 LLM 實際上學到了怎樣的 world knowledge
- 目前只能在文字訊息上處理，還未能做到 multi-modal
- WKM 無法及時針對環境與 agent 給出的 feedback 做調整
    - local state knowledge 仰賴 training 階段的知識
    - global task knowledge 在 inference 階段也是固定的
- 多了 WKM 或產生額外的 overhead，大約會比沒使用 WKM 的方法多 2.5 倍的時間花費
