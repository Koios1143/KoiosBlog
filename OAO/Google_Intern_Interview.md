---
date: 2024-01-29
category:
  - Feedbacks
tag:
  - Interview
  - Google
---

# 2024 Google SWE Intern Interview 心得

大概在今年的一月初投了履歷到 Google SWE，希望暑假可以去累積累積經驗，大概在幾天後就收到要填寫面試時間的通知信件，三周後收到了面試的通知。

在面試前收到的信件都會很貼心地說明，也有提供一些線上的資源協助你去了解如何準備面試，甚至也可以直接跟對方來往信件去確認，我覺得這一點是還蠻不錯的。

面試的流程大致上會分成兩場，兩場都是針對算法題目，不太會對簡歷或是個人背景去做提問，結束前也會有一小段可以任意問問題的時間。

兩場面試基本上會被放在同一天當中，但是會在不同時段。以我自己來說，我分別是在早上跟下午各一場。一場時間大概是 45 分鐘。

:::info
有趣的是，我們並不會在一般的 IDE 或是 Text Editor 上寫 code，而是 Google 會提供 Google Doc 在上面寫，而你的面試官也可以一起編輯，就像是實體面試的時候"白板"這個工具。
:::

## 第一場面試

### Warnup

#### 題目敘述

給定 $N \times M$ 的地圖，其中

- `#` 表示 wall
- `.` 表示 road
- `a` 表示起點
- `A` 表示終點

每次只能選上下左右一個方向去移動，問從 `a` 是否有辦法走到 `A`。

#### 思路

從起點做 BFS，看能不能走到終點

```c++
bool FindExit(vector<vector<char>> maps){
    // Find start point
    int rows = maps.size();
    int cols = maps[0].size();
    pair<int, int> start;
    for(int i=0 ; i<rows ; i++){
        for(int j=0 ; j<cols ; j++){
            if(maps[i][j] == 'a'){
                start = make_pair(i, j);
            }
        }
    }

    // BFS from start point
    // initialize vis array
    bool vis[rows][cols];
    for(int i=0 ; i<rows ; i++){
        for(int j=0 ; j<cols ; j++){
            vis[i][j] = false;
        }
    }
    // put start point
    queue<pair<int, int>> q;
    q.emplace(start);
    vis[start.first][start.second] = true;
    // For implementation simplicity
    int dx[] = {-1, 0, 1, 0};
    int dy[] = {0, -1, 0, 1};
    // BFS
    while(!q.empty()){
        int x = q.front().first;
        int y = q.front().second;
        q.pop();

        // check whether is answer or not
        if(maps[x][y] == 'A'){
            return true;
        }

        for(int i=0 ; i<4 ; i++){
            int nx = x + dx[i];
            int ny = y + dy[i];
            // check whether nx and ny is valid
            if(nx < 0 || nx >= rows || ny < 0 || ny >= cols) continue;
            // check whether visited
            if(vis[nx][ny]) continue;
            // check wall
            if(maps[nx][ny] == '#') continue;
            // otherwise, road or end point
            q.emplace(nx, ny);
            vis[nx][ny] = true;
        }
    }
    // Cannot reach the end point
    return false;
}
```

### Follow-up

#### 題目敘述

現在把 `a` 當成是一台車，又再多一台車 `b`。

`a` 一樣要到終點 `A`，`b` 則是要到終點 `B`。

兩台車行走的時間可以隨意(也就是說，不需要一起移動一格)。

問是否可以在不碰撞的情況下兩台車都到終點。

#### 思路

兩台車並不需要同時去運行，所以可以等某一台車通過之後再回到原本的位置。

:::tip
舉例來說

```
#B#
a.A
#b#
```

這個 case 下可以先等 `a` 走到 `A` 之後 `b` 再移動，所以不會有碰撞出現

不過底下這個 case 就無法避免碰撞而造成無解。

```
####
aBAb
####
```
:::

這一題面試當下並沒有好的想法，最後也並沒有實際實作出來。

不過大抵還是會使用 BFS 去解。

### 第一場面試小結

當初填表單的時候好像不小心填到一場英文一場中文，所以那一場我是用英文面試的，還是有一定程度影響到輸出效果www

第一場面試下來自己覺得有盡可能去溝通，也有先好好確認過題目的條件，不過 Follow-up 沒有找到一個好的想法覺得很可惜。即便是在面試結束之後想這個題目目前也還沒有什麼好的想法。

我不太確定時間複雜度的部分是不是需要自己說明，面試官當時沒有提問到，我也就沒有額外補充說明這一塊，面試結束後覺得應該還是需要提的，因此下一場面試就有比較積極去說明自己的解法時間複雜度如何。

## 第二場面試

經過了午餐的小休息之後，接下來就是第二場的面試。

### Warn-up

#### 題目敘述

給三個整數，找出中位數

#### 思路

只有三個數字，簡單條件判斷就可以出來了。

```c++
int FindMedium(int a, int b, int c){
    if(a <= b && b <= c) return b;
    if(a <= c && c <= b) return c;
    if(b <= a && a <= c) return a;
    if(b <= c && c <= a) return c;
    if(c <= b && b <= a) return b;
    if(c <= a && a <= b) return a;
}
```

### Question 1

現在有 10 個人要安排會議日期，每個人都有一些 unavaliable 的日期，詢問有哪些大家都可行的日期。

每個不能的日期會用下列的 struct 表示

```c++
struct block{
    int personId; // 表示是哪一個人的日期
    int startDay; // 表示不能的開始時間
    int endDay; // 表示不能的結束時間
};
```

整體安排的日期為 `0`~`最大的 endDay + 1`

#### 思路 1

對於每一天去看看有沒有任何不能的時段有重疊到他，沒有的話就放進答案，否則繼續看下一個。

假設總共 $M$ 個 block，安排的日期最多 $N$ 天。每一天都需要去對 $M$ 個 block 檢查，檢查一個 block 是 $O(1)$，因此總共時間複雜度是 $O(NM)$。

#### 思路 2

上面的解法也許有些暴力，況且 $N$ 也有可能很大。

考慮到排序往往會帶來不錯的性質，這裡先依照 `startDay` 由小到大排序，相同時則依照 `endDay` 由小到大。

對於相鄰的兩個時段，如果說他們是有重疊的，那我們可以把它們合併在一起。否則就表示從現在的 `endDay + 1` 到下一個 `startDay - 1` 之間都會是可以的日期。

如此一來對於每個 block 我們就可以只去考慮 `startDay` 以及 `endDay`。

排序採用 merge sort 可以得到 $O(MlogM)$ 的時間複雜度。

看過一個 block 以及處理合併都只需要 $O(1)$，因此這一個步驟對每個 block 總共需要 $O(M)$。

對於可以的時間需要逐一加入到答案當中，因此會需要額外 $O(N)$ 的時間處理插入。

總時間複雜度來到 $O(MlogM + N + M)$

```c++
bool cmp(struct block p, struct block q){
    if(p.startDay != q.startDay) return p.startDay < q.startDay;
    else return p.endDay < q.endDay;
}

vector<int> FindAvaliable(vector<block> v){
    vector<int> ans;
    // sort the blocks
    sort(v.begin(), v.end(), cmp);
    // [0, v[0].startDay) should also be avaliable
    for(int i=0 ; i<v[0].startDay ; i++){
        ans.emplace_back(i);
    }
    // Look over the whole blocks
    for(int i=0 ; i<v.size() ; i++){
        // check whether there's next block
        if(i + 1 < v.size()){
            // check whether we can merge two blocks
            if(v[i+1].endDay <= v[i].startDay){
                // merge result to next block
                v[i+1].startDay = v[i].startDay;
            }
            else{
                // otherwise, add (v[i].endDay, v[i+1].startDay) to answer
                for(int i=v[i].endDay+1 ; i<v[i+1].startDay ; i++){
                    ans.emplace_back(i);
                }
            }
        }
        else{
            // No more blocks, we should add v[i].endDay + 1 to answer
            ans.emplace_back(v[i].endDay + 1);
        }
    }
    return ans;
}
```

### Question 1 - Follow-up

現在多給你一個參數 `k`，請把所有 `<=k` 個人 unavaliable 的日期都列出來。

> 保證不會有一個人自己填到重疊的時段

#### 思路 1

同樣可以採取暴力的做法，去看過每一個時間，check 是否 unavaliable 的人數是 `<=k`。

每一個時間需要花 $O(M)$ 去確認，總共有 $N$ 個時間，總時間複雜度 $O(NM)$。

#### 思路 2

不妨把時段畫成圖，像是底下的樣子。

```
---------
 -----
   -------
```

其實是哪一個人佔據了哪一個時段我們並不在乎，因為已經事先知道不會有人自己填到跟自己重疊的時段，所以我們只在乎**美個時段有多少個重疊**。

當然，我們可以像 `思路 1` 一樣去算每一個時間點的人數，但是實際上人數**只會在那些開始以及結束的點被更新**。

- 進到 `startDay` 人數就 +1
- 進到 `endDat` 人數就 -1

所以我們可以單純的去把所有的時間標記上他是 `start` 或是 `end`，然後拿去 sort。

如此一來我們就能夠知道每一個時間段 unavaliable 的人數有多少了。

製作出 `Days` 需要花 $O(M)$ 的時間。排序需要 $O(MlogM)$ 的時間。最後只需要掃過每一個 block 就可以知道有哪些時段需要插入，時間是 $O(M)$。記得也要考慮插入答案的 $O(N)$。

總時間複雜度落在 $O(M + MlogM + M + N)$。

### 第二場面試小結

這一場的發揮還算不錯，也頗主動去跟面試官互動，整體感覺下來很棒。

## 總結

Google 的面試往往要求的是跟面試官之間的互動，以及有沒有辦法好好說明與想到解決方案。

面試前沒有做什麼準備，實際上面試起來也覺得狀態還不錯，不過面試官主動的互動有比想像中少一些，像是時間複雜度的部分他們並不會直接詢問，說出一個解法之後似乎也不會說「有沒有更好的解法呢」之類的，而是說「如果你只想到這樣的話也可以開始寫 code，或是是著想想看有沒有其他想法」。

一些小細節上跟預期有些落差，不過整體上我覺得是還蠻不錯的體驗。題目其實並不會過難，真的就是期待你可以現場去好好思考跟說明你的想法，把面試官當成是你的隊友去討論討論作法。

最重要的也許是心態上的調整。面對面試如果過於緊張往往會無法順利面對突發狀況，需要好好整理心態，去面對未知的題目跟挑戰。

期待後續的消息 ouob。