const e=JSON.parse('{"key":"v-ad1b5b16","path":"/posts/WKM.html","title":"Agent Planning with World Knowledge Model","lang":"en-US","frontmatter":{"date":"2024-06-03T00:00:00.000Z","category":["Note"],"tag":["Paper Read","Reinforcement Learning","LLM"],"description":"Agent Planning with World Knowledge Model Basic Information 2024/05/13 發布 (尚未正式於 Conf. 發表) Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. @ Zhejiang University, National University of Singapore, Alibaba Group 問題描述 近年來大型語言模型(LLM)在許多自然語言處理的問題有很快速的成長，而近期開始出現一些使用 LLM 作為 agent model 來處理物理環境中的規劃問題。然而由於當前 SOTA 的 LLM 幾乎都是 autoregressive model，模型實際上會做的事情是去預測下一個 output token 要是什麼，實際上他們對於物理環境是沒有任何理解的。","head":[["meta",{"property":"og:url","content":"https://mister-hope.github.io/posts/WKM.html"}],["meta",{"property":"og:site_name","content":"Koios Blog"}],["meta",{"property":"og:title","content":"Agent Planning with World Knowledge Model"}],["meta",{"property":"og:description","content":"Agent Planning with World Knowledge Model Basic Information 2024/05/13 發布 (尚未正式於 Conf. 發表) Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. @ Zhejiang University, National University of Singapore, Alibaba Group 問題描述 近年來大型語言模型(LLM)在許多自然語言處理的問題有很快速的成長，而近期開始出現一些使用 LLM 作為 agent model 來處理物理環境中的規劃問題。然而由於當前 SOTA 的 LLM 幾乎都是 autoregressive model，模型實際上會做的事情是去預測下一個 output token 要是什麼，實際上他們對於物理環境是沒有任何理解的。"}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"en-US"}],["meta",{"property":"og:updated_time","content":"2024-06-04T00:24:35.000Z"}],["meta",{"property":"article:tag","content":"Paper Read"}],["meta",{"property":"article:tag","content":"Reinforcement Learning"}],["meta",{"property":"article:tag","content":"LLM"}],["meta",{"property":"article:published_time","content":"2024-06-03T00:00:00.000Z"}],["meta",{"property":"article:modified_time","content":"2024-06-04T00:24:35.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"Agent Planning with World Knowledge Model\\",\\"image\\":[\\"\\"],\\"datePublished\\":\\"2024-06-03T00:00:00.000Z\\",\\"dateModified\\":\\"2024-06-04T00:24:35.000Z\\",\\"author\\":[]}"]]},"headers":[{"level":2,"title":"Basic Information","slug":"basic-information","link":"#basic-information","children":[]},{"level":2,"title":"問題描述","slug":"問題描述","link":"#問題描述","children":[]},{"level":2,"title":"Related Works","slug":"related-works","link":"#related-works","children":[{"level":3,"title":"Knowledge Augumented Agent Planning","slug":"knowledge-augumented-agent-planning","link":"#knowledge-augumented-agent-planning","children":[]}]},{"level":2,"title":"Methodology","slug":"methodology","link":"#methodology","children":[{"level":3,"title":"Preliminaries","slug":"preliminaries","link":"#preliminaries","children":[]},{"level":3,"title":"整體流程","slug":"整體流程","link":"#整體流程","children":[]},{"level":3,"title":"Task Knowledge Synthesis","slug":"task-knowledge-synthesis","link":"#task-knowledge-synthesis","children":[]},{"level":3,"title":"State Knowledge Summarization","slug":"state-knowledge-summarization","link":"#state-knowledge-summarization","children":[]},{"level":3,"title":"Model Training","slug":"model-training","link":"#model-training","children":[]},{"level":3,"title":"Agent Planning with World Knowledge Model","slug":"agent-planning-with-world-knowledge-model-1","link":"#agent-planning-with-world-knowledge-model-1","children":[]}]},{"level":2,"title":"Results","slug":"results","link":"#results","children":[{"level":3,"title":"實驗設定","slug":"實驗設定","link":"#實驗設定","children":[]},{"level":3,"title":"實驗結果","slug":"實驗結果","link":"#實驗結果","children":[]}]},{"level":2,"title":"Contribution","slug":"contribution","link":"#contribution","children":[]}],"git":{"createdTime":1717408994000,"updatedTime":1717460675000,"contributors":[{"name":"Koios","email":"ken1357924681010@gmail.com","commits":2}]},"readingTime":{"minutes":19.62,"words":5885},"filePathRelative":"posts/WKM.md","localizedDate":"June 3, 2024","excerpt":"<h1> Agent Planning with World Knowledge Model</h1>\\n<h2> Basic Information</h2>\\n<ul>\\n<li>2024/05/13 發布 (尚未正式於 Conf. 發表)</li>\\n<li>Shuofei Qiao, Runnan Fang, Ningyu Zhang et al. @ Zhejiang University, National University of Singapore, Alibaba Group</li>\\n</ul>\\n<h2> 問題描述</h2>\\n<p>近年來大型語言模型(LLM)在許多自然語言處理的問題有很快速的成長，而近期開始出現一些使用 LLM 作為 agent model 來處理物理環境中的規劃問題。然而由於當前 SOTA 的 LLM 幾乎都是 autoregressive model，模型實際上會做的事情是去預測下一個 output token 要是什麼，實際上他們對於物理環境是沒有任何理解的。</p>","autoDesc":true}');export{e as data};
