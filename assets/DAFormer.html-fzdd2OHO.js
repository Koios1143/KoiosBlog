const e=JSON.parse('{"key":"v-6fdb6976","path":"/posts/DAFormer.html","title":"DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation","lang":"en-US","frontmatter":{"date":"2024-03-11T00:00:00.000Z","category":["Note"],"tag":["Paper Read","Domain Adaption","Computer Vision","CVPR"],"description":"DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation Basic Information Lukas Hoyer, Dengxin Dai, Luc Van Gool @ ETH Zurich &amp; MPI for Informatics 2022 CVPR Image from Lukas Hoyer, Dengxin Dai, Luc Van Gool (2022)","head":[["meta",{"property":"og:url","content":"https://mister-hope.github.io/KoiosBlog/posts/DAFormer.html"}],["meta",{"property":"og:site_name","content":"Koios Blog"}],["meta",{"property":"og:title","content":"DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation"}],["meta",{"property":"og:description","content":"DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation Basic Information Lukas Hoyer, Dengxin Dai, Luc Van Gool @ ETH Zurich &amp; MPI for Informatics 2022 CVPR Image from Lukas Hoyer, Dengxin Dai, Luc Van Gool (2022)"}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"en-US"}],["meta",{"property":"og:updated_time","content":"2024-03-14T10:33:38.000Z"}],["meta",{"property":"article:tag","content":"Paper Read"}],["meta",{"property":"article:tag","content":"Domain Adaption"}],["meta",{"property":"article:tag","content":"Computer Vision"}],["meta",{"property":"article:tag","content":"CVPR"}],["meta",{"property":"article:published_time","content":"2024-03-11T00:00:00.000Z"}],["meta",{"property":"article:modified_time","content":"2024-03-14T10:33:38.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation\\",\\"image\\":[\\"\\"],\\"datePublished\\":\\"2024-03-11T00:00:00.000Z\\",\\"dateModified\\":\\"2024-03-14T10:33:38.000Z\\",\\"author\\":[]}"]]},"headers":[{"level":2,"title":"Basic Information","slug":"basic-information","link":"#basic-information","children":[]},{"level":2,"title":"問題描述","slug":"問題描述","link":"#問題描述","children":[]},{"level":2,"title":"Related Works","slug":"related-works","link":"#related-works","children":[]},{"level":2,"title":"Methodology","slug":"methodology","link":"#methodology","children":[{"level":3,"title":"Self training for UDA","slug":"self-training-for-uda","link":"#self-training-for-uda","children":[]},{"level":3,"title":"DAFormer Network Architecture","slug":"daformer-network-architecture","link":"#daformer-network-architecture","children":[]},{"level":3,"title":"Rare Class Sampling (RCS)","slug":"rare-class-sampling-rcs","link":"#rare-class-sampling-rcs","children":[]},{"level":3,"title":"Thing-Class ImageNet Feature Distance (FD)","slug":"thing-class-imagenet-feature-distance-fd","link":"#thing-class-imagenet-feature-distance-fd","children":[]},{"level":3,"title":"Learning Rate Warmup for UDA","slug":"learning-rate-warmup-for-uda","link":"#learning-rate-warmup-for-uda","children":[]}]},{"level":2,"title":"Results","slug":"results","link":"#results","children":[{"level":3,"title":"實驗設定","slug":"實驗設定","link":"#實驗設定","children":[]},{"level":3,"title":"Summary","slug":"summary","link":"#summary","children":[]},{"level":3,"title":"Learning Rate Warmup","slug":"learning-rate-warmup","link":"#learning-rate-warmup","children":[]},{"level":3,"title":"Rare Class Sampling (RCS)","slug":"rare-class-sampling-rcs-1","link":"#rare-class-sampling-rcs-1","children":[]},{"level":3,"title":"Thing-Class ImageNet Feature Distance (FD)","slug":"thing-class-imagenet-feature-distance-fd-1","link":"#thing-class-imagenet-feature-distance-fd-1","children":[]},{"level":3,"title":"DAFormer Decoder","slug":"daformer-decoder","link":"#daformer-decoder","children":[]}]},{"level":2,"title":"Contribution","slug":"contribution","link":"#contribution","children":[]},{"level":2,"title":"值得一看的文章們","slug":"值得一看的文章們","link":"#值得一看的文章們","children":[]}],"git":{"createdTime":1710140481000,"updatedTime":1710412418000,"contributors":[{"name":"Koios","email":"ken1357924681010@gmail.com","commits":2}]},"readingTime":{"minutes":12.3,"words":3689},"filePathRelative":"posts/DAFormer.md","localizedDate":"March 11, 2024","excerpt":"<h1> DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation</h1>\\n<h2> Basic Information</h2>\\n<ul>\\n<li>Lukas Hoyer, Dengxin Dai, Luc Van Gool @ ETH Zurich &amp; MPI for Informatics</li>\\n<li>2022 CVPR</li>\\n</ul>\\n\\n<br>\\n<blockquote>\\n<p>Image from <a href=\\"https://arxiv.org/abs/2111.14887\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\">Lukas Hoyer, Dengxin Dai, Luc Van Gool (2022)</a></p>\\n</blockquote>","autoDesc":true}');export{e as data};
