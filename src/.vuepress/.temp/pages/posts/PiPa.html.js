export const data = JSON.parse("{\"key\":\"v-d4413c4c\",\"path\":\"/posts/PiPa.html\",\"title\":\"PiPa: Pixel- and Patch-wise Self-supervised Learning for Domain Adaptative Semantic Segmentation\",\"lang\":\"en-US\",\"frontmatter\":{\"date\":\"2024-05-18T00:00:00.000Z\",\"category\":[\"Note\"],\"tag\":[\"Paper Read\",\"Domain Adaptation\",\"Computer Vision\",\"ACM Multimedia\"],\"description\":\"PiPa: Pixel- and Patch-wise Self-supervised Learning for Domain Adaptative Semantic Segmentation Basic Information Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua 2022 ACM Multimedia 問題描述 這一篇與過去看過的 DACS, ProDA, DAFormer, HRDA 同樣都是以 Unsupervised 的方式解決 Semantic Segmentationb 的 Domain Adaptation問題。\",\"head\":[[\"meta\",{\"property\":\"og:url\",\"content\":\"https://mister-hope.github.io/KoiosBlog/posts/PiPa.html\"}],[\"meta\",{\"property\":\"og:site_name\",\"content\":\"Koios Blog\"}],[\"meta\",{\"property\":\"og:title\",\"content\":\"PiPa: Pixel- and Patch-wise Self-supervised Learning for Domain Adaptative Semantic Segmentation\"}],[\"meta\",{\"property\":\"og:description\",\"content\":\"PiPa: Pixel- and Patch-wise Self-supervised Learning for Domain Adaptative Semantic Segmentation Basic Information Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua 2022 ACM Multimedia 問題描述 這一篇與過去看過的 DACS, ProDA, DAFormer, HRDA 同樣都是以 Unsupervised 的方式解決 Semantic Segmentationb 的 Domain Adaptation問題。\"}],[\"meta\",{\"property\":\"og:type\",\"content\":\"article\"}],[\"meta\",{\"property\":\"og:locale\",\"content\":\"en-US\"}],[\"meta\",{\"property\":\"og:updated_time\",\"content\":\"2024-05-18T09:13:59.000Z\"}],[\"meta\",{\"property\":\"article:tag\",\"content\":\"Paper Read\"}],[\"meta\",{\"property\":\"article:tag\",\"content\":\"Domain Adaptation\"}],[\"meta\",{\"property\":\"article:tag\",\"content\":\"Computer Vision\"}],[\"meta\",{\"property\":\"article:tag\",\"content\":\"ACM Multimedia\"}],[\"meta\",{\"property\":\"article:published_time\",\"content\":\"2024-05-18T00:00:00.000Z\"}],[\"meta\",{\"property\":\"article:modified_time\",\"content\":\"2024-05-18T09:13:59.000Z\"}],[\"script\",{\"type\":\"application/ld+json\"},\"{\\\"@context\\\":\\\"https://schema.org\\\",\\\"@type\\\":\\\"Article\\\",\\\"headline\\\":\\\"PiPa: Pixel- and Patch-wise Self-supervised Learning for Domain Adaptative Semantic Segmentation\\\",\\\"image\\\":[\\\"\\\"],\\\"datePublished\\\":\\\"2024-05-18T00:00:00.000Z\\\",\\\"dateModified\\\":\\\"2024-05-18T09:13:59.000Z\\\",\\\"author\\\":[]}\"]]},\"headers\":[{\"level\":2,\"title\":\"Basic Information\",\"slug\":\"basic-information\",\"link\":\"#basic-information\",\"children\":[]},{\"level\":2,\"title\":\"問題描述\",\"slug\":\"問題描述\",\"link\":\"#問題描述\",\"children\":[]},{\"level\":2,\"title\":\"Related Works\",\"slug\":\"related-works\",\"link\":\"#related-works\",\"children\":[]},{\"level\":2,\"title\":\"Methodology\",\"slug\":\"methodology\",\"link\":\"#methodology\",\"children\":[{\"level\":3,\"title\":\"基本的 UDA Loss 設定\",\"slug\":\"基本的-uda-loss-設定\",\"link\":\"#基本的-uda-loss-設定\",\"children\":[]},{\"level\":3,\"title\":\"Pixel-wise Contrastive Learning\",\"slug\":\"pixel-wise-contrastive-learning\",\"link\":\"#pixel-wise-contrastive-learning\",\"children\":[]},{\"level\":3,\"title\":\"Patch-wise Contrastive Learning\",\"slug\":\"patch-wise-contrastive-learning\",\"link\":\"#patch-wise-contrastive-learning\",\"children\":[]},{\"level\":3,\"title\":\"結合\",\"slug\":\"結合\",\"link\":\"#結合\",\"children\":[]}]},{\"level\":2,\"title\":\"Results\",\"slug\":\"results\",\"link\":\"#results\",\"children\":[{\"level\":3,\"title\":\"實驗設定\",\"slug\":\"實驗設定\",\"link\":\"#實驗設定\",\"children\":[]},{\"level\":3,\"title\":\"Quantitative Comparison\",\"slug\":\"quantitative-comparison\",\"link\":\"#quantitative-comparison\",\"children\":[]},{\"level\":3,\"title\":\"Qualitative Results\",\"slug\":\"qualitative-results\",\"link\":\"#qualitative-results\",\"children\":[]},{\"level\":3,\"title\":\"Ablation Studies\",\"slug\":\"ablation-studies\",\"link\":\"#ablation-studies\",\"children\":[]}]},{\"level\":2,\"title\":\"Contribution\",\"slug\":\"contribution\",\"link\":\"#contribution\",\"children\":[]}],\"git\":{\"createdTime\":1716023639000,\"updatedTime\":1716023639000,\"contributors\":[{\"name\":\"Koios\",\"email\":\"ken1357924681010@gmail.com\",\"commits\":1}]},\"readingTime\":{\"minutes\":6.86,\"words\":2059},\"filePathRelative\":\"posts/PiPa.md\",\"localizedDate\":\"May 18, 2024\",\"excerpt\":\"<h1> PiPa: Pixel- and Patch-wise Self-supervised Learning for Domain Adaptative Semantic Segmentation</h1>\\n<h2> Basic Information</h2>\\n<ul>\\n<li>Mu Chen, Zhedong Zheng, Yi Yang, Tat-Seng Chua</li>\\n<li>2022 ACM Multimedia</li>\\n</ul>\\n<h2> 問題描述</h2>\\n<p>這一篇與過去看過的 <a href=\\\"https://koios1143.github.io/PaperBlog/posts/DACS.html\\\" target=\\\"_blank\\\" rel=\\\"noopener noreferrer\\\">DACS</a>, <a href=\\\"https://koios1143.github.io/PaperBlog/posts/ProDA.html\\\" target=\\\"_blank\\\" rel=\\\"noopener noreferrer\\\">ProDA</a>, <a href=\\\"https://koios1143.github.io/PaperBlog/posts/DAFormer.html\\\" target=\\\"_blank\\\" rel=\\\"noopener noreferrer\\\">DAFormer</a>, <a href=\\\"https://koios1143.github.io/PaperBlog/posts/HRDA.html\\\" target=\\\"_blank\\\" rel=\\\"noopener noreferrer\\\">HRDA</a> 同樣都是以 Unsupervised 的方式解決 Semantic Segmentationb 的 Domain Adaptation問題。</p>\",\"autoDesc\":true}")
