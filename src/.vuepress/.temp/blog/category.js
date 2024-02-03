export const categoryMap = {"category":{"/":{"path":"/category/","map":{"Feedbacks":{"path":"/category/feedbacks/","keys":["v-620a6165"]},"Note":{"path":"/category/note/","keys":["v-5b18c8c4","v-c0336012"]}}}},"tag":{"/":{"path":"/tag/","map":{"TSMC":{"path":"/tag/tsmc/","keys":["v-620a6165"]},"CareerHack":{"path":"/tag/careerhack/","keys":["v-620a6165"]},"Paper Read":{"path":"/tag/paper-read/","keys":["v-5b18c8c4","v-c0336012"]},"Domain Adaption":{"path":"/tag/domain-adaption/","keys":["v-c0336012"]},"Computer Vision":{"path":"/tag/computer-vision/","keys":["v-c0336012"]},"WACV":{"path":"/tag/wacv/","keys":["v-c0336012"]},"Reinforcement Learning":{"path":"/tag/reinforcement-learning/","keys":["v-5b18c8c4"]},"ICLR":{"path":"/tag/iclr/","keys":["v-5b18c8c4"]}}}}};

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept();
  if (__VUE_HMR_RUNTIME__.updateBlogCategory)
    __VUE_HMR_RUNTIME__.updateBlogCategory(categoryMap);
}

if (import.meta.hot)
  import.meta.hot.accept(({ categoryMap }) => {
    __VUE_HMR_RUNTIME__.updateBlogCategory(categoryMap);
  });


