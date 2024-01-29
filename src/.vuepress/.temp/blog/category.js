export const categoryMap = {"category":{"/":{"path":"/category/","map":{"Feedbacks":{"path":"/category/feedbacks/","keys":["v-6e6058d4","v-620a6165"]},"Note":{"path":"/category/note/","keys":["v-c0336012"]}}}},"tag":{"/":{"path":"/tag/","map":{"TSMC":{"path":"/tag/tsmc/","keys":["v-620a6165"]},"CareerHack":{"path":"/tag/careerhack/","keys":["v-620a6165"]},"Paper Read":{"path":"/tag/paper-read/","keys":["v-c0336012"]},"Domain Adaption":{"path":"/tag/domain-adaption/","keys":["v-c0336012"]},"Computer Vision":{"path":"/tag/computer-vision/","keys":["v-c0336012"]},"WACV":{"path":"/tag/wacv/","keys":["v-c0336012"]},"Interview":{"path":"/tag/interview/","keys":["v-6e6058d4"]},"Google":{"path":"/tag/google/","keys":["v-6e6058d4"]}}}}};

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept();
  if (__VUE_HMR_RUNTIME__.updateBlogCategory)
    __VUE_HMR_RUNTIME__.updateBlogCategory(categoryMap);
}

if (import.meta.hot)
  import.meta.hot.accept(({ categoryMap }) => {
    __VUE_HMR_RUNTIME__.updateBlogCategory(categoryMap);
  });


