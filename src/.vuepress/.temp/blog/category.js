export const categoryMap = {"category":{"/":{"path":"/category/","map":{"Note":{"path":"/category/note/","keys":["v-c0336012"]}}}},"tag":{"/":{"path":"/tag/","map":{"Paper Read":{"path":"/tag/paper-read/","keys":["v-c0336012"]},"Domain Adaption":{"path":"/tag/domain-adaption/","keys":["v-c0336012"]},"Computer Vision":{"path":"/tag/computer-vision/","keys":["v-c0336012"]},"WACV":{"path":"/tag/wacv/","keys":["v-c0336012"]}}}}};

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept();
  if (__VUE_HMR_RUNTIME__.updateBlogCategory)
    __VUE_HMR_RUNTIME__.updateBlogCategory(categoryMap);
}

if (import.meta.hot)
  import.meta.hot.accept(({ categoryMap }) => {
    __VUE_HMR_RUNTIME__.updateBlogCategory(categoryMap);
  });


