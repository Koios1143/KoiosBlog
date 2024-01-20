import { defineClientConfig } from "@vuepress/client";
import { hasGlobalComponent } from "/root/blog_v3/node_modules/.pnpm/vuepress-shared@2.0.0-rc.11_vuepress@2.0.0-rc.0/node_modules/vuepress-shared/lib/client/index.js";
import { h } from "vue";

import { useScriptTag } from "/root/blog_v3/node_modules/.pnpm/@vueuse+core@10.7.2_vue@3.4.15/node_modules/@vueuse/core/index.mjs";
import FontIcon from "/root/blog_v3/node_modules/.pnpm/vuepress-plugin-components@2.0.0-rc.11_vuepress@2.0.0-rc.0/node_modules/vuepress-plugin-components/lib/client/components/FontIcon.js";
import Badge from "/root/blog_v3/node_modules/.pnpm/vuepress-plugin-components@2.0.0-rc.11_vuepress@2.0.0-rc.0/node_modules/vuepress-plugin-components/lib/client/components/Badge.js";
import VPCard from "/root/blog_v3/node_modules/.pnpm/vuepress-plugin-components@2.0.0-rc.11_vuepress@2.0.0-rc.0/node_modules/vuepress-plugin-components/lib/client/components/VPCard.js";
import BackToTop from "/root/blog_v3/node_modules/.pnpm/vuepress-plugin-components@2.0.0-rc.11_vuepress@2.0.0-rc.0/node_modules/vuepress-plugin-components/lib/client/components/BackToTop.js";

import "/root/blog_v3/node_modules/.pnpm/vuepress-plugin-components@2.0.0-rc.11_vuepress@2.0.0-rc.0/node_modules/vuepress-plugin-components/lib/client/styles/sr-only.scss";

export default defineClientConfig({
  enhance: ({ app }) => {
    if(!hasGlobalComponent("FontIcon")) app.component("FontIcon", FontIcon);
    if(!hasGlobalComponent("Badge")) app.component("Badge", Badge);
    if(!hasGlobalComponent("VPCard")) app.component("VPCard", VPCard);
    
  },
  setup: () => {
    useScriptTag(
  `https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6/js/brands.min.js`,
  () => {},
  { attrs: { "data-auto-replace-svg": "nest" } }
);

    useScriptTag(
  `https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6/js/solid.min.js`,
  () => {},
  { attrs: { "data-auto-replace-svg": "nest" } }
);

    useScriptTag(
  `https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6/js/fontawesome.min.js`,
  () => {},
  { attrs: { "data-auto-replace-svg": "nest" } }
);

  },
  rootComponents: [
    () => h(BackToTop, {}),
  ],
});
