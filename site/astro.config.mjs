// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { katexGuard, katexGuardIntegration } from './scripts/lib/katex-guard.mjs';
import { rehypeHeadingIds } from './scripts/lib/rehype-heading-ids.mjs';

// https://alanfokco.github.io/ai-infra-primer/
export default defineConfig({
  site: 'https://alanfokco.github.io',
  base: '/ai-infra-primer',
  trailingSlash: 'always',
  output: 'static',
  integrations: [sitemap(), katexGuardIntegration()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      [rehypeKatex, { strict: false, throwOnError: false, trust: true }],
      katexGuard,
      rehypeHeadingIds,
    ],
    shikiConfig: {
      theme: 'github-dark-default',
      langs: ['python', 'bash', 'json', 'yaml', 'text'],
    },
  },
});
