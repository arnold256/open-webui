import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
	plugins: [
		sveltekit(),
		viteStaticCopy({
			targets: [
				{
					src: 'node_modules/onnxruntime-web/dist/*.jsep.*',

					dest: 'wasm'
				}
			]
		})
	],
	define: {
		APP_VERSION: JSON.stringify(process.env.npm_package_version),
		APP_BUILD_HASH: JSON.stringify(process.env.APP_BUILD_HASH || 'dev-build')
	},
	build: {
		// Source maps are the largest single memory consumer in this build:
		// rollup holds the mapping structures for every module alongside the
		// modules themselves. On a build agent with 3.9 GB and no swap that is
		// the difference between finishing and being killed - see the Dockerfile,
		// where the image build sets VITE_SOURCEMAP=false.
		//
		// Unset, the behaviour is exactly as before. Only a build that opts out
		// loses them, and a deployed container has no use for them.
		sourcemap: process.env.VITE_SOURCEMAP !== 'false'
	},
	worker: {
		format: 'es'
	},
	esbuild: {
		pure: process.env.ENV === 'dev' ? [] : ['console.log', 'console.debug', 'console.error']
	}
});
