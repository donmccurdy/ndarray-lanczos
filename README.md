# ndarray-lanczos

[![Latest NPM release](https://img.shields.io/npm/v/ndarray-lanczos.svg)](https://www.npmjs.com/package/ndarray-lanczos)
[![License](https://img.shields.io/badge/license-MIT-007ec6.svg)](https://github.com/donmccurdy/ndarray-lanczos/blob/main/LICENSE)
[![Build Status](https://github.com/donmccurdy/ndarray-lanczos/actions/workflows/ci.yml/badge.svg)](https://github.com/donmccurdy/ndarray-lanczos/actions?query=workflow%3Abuild+branch%3Amain)
[![Coverage](https://codecov.io/gh/donmccurdy/ndarray-pixels/branch/main/graph/badge.svg?token=S30LCC3L04)](https://codecov.io/gh/donmccurdy/ndarray-pixels)

Resize an [ndarray](https://www.npmjs.com/package/ndarray) with [Lanczos resampling](https://en.wikipedia.org/wiki/Lanczos_resampling).

## Quickstart

Installation:

```
npm install --save ndarray-lanczos
```

Use:

```ts
import ndarray from 'ndarray';
import { getPixels, savePixels } from 'ndarray-pixels';
import { lanczos3 } from 'ndarray-lanczos';

// Read PNG.
const srcPixels = await getPixels('full-size.png');

// Resize with Lanczos 3 resampling.
const dstPixels = ndarray(new Uint8Array(width * height * 4).fill(0), [width, height, 4]);
lanczos3(srcPixels, dstPixels);

// Write PNG.
const data = await savePixels(dstPixels, 'image/png'); // → Uint8Array
```

Two filtering methods, `lanczos3` and `lanczos2`, are included.

## Credits

Thanks to https://github.com/rgba-image/lanczos and https://github.com/nodeca/pica.
