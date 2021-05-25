require('source-map-support').install();

import ndarray from 'ndarray';
import { getPixels } from 'ndarray-pixels';
import test from 'tape';
import { lanczos2, lanczos3 } from '../';

const createImage = (w: number, h: number): ndarray.NdArray => {
	const image = ndarray(new Uint8Array(w * h * 4).fill(0), [h, w, 4]);
	return image.transpose(1, 0); // https://github.com/scijs/get-pixels/issues/52
};

const FIXTURES: Promise<Record<string, ndarray.NdArray>> = new Promise(async (resolve, reject) => {
	try {
		resolve({
			pattern: await getPixels(`${__dirname}/fixtures/pattern.png`),
			patternTiled: await getPixels(`${__dirname}/fixtures/pattern-tiled.png`),
			expectedPatternHalf: await getPixels(`${__dirname}/fixtures/pattern-half.png`),
			expectedPatternHalf2: await getPixels(`${__dirname}/fixtures/pattern-half-2.png`),
			expectedPatternDouble: await getPixels(`${__dirname}/fixtures/pattern-double.png`),
			expectedPatternTiledHalf: await getPixels(`${__dirname}/fixtures/pattern-tiled-half.png`),
		});
	} catch (e) {
		reject(e);
	}
});

test('resize down - lanczos3', async (t) => {
	const {pattern, expectedPatternHalf} = await FIXTURES;
	let patternHalf = createImage(4, 4);

	lanczos3(pattern, patternHalf);

	t.deepEqual(patternHalf.shape, expectedPatternHalf.shape, 'shape');
	t.deepEqual(patternHalf.stride, expectedPatternHalf.stride, 'stride');
	t.deepEqual(Array.from(patternHalf.data), Array.from(expectedPatternHalf.data), 'data');
	t.end();
});

test('resize down - lanczos2', async (t) => {
	const {pattern, expectedPatternHalf2} = await FIXTURES;
	const patternHalf = createImage(4, 4);

	lanczos2(pattern, patternHalf);

	t.deepEqual(patternHalf.shape, expectedPatternHalf2.shape, 'shape');
	t.deepEqual(patternHalf.stride, expectedPatternHalf2.stride, 'stride');
	t.deepEqual(patternHalf, expectedPatternHalf2, 'data');
	t.end();
});

test('resize up - lanczos3', async (t) => {
	const {pattern, expectedPatternDouble} = await FIXTURES;
	const patternDouble = createImage(16, 16);

	lanczos3(pattern, patternDouble);

	t.deepEqual(patternDouble.shape, expectedPatternDouble.shape, 'shape');
	t.deepEqual(patternDouble.stride, expectedPatternDouble.stride, 'stride');
	t.deepEqual(patternDouble, expectedPatternDouble, 'data');
	t.end();
});

test('resize down - lanczos3 non-square', async (t) => {
	const {patternTiled, expectedPatternTiledHalf} = await FIXTURES;
	const patternTiledHalf = createImage(16, 4);

	lanczos3(patternTiled, patternTiledHalf);

	t.deepEqual(patternTiledHalf.shape, expectedPatternTiledHalf.shape, 'shape');
	t.deepEqual(patternTiledHalf.stride, expectedPatternTiledHalf.stride, 'stride');
	t.deepEqual(patternTiledHalf, expectedPatternTiledHalf, 'data');
	t.end();
});
