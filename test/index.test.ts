require('source-map-support').install();

import fs from 'fs';
import ndarray from 'ndarray';
import { getPixels, savePixels } from 'ndarray-pixels';
import test from 'tape';
import { lanczos2, lanczos3 } from '../';

const createImage = (w: number, h: number): ndarray.NdArray => {
	let image = ndarray(new Uint8Array(w * h * 4).fill(0), [w, h, 4]);
	image = image.transpose(1, 0); // https://github.com/scijs/get-pixels/issues/52
	return image;
};

const FIXTURES: Promise<Record<string, ndarray.NdArray>> = new Promise(async (resolve, reject) => {
	try {
		resolve({
			pattern: await getPixels(`${__dirname}/fixtures/pattern.png`),
			expectedPatternHalf: await getPixels(`${__dirname}/fixtures/pattern-half.png`),
			expectedPatternHalf2: await getPixels(`${__dirname}/fixtures/pattern-half-2.png`),
			expectedPatternDouble: await getPixels(`${__dirname}/fixtures/pattern-double.png`),
		});
	} catch (e) {
		reject(e);
	}
});

test.only('resize down - lanczos3', async (t) => {
	const {pattern, expectedPatternHalf} = await FIXTURES;
	let patternHalf = createImage(4, 4);
	lanczos3(pattern, patternHalf);
	patternHalf = patternHalf.transpose(1, 0); // TODO(bug)
	t.deepEqual(patternHalf.shape, expectedPatternHalf.shape, 'shape');
	t.deepEqual(patternHalf.stride, expectedPatternHalf.stride, 'stride');
	t.deepEqual(Array.from(patternHalf.data), Array.from(expectedPatternHalf.data), 'data');

	// TODO(cleanup): Debugging.
	fs.writeFileSync(`${__dirname}/pattern.png`, await savePixels(patternHalf, 'image/png'));

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
