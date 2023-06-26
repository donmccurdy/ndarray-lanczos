import test from 'ava';
import { readFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import ndarray from 'ndarray';
import { getPixels } from 'ndarray-pixels';
import { lanczos2, lanczos3 } from 'ndarray-lanczos';

const __dirname = dirname(fileURLToPath(import.meta.url));

const createImage = (w: number, h: number): ndarray.NdArray<Uint8Array> => {
	const image = ndarray(new Uint8Array(w * h * 4).fill(0), [h, w, 4]);
	return image.transpose(1, 0); // https://github.com/scijs/get-pixels/issues/52
};

const FIXTURES: Promise<Record<string, ndarray.NdArray<Uint8Array>>> = Promise.all([
	readFile(`${__dirname}/fixtures/pattern.png`),
	readFile(`${__dirname}/fixtures/pattern-tiled.png`),
	readFile(`${__dirname}/fixtures/pattern-half.png`),
	readFile(`${__dirname}/fixtures/pattern-half-2.png`),
	readFile(`${__dirname}/fixtures/pattern-double.png`),
	readFile(`${__dirname}/fixtures/pattern-tiled-half.png`),
])
	.then((images) => Promise.all(images.map((image) => getPixels(image, 'image/png'))))
	.then((pixels) => ({
		pattern: pixels[0],
		patternTiled: pixels[1],
		expectedPatternHalf: pixels[2],
		expectedPatternHalf2: pixels[3],
		expectedPatternDouble: pixels[4],
		expectedPatternTiledHalf: pixels[5],
	}));

test('resize down - lanczos3', async (t) => {
	const { pattern, expectedPatternHalf } = await FIXTURES;
	let patternHalf = createImage(4, 4);

	lanczos3(pattern, patternHalf);

	t.deepEqual(patternHalf.shape, expectedPatternHalf.shape, 'shape');
	t.deepEqual(patternHalf.stride, expectedPatternHalf.stride, 'stride');
	t.deepEqual(Array.from(patternHalf.data), Array.from(expectedPatternHalf.data), 'data');
});

test('resize down - lanczos2', async (t) => {
	const { pattern, expectedPatternHalf2 } = await FIXTURES;
	const patternHalf = createImage(4, 4);

	lanczos2(pattern, patternHalf);

	t.deepEqual(patternHalf.shape, expectedPatternHalf2.shape, 'shape');
	t.deepEqual(patternHalf.stride, expectedPatternHalf2.stride, 'stride');
	t.deepEqual(patternHalf, expectedPatternHalf2, 'data');
});

test('resize up - lanczos3', async (t) => {
	const { pattern, expectedPatternDouble } = await FIXTURES;
	const patternDouble = createImage(16, 16);

	lanczos3(pattern, patternDouble);

	t.deepEqual(patternDouble.shape, expectedPatternDouble.shape, 'shape');
	t.deepEqual(patternDouble.stride, expectedPatternDouble.stride, 'stride');
	t.deepEqual(patternDouble, expectedPatternDouble, 'data');
});

test('resize down - lanczos3 non-square', async (t) => {
	const { patternTiled, expectedPatternTiledHalf } = await FIXTURES;
	const patternTiledHalf = createImage(16, 4);

	lanczos3(patternTiled, patternTiledHalf);

	t.deepEqual(patternTiledHalf.shape, expectedPatternTiledHalf.shape, 'shape');
	t.deepEqual(patternTiledHalf.stride, expectedPatternTiledHalf.stride, 'stride');
	t.deepEqual(patternTiledHalf, expectedPatternTiledHalf, 'data');
});

test('upscale Uint16 data', async (t) => {
	const pattern = ndarray(new Uint16Array([0, 500, 1000, 500, 500, 500, 500, 500, 0, 500, 1000, 500]), [3, 4, 1]);
	const output = ndarray(new Uint16Array(18), [3, 6, 1]);
	const expected = ndarray(
		new Uint16Array([0, 164, 640, 1011, 764, 440, 500, 500, 500, 500, 500, 500, 0, 164, 640, 1011, 764, 440]),
		[3, 6, 1]
	);

	lanczos3(pattern, output);

	t.deepEqual(output.shape, expected.shape, 'shape');
	t.deepEqual(output.stride, expected.stride, 'stride');
	t.deepEqual(output, expected, 'data');
});

test('downscale Uint16 data', async (t) => {
	const pattern = ndarray(
		new Uint16Array([0, 164, 640, 1011, 764, 440, 500, 500, 500, 500, 500, 500, 0, 164, 640, 1011, 764, 440]),
		[3, 6, 1]
	);
	const output = ndarray(new Uint16Array(12), [3, 4, 1]);
	const expected = ndarray(new Uint16Array([6, 520, 969, 525, 500, 500, 500, 500, 6, 520, 969, 525]), [3, 4, 1]);

	lanczos3(pattern, output);

	t.deepEqual(output.shape, expected.shape, 'shape');
	t.deepEqual(output.stride, expected.stride, 'stride');
	t.deepEqual(output, expected, 'data');
});

test('upscale Uint32 data', async (t) => {
	const pattern = ndarray(new Uint32Array([0, 500, 1000, 500, 500, 500, 500, 500, 0, 500, 1000, 500]), [3, 4, 1]);
	const output = ndarray(new Uint32Array(18), [3, 6, 1]);
	const expected = ndarray(
		new Uint32Array([0, 164, 640, 1011, 764, 440, 500, 500, 500, 500, 500, 500, 0, 164, 640, 1011, 764, 440]),
		[3, 6, 1]
	);

	lanczos3(pattern, output);

	t.deepEqual(output.shape, expected.shape, 'shape');
	t.deepEqual(output.stride, expected.stride, 'stride');
	t.deepEqual(output, expected, 'data');
});

test('downscale Uint32 data', async (t) => {
	const pattern = ndarray(
		new Uint32Array([0, 164, 640, 1011, 764, 440, 500, 500, 500, 500, 500, 500, 0, 164, 640, 1011, 764, 440]),
		[3, 6, 1]
	);
	const output = ndarray(new Uint32Array(12), [3, 4, 1]);
	const expected = ndarray(new Uint32Array([6, 520, 969, 525, 500, 500, 500, 500, 6, 520, 969, 525]), [3, 4, 1]);

	lanczos3(pattern, output);

	t.deepEqual(output.shape, expected.shape, 'shape');
	t.deepEqual(output.stride, expected.stride, 'stride');
	t.deepEqual(output, expected, 'data');
});
