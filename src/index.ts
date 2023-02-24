import ndarray, { NdArray } from 'ndarray';
import { filters, TypedArrayConstructor } from '../vendor/filters.js';
import { convolve } from './convolve.js';

enum Method {
	LANCZOS_3 = 3,
	LANCZOS_2 = 2,
}

export type SupportedTypes = Uint8Array | Uint8ClampedArray | Uint16Array | Uint32Array

function resize(
	src: NdArray<SupportedTypes | number[]>,
	dst: NdArray<SupportedTypes>, method: Method
): void {
	if (src.shape.length !== 3 || dst.shape.length !== 3)
		throw new TypeError
			('Input and output must have exactly 3 dimensions (width, height and colorspace)');

	const [srcWidth, srcHeight] = src.shape;
	const [dstWidth, dstHeight] = dst.shape;

	const ratioX = dstWidth / srcWidth;
	const ratioY = dstHeight / srcHeight;

	let floatType, intType;
	switch (dst.dtype) {
		case 'uint8_clamped':
		case 'uint8':
			floatType = Float32Array;
			intType = Int16Array;
			break;
		case 'uint16':
		case 'uint32':
			floatType = Float64Array;
			intType = Int32Array;
			break;
		default:
			throw TypeError(`Unsupported data type ${dst.dtype}`);
	}
	const fixedFracBits = intType.BYTES_PER_ELEMENT * 7;

	const filtersX = filters(srcWidth, dstWidth, ratioX, 0, method === Method.LANCZOS_2,
		floatType, intType, fixedFracBits);
	const filtersY = filters(srcHeight, dstHeight, ratioY, 0, method === Method.LANCZOS_2,
		floatType, intType, fixedFracBits);

	const constructor = dst.data.constructor as TypedArrayConstructor;
	const tmp = ndarray(new constructor(dstWidth * srcHeight * 4), [srcHeight, dstWidth, 4]);
	const tmpTranspose = tmp.transpose(1, 0);
	const dstTranspose = dst.transpose(1, 0);

	convolve(src, tmpTranspose, filtersX, fixedFracBits);
	convolve(tmp, dstTranspose, filtersY, fixedFracBits);
}

export function lanczos3(src: NdArray<SupportedTypes | number[]>, dst: NdArray<SupportedTypes>): void {
	resize(src, dst, Method.LANCZOS_3);
}

export function lanczos2(src: NdArray<SupportedTypes | number[]>, dst: NdArray<SupportedTypes>): void {
	resize(src, dst, Method.LANCZOS_2);
}
