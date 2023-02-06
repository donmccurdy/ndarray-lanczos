import ndarray, { NdArray, TypedArray } from 'ndarray';
import { filters, TypedArrayConstructor } from '../vendor/filters';
import { convolve } from './convolve';

enum Method {
	LANCZOS_3 = 3,
	LANCZOS_2 = 2,
}

function resize(src: NdArray<TypedArray>, dst: NdArray<TypedArray>, method: Method): void {
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

export function lanczos3(src: NdArray<TypedArray>, dst: NdArray<TypedArray>): void {
	resize(src, dst, Method.LANCZOS_3);
}

export function lanczos2(src: NdArray<TypedArray>, dst: NdArray<TypedArray>): void {
	resize(src, dst, Method.LANCZOS_2);
}
