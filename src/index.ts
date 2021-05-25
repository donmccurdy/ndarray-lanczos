import ndarray, { NdArray } from 'ndarray';
import { filters } from '../vendor/filters';
import { convolve } from './convolve';

enum Method {
	LANCZOS_3 = 3,
	LANCZOS_2 = 2,
}

function resize(src: NdArray, dst: NdArray, method: Method): void {
	const [srcWidth, srcHeight] = src.shape;
	const [dstWidth, dstHeight] = dst.shape;

	const ratioX = dstWidth / srcWidth;
	const ratioY = dstHeight / srcHeight;

	const filtersX = filters(srcWidth, dstWidth, ratioX, 0, method === Method.LANCZOS_2);
	const filtersY = filters(srcHeight, dstHeight, ratioY, 0, method === Method.LANCZOS_2);

	let tmp = ndarray(new Uint8Array(dstWidth * srcHeight * 4), [dstWidth, srcHeight, 4]);

	convolve(src, tmp, filtersX);
	tmp = tmp.transpose(1, 0);
	convolve(tmp, dst, filtersY);
}

export function lanczos3(src: NdArray, dst: NdArray): void {
	resize(src, dst, Method.LANCZOS_3);
}

export function lanczos2(src: NdArray, dst: NdArray): void {
	resize(src, dst, Method.LANCZOS_2);
}
