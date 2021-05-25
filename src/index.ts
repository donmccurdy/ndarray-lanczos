import type { NdArray } from 'ndarray';
import { convolve } from '../vendor/convolve';
import { filters } from '../vendor/filters';

const resize = (source: NdArray, dest: NdArray, use2 = false) => {
	const xRatio = dest.shape[0] / source.shape[0];
	const yRatio = dest.shape[1] / source.shape[1];

	const filtersX = filters(source.shape[0], dest.shape[0], xRatio, 0, use2);
	const filtersY = filters(source.shape[1], dest.shape[1], yRatio, 0, use2);

	const tmp = new Uint8ClampedArray(dest.shape[0] * source.shape[1] * 4);

	convolve(source.data as Uint8ClampedArray, tmp, source.shape[0], source.shape[1], dest.shape[0], filtersX);
	convolve(tmp, dest.data as Uint8ClampedArray, source.shape[1], dest.shape[0], dest.shape[1], filtersY);
}

export const lanczos = (source: NdArray, dest: NdArray): void => {
	resize(source, dest);
}

export const lanczos2 = (source: NdArray, dest: NdArray): void => {
	resize(source, dest, true);
}
