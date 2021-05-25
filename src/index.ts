import ndarray from 'ndarray';
import { filters } from '../vendor/filters';
import { convolve } from './convolve';

enum Method {
	LANCZOS_3 = 3,
	LANCZOS_2 = 2,
}

const resize = (source: ndarray.NdArray, dest: ndarray.NdArray, method: Method) => {
	const xRatio = dest.shape[0] / source.shape[0];
	const yRatio = dest.shape[1] / source.shape[1];

	const filtersX = filters(source.shape[0], dest.shape[0], xRatio, 0, method === Method.LANCZOS_2);
	const filtersY = filters(source.shape[1], dest.shape[1], yRatio, 0, method === Method.LANCZOS_2);

	let tmp = ndarray(
		new Uint8ClampedArray(dest.shape[0] * source.shape[1] * 4),
		[dest.shape[0], source.shape[1], 4]
	);

	convolve(source, tmp, source.shape[0], source.shape[1], dest.shape[0], filtersX);
	tmp = tmp.transpose(1, 0);
	convolve(tmp, dest, source.shape[1], dest.shape[0], dest.shape[1], filtersY);
}

export const lanczos3 = (source: ndarray.NdArray, dest: ndarray.NdArray): void => {
	resize(source, dest, Method.LANCZOS_3);
}

export const lanczos2 = (source: ndarray.NdArray, dest: ndarray.NdArray): void => {
	resize(source, dest, Method.LANCZOS_2);
}
