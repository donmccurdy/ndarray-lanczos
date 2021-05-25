import type { NdArray } from 'ndarray';

const fixedFracBits = 14;

const clamp = (v: number): number => v < 0 ? 0 : (v > 255 ? 255 : v);

export const convolve = (source: NdArray, dest: NdArray, sw: number, sh: number, dw: number, filters: Int16Array) => {
	let destY = 0

	// For each row
	for ( let sourceY = 0; sourceY < sh; sourceY++ ) {
		let filterPtr = 0

		// Apply precomputed filters to each destination row point
		for ( let destX = 0; destX < dw; destX++ ) {
			// Get the filter that determines the current output pixel.
			const filterShift = filters[ filterPtr++ ]

			let sourceX = filterShift; // ( srcOffset + ( filterShift * 4 ) ) | 0

			let r = 0;
			let g = 0;
			let b = 0;
			let a = 0;

			// Apply the filter to the row to get the destination pixel r, g, b, a
			for (let filterSize = filters[filterPtr++]; filterSize > 0; filterSize--) {
				const filterValue = filters[filterPtr++]

				r = ( r + filterValue * source.get(sourceX, sourceY, 0) );
				g = ( g + filterValue * source.get(sourceX, sourceY, 1) );
				b = ( b + filterValue * source.get(sourceX, sourceY, 2) );
				a = ( a + filterValue * source.get(sourceX, sourceY, 3) );

				sourceX++;
			}

			// Bring this value back in range. All of the filter scaling factors
			// are in fixed point with fixedFracBits bits of fractional part.
			//
			// (!) Add 1/2 of value before clamping to get proper rounding. In other
			// case brightness loss will be noticeable if you resize image with white
			// border and place it on white background.
			//
			dest.set(destX, destY, 0, clamp( ( r + ( 1 << 13 ) ) >> fixedFracBits ) );
			dest.set(destX, destY, 1, clamp( ( g + ( 1 << 13 ) ) >> fixedFracBits ) );
			dest.set(destX, destY, 2, clamp( ( b + ( 1 << 13 ) ) >> fixedFracBits ) );
			dest.set(destX, destY, 3, clamp( ( a + ( 1 << 13 ) ) >> fixedFracBits ) );
		}

		destY++;
	}
}
