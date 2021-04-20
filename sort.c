float *x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
initializeArray(x, ARRAY_SIZE, ...);

// Allocate the copy of the array on the GPU
cudaMalloc(&x, ARRAY_SIZE * sizeof(float));

sort<<<grid, block>>>(x, ARRAY_SIZE, ...);