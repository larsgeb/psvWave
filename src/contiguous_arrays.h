#include <iostream>

template <class T>
void allocate_array(T ****&pointer, int dim1, int dim2, int dim3, int dim4) {

  pointer = new T ***[dim1];
  pointer[0] = new T **[dim1 * dim2];
  pointer[0][0] = new T *[dim1 * dim2 * dim3];
  pointer[0][0][0] = new T[dim1 * dim2 * dim3 * dim4];

  // Pointer arithmetic
  for (int h = 0; h < dim1; h++) {
    if (h > 0) {
      pointer[h] = pointer[h - 1] + dim2;
      pointer[h][0] = pointer[h - 1][0] + (dim2 * dim3);
      pointer[h][0][0] = pointer[h - 1][0][0] + (dim2 * dim3 * dim4);
    }

    for (int i = 0; i < dim2; i++) {
      if (i > 0) {
        pointer[h][i] = pointer[h][i - 1] + dim3;
        pointer[h][i][0] = pointer[h][i - 1][0] + (dim3 * dim4);
      }

      for (int j = 1; j < dim3; j++) {
        pointer[h][i][j] = pointer[h][i][j - 1] + dim4;
      }
    }
  }
}

template <class T> void allocate_array(T ***&pointer, int dim1, int dim2, int dim3) {

  pointer = new T **[dim1];
  pointer[0] = new T *[dim1 * dim2];
  pointer[0][0] = new T[dim1 * dim2 * dim3];

  for (int i = 0; i < dim1; i++) {
    if (i > 0) {
      pointer[i] = pointer[i - 1] + dim2;
      pointer[i][0] = pointer[i - 1][0] + dim2 * dim3;
    }

    for (int j = 1; j < dim2; j++) {
      pointer[i][j] = pointer[i][j - 1] + dim3;
    }
  }
}

template <class T> void allocate_array(T **&pointer, int dim1, int dim2) {
  pointer = new T *[dim1];

  int size = dim1 * dim2;

  pointer[0] = new T[size];

  for (int ir = 1; ir < dim1; ir++) {
    pointer[ir] = pointer[ir - 1] + dim2;
  }
}

template <class T> void allocate_array(T *&pointer, int dim1) { pointer = new T[dim1]; }
template <class T> void deallocate_array(T *&pointer) { delete[] pointer; }
template <class T> void deallocate_array(T **&pointer) {
  delete[] pointer[0];
  delete[] pointer;
}
template <class T> void deallocate_array(T ***&pointer) {
  delete[] pointer[0][0];
  delete[] pointer[0];
  delete[] pointer;
}
template <class T> void deallocate_array(T ****&pointer) {
  delete[] pointer[0][0][0];
  delete[] pointer[0][0];
  delete[] pointer[0];
  delete[] pointer;
}
