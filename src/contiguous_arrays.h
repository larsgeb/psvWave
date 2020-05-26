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

template <class T>
void allocate_array(T ***&pointer, int dim1, int dim2, int dim3) {

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

template <class T> void allocate_array(T *&pointer, int dim1) {
  pointer = new T[dim1];
}
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

// int test_arrays() {

//   double *pointer_1d;
//   double **pointer_2d;
//   double ***pointer_3d;
//   double ****pointer_4d;

//   int dim1 = 5;
//   int dim2 = 4;
//   int dim3 = 3;
//   int dim4 = 2;

//   allocate_1d_array(pointer_1d, dim1);
//   allocate_2d_array(pointer_2d, dim1, dim2);
//   allocate_3d_array(pointer_3d, dim1, dim2, dim3);
//   allocate_4d_array(pointer_4d, dim1, dim2, dim3, dim4);

//   std::cout << std::endl << "Memory addresses for 1d array:" << std::endl;
//   for (int j = 0; j < dim1; j++) {
//     std::cout << j << " " << &pointer_1d[j] << std::endl;
//   }

//   std::cout << std::endl << "Memory addresses for 2d array:" << std::endl;
//   for (int j = 0; j < dim1; j++) {
//     for (int k = 0; k < dim2; k++) {
//       std::cout << j << " " << k << " " << &pointer_2d[j][k] << std::endl;
//     }
//   }

//   std::cout << std::endl << "Memory addresses for 3d array:" << std::endl;
//   for (int j = 0; j < dim1; j++) {
//     for (int k = 0; k < dim2; k++) {
//       for (int l = 0; l < dim3; l++) {
//         std::cout << j << " " << k << " " << l << " " << &pointer_3d[j][k][l]
//                   << std::endl;
//       }
//     }
//   }

//   std::cout << std::endl << "Memory addresses for 4d array:" << std::endl;
//   for (int j = 0; j < dim1; j++) {
//     for (int k = 0; k < dim2; k++) {
//       for (int l = 0; l < dim3; l++) {
//         for (int m = 0; m < dim4; m++) {

//           std::cout << j << " " << k << " " << l << " " << m << " "
//                     << &pointer_4d[j][k][l][m] << std::endl;
//         }
//       }
//     }
//   }

//   deallocate_1d_array(pointer_1d);
//   deallocate_2d_array(pointer_2d);
//   deallocate_3d_array(pointer_3d);
//   deallocate_4d_array(pointer_4d);

//   return 0;
// }
