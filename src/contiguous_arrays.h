#ifndef CONTIGUOUS_H
#define CONTIGUOUS_H

#include <vector>

template <class T>
void allocate_array(T *&pointer, std::vector<int> shape)
{
    int total_size = 1;
    for (auto &&dimension_length : shape)
    {
        total_size *= dimension_length;
    }
    pointer = new T[total_size];
};

template <class T>
void deallocate_array(T *pointer)
{
    delete[] pointer;
};

int linear_IDX(int pos1, int shape1);

int linear_IDX(int pos1, int pos2, int shape1, int shape2);

int linear_IDX(int pos1, int pos2, int pos3, int shape1, int shape2, int shape3);

int linear_IDX(int pos1, int pos2, int pos3, int pos4, int shape1, int shape2, int shape3, int shape4);
#endif