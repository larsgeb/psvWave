#ifndef CONTIGUOUS_H
#define CONTIGUOUS_H

#include <Metal/Metal.hpp>
#include <vector>

template <class T>
void allocate_array(MTL::Device *gpu_device,
                    MTL::Buffer *buffer_pointer,
                    T *&pointer,
                    std::vector<int> shape)
{
    // Compute number of elements and byte size
    int total_size = 1;
    for (auto &&dimension_length : shape)
    {
        total_size *= dimension_length;
    }
    auto byte_size = total_size * sizeof(T);

    // Create buffer on gpu device
    buffer_pointer = gpu_device->newBuffer(byte_size, MTL::ResourceStorageModeManaged);

    // Get contents pointer to first element
    pointer = (T *)buffer_pointer->contents(); // M1 allocation
    // pointer = new T[total_size]; // CPU allocation
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