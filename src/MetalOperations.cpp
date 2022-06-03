#include "MetalOperations.hpp"
#include <iostream>
#include <list>
#include <map>
#include <vector>

MetalOperations::MetalOperations(MTL::Device *device)
{

    _mDevice = device;

    NS::Error *error = nullptr;

    // Load the shader files with a .metal file extension in the project
    auto filepath = NS::String::string("./ops.metallib", NS::ASCIIStringEncoding);
    MTL::Library *opLibrary = _mDevice->newLibrary(filepath, &error);

    if (opLibrary == nullptr)
    {
        std::cout << "Failed to find the default library. Error: "
                  << error->description()->utf8String() << std::endl;
        return;
    }

    // Get all function names
    auto fnNames = opLibrary->functionNames();

    for (size_t i = 0; i < fnNames->count(); i++)
    {

        auto name_nsstring = fnNames->object(i)->description();
        auto name_utf8 = name_nsstring->utf8String();

        // Load function into a map
        functionMap[name_utf8] =
            (opLibrary->newFunction(name_nsstring));

        // Create pipeline from function
        functionPipelineMap[name_utf8] =
            _mDevice->newComputePipelineState(functionMap[name_utf8], &error);

        if (functionPipelineMap[name_utf8] == nullptr)
        {
            std::cout << "Failed to created pipeline state object for "
                      << name_utf8 << ", error "
                      << error->description()->utf8String() << std::endl;
            return;
        }
    }

    _mCommandQueue = _mDevice->newCommandQueue();
    if (_mCommandQueue == nullptr)
    {
        std::cout << "Failed to find the command queue." << std::endl;
        return;
    }
}

void MetalOperations::Blocking1D(std::vector<MTL::Buffer *> buffers,
                                 size_t arrayLength,
                                 const char *method)
{

    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    computeEncoder->setComputePipelineState(functionPipelineMap[method]);
    for (std::size_t i = 0; i < buffers.size(); ++i)
    {
        computeEncoder->setBuffer(buffers[i], 0, i);
    }

    NS::UInteger threadGroupSize =
        functionPipelineMap[method]->maxTotalThreadsPerThreadgroup();

    if (threadGroupSize > arrayLength)
        threadGroupSize = arrayLength;

    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);
    MTL::Size gridSize = MTL::Size::Make(arrayLength, 1, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();

    auto blyat = commandBuffer->blitCommandEncoder();
    for (auto &&buffer : buffers)
    {
        blyat->synchronizeResource(buffer);
    }
    blyat->endEncoding();

    // inline void fn(){
    //     // function code
    // };

    // std::function<void(MTL::CommandBuffer *)> f_display_42 = [](MTL::CommandBuffer *)
    // { std::cout << 42 << std::endl; };
    // auto handler = MTL::HandlerFunction(f_display_42);
    // commandBuffer->addCompletedHandler(handler);

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void MetalOperations::Blocking2D(std::vector<MTL::Buffer *> buffers,
                                 size_t rows,
                                 size_t columns,
                                 const char *method)
{

    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    computeEncoder->setComputePipelineState(functionPipelineMap[method]);

    for (std::size_t i = 0; i < buffers.size(); ++i)
    {
        // Needed??
        // buffers[i]->didModifyRange(NS::Range(0, buffers[i]->allocatedSize()));
        computeEncoder->setBuffer(buffers[i], 0, i);
    }

    NS::UInteger maxThreadGroupSize =
        functionPipelineMap[method]->maxTotalThreadsPerThreadgroup();

    // aspect ratio of kernel operation is threadGroupSize:1
    MTL::Size threadgroupSize = MTL::Size::Make(32, 16, 1);

    std::cout << "Max thread group size" << maxThreadGroupSize << std::endl;
    assert(threadgroupSize.width * threadgroupSize.height * threadgroupSize.depth < maxThreadGroupSize);

    MTL::Size gridSize = MTL::Size::Make(rows, columns, 1);

    // Encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->endEncoding();

    // Needed?
    // auto blyat = commandBuffer->blitCommandEncoder();
    // for (auto &&buffer : buffers)
    // {
    //     blyat->synchronizeResource(buffer);
    // }
    // blyat->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void MetalOperations::addArrays(MTL::Buffer *x_array,
                                MTL::Buffer *y_array,
                                MTL::Buffer *r_array,
                                size_t arrayLength)
{
    std::vector<MTL::Buffer *> buffers = {x_array,
                                          y_array,
                                          r_array};
    const char *method = "add_arrays";

    Blocking1D(buffers, arrayLength, method);
}

void MetalOperations::multiplyArrays(MTL::Buffer *x_array,
                                     MTL::Buffer *y_array,
                                     MTL::Buffer *r_array,
                                     size_t arrayLength)
{
    std::vector<MTL::Buffer *> buffers = {x_array,
                                          y_array,
                                          r_array};
    const char *method = "multiply_arrays";

    Blocking1D(buffers, arrayLength, method);
}

void MetalOperations::multiplyArrayConstant(MTL::Buffer *x_array,
                                            MTL::Buffer *alpha,
                                            MTL::Buffer *r_array,
                                            size_t arrayLength)
{
    std::vector<MTL::Buffer *> buffers = {x_array,
                                          alpha,
                                          r_array};
    const char *method = "multiply_array_constant";

    Blocking1D(buffers, arrayLength, method);
}

void MetalOperations::saxpyArrays(MTL::Buffer *alpha,
                                  MTL::Buffer *x_array,
                                  MTL::Buffer *y_array,
                                  MTL::Buffer *r_array,
                                  size_t arrayLength)
{
    std::vector<MTL::Buffer *> buffers = {alpha,
                                          x_array,
                                          y_array,
                                          r_array};
    const char *method = "saxpy";

    Blocking1D(buffers, arrayLength, method);
}

void MetalOperations::central_difference(MTL::Buffer *delta,
                                         MTL::Buffer *x_array,
                                         MTL::Buffer *r_array,
                                         size_t arrayLength)
{
    std::vector<MTL::Buffer *> buffers = {delta,
                                          x_array,
                                          r_array};
    const char *method = "central_difference";

    Blocking1D(buffers, arrayLength, method);
}

void MetalOperations::quadratic2d(MTL::Buffer *X,
                                  MTL::Buffer *Y,
                                  MTL::Buffer *R,
                                  size_t rows,
                                  size_t columns)
{
    std::vector<MTL::Buffer *> buffers = {X,
                                          Y,
                                          R};
    const char *method = "quadratic2d";

    Blocking2D(buffers, rows, columns, method);
}

void MetalOperations::laplacian2d(MTL::Buffer *X,
                                  MTL::Buffer *R,
                                  size_t rows,
                                  size_t columns)
{
    std::vector<MTL::Buffer *> buffers = {X,
                                          R};
    const char *method = "laplacian2d";

    Blocking2D(buffers, rows, columns, method);
}

void MetalOperations::stress_integrate_2d(MTL::Buffer *&txx,
                                          MTL::Buffer *&tzz,
                                          MTL::Buffer *&txz,
                                          MTL::Buffer *&taper,
                                          MTL::Buffer *&_dt,
                                          MTL::Buffer *&_dx,
                                          MTL::Buffer *&_dz,
                                          MTL::Buffer *&vx,
                                          MTL::Buffer *&vz,
                                          MTL::Buffer *&lm,
                                          MTL::Buffer *&la,
                                          MTL::Buffer *&mu,
                                          MTL::Buffer *&b_vx,
                                          MTL::Buffer *&b_vz,
                                          size_t rows,
                                          size_t columns)
{
    std::vector<MTL::Buffer *> buffers =
        {txx, tzz, txz, taper, _dt, _dx, _dz, vx, vz, lm, la, mu, b_vx, b_vz};

    for (auto &&buffer : buffers)
    {
        buffer->didModifyRange(NS::Range::Make(0, buffer->length()));
        if (buffer == nullptr)
        {
            throw std::invalid_argument("Some buffers don't exist!");
        }
    }

    const char *method = "stress_integrate_2d";

    Blocking2D(buffers, rows, columns, method);
}

void MetalOperations::velocity_integrate_2d(MTL::Buffer *&txx,
                                            MTL::Buffer *&tzz,
                                            MTL::Buffer *&txz,
                                            MTL::Buffer *&taper,
                                            MTL::Buffer *&_dt,
                                            MTL::Buffer *&_dx,
                                            MTL::Buffer *&_dz,
                                            MTL::Buffer *&vx,
                                            MTL::Buffer *&vz,
                                            MTL::Buffer *&lm,
                                            MTL::Buffer *&la,
                                            MTL::Buffer *&mu,
                                            MTL::Buffer *&b_vx,
                                            MTL::Buffer *&b_vz,
                                            size_t rows,
                                            size_t columns)
{

    std::vector<MTL::Buffer *> buffers =
        {txx, tzz, txz, taper, _dt, _dx, _dz, vx, vz, lm, la, mu, b_vx, b_vz};

    for (auto &&buffer : buffers)
    {
        buffer->didModifyRange(NS::Range::Make(0, buffer->length()));
        if (buffer == nullptr)
        {
            throw std::invalid_argument("Some buffers don't exist!");
        }
    }

    const char *method = "velocity_integrate_2d";

    Blocking2D(buffers, rows, columns, method);
}

void MetalOperations::combined_integrate_2d(MTL::Buffer *&txx,
                                            MTL::Buffer *&tzz,
                                            MTL::Buffer *&txz,
                                            MTL::Buffer *&taper,
                                            MTL::Buffer *&_dt,
                                            MTL::Buffer *&_dx,
                                            MTL::Buffer *&_dz,
                                            MTL::Buffer *&vx,
                                            MTL::Buffer *&vz,
                                            MTL::Buffer *&lm,
                                            MTL::Buffer *&la,
                                            MTL::Buffer *&mu,
                                            MTL::Buffer *&b_vx,
                                            MTL::Buffer *&b_vz,
                                            size_t rows,
                                            size_t columns)
{

    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    NS::UInteger maxThreadGroupSize =
        functionPipelineMap["stress_integrate_2d"]->maxTotalThreadsPerThreadgroup();
    MTL::Size threadgroupSize = MTL::Size::Make(4, 256, 1);
    assert(threadgroupSize.width * threadgroupSize.height * threadgroupSize.depth <= maxThreadGroupSize);
    MTL::Size gridSize = MTL::Size::Make(rows, columns, 1);

    computeEncoder->setBuffer(txx, 0, 0);
    computeEncoder->setBuffer(tzz, 0, 1);
    computeEncoder->setBuffer(txz, 0, 2);
    computeEncoder->setBuffer(taper, 0, 3);
    computeEncoder->setBuffer(_dt, 0, 4);
    computeEncoder->setBuffer(_dx, 0, 5);
    computeEncoder->setBuffer(_dz, 0, 6);
    computeEncoder->setBuffer(vx, 0, 7);
    computeEncoder->setBuffer(vz, 0, 8);
    computeEncoder->setBuffer(lm, 0, 9);
    computeEncoder->setBuffer(la, 0, 10);
    computeEncoder->setBuffer(mu, 0, 11);
    computeEncoder->setBuffer(b_vx, 0, 12);
    computeEncoder->setBuffer(b_vz, 0, 13);

    computeEncoder->setComputePipelineState(functionPipelineMap["stress_integrate_2d"]);
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->setComputePipelineState(functionPipelineMap["velocity_integrate_2d"]);
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    computeEncoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void MetalOperations::combined_integrate_2d_non_blocking(MTL::Buffer *&txx,
                                                         MTL::Buffer *&tzz,
                                                         MTL::Buffer *&txz,
                                                         MTL::Buffer *&taper,
                                                         MTL::Buffer *&_dt,
                                                         MTL::Buffer *&_dx,
                                                         MTL::Buffer *&_dz,
                                                         MTL::Buffer *&vx,
                                                         MTL::Buffer *&vz,
                                                         MTL::Buffer *&lm,
                                                         MTL::Buffer *&la,
                                                         MTL::Buffer *&mu,
                                                         MTL::Buffer *&b_vx,
                                                         MTL::Buffer *&b_vz,
                                                         size_t rows,
                                                         size_t columns,
                                                         MTL::CommandBuffer *&commandBuffer)
{

    commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    NS::UInteger maxThreadGroupSize =
        functionPipelineMap["stress_integrate_2d"]->maxTotalThreadsPerThreadgroup();
    MTL::Size threadgroupSize = MTL::Size::Make(4, 256, 1);
    assert(threadgroupSize.width * threadgroupSize.height * threadgroupSize.depth <= maxThreadGroupSize);
    MTL::Size gridSize = MTL::Size::Make(rows, columns, 1);

    computeEncoder->setBuffer(txx, 0, 0);
    computeEncoder->setBuffer(tzz, 0, 1);
    computeEncoder->setBuffer(txz, 0, 2);
    computeEncoder->setBuffer(taper, 0, 3);
    computeEncoder->setBuffer(_dt, 0, 4);
    computeEncoder->setBuffer(_dx, 0, 5);
    computeEncoder->setBuffer(_dz, 0, 6);
    computeEncoder->setBuffer(vx, 0, 7);
    computeEncoder->setBuffer(vz, 0, 8);
    computeEncoder->setBuffer(lm, 0, 9);
    computeEncoder->setBuffer(la, 0, 10);
    computeEncoder->setBuffer(mu, 0, 11);
    computeEncoder->setBuffer(b_vx, 0, 12);
    computeEncoder->setBuffer(b_vz, 0, 13);

    computeEncoder->setComputePipelineState(functionPipelineMap["stress_integrate_2d"]);
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    computeEncoder->setComputePipelineState(functionPipelineMap["velocity_integrate_2d"]);
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    computeEncoder->endEncoding();

    commandBuffer->commit();
    // commandBuffer->waitUntilCompleted();
}

void MetalOperations::shared_cpu_txx_tzz(MTL::Buffer *&txx,
                                         MTL::Buffer *&tzz,
                                         MTL::Buffer *&txz,
                                         MTL::Buffer *&taper,
                                         MTL::Buffer *&_dt,
                                         MTL::Buffer *&_dx,
                                         MTL::Buffer *&_dz,
                                         MTL::Buffer *&vx,
                                         MTL::Buffer *&vz,
                                         MTL::Buffer *&lm,
                                         MTL::Buffer *&la,
                                         MTL::Buffer *&mu,
                                         MTL::Buffer *&b_vx,
                                         MTL::Buffer *&b_vz,
                                         size_t rows,
                                         size_t columns,
                                         MTL::CommandBuffer *&commandBuffer)
{

    commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    NS::UInteger maxThreadGroupSize =
        functionPipelineMap["txx_tzz_integrate_2d"]->maxTotalThreadsPerThreadgroup();
    MTL::Size threadgroupSize = MTL::Size::Make(4, 256, 1);
    assert(threadgroupSize.width * threadgroupSize.height * threadgroupSize.depth <= maxThreadGroupSize);
    MTL::Size gridSize = MTL::Size::Make(rows, columns, 1);

    computeEncoder->setBuffer(txx, 0, 0);
    computeEncoder->setBuffer(tzz, 0, 1);
    computeEncoder->setBuffer(txz, 0, 2);
    computeEncoder->setBuffer(taper, 0, 3);
    computeEncoder->setBuffer(_dt, 0, 4);
    computeEncoder->setBuffer(_dx, 0, 5);
    computeEncoder->setBuffer(_dz, 0, 6);
    computeEncoder->setBuffer(vx, 0, 7);
    computeEncoder->setBuffer(vz, 0, 8);
    computeEncoder->setBuffer(lm, 0, 9);
    computeEncoder->setBuffer(la, 0, 10);
    computeEncoder->setBuffer(mu, 0, 11);
    computeEncoder->setBuffer(b_vx, 0, 12);
    computeEncoder->setBuffer(b_vz, 0, 13);

    computeEncoder->setComputePipelineState(functionPipelineMap["txx_tzz_integrate_2d"]);
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    computeEncoder->endEncoding();

    commandBuffer->commit();
    // commandBuffer->waitUntilCompleted();
}

void MetalOperations::shared_cpu_vx(MTL::Buffer *&txx,
                                    MTL::Buffer *&tzz,
                                    MTL::Buffer *&txz,
                                    MTL::Buffer *&taper,
                                    MTL::Buffer *&_dt,
                                    MTL::Buffer *&_dx,
                                    MTL::Buffer *&_dz,
                                    MTL::Buffer *&vx,
                                    MTL::Buffer *&vz,
                                    MTL::Buffer *&lm,
                                    MTL::Buffer *&la,
                                    MTL::Buffer *&mu,
                                    MTL::Buffer *&b_vx,
                                    MTL::Buffer *&b_vz,
                                    size_t rows,
                                    size_t columns,
                                    MTL::CommandBuffer *&commandBuffer)
{

    commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    NS::UInteger maxThreadGroupSize =
        functionPipelineMap["vx_integrate_2d"]->maxTotalThreadsPerThreadgroup();
    MTL::Size threadgroupSize = MTL::Size::Make(4, 256, 1);
    assert(threadgroupSize.width * threadgroupSize.height * threadgroupSize.depth <= maxThreadGroupSize);
    MTL::Size gridSize = MTL::Size::Make(rows, columns, 1);

    computeEncoder->setBuffer(txx, 0, 0);
    computeEncoder->setBuffer(tzz, 0, 1);
    computeEncoder->setBuffer(txz, 0, 2);
    computeEncoder->setBuffer(taper, 0, 3);
    computeEncoder->setBuffer(_dt, 0, 4);
    computeEncoder->setBuffer(_dx, 0, 5);
    computeEncoder->setBuffer(_dz, 0, 6);
    computeEncoder->setBuffer(vx, 0, 7);
    computeEncoder->setBuffer(vz, 0, 8);
    computeEncoder->setBuffer(lm, 0, 9);
    computeEncoder->setBuffer(la, 0, 10);
    computeEncoder->setBuffer(mu, 0, 11);
    computeEncoder->setBuffer(b_vx, 0, 12);
    computeEncoder->setBuffer(b_vz, 0, 13);

    computeEncoder->setComputePipelineState(functionPipelineMap["vx_integrate_2d"]);
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    computeEncoder->endEncoding();

    commandBuffer->commit();
    // commandBuffer->waitUntilCompleted();
}

void MetalOperations::inspector(MTL::Buffer *x_array,
                                MTL::Buffer *r_array,
                                MTL::Buffer *store,
                                size_t arrayLength)
{
    std::vector<MTL::Buffer *> buffers = {x_array,
                                          r_array,
                                          store};
    const char *method = "inspector";

    Blocking1D(buffers, arrayLength, method);
}

void MetalOperations::addMultiply(MTL::Buffer *x_array,
                                  MTL::Buffer *y_array,
                                  MTL::Buffer *r_array,
                                  size_t arrayLength)
{
    // Example compound operator. Computes (x + y) * y.
    // Cannot use Blocking 1D, as it requires multiple buffer and PSO encodings.

    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);

    MTL::Size gridSize = MTL::Size::Make(arrayLength, 1, 1);
    NS::UInteger threadGroupSize =
        functionPipelineMap["add_arrays"]->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > arrayLength)
        threadGroupSize = arrayLength;
    MTL::Size threadgroupSize = MTL::Size::Make(threadGroupSize, 1, 1);

    computeEncoder->setComputePipelineState(functionPipelineMap["add_arrays"]);
    computeEncoder->setBuffer(x_array, 0, 0);
    computeEncoder->setBuffer(y_array, 0, 1);
    computeEncoder->setBuffer(r_array, 0, 2);
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    computeEncoder->setComputePipelineState(functionPipelineMap["multiply_arrays"]);
    computeEncoder->setBuffer(r_array, 0, 0);
    computeEncoder->setBuffer(y_array, 0, 1);
    computeEncoder->setBuffer(r_array, 0, 2);
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    computeEncoder->endEncoding();
    commandBuffer->commit();

    commandBuffer->waitUntilCompleted();
}
