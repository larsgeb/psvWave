#include <metal_stdlib>
using namespace metal;


kernel void add_arrays(device const float* X [[buffer(0)]],
                       device const float* Y [[buffer(1)]],
                       device float* result  [[buffer(2)]],
                       uint index            [[thread_position_in_grid]])
{
    result[index] = X[index] + Y[index];
}


kernel void multiply_arrays(device const float* X [[buffer(0)]],
                            device const float* Y [[buffer(1)]],
                            device float* result  [[buffer(2)]],
                            uint index            [[thread_position_in_grid]])
{
    result[index] = X[index] * Y[index];
}
kernel void multiply_array_constant(device const float* X     [[buffer(0)]],
                                    device const float* alpha [[buffer(1)]],
                                    device float* result      [[buffer(2)]],
                                    uint index                [[thread_position_in_grid]])
{
    result[index] = X[index] * alpha[0];
}

kernel void saxpy(device const float* a [[buffer(0)]],
                  device const float* X [[buffer(1)]],
                  device const float* Y [[buffer(2)]],
                  device float* result  [[buffer(3)]],
                  uint index            [[thread_position_in_grid]])
{
    result[index] = (*a) * X[index] + Y[index];
}

kernel void central_difference(
                  device const float* delta [[buffer(0)]],
                  device const float* X     [[buffer(1)]],
                  device float* result      [[buffer(2)]],
                  uint index                [[thread_position_in_grid]],
                  uint arrayLength          [[threads_per_grid]])
{
    if (index == 0)
    {
        result[index] = (X[index + 1] - X[index]) /  *delta;
    }
    else if (index == arrayLength - 1)
    {
        result[index] = (X[index] - X[index - 1]) /  *delta;
    }
    else
    {
        result[index] = (X[index + 1] - X[index - 1]) / (2 * *delta);
    }
}


int linear_IDX(int pos1, int pos2, int shape1, int shape2)
{
  return pos1 * shape2 + pos2;
}


kernel void stress_integrate_2d(
                  device float* txx         [[buffer(0)]],
                  device float* tzz         [[buffer(1)]],
                  device float* txz         [[buffer(2)]],
                  device const float* taper [[buffer(3)]],
                  device const float* _dt   [[buffer(4)]],
                  device const float* _dx   [[buffer(5)]],
                  device const float* _dz   [[buffer(6)]],
                  device const float* vx    [[buffer(7)]],
                  device const float* vz    [[buffer(8)]],
                  device const float* lm    [[buffer(9)]],
                  device const float* la    [[buffer(10)]],
                  device const float* mu    [[buffer(11)]],
                  device const float* b_vx  [[buffer(12)]],
                  device const float* b_vz  [[buffer(13)]],
                  uint2 index               [[thread_position_in_grid]],
                  uint2 grid                [[threads_per_grid]])
{

    int ix = index.x;
    int iz = index.y;
    int nx = grid.x;
    int nz = grid.y;

    int idx = linear_IDX(ix, iz, nx, nz);

    // Skip the outer 2 points on all sides (stencil can't be applied). This introduces
    // reflecting boundaries which are supressed using the taper.
    if (ix > 1 and ix < nx - 2 and iz > 1 and iz < nz - 2){

        float c1 = float(9.0 / 8.0);
        float c2 = float(1.0 / 24.0);

        float dx = *_dx;
        float dz = *_dz;
        float dt = *_dt;

        int idx_xp1 = linear_IDX(ix + 1, iz, nx, nz);
        int idx_xp2 = linear_IDX(ix + 2, iz, nx, nz);
        int idx_xm1 = linear_IDX(ix - 1, iz, nx, nz);
        int idx_xm2 = linear_IDX(ix - 2, iz, nx, nz);
        int idx_zm1 = linear_IDX(ix, iz - 1, nx, nz);
        int idx_zm2 = linear_IDX(ix, iz - 2, nx, nz);
        int idx_zp1 = linear_IDX(ix, iz + 1, nx, nz);
        int idx_zp2 = linear_IDX(ix, iz + 2, nx, nz);

        txx[idx] =
            taper[idx] *
            (txx[idx] + dt * 
                (
                    lm[idx] * (
                            c1 * (vx[idx_xp1] - vx[idx]) +
                            c2 * (vx[idx_xm1] - vx[idx_xp2])
                        ) / dx +
                    la[idx] * (
                            c1 * (vz[idx] - vz[idx_zm1]) +
                            c2 * (vz[idx_zm2] - vz[idx_zp1])
                        ) / dz
                )
            );
        tzz[idx] =
            taper[idx] *
            (tzz[idx] + dt * (la[idx] *
                                    (c1 * (vx[idx_xp1] - vx[idx]) +
                                    c2 * (vx[idx_xm1] - vx[idx_xp2])) /
                                    dx +
                                (lm[idx]) *
                                    (c1 * (vz[idx] - vz[idx_zm1]) +
                                    c2 * (vz[idx_zm2] - vz[idx_zp1])) /
                                    dz));
        txz[idx] = taper[idx] *
                    (txz[idx] + dt * mu[idx] *
                                    ((c1 * (vx[idx_zp1] - vx[idx]) +
                                        c2 * (vx[idx_zm1] - vx[idx_zp2])) /
                                        dz +
                                    (c1 * (vz[idx] - vz[idx_xm1]) +
                                        c2 * (vz[idx_xm2] - vz[idx_xp1])) /
                                        dx));
    } 
}



kernel void txx_tzz_integrate_2d(
                  device float* txx         [[buffer(0)]],
                  device float* tzz         [[buffer(1)]],
                  device float* txz         [[buffer(2)]],
                  device const float* taper [[buffer(3)]],
                  device const float* _dt   [[buffer(4)]],
                  device const float* _dx   [[buffer(5)]],
                  device const float* _dz   [[buffer(6)]],
                  device const float* vx    [[buffer(7)]],
                  device const float* vz    [[buffer(8)]],
                  device const float* lm    [[buffer(9)]],
                  device const float* la    [[buffer(10)]],
                  device const float* mu    [[buffer(11)]],
                  device const float* b_vx  [[buffer(12)]],
                  device const float* b_vz  [[buffer(13)]],
                  uint2 index               [[thread_position_in_grid]],
                  uint2 grid                [[threads_per_grid]])
{

    int ix = index.x;
    int iz = index.y;
    int nx = grid.x;
    int nz = grid.y;

    int idx = linear_IDX(ix, iz, nx, nz);

    // Skip the outer 2 points on all sides (stencil can't be applied). This introduces
    // reflecting boundaries which are supressed using the taper.
    if (ix > 1 and ix < nx - 2 and iz > 1 and iz < nz - 2){

        float c1 = float(9.0 / 8.0);
        float c2 = float(1.0 / 24.0);

        float dx = *_dx;
        float dz = *_dz;
        float dt = *_dt;

        int idx_xp1 = linear_IDX(ix + 1, iz, nx, nz);
        int idx_xp2 = linear_IDX(ix + 2, iz, nx, nz);
        int idx_xm1 = linear_IDX(ix - 1, iz, nx, nz);
        int idx_zm1 = linear_IDX(ix, iz - 1, nx, nz);
        int idx_zm2 = linear_IDX(ix, iz - 2, nx, nz);
        int idx_zp1 = linear_IDX(ix, iz + 1, nx, nz);

        txx[idx] =
            taper[idx] *
            (txx[idx] + dt * 
                (
                    lm[idx] * (
                            c1 * (vx[idx_xp1] - vx[idx]) +
                            c2 * (vx[idx_xm1] - vx[idx_xp2])
                        ) / dx +
                    la[idx] * (
                            c1 * (vz[idx] - vz[idx_zm1]) +
                            c2 * (vz[idx_zm2] - vz[idx_zp1])
                        ) / dz
                )
            );
        tzz[idx] =
            taper[idx] *
            (tzz[idx] + dt * (la[idx] *
                                    (c1 * (vx[idx_xp1] - vx[idx]) +
                                    c2 * (vx[idx_xm1] - vx[idx_xp2])) /
                                    dx +
                                (lm[idx]) *
                                    (c1 * (vz[idx] - vz[idx_zm1]) +
                                    c2 * (vz[idx_zm2] - vz[idx_zp1])) /
                                    dz));
    } 
}



kernel void velocity_integrate_2d(
                  device const float* txx   [[buffer(0)]],
                  device const float* tzz   [[buffer(1)]],
                  device const float* txz   [[buffer(2)]],
                  device const float* taper [[buffer(3)]],
                  device const float* _dt   [[buffer(4)]],
                  device const float* _dx   [[buffer(5)]],
                  device const float* _dz   [[buffer(6)]],
                  device float* vx          [[buffer(7)]],
                  device float* vz          [[buffer(8)]],
                  device const float* lm    [[buffer(9)]],
                  device const float* la    [[buffer(10)]],
                  device const float* mu    [[buffer(11)]],
                  device const float* b_vx  [[buffer(12)]],
                  device const float* b_vz  [[buffer(13)]],
                  uint2 index               [[thread_position_in_grid]],
                  uint2 grid                [[threads_per_grid]])
{

    int ix = index.x;
    int iz = index.y;
    int nx = grid.x;
    int nz = grid.y;
    
    // Skip the outer 2 points on all sides (stencil can't be applied). This introduces
    // reflecting boundaries which are supressed using the taper.
    if (ix > 1 and ix < nx - 2 and iz > 1 and iz < nz - 2){

        float c1 = float(9.0 / 8.0);
        float c2 = float(1.0 / 24.0);

        float dx = *_dx;
        float dz = *_dz;
        float dt = *_dt;

        int idx = linear_IDX(ix, iz, nx, nz);
        int idx_xp1 = linear_IDX(ix + 1, iz, nx, nz);
        int idx_xp2 = linear_IDX(ix + 2, iz, nx, nz);
        int idx_xm1 = linear_IDX(ix - 1, iz, nx, nz);
        int idx_xm2 = linear_IDX(ix - 2, iz, nx, nz);
        int idx_zm1 = linear_IDX(ix, iz - 1, nx, nz);
        int idx_zm2 = linear_IDX(ix, iz - 2, nx, nz);
        int idx_zp1 = linear_IDX(ix, iz + 1, nx, nz);
        int idx_zp2 = linear_IDX(ix, iz + 2, nx, nz);

        vx[idx] = taper[idx] *
                (vx[idx] + b_vx[idx] * dt *
                                ((c1 * (txx[idx] - txx[idx_xm1]) +
                                    c2 * (txx[idx_xm2] - txx[idx_xp1])) /
                                    dx +
                                (c1 * (txz[idx] - txz[idx_zm1]) +
                                    c2 * (txz[idx_zm2] - txz[idx_zp1])) /
                                    dz));
        vz[idx] = taper[idx] *
                (vz[idx] + b_vz[idx] * dt *
                                ((c1 * (txz[idx_xp1] - txz[idx]) +
                                    c2 * (txz[idx_xm1] - txz[idx_xp2])) /
                                    dx +
                                (c1 * (tzz[idx_zp1] - tzz[idx]) +
                                    c2 * (tzz[idx_zm1] - tzz[idx_zp2])) /
                                    dz));

    } 
}

kernel void vx_integrate_2d(
                  device const float* txx   [[buffer(0)]],
                  device const float* tzz   [[buffer(1)]],
                  device const float* txz   [[buffer(2)]],
                  device const float* taper [[buffer(3)]],
                  device const float* _dt   [[buffer(4)]],
                  device const float* _dx   [[buffer(5)]],
                  device const float* _dz   [[buffer(6)]],
                  device float* vx          [[buffer(7)]],
                  device float* vz          [[buffer(8)]],
                  device const float* lm    [[buffer(9)]],
                  device const float* la    [[buffer(10)]],
                  device const float* mu    [[buffer(11)]],
                  device const float* b_vx  [[buffer(12)]],
                  device const float* b_vz  [[buffer(13)]],
                  uint2 index               [[thread_position_in_grid]],
                  uint2 grid                [[threads_per_grid]])
{

    int ix = index.x;
    int iz = index.y;
    int nx = grid.x;
    int nz = grid.y;
    
    // Skip the outer 2 points on all sides (stencil can't be applied). This introduces
    // reflecting boundaries which are supressed using the taper.
    if (ix > 1 and ix < nx - 2 and iz > 1 and iz < nz - 2){

        float c1 = float(9.0 / 8.0);
        float c2 = float(1.0 / 24.0);

        float dx = *_dx;
        float dz = *_dz;
        float dt = *_dt;

        int idx = linear_IDX(ix, iz, nx, nz);
        int idx_xp1 = linear_IDX(ix + 1, iz, nx, nz);
        int idx_xm1 = linear_IDX(ix - 1, iz, nx, nz);
        int idx_xm2 = linear_IDX(ix - 2, iz, nx, nz);
        int idx_zm1 = linear_IDX(ix, iz - 1, nx, nz);
        int idx_zm2 = linear_IDX(ix, iz - 2, nx, nz);
        int idx_zp1 = linear_IDX(ix, iz + 1, nx, nz);

        vx[idx] = taper[idx] *
                (vx[idx] + b_vx[idx] * dt *
                                ((c1 * (txx[idx] - txx[idx_xm1]) +
                                    c2 * (txx[idx_xm2] - txx[idx_xp1])) /
                                    dx +
                                (c1 * (txz[idx] - txz[idx_zm1]) +
                                    c2 * (txz[idx_zm2] - txz[idx_zp1])) /
                                    dz));

    } 
}

kernel void inspector(
                  device const float* X                        [[buffer(0)]],
                  device float* result                         [[buffer(1)]],
                  device uint* store                           [[buffer(2)]],
                  uint thread_position_in_grid                 [[thread_position_in_grid]],
                  uint threads_per_grid                        [[threads_per_grid]],
                  uint dispatch_quadgroups_per_threadgroup     [[dispatch_quadgroups_per_threadgroup]],
                  uint dispatch_simdgroups_per_threadgroup     [[dispatch_simdgroups_per_threadgroup]], 
                  uint dispatch_threads_per_threadgroup        [[dispatch_threads_per_threadgroup]], 
                  uint grid_origin                             [[grid_origin]], 
                  uint grid_size                               [[grid_size]], 
                  uint quadgroup_index_in_threadgroup          [[quadgroup_index_in_threadgroup]], 
                  uint quadgroups_per_threadgroup              [[quadgroups_per_threadgroup]], 
                  uint simdgroup_index_in_threadgroup          [[simdgroup_index_in_threadgroup]], 
                  uint simdgroups_per_threadgroup              [[simdgroups_per_threadgroup]], 
                  uint thread_execution_width                  [[thread_execution_width]], 
                  uint thread_index_in_quadgroup               [[thread_index_in_quadgroup]], 
                  uint thread_index_in_simdgroup               [[thread_index_in_simdgroup]], 
                  uint thread_index_in_threadgroup             [[thread_index_in_threadgroup]], 
                  uint thread_position_in_threadgroup          [[thread_position_in_threadgroup]], 
                  uint threadgroup_position_in_grid            [[threadgroup_position_in_grid]],
                  uint threadgroups_per_grid                   [[threadgroups_per_grid]], 
                  uint threads_per_simdgroup                   [[threads_per_simdgroup]], 
                  uint threads_per_threadgroup                 [[threads_per_threadgroup]])
{
    result[thread_position_in_grid] = X[thread_position_in_grid] + 1.0;

    if (thread_position_in_grid == 200){
        store[0] = thread_position_in_grid;
        store[1] = threads_per_grid;
        store[2] = dispatch_quadgroups_per_threadgroup; // quadgroup is 4 simd groups
        store[3] = dispatch_simdgroups_per_threadgroup;
        store[4] = dispatch_threads_per_threadgroup;
        store[5] = grid_origin;
        store[6] = grid_size;
        store[7] = quadgroup_index_in_threadgroup;
        store[8] = quadgroups_per_threadgroup;      
        store[9] = simdgroup_index_in_threadgroup;
        store[10] = simdgroups_per_threadgroup;
        store[11] = thread_execution_width;
        store[12] = thread_index_in_quadgroup;
        store[13] = thread_index_in_simdgroup;
        store[14] = thread_index_in_threadgroup;
        store[15] = thread_position_in_threadgroup;
        store[16] = threadgroup_position_in_grid;
        store[17] = threadgroups_per_grid;
        store[18] = threads_per_simdgroup;
        store[19] = threads_per_threadgroup;
    }

}
