# forwardGPU
Forward 2D elastic wave equation modelling using either OpenMP or OpenACC. Compiles with PGI compiler.

Compilation is fairly easy with CMake. Just make sure to point towards your C++ and C compiler in the CMakeLists.txt. Compilation is done by:

```
    $ cmake . -DFLOATS=OFF    // or -DFLOATS=ON
    $ make gpuWave
    $ make cpuWave
```

Running the GPU code is straightforward:
```
    $./gpuWave
```
Running the CPU code requires setting the OMP_NUM_THREADS environment variable to correspond to your preference (usually the amount of physical, 
not logical, cores in your pc). In my case, I use a Intel i7-8850H, 12 threads, 6 cores. Although I could use 12 threads, it probably won't be any 
faster as the process would be using all available physical cores anyway. If I want
 to use 6 threads for 
just one run:
```
    $ OMP_NUM_THREADS=6 ./cpu.program
 ```
 ## Main controls on speed
GPU's are very fast in some very specific cases. They are fastest when there is a lot of work (computations) to be done, with limited memory 
copies to the host machine. Conditional statements typically decrease GPU performance. However, porting the wave propagation code required minimal
alteration from the CPU code. One source code now can be compiled to both targets.
  
GPU's are fastest when the blocks they work on are not too small such that they must shift positions often, but also not too big such that only a
few blocks fit in the computational domain. Very small physical problems will therefore likely be faster on CPU code.
   
The type of computation performed also affects running time. GPU's are ideal for float operations, but are on par with CPU's on double operations. 
See also the benchmarks below. 
 
For extended computation, GPU seems to have better performance even on floats. 
 
 
 ## Benchmark
 Benchmark on a Dell Precision 5530 using a Quadro P2000 (4GB) vs. an Intel i7-8850H, 16GB ram. The wave problem solved had dimensions:
 ```
    nt = 250
    nx = 4096
    nz = 1024
```
 The dimension nt only affects time linearly, and does typically not affect memory usage when not storing wavefields.
 
 The number shown at the end of the computation is the summation over 1 array of the wavefield vx, to ensure deterministic computations. Re-rerunning should give the same result, CPU/GPU should give the same result, double vs. float should not give the same result..
 
 
 **Using floats:**
 
 ```
$ ./gpuWave && ./cpuWave 

OpenACC acceleration enabled from cmake, code should run on GPU.
Code compiled with f (d for double, accurate, f for float, fast)
Seconds elapsed for wave simulation: 2.27787
-3.28162e-17

OpenACC acceleration not enabled from cmake, code should run on CPU.
Code compiled with f (d for double, accurate, f for float, fast)
Seconds elapsed for wave simulation: 5.87679
-3.28162e-17
```
**Using doubles:**
```
$ ./gpuWave && ./cpuWave 

OpenACC acceleration enabled from cmake, code should run on GPU.
Code compiled with d (d for double, accurate, f for float, fast)
Seconds elapsed for wave simulation: 7.25039
-3.2829e-17

OpenACC acceleration not enabled from cmake, code should run on CPU.
Code compiled with d (d for double, accurate, f for float, fast)
Seconds elapsed for wave simulation: 7.20166
-3.2829e-17
 
```
As expected, different precisions give different deterministic results, to within 1%.


Running nvidia-smi during a GPU run shows full utilization of cores, not nearly full utilization of memory:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.79       Driver Version: 410.79       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Quadro P2000        Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   54C    P0    N/A /  N/A |    747MiB /  4042MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     xxxxx      G   ---- other processes ----                    166MiB |
|    0     xxxxx      G   ---- other processes ----                     84MiB |
|    0     xxxxx      G   ---- other processes ----                      4MiB |
|    0     xxxxx      G   ---- other processes ----                     44MiB |
|    0     xxxxx      G   ---- other processes ----                     35MiB |
|    0     28689      C   ./gpuWave                                    401MiB |
+-----------------------------------------------------------------------------+
```


**Large computations: GPU outperforms CPU on doubles**

Rerunning with:
 ```
    nt = 2500   // This changed
    nx = 4096
    nz = 1024
```
Gives:
```
$ ./gpuWave && ./cpuWave 

OpenACC acceleration enabled from cmake, code should run on GPU.
Code compiled with d (d for double, accurate, f for float, fast)
Seconds elapsed for wave simulation: 64.0353
-5.00058e-20

OpenACC acceleration not enabled from cmake, code should run on CPU.
Code compiled with d (d for double, accurate, f for float, fast)
Seconds elapsed for wave simulation: 78.9906
-5.00058e-20
```
Faster on GPU!

Also on floats of course:
```
$ ./gpuWave && ./cpuWave 

OpenACC acceleration enabled from cmake, code should run on GPU.
Code compiled with f (d for double, accurate, f for float, fast)
Seconds elapsed for wave simulation: 20.8963
-7.10227e-20

OpenACC acceleration not enabled from cmake, code should run on CPU.
Code compiled with f (d for double, accurate, f for float, fast)
Seconds elapsed for wave simulation: 62.6243
-7.10227e-20
```
Mind the strong deviation in deterministic sums.