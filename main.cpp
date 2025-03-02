#include "opencl_wrapper.h"
#include <iostream>
#include <vector>

int main() {
    const char *source = R"(
        __kernel void add(__global const float *a, __global const float *b, __global float *c) {
            int i = get_global_id(0);
            c[i] = a[i] + b[i];
        }
    )";

    ocl::proque pq(source, CL_QUEUE_PROFILING_ENABLE);

    const int N = 1'000'000'00;

    std::vector<float> a(N), b(N), c(N);

    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(N - i);
    }

    cl_mem a_buf = pq.create_buffer(N * sizeof(float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.data());
    cl_mem b_buf = pq.create_buffer(N * sizeof(float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.data());
    cl_mem c_buf = pq.create_buffer(N * sizeof(float), CL_MEM_WRITE_ONLY, nullptr);

    pq.create_kernel("add");
    pq.set_arg(0, &a_buf);
    pq.set_arg(1, &b_buf);
    pq.set_arg(2, &c_buf);

    double duration = pq.run_kernel_with_profiling(N);

    pq.finish();

    pq.read_buffer(c_buf, N * sizeof(float), c.data());

    const float eps = 1e-6;
    for (int i = 0; i < N; i++) {
        if (std::fabs(c[i] - (a[i] + b[i])) > eps) {
            std::cerr << "Error at index " << i << ": " << c[i] << " != " << a[i] << " + " << b[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Success! Time: " 
        << duration << " ms" 
        << std::endl;
    return 0;
}