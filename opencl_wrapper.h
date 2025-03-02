#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdexcept>
#include <string>

namespace ocl {

    inline std::string to_error_string(cl_int err);

    inline std::runtime_error cl_error(cl_int err);

    struct proque {
        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;
        cl_device_id device = nullptr;

        ~proque();

        proque(const char *source, cl_queue_flags flags = 0);

        cl_mem create_buffer(size_t size, cl_mem_flags flags, void* data);

        void create_kernel(const char *name);

        void set_arg(cl_uint index, const void *value, size_t size);

        void set_arg(cl_uint index, const void* value);

        void run_kernel(size_t work_size, size_t local_work_size = 0);

        double run_kernel_with_profiling(size_t work_size, size_t local_work_size = 0);

        void finish();

        void read_buffer(cl_mem buffer, size_t size, void* ptr);

        void write_buffer(cl_mem buffer, size_t size, const void* ptr);
    };
}