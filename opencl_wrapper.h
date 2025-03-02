#include <OpenCL/opencl.h>
#include <stdexcept>
#include <iostream>

namespace ocl {

    inline std::string to_error_string(cl_int err) {
        switch (err) {
            case CL_SUCCESS: return "Success";
            case CL_DEVICE_NOT_FOUND: return "Device not found";
            case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
            case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
            case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
            case CL_OUT_OF_RESOURCES: return "Out of resources";
            case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
            case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling information not available";
            case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
            case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
            case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
            case CL_BUILD_PROGRAM_FAILURE: return "Program build failure";
            case CL_MAP_FAILURE: return "Map failure";

            default: {
                return "Unknown OpenCL error: " + std::to_string(err);
            }
        }
    }

    inline std::runtime_error cl_error(cl_int err) {
        return std::runtime_error("OpenCL error: " + to_error_string(err));
    }

    struct proque {
        cl_context context = nullptr;
        cl_command_queue queue = nullptr;
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;
        cl_device_id device = nullptr;

        ~proque() {
            if (kernel) {
                clReleaseKernel(kernel);
            }
            if (program) {
                clReleaseProgram(program);
            }
            if (queue) {
                clReleaseCommandQueue(queue);
            }
            if (context) {
                clReleaseContext(context);
            }
        }


        proque(const char *source, cl_queue_flags flags = 0) {
            cl_int err;
            cl_platform_id platform;
            err = clGetPlatformIDs(1, &platform, nullptr);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }

            context = clCreateContextFromType(
                (cl_context_properties[]){CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0},
                CL_DEVICE_TYPE_GPU,
                nullptr, nullptr, &err);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }

            cl_device_id device;
            err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }

            queue = clCreateCommandQueue(context, device, flags, &err);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }

            program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }

            err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                size_t log_size;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::string log(log_size, '\0');
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
                throw std::runtime_error("Failed to build OpenCL program:\n" + log);
            }
        }

        cl_mem create_buffer(size_t size, cl_mem_flags flags, void* data) {
            cl_int err;
            cl_mem buffer = clCreateBuffer(context, flags, size, data, &err);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }
            return buffer;
        }

        void create_kernel(const char *name) {
            cl_int err;
            kernel = clCreateKernel(program, name, &err);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }
        }

        void set_arg(cl_uint index, const void *value, size_t size) {
            // maybe add check kernel != nullptr
            cl_int err = clSetKernelArg(kernel, index, size, value);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to set OpenCL kernel argument");
            }
        }

        void set_arg(cl_uint index, const void* value) {
            set_arg(index, value, sizeof(value));
        }

        void run_kernel(size_t work_size, size_t local_work_size = 0) {
            const size_t* lws = (local_work_size != 0 ? &local_work_size : nullptr);
            cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &work_size,
                lws, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }
        }

        double run_kernel_with_profiling(size_t work_size, size_t local_work_size = 0) {
            const size_t* lws = (local_work_size != 0 ? &local_work_size : nullptr);
            cl_event kernel_event;
            cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &work_size, lws, 0, nullptr, &kernel_event);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }

            finish();

            cl_ulong start_time, end_time;

            err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }
            err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }
            clReleaseEvent(kernel_event);
            double execution_time = (end_time - start_time) * 1e-6;
            return execution_time;

        }

        void finish() {
            cl_int err = clFinish(queue);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }
        }

        void read_buffer(cl_mem buffer, size_t size, void* ptr) {
            cl_int err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size, ptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }
        }

        void write_buffer(cl_mem buffer, size_t size, const void* ptr) {
            cl_int err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, size, ptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                throw cl_error(err);
            }
        }

    };
}