package at.hollanderkalauner.gpgpu;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.*;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;

import static at.hollanderkalauner.gpgpu.Util.readFully;
import static at.hollanderkalauner.gpgpu.Util.toFloatBuffer;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CLUtil.checkCLError;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memDecodeUTF8;

public class Main {

    // Data buffers to store the input and result data in
    static final FloatBuffer a = toFloatBuffer(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    static final FloatBuffer b = toFloatBuffer(new float[]{9, 8, 7, 6, 5, 4, 3, 2, 1, 0});
    static final FloatBuffer answer = BufferUtils.createFloatBuffer(a.capacity());

    private static final CLContextCallback CREATE_CONTEXT_CALLBACK = new CLContextCallback() {
        @Override
        public void invoke(long errinfo, long private_info, long cb, long user_data) {
            System.err.println("[LWJGL] cl_create_context_callback");
            System.err.println("\tInfo: " + memDecodeUTF8(errinfo));
        }
    };

    public static void main(String[] args) throws Exception {
        // System.out.println(CL.getICD().toString());

        CLPlatform platform = CLPlatform.getPlatforms().get(0);

        PointerBuffer ctxProps = BufferUtils.createPointerBuffer(3);
        ctxProps.put(CL_CONTEXT_PLATFORM).put(platform).put(0).flip();

        IntBuffer errcode_ret = BufferUtils.createIntBuffer(1);

        List<CLDevice> devices = platform.getDevices(CL_DEVICE_TYPE_GPU);
        long context = clCreateContext(ctxProps, devices.get(0).address(), CREATE_CONTEXT_CALLBACK, NULL, errcode_ret);

        checkCLError(errcode_ret);
        long queue = clCreateCommandQueue(context, devices.get(0).address(), CL_QUEUE_PROFILING_ENABLE, errcode_ret);

        // Allocate memory for our two input buffers and our result buffer
        long aMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a, null);
        clEnqueueWriteBuffer(queue, aMem, 1, 0, a, null, null);
        long bMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b, null);
        clEnqueueWriteBuffer(queue, bMem, 1, 0, b, null, null);
        long answerMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, answer, null);
        clFinish(queue);

        // Create our program and kernel
        long program = clCreateProgramWithSource(context, readFully("kernels/helloworld.cl"), null);

        CLUtil.checkCLError(clBuildProgram(program, devices.get(0).address(), "", null, 0L));
        // sum has to match a kernel method name in the OpenCL source
        long kernel = clCreateKernel(program, "sum", null);

        // Execution our kernel
        PointerBuffer kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
        kernel1DGlobalWorkSize.put(0, a.capacity());

        clSetKernelArg1p(kernel, 0, aMem);
        clSetKernelArg1p(kernel, 1, bMem);
        clSetKernelArg1p(kernel, 2, answerMem);

        clEnqueueNDRangeKernel(queue, kernel, 1, null, kernel1DGlobalWorkSize, null, null, null);

        // Read the results memory back into our result buffer
        clEnqueueReadBuffer(queue, answerMem, 1, 0, answer, null, null);
        clFinish(queue);

        // Print the result memory
        System.out.println(Util.toString(a));
        System.out.println("+");
        System.out.println(Util.toString(b));
        System.out.println("=");
        System.out.println(Util.toString(answer));

        // Clean up OpenCL resources
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(aMem);
        clReleaseMemObject(bMem);
        clReleaseMemObject(answerMem);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        CL.destroy();
    }


}