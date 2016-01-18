package at.hollanderkalauner.gpgpu;

import com.google.common.base.Splitter;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.*;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;

import static at.hollanderkalauner.gpgpu.Util.*;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CLUtil.checkCLError;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memDecodeUTF8;

public class BruteforceMain {

    private static final int MAX_SOURCE_SIZE = 0x100000;

    private static final int GLOBAL_ITEM_SIZE = 8192;
    private static final int LOCAL_ITEM_SIZE = 64;
    private static final double BILLION = 1E9;

    private static final String PW_HASH = "c75e86c6362f42a5b07cfe0f66d3d10a";


    private static final CLContextCallback CREATE_CONTEXT_CALLBACK = new CLContextCallback() {
        @Override
        public void invoke(long errinfo, long private_info, long cb, long user_data) {
            System.err.println("[LWJGL] cl_create_context_callback");
            System.err.println("\tInfo: " + memDecodeUTF8(errinfo));
        }
    };

    public static void main(String[] args) throws Exception {
        int maxlen = 7;

        long permutations = 0;
        for (long i = 1; i <= maxlen; i++) {
            permutations += Math.pow(26, i);
        }
        long permutationsPerThread = permutations / GLOBAL_ITEM_SIZE;
        long missingPermutations = permutations - permutationsPerThread * GLOBAL_ITEM_SIZE;

        System.out.format("Permutations globally: %d!\n", permutations);

        System.out.format("Permutations per thread: %d!\n", permutationsPerThread);
        System.out.format("Permutations missing: %d!\n", missingPermutations);
        System.out.println();


        int[] starts = new int[GLOBAL_ITEM_SIZE];
        int[] stops = new int[GLOBAL_ITEM_SIZE];
        int[] pw_hash = new int[4];

        List<String> pwHashList = Splitter.fixedLength(8).splitToList(PW_HASH);
        for (int i = 0; i < pwHashList.size(); i++) {
            pw_hash[i] = (int) Long.parseLong(pwHashList.get(i), 16);
        }

        int count = 0;
        for (int i = 0; i < GLOBAL_ITEM_SIZE; i++) {
            starts[i] = count;
            count += permutationsPerThread;
            stops[i] = count;
            count++;
        }
        stops[GLOBAL_ITEM_SIZE - 1] += missingPermutations;

        IntBuffer startsBuf = toIntBuffer(starts);
        IntBuffer stopsBuf = toIntBuffer(stops);
        IntBuffer pwHashBuf = toIntBuffer(pw_hash);
        IntBuffer maxlenBuf = asIntBuffer(maxlen);
        ByteBuffer crackedPwBuf = BufferUtils.createByteBuffer(maxlen + 1);


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
        long startsMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, startsBuf, null);
        clEnqueueWriteBuffer(queue, startsMem, 1, 0, startsBuf, null, null);
        long stopsMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, stopsBuf, null);
        clEnqueueWriteBuffer(queue, stopsMem, 1, 0, stopsBuf, null, null);
        long pwHashMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pwHashBuf, null);
        clEnqueueWriteBuffer(queue, pwHashMem, 1, 0, pwHashBuf, null, null);
        long maxlenMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, maxlenBuf, null);
        clEnqueueWriteBuffer(queue, maxlenMem, 1, 0, maxlenBuf, null, null);
        long crackedMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, crackedPwBuf, null);
        clFinish(queue);

        // Create our program and kernel
        long program = clCreateProgramWithSource(context, readFully("kernels/bruteforce.cl"), null);

        int err = clBuildProgram(program, devices.get(0).address(), "-I kernels", null, 0L);
        if (err == CL_BUILD_PROGRAM_FAILURE) {
            System.out.println(Info.clGetProgramBuildInfoStringUTF8(program, devices.get(0).address(), CL_PROGRAM_BUILD_LOG));
        }
        // sum has to match a kernel method name in the OpenCL source
        long kernel = clCreateKernel(program, "vector_add", null);

        // Execution our kernel
        PointerBuffer kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
        kernel1DGlobalWorkSize.put(0, GLOBAL_ITEM_SIZE);

        clSetKernelArg1p(kernel, 0, startsMem);
        clSetKernelArg1p(kernel, 1, stopsMem);
        clSetKernelArg1p(kernel, 2, maxlenMem);
        clSetKernelArg1p(kernel, 3, pwHashMem);
        clSetKernelArg1p(kernel, 4, crackedMem);

        clEnqueueNDRangeKernel(queue, kernel, 1, null, kernel1DGlobalWorkSize, null, null, null);

        // Read the results memory back into our result buffer
        clEnqueueReadBuffer(queue, crackedMem, 1, 0, crackedPwBuf, null, null);
        clFinish(queue);

        // Print the result memory
        System.out.println("Cracked password: " + Util.toString(crackedPwBuf));

        // Clean up OpenCL resources
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(startsMem);
        clReleaseMemObject(stopsMem);
        clReleaseMemObject(maxlenMem);
        clReleaseMemObject(pwHashMem);
        clReleaseMemObject(crackedMem);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        CL.destroy();
    }


}