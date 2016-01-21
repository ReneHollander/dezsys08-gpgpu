package at.hollanderkalauner.gpgpu;

import at.hollanderkalauner.gpgpu.simplecl.*;
import com.google.common.base.Splitter;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.List;

import static at.hollanderkalauner.gpgpu.Util.*;
import static org.lwjgl.opencl.CL10.*;

public class Main {

    private static final int MAX_SOURCE_SIZE = 0x100000;

    private static final int GLOBAL_ITEM_SIZE = 8192;
    private static final String PW_HASH = "c75e86c6362f42a5b07cfe0f66d3d10a";

    public static void main(String[] args) throws Exception {

        Platform platform = Platform.createPlatform();
        System.out.println(platform);
        Device device = platform.createDevice(CL_DEVICE_TYPE_GPU);
        System.out.println(device);
        Context context = device.createContext();
        CommandQueue commandQueue = context.createCommandQueue();
        Program program = context.createProgram(readFully("kernels/bruteforce.cl"));
        program.build("-I kernels");
        Kernel kernel = program.createKernel("vector_add");

        int maxlen = 6;

        long permutations = 0;
        for (long i = 1; i <= maxlen; i++) {
            permutations += Math.pow(26, i);
        }
        long permutationsPerThread = permutations / GLOBAL_ITEM_SIZE;
        long missingPermutations = permutations - permutationsPerThread * GLOBAL_ITEM_SIZE;

        System.out.format("Permutations globally: %d!\n", permutations);
        System.out.format("Permutations per thread: %d!\n", permutationsPerThread);
        System.out.format("Permutations missing: %d!\n", missingPermutations);


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

        // Allocate memory for our two input buffers and our result buffer
        long startsMem = clCreateBuffer(context.address(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, startsBuf, null);
        clEnqueueWriteBuffer(commandQueue.address(), startsMem, 1, 0, startsBuf, null, null);
        long stopsMem = clCreateBuffer(context.address(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, stopsBuf, null);
        clEnqueueWriteBuffer(commandQueue.address(), stopsMem, 1, 0, stopsBuf, null, null);
        long pwHashMem = clCreateBuffer(context.address(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pwHashBuf, null);
        clEnqueueWriteBuffer(commandQueue.address(), pwHashMem, 1, 0, pwHashBuf, null, null);
        long maxlenMem = clCreateBuffer(context.address(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, maxlenBuf, null);
        clEnqueueWriteBuffer(commandQueue.address(), maxlenMem, 1, 0, maxlenBuf, null, null);
        long crackedMem = clCreateBuffer(context.address(), CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, crackedPwBuf, null);
        commandQueue.finish();

        // Execution our kernel
        PointerBuffer kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
        kernel1DGlobalWorkSize.put(0, GLOBAL_ITEM_SIZE);

        clSetKernelArg1p(kernel.address(), 0, startsMem);
        clSetKernelArg1p(kernel.address(), 1, stopsMem);
        clSetKernelArg1p(kernel.address(), 2, maxlenMem);
        clSetKernelArg1p(kernel.address(), 3, pwHashMem);
        clSetKernelArg1p(kernel.address(), 4, crackedMem);

        long start = System.nanoTime();
        clEnqueueNDRangeKernel(commandQueue.address(), kernel.address(), 1, null, kernel1DGlobalWorkSize, null, null, null);

        // Read the results memory back into our result buffer
        clEnqueueReadBuffer(commandQueue.address(), crackedMem, 1, 0, crackedPwBuf, null, null);
        commandQueue.finish();
        long time = System.nanoTime() - start;

        // Print the result memory
        System.out.println("Cracked password: " + Util.toString(crackedPwBuf));
        System.out.println("Time: " + (time / 1000000) + "ms");

        // Clean up OpenCL resources
        clReleaseMemObject(startsMem);
        clReleaseMemObject(stopsMem);
        clReleaseMemObject(maxlenMem);
        clReleaseMemObject(pwHashMem);
        clReleaseMemObject(crackedMem);
        kernel.release();
        program.release();
        commandQueue.release();
        context.release();
    }


}