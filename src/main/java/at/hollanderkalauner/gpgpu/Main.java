package at.hollanderkalauner.gpgpu;

import at.hollanderkalauner.gpgpu.simplecl.*;
import org.lwjgl.BufferUtils;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import static at.hollanderkalauner.gpgpu.Util.*;
import static org.lwjgl.opencl.CL10.*;

public class Main {
    private static final int ITERATIONS = 8;
    private static final int LOCAL_ITEM_SIZE = 64;
    private static final int GLOBAL_ITEM_SIZE = 8192;
    private static final String PW_HASH = "c75e86c6362f42a5b07cfe0f66d3d10a";

    public static void main(String[] args) throws Exception {
        int device_type = CL_DEVICE_TYPE_GPU;

        if (args.length == 1) {
            if (args[0].equals("cpu")) {
                device_type = CL_DEVICE_TYPE_CPU;
            }
        }

        System.out.println("Using device type: " + Device.deviceTypeToString(device_type));

        Platform platform = Platform.createPlatform();
        System.out.println(platform);
        Device device = platform.createDevice(device_type);
        System.out.println(device);
        Context context = device.createContext();
        CommandQueue commandQueue = context.createCommandQueue();
        Program program = context.createProgram(Util.readResource("/kernels/bruteforce.cl"));
        program.build();
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
        int[] pw_hash = hexStringToIntArray(PW_HASH);

        int count = 0;
        for (int i = 0; i < GLOBAL_ITEM_SIZE; i++) {
            starts[i] = count;
            count += permutationsPerThread;
            stops[i] = count;
            count++;
        }
        stops[GLOBAL_ITEM_SIZE - 1] += missingPermutations;

        for (int i = 0; i < ITERATIONS; i++) {
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

            kernel.clSetKernelArg1p(0, startsMem);
            kernel.clSetKernelArg1p(1, stopsMem);
            kernel.clSetKernelArg1p(2, maxlenMem);
            kernel.clSetKernelArg1p(3, pwHashMem);
            kernel.clSetKernelArg1p(4, crackedMem);

            commandQueue.finish();
            long start = System.nanoTime();
            commandQueue.enqueueNDRangeKernel(kernel, 1, null, GLOBAL_ITEM_SIZE, LOCAL_ITEM_SIZE, null, null);
            commandQueue.finish();
            long time = System.nanoTime() - start;

            // Read the results memory back into our result buffer
            clEnqueueReadBuffer(commandQueue.address(), crackedMem, 1, 0, crackedPwBuf, null, null);
            commandQueue.finish();

            // Print the result memory
            //System.out.println("Cracked password: " + Util.toString(crackedPwBuf));
            System.out.println(/*"Time: " + */(time / 1000000) /*+ "ms" */);

            // Clean up OpenCL resources
            clReleaseMemObject(startsMem);
            clReleaseMemObject(stopsMem);
            clReleaseMemObject(maxlenMem);
            clReleaseMemObject(pwHashMem);
            clReleaseMemObject(crackedMem);
        }
        kernel.release();
        program.release();
        commandQueue.release();
        context.release();
    }


}