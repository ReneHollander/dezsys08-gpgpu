package at.hollanderkalauner.gpgpu.simplecl;

import org.lwjgl.BufferUtils;
import org.lwjgl.opencl.Info;

import java.nio.IntBuffer;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CLUtil.checkCLError;

public class Program implements Releaseable {
    private final Context context;
    private final long address;

    public Program(Context context, long address) {
        this.context = context;
        this.address = address;
    }

    public void build(String options) {
        int errCode = clBuildProgram(address(), getContext().getDevice().address(), options, null, 0L);
        if (errCode == CL_BUILD_PROGRAM_FAILURE) {
            System.err.println(Info.clGetProgramBuildInfoStringUTF8(address(), getContext().getDevice().address(), CL_PROGRAM_BUILD_LOG));
        } else if (errCode != CL_SUCCESS) {
            System.err.println(Info.clGetProgramBuildInfoStringUTF8(address(), getContext().getDevice().address(), CL_PROGRAM_BUILD_LOG));
            checkCLError(errCode);
        }
    }

    public void build() {
        build("");
    }

    public Kernel createKernel(String name) {
        IntBuffer errcode_ret = BufferUtils.createIntBuffer(1);
        long kernelAddress = clCreateKernel(address(), name, null);
        checkCLError(errcode_ret);
        return new Kernel(this, kernelAddress);
    }

    public Context getContext() {
        return context;
    }

    public long address() {
        return address;
    }

    @Override
    public void release() {
        clReleaseProgram(address());
    }
}
