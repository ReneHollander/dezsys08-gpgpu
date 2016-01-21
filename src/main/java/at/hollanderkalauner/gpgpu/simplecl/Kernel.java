package at.hollanderkalauner.gpgpu.simplecl;

import org.lwjgl.opencl.CL10;

import static org.lwjgl.opencl.CL10.clReleaseKernel;

public class Kernel implements Releaseable {
    private final Program program;
    private final long address;

    public Kernel(Program program, long address) {
        this.program = program;
        this.address = address;
    }

    public Program getProgram() {
        return program;
    }

    public long address() {
        return address;
    }

    public void clSetKernelArg1p(int idx, long arg) {
        CL10.clSetKernelArg1p(address(), idx, arg);
    }

    @Override
    public void release() {
        clReleaseKernel(address());
    }
}
