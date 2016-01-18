package at.hollanderkalauner.gpgpu.simplecl;

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

    @Override
    public void release() {
        clReleaseKernel(address());
    }
}
