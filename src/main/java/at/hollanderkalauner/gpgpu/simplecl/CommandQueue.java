package at.hollanderkalauner.gpgpu.simplecl;

import static org.lwjgl.opencl.CL10.clFinish;
import static org.lwjgl.opencl.CL10.clReleaseCommandQueue;

public class CommandQueue implements Releaseable {

    private final Context context;
    private final long address;

    public CommandQueue(Context context, long address) {
        this.context = context;
        this.address = address;
    }

    public Context getContext() {
        return context;
    }

    public long address() {
        return address;
    }

    @Override
    public void release() {
        clReleaseCommandQueue(address());
    }

    public void finish() {
        clFinish(address());
    }
}
