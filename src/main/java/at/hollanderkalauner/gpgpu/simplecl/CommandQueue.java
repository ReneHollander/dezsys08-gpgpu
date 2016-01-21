package at.hollanderkalauner.gpgpu.simplecl;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import static org.lwjgl.opencl.CL10.*;

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

    public void enqueueNDRangeKernel(Kernel kernel, int work_dim, PointerBuffer global_work_offset, int global_work_size, int local_work_size, PointerBuffer event_wait_list, PointerBuffer event) {
        PointerBuffer globalWorkSize = null;
        PointerBuffer localWorkSize = null;
        if (global_work_size > 0) {
            globalWorkSize = BufferUtils.createPointerBuffer(1);
            globalWorkSize.put(0, global_work_size);
        }
        if (local_work_size > 0) {
            localWorkSize = BufferUtils.createPointerBuffer(1);
            localWorkSize.put(0, local_work_size);
        }
        clEnqueueNDRangeKernel(address(), kernel.address(), work_dim, global_work_offset, globalWorkSize, localWorkSize, event_wait_list, event);
    }
}
