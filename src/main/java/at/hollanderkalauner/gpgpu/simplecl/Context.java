package at.hollanderkalauner.gpgpu.simplecl;

import org.lwjgl.BufferUtils;
import org.lwjgl.opencl.CLContextCallback;

import java.nio.IntBuffer;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CLUtil.checkCLError;
import static org.lwjgl.system.MemoryUtil.memDecodeUTF8;

public class Context implements Releaseable {

    public static final CLContextCallback CREATE_CONTEXT_CALLBACK = new CLContextCallback() {
        @Override
        public void invoke(long errinfo, long private_info, long cb, long user_data) {
            System.err.println("[LWJGL] cl_create_context_callback");
            System.err.println("\tInfo: " + memDecodeUTF8(errinfo));
        }
    };
    private final Device device;
    private final long address;

    public Context(Device device, long address) {
        this.device = device;
        this.address = address;
    }

    public Device getDevice() {
        return device;
    }

    public long address() {
        return address;
    }

    public CommandQueue createCommandQueue() {
        IntBuffer errcode_ret = BufferUtils.createIntBuffer(1);
        long queueAddress = clCreateCommandQueue(address(), getDevice().address(), CL_QUEUE_PROFILING_ENABLE, errcode_ret);
        checkCLError(errcode_ret);
        return new CommandQueue(this, queueAddress);
    }

    @Override
    public void release() {
        clReleaseContext(address());
    }

    public Program createProgram(String source) {
        IntBuffer errcode_ret = BufferUtils.createIntBuffer(1);
        long programAddress = clCreateProgramWithSource(address(), source, null);
        checkCLError(errcode_ret);
        return new Program(this, programAddress);
    }

    public Program createProgram(String... sources) {
        IntBuffer errcode_ret = BufferUtils.createIntBuffer(1);
        long programAddress = clCreateProgramWithSource(address(), sources, null);
        checkCLError(errcode_ret);
        return new Program(this, programAddress);
    }
}
