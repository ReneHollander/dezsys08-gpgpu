package at.hollanderkalauner.gpgpu.simplecl;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.opencl.CLCapabilities;
import org.lwjgl.opencl.CLDevice;
import org.lwjgl.opencl.Info;

import java.nio.IntBuffer;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CLUtil.checkCLError;

public class Device {

    private final Platform platform;
    private final CLDevice clDevice;

    public Device(Platform platform, CLDevice clDevice) {
        this.platform = platform;
        this.clDevice = clDevice;
    }

    public Context createContext() {
        PointerBuffer ctxProps = BufferUtils.createPointerBuffer(3);
        ctxProps.put(CL10.CL_CONTEXT_PLATFORM).put(getPlatform().getClPlatform()).put(0).flip();
        IntBuffer errcode_ret = BufferUtils.createIntBuffer(1);
        long address = clCreateContext(ctxProps, address(), Context.CREATE_CONTEXT_CALLBACK, 0, errcode_ret);
        checkCLError(errcode_ret);
        return new Context(this, address);
    }

    public Platform getPlatform() {
        return platform;
    }

    public CLDevice getClDevice() {
        return clDevice;
    }

    public CLCapabilities getCapabilities() {
        return getClDevice().getCapabilities();
    }

    public static CLCapabilities createCapabilities(long cl_device_id, CLCapabilities platformCapabilities) {
        return CLDevice.createCapabilities(cl_device_id, platformCapabilities);
    }

    public long address() {
        return getClDevice().address();
    }

    public String getName() {
        return Info.clGetDeviceInfoStringUTF8(address(), CL_DEVICE_NAME).trim();
    }

    public String getProfile() {
        return Info.clGetDeviceInfoStringUTF8(address(), CL_DEVICE_PROFILE).trim();
    }

    public String getVendor() {
        return Info.clGetDeviceInfoStringUTF8(address(), CL_DEVICE_VENDOR).trim();
    }

    public String getVersion() {
        return Info.clGetDeviceInfoStringUTF8(address(), CL_DEVICE_VERSION).trim();
    }

    public int getType() {
        return (int) Info.clGetDeviceInfoLong(address(), CL_DEVICE_TYPE);
    }

    public String getDriverVersion() {
        int i = CL_DEVICE_TYPE_ALL;
        return Info.clGetDeviceInfoStringUTF8(address(), CL_DRIVER_VERSION);
    }

    public static String deviceTypeToString(int type) {
        if (type == CL_DEVICE_TYPE_DEFAULT) return "DEFAULT";
        else if (type == CL_DEVICE_TYPE_CPU) return "CUP";
        else if (type == CL_DEVICE_TYPE_GPU) return "GPU";
        else if (type == CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
        else throw new IllegalArgumentException("Specified integer not a device type: " + type);
    }

    @Override
    public String toString() {
        return "Device{" +
                "driverVersion='" + getDriverVersion() + '\'' +
                ", type='" + deviceTypeToString(getType()) + '\'' +
                ", version='" + getVersion() + '\'' +
                ", vendor='" + getVendor() + '\'' +
                ", profile='" + getProfile() + '\'' +
                ", name='" + getName() + '\'' +
                ", platform=" + platform +
                '}';
    }
}
