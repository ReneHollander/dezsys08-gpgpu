package at.hollanderkalauner.gpgpu.simplecl;

import org.lwjgl.opencl.*;

import java.util.Arrays;
import java.util.List;

public class Platform {

    private final CLPlatform clPlatform;

    public Platform(CLPlatform clPlatform) {
        this.clPlatform = clPlatform;
    }

    public CLPlatform getClPlatform() {
        return clPlatform;
    }

    public Device createDevice(int device_type) {
        List<CLDevice> devices = this.getClPlatform().getDevices(device_type);
        return new Device(this, devices.get(0));
    }

    public static Platform createPlatform() {
        return new Platform(CLPlatform.getPlatforms().get(0));
    }

    public CLCapabilities getCapabilities() {
        return getClPlatform().getCapabilities();
    }

    public static List<CLPlatform> getPlatforms() {
        return CLPlatform.getPlatforms();
    }

    public static List<CLPlatform> getPlatforms(CLPlatform.Filter<CLPlatform> filter) {
        return CLPlatform.getPlatforms(filter);
    }

    public long address() {
        return getClPlatform().address();
    }

    public List<CLDevice> getDevices(int device_type) {
        return getClPlatform().getDevices(device_type);
    }

    public static CLCapabilities createCapabilities(long cl_platform_id) {
        return CLPlatform.createCapabilities(cl_platform_id);
    }

    public List<CLDevice> getDevices(int device_type, CLPlatform.Filter<CLDevice> filter) {
        return getClPlatform().getDevices(device_type, filter);
    }

    public String getProfile() {
        return Info.clGetPlatformInfoStringUTF8(address(), CL10.CL_PLATFORM_PROFILE).trim();
    }

    public String getVersion() {
        return Info.clGetPlatformInfoStringUTF8(address(), CL10.CL_PLATFORM_VERSION).trim();
    }

    public String getName() {
        return Info.clGetPlatformInfoStringUTF8(address(), CL10.CL_PLATFORM_NAME).trim();
    }

    public String getVendor() {
        return Info.clGetPlatformInfoStringUTF8(address(), CL10.CL_PLATFORM_VENDOR).trim();
    }

    public String[] getExtensions() {
        return Info.clGetPlatformInfoStringUTF8(address(), CL10.CL_PLATFORM_EXTENSIONS).split(" ");
    }

    @Override
    public String toString() {
        return "Platform{" +
                "profile='" + getProfile() + '\'' +
                ", version='" + getVersion() + '\'' +
                ", name='" + getName() + '\'' +
                ", vendor='" + getVendor() + '\'' +
                ", extensions=" + Arrays.toString(getExtensions()) +
                '}';
    }
}
