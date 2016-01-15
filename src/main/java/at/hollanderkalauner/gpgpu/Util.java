package at.hollanderkalauner.gpgpu;

import org.apache.commons.io.IOUtils;
import org.lwjgl.BufferUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.FloatBuffer;

public class Util {

    public static String readFully(String file) throws IOException {
        return readFully(new File(file));
    }

    public static String readFully(File file) throws IOException {
        return IOUtils.toString(new FileInputStream(file), java.nio.charset.StandardCharsets.UTF_8);
    }

    /**
     * Utility method to convert float array to float buffer
     *
     * @param floats - the float array to convert
     * @return a float buffer containing the input float array
     */
    public static FloatBuffer toFloatBuffer(float[] floats) {
        FloatBuffer buf = BufferUtils.createFloatBuffer(floats.length).put(floats);
        buf.rewind();
        return buf;
    }


    /**
     * Utility method to toString a float buffer
     *
     * @param buffer - the float buffer to toString to System.out
     */
    public static String toString(FloatBuffer buffer) {
        StringBuilder str = new StringBuilder(buffer.capacity() * 2);
        str.append('[');
        for (int i = 0; i < buffer.capacity(); i++) {
            str.append(buffer.get(i));
            if (i != buffer.capacity() - 1) {
                str.append(", ");
            }
        }
        str.append(']');
        return str.toString();
    }

}
