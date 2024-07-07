package com.baidu.paddle.lite.demo.fom;

import android.content.Context;

public class Native {
    static {
        System.loadLibrary("Native");
    }

    private long ctx = 0;

    private boolean isInit = false;

    public boolean isLoaded() {
        return isInit;
    }

    public boolean init(String modelDir,
                        String fommodelDir,
                        String labelPath,
                        int cpuThreadNum,
                        String cpuPowerMode,
                        int inputWidth,
                        int inputHeight,
                        float[] inputMean,
                        float[] inputStd) {
        ctx = nativeInit(
                modelDir,
                fommodelDir,
                labelPath,
                cpuThreadNum,
                cpuPowerMode,
                inputWidth,
                inputHeight,
                inputMean,
                inputStd, 0);
        System.out.println("=============");
        System.out.println(modelDir);
        System.out.println(fommodelDir);
        System.out.println(labelPath);
        isInit = true;
        return true;
        //return ctx == 0;
    }

    public boolean release() {
        if (ctx == 0) {
            return false;
        }
        return nativeRelease(ctx);
    }

    public boolean process(int inTextureId, int outTextureId, int textureWidth, int textureHeight, String savedImagePath) {
        if (ctx == 0) {
            return false;
        }
        return nativeProcess(ctx, inTextureId, outTextureId, textureWidth, textureHeight, savedImagePath);
    }

    public boolean fomProcess(String imagePath, String videoPath) {
        if (ctx == 0) {
            return false;
        }
        return nativeFomProcess(ctx, imagePath, videoPath);
    }

    public static native long nativeInit(String modelDir,
                                         String fommodelDir,
                                         String labelPath,
                                         int cpuThreadNum,
                                         String cpuPowerMode,
                                         int inputWidth,
                                         int inputHeight,
                                         float[] inputMean,
                                         float[] inputStd, float score);

    public static native boolean nativeRelease(long ctx);

    public static native boolean nativeProcess(long ctx, int inTextureId, int outTextureId, int textureWidth, int textureHeight, String savedImagePath);

    public static native boolean nativeFomProcess(long ctx, String imagePath, String videoPath);

}

