package com.baidu.paddle.lite.demo.fom.config;

import android.graphics.Bitmap;

public class Config {

    public String modelPath = "";
    public String fommodelPath = "";
    public String labelPath = "";
    public String imagePath = "";
    public String videoPath = "";
    public int cpuThreadNum = 1;
    public String cpuPowerMode = "";
    public String inputColorFormat = "";
    public int[] inputShape = new int[]{};
    public float[] inputMean = new float[]{};
    public float[] inputStd = new float[]{};

    public void init(String modelPath, String fommodelPath, String labelPath,
                     String imagePath, String videoPath, int cpuThreadNum,
                     String cpuPowerMode, String inputColorFormat,int[] inputShape,
                     float[] inputMean,float[] inputStd ){

        this.modelPath = modelPath;
        this.fommodelPath = fommodelPath;
        this.labelPath = labelPath;
        this.imagePath = imagePath;
        this.videoPath = videoPath;
        this.cpuThreadNum = cpuThreadNum;
        this.cpuPowerMode = cpuPowerMode;
        this.inputColorFormat = inputColorFormat;
        this.inputShape = inputShape;
        this.inputMean = inputMean;
        this.inputStd = inputStd;
    }

    public void setInputShape(Bitmap inputImage){
        this.inputShape[0] = 1;
        this.inputShape[1] = 3;
        this.inputShape[2] = inputImage.getHeight();
        this.inputShape[3] = inputImage.getWidth();
    }

}
