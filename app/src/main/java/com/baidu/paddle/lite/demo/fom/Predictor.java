package com.baidu.paddle.lite.demo.fom;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;
import com.baidu.paddle.lite.demo.fom.config.Config;
import com.baidu.paddle.lite.demo.fom.preprocess.Preprocess;
import com.baidu.paddle.lite.demo.fom.visual.Visualize;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Vector;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class Predictor {
    private static final String TAG = Predictor.class.getSimpleName();
    protected Vector<String> wordLabels = new Vector<String>();

    Config config;
    Config fom_config;

    public boolean isLoaded = false;
    public int warmupIterNum = 1;
    public int inferIterNum = 1;
    protected Context appCtx = null;
    public int cpuThreadNum = 1;
    public String cpuPowerMode = "LITE_POWER_HIGH";
    public String modelPath = "";
    public String fommodelPath = "";
    public String modelName = "";
    public String fommodelName = "";
    public String cachePath = "";
    public String videoPath = "";
    protected PaddlePredictor paddlePredictor = null;
    protected PaddlePredictor kpPredictor = null;
    protected PaddlePredictor generatorPredictor = null;
    protected float inferenceTime = 0;
    protected float kpInferenceTime = 0;
    protected float generatorInferenceTime = 0;

    protected Bitmap inputImage = null;
    protected List<Bitmap> inputVideo = null;
    protected Bitmap scaledImage = null;
    protected Bitmap outputImage = null;
    protected String outputResult = "";
    protected float preprocessTime = 0;
    protected float postprocessTime = 0;

    public Predictor() {
        super();
    }



    public boolean init(Context appCtx, String modelPath, String fommodelPath, String videoPath,
                        int cpuThreadNum, String cpuPowerMode) {
        this.appCtx = appCtx;
        isLoaded = loadModel(modelPath, fommodelPath, cpuThreadNum, cpuPowerMode);
        cachePath = appCtx.getCacheDir() + "/";
        setVideo(videoPath);
        return isLoaded;
    }


    public boolean init(Context appCtx, Config config) {

        if (config.inputShape.length != 4) {
            Log.i(TAG, "size of input shape should be: 4");
            return false;
        }
        if (config.inputMean.length != config.inputShape[1]) {
            Log.i(TAG, "size of input mean should be: " + Long.toString(config.inputShape[1]));
            return false;
        }
        if (config.inputStd.length != config.inputShape[1]) {
            Log.i(TAG, "size of input std should be: " + Long.toString(config.inputShape[1]));
            return false;
        }
        if (config.inputShape[0] != 1) {
            Log.i(TAG, "only one batch is supported in the image classification demo, you can use any batch size in " +
                    "your Apps!");
            return false;
        }
        if (config.inputShape[1] != 1 && config.inputShape[1] != 3) {
            Log.i(TAG, "only one/three channels are supported in the image classification demo, you can use any " +
                    "channel size in your Apps!");
            return false;
        }
        if (!config.inputColorFormat.equalsIgnoreCase("RGB") && !config.inputColorFormat.equalsIgnoreCase("BGR")) {
            Log.i(TAG, "only RGB and BGR color format is supported.");
            return false;
        }
        init(appCtx, config.modelPath, config.fommodelPath, config.videoPath, config.cpuThreadNum, config.cpuPowerMode);
        if (!isLoaded()) {
            return false;
        }
        isLoaded &= loadLabel(config.labelPath);
        this.config = config;

        return isLoaded;
    }

    protected boolean loadLabel(String labelPath) {
        wordLabels.clear();
        // load word labels from file
        try {
            InputStream assetsInputStream = appCtx.getAssets().open(labelPath);
            int available = assetsInputStream.available();
            byte[] lines = new byte[available];
            assetsInputStream.read(lines);
            assetsInputStream.close();
            String words = new String(lines);
            String[] contents = words.split("\n");
            for (String content : contents) {
                wordLabels.add(content);
            }
            Log.i(TAG, "word label size: " + wordLabels.size());
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            return false;
        }
        return true;
    }

    public boolean runModel(Bitmap image, List<Bitmap> videos) {
        setInputImage(image, videos);
        return runModel();
    }

    public boolean runModel(Preprocess preprocess, Visualize visualize) {
        if (inputImage == null) {
            return false;
        }
        // set input shape
        Tensor inputTensor = getInput(0);
        //inputTensor.resize(config.inputShape);

        // pre-process image
        Date start = new Date();
        preprocess.init(config);
        preprocess.process(scaledImage);
        // feed input tensor with pre-processed data
        inputTensor.setData(preprocess.inputData);
        Date end = new Date();
        preprocessTime = (float) (end.getTime() - start.getTime());

        // inference
        runModel();

        start = new Date();
        this.outputImage = inputImage;
        // post-process
        //visualize.nms(this.outputImage, this.paddlePredictor);
        //visualize.draw(visualize.boxAndScores,this.outputImage, this.outputResult);
        postprocessTime = (float) (end.getTime() - start.getTime());

        outputResult = new String();
        return true;
    }

    public boolean runFomModel(Preprocess preprocess) {
        if (inputImage == null) {
            return false;
        }
        // set input shape
        Tensor inputTensorImage = getKpInput(0);

        config.inputShape[2] = 256;
        config.inputShape[3] = 256;
        preprocess.init(config);
        preprocess.process(inputImage);
        //inputTensorImage.resize(config.inputShape);
        // feed input tensor with pre-processed data
        inputTensorImage.setData(preprocess.inputData);
        // inference
        runKpModel();

        Tensor source_j = this.kpPredictor.getOutput(0);
        Tensor source_v = this.kpPredictor.getOutput(1);



        Preprocess preprocess_driving = new Preprocess();;
        preprocess_driving.init(config);
        Bitmap driving_init = inputVideo.get(0);
        preprocess_driving.process(driving_init);
        // feed input tensor with pre-processed data
        inputTensorImage.setData(preprocess_driving.inputData);

        // inference
        runKpModel();
        Tensor driving_init_j = this.kpPredictor.getOutput(0);
        Tensor driving_init_v = this.kpPredictor.getOutput(1);

        Bitmap driving = inputVideo.get(1);
        preprocess_driving.process(driving);
        // feed input tensor with pre-processed data
        inputTensorImage.setData(preprocess_driving.inputData);

        // inference
        runKpModel();
        Tensor driving_j = this.kpPredictor.getOutput(0);
        Tensor driving_v = this.kpPredictor.getOutput(1);
        List<Tensor> GeneratorInput = new ArrayList<>();
        GeneratorInput.add(source_j);
        GeneratorInput.add(source_v);
        GeneratorInput.add(driving_j);
        GeneratorInput.add(driving_v);
        GeneratorInput.add(driving_init_j);
        GeneratorInput.add(driving_init_v);

        Tensor inputTensorGen = getGeneratorInput(0);
        //inputTensorGen.resize(config.inputShape);
        inputTensorGen.setData(preprocess.inputData);
        for(int i = 1; i < 7; i++) {
            inputTensorGen = getGeneratorInput(i);
            inputTensorGen.resize(GeneratorInput.get(i-1).shape());
            inputTensorGen.setData(GeneratorInput.get(i-1).getFloatData());
        }

        runGeneratorModel();

        Tensor res = this.generatorPredictor.getOutput(0);

        this.outputImage = inputVideo.get(1);;

        outputResult = new String();

        return true;
    }

    public boolean isLoaded() {
        return paddlePredictor != null && isLoaded;
    }

    public String modelPath() {
        return modelPath;
    }

    public String modelName() {
        return modelName;
    }
    public String fommodelName() {
        return fommodelName;
    }

    public int cpuThreadNum() {
        return cpuThreadNum;
    }

    public String cpuPowerMode() {
        return cpuPowerMode;
    }

    public float inferenceTime() {
        return inferenceTime;
    }

    public float kpInferenceTime() {
        return kpInferenceTime;
    }

    public float generatorInferenceTime() {
        return generatorInferenceTime;
    }

    public void setConfig(Config config){
        this.config = config;
    }

    public Bitmap inputImage() {
        return inputImage;
    }

    public Bitmap outputImage() {
        return outputImage;
    }

    public String outputResult() {
        return outputResult;
    }

    public float preprocessTime() {
        return preprocessTime;
    }

    public float postprocessTime() {
        return postprocessTime;
    }

    protected boolean setVideo(String videoPath) {
        String videorealPath = videoPath;
        if (!videoPath.substring(0, 1).equals("/")) {
            // read model files from custom path if the first character of mode path is '/'
            // otherwise copy model to cache from assets

            videorealPath = appCtx.getCacheDir() + "/fom/images";
            Utils.copyDirectoryFromAssets(appCtx, "fom/images", videorealPath);
            videorealPath = appCtx.getCacheDir() + "/" + videoPath;
        }
        this.videoPath = videorealPath;
        return true;
    }

    protected boolean loadModel(String modelPath, String fommodelPath,
                                int cpuThreadNum, String cpuPowerMode) {
        // release model if exists
        releaseModel();

        // load model
        if (modelPath.isEmpty()) {
            return false;
        }
        if (fommodelPath.isEmpty()) {
            return false;
        }
        String realPath = modelPath;
        if (!modelPath.substring(0, 1).equals("/")) {
            // read model files from custom path if the first character of mode path is '/'
            // otherwise copy model to cache from assets
            realPath = appCtx.getCacheDir() + "/" + modelPath;
            Utils.copyDirectoryFromAssets(appCtx, modelPath, realPath);
        }
        if (realPath.isEmpty()) {
            return false;
        }
        System.out.println(realPath);
        String fomrealPath = fommodelPath;
        if (!fommodelPath.substring(0, 1).equals("/")) {
            // read model files from custom path if the first character of mode path is '/'
            // otherwise copy model to cache from assets
            fomrealPath = appCtx.getCacheDir() + "/" + fommodelPath;
            Utils.copyDirectoryFromAssets(appCtx, fommodelPath, fomrealPath);
        }

        if (fomrealPath.isEmpty()) {
            return false;
        }

        MobileConfig config = new MobileConfig();
        config.setModelFromFile(realPath + File.separator + "model.nb");
        config.setThreads(cpuThreadNum);
        MobileConfig generator_config = new MobileConfig();
        generator_config.setModelFromFile(fomrealPath + File.separator + "generator_lite.nb");
        generator_config.setThreads(cpuThreadNum);
        MobileConfig kp_detector_config = new MobileConfig();
        kp_detector_config.setModelFromFile(fomrealPath + File.separator + "kp_detector_lite.nb");
        kp_detector_config.setThreads(cpuThreadNum);
        if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_HIGH);
            kp_detector_config.setPowerMode(PowerMode.LITE_POWER_HIGH);
            generator_config.setPowerMode(PowerMode.LITE_POWER_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_LOW);
            kp_detector_config.setPowerMode(PowerMode.LITE_POWER_LOW);
            generator_config.setPowerMode(PowerMode.LITE_POWER_LOW);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_FULL")) {
            config.setPowerMode(PowerMode.LITE_POWER_FULL);
            kp_detector_config.setPowerMode(PowerMode.LITE_POWER_FULL);
            generator_config.setPowerMode(PowerMode.LITE_POWER_FULL);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_NO_BIND")) {
            config.setPowerMode(PowerMode.LITE_POWER_NO_BIND);
            kp_detector_config.setPowerMode(PowerMode.LITE_POWER_NO_BIND);
            generator_config.setPowerMode(PowerMode.LITE_POWER_NO_BIND);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_HIGH);
            kp_detector_config.setPowerMode(PowerMode.LITE_POWER_RAND_HIGH);
            generator_config.setPowerMode(PowerMode.LITE_POWER_RAND_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_LOW);
            kp_detector_config.setPowerMode(PowerMode.LITE_POWER_RAND_LOW);
            generator_config.setPowerMode(PowerMode.LITE_POWER_RAND_LOW);
        } else {
            Log.e(TAG, "unknown cpu power mode!");
            return false;
        }
        paddlePredictor = PaddlePredictor.createPaddlePredictor(config);
        kpPredictor = PaddlePredictor.createPaddlePredictor(kp_detector_config);
        generatorPredictor = PaddlePredictor.createPaddlePredictor(generator_config);

        this.cpuThreadNum = cpuThreadNum;
        this.cpuPowerMode = cpuPowerMode;
        this.modelPath = realPath;
        this.modelName = realPath.substring(realPath.lastIndexOf("/") + 1);
        this.fommodelPath = fomrealPath;
        this.fommodelName = fomrealPath.substring(fomrealPath.lastIndexOf("/") + 1);
        return true;
    }

    public void releaseModel() {
        paddlePredictor = null;
        kpPredictor = null;
        generatorPredictor = null;
        isLoaded = false;
        cpuThreadNum = 1;
        cpuPowerMode = "LITE_POWER_HIGH";
        modelPath = "";
        fommodelPath = "";
        modelName = "";
        fommodelName = "";
    }

    public Tensor getInput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getInput(idx);
    }

    public Tensor getKpInput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return kpPredictor.getInput(idx);
    }

    public Tensor getGeneratorInput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return generatorPredictor.getInput(idx);
    }

    public Tensor getOutput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getOutput(idx);
    }

    public boolean runModel() {
        if (!isLoaded()) {
            return false;
        }
        // warm up
        for (int i = 0; i < warmupIterNum; i++){
            paddlePredictor.run();
        }
        // inference
        Date start = new Date();
        for (int i = 0; i < inferIterNum; i++) {
            paddlePredictor.run();
        }
        Date end = new Date();
        inferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;
        return true;
    }

    public boolean runKpModel() {
        if (!isLoaded()) {
            return false;
        }
        // warm up
        for (int i = 0; i < warmupIterNum; i++){
            kpPredictor.run();
        }
        // inference
        Date start = new Date();
        for (int i = 0; i < inferIterNum; i++) {
            kpPredictor.run();
        }
        Date end = new Date();
        kpInferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;
        return true;
    }


    public boolean runGeneratorModel() {
        if (!isLoaded()) {
            return false;
        }
        // warm up
        for (int i = 0; i < warmupIterNum; i++){
            generatorPredictor.run();
        }
        // inference
        Date start = new Date();
        for (int i = 0; i < inferIterNum; i++) {
            generatorPredictor.run();
        }
        Date end = new Date();
        generatorInferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;
        return true;
    }


    public void setInputImage(Bitmap image, List<Bitmap> videos) {
        if (image == null) {
            return;
        }
        // scale image to the size of input tensor
        Bitmap rgbaImage = image.copy(Bitmap.Config.ARGB_8888, true);
        Log.i(TAG, "config inputShape:"+this.config.inputShape[3]);
        Bitmap scaleImage = Bitmap.createScaledBitmap(rgbaImage, (int) this.config.inputShape[3], (int) this.config.inputShape[2], true);
        this.inputImage = rgbaImage;
        this.scaledImage = scaleImage;
        this.inputVideo = videos;
    }
}
