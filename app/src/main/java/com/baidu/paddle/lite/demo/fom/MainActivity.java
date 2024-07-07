package com.baidu.paddle.lite.demo.fom;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaMetadataRetriever;
import android.media.MediaPlayer;
import android.os.Build;
import android.view.View;
import android.widget.Button;
import android.widget.MediaController;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.appcompat.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VideoView;
import android.os.Environment;


import com.baidu.paddle.lite.demo.fom.config.Config;
import com.baidu.paddle.lite.demo.fom.preprocess.Preprocess;
//import com.baidu.paddle.lite.demo.fom.visual.Visualize;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import nl.bravobit.ffmpeg.ExecuteBinaryResponseHandler;
import nl.bravobit.ffmpeg.FFmpeg;
import nl.bravobit.ffmpeg.FFprobe;
import nl.bravobit.ffmpeg.FFtask;


public class MainActivity extends AppCompatActivity {
    public static final int PICK_IMAGE_REQUEST = 1;
    private static final String TAG = MainActivity.class.getSimpleName();
    public static final int OPEN_GALLERY_REQUEST_CODE = 0;
    public static final int TAKE_PHOTO_REQUEST_CODE = 1;

    public static final int REQUEST_LOAD_MODEL = 0;
    public static final int REQUEST_RUN_MODEL = 1;
    public static final int RESPONSE_LOAD_MODEL_SUCCESSED = 0;
    public static final int RESPONSE_LOAD_MODEL_FAILED = 1;
    public static final int RESPONSE_RUN_MODEL_SUCCESSED = 2;
    public static final int RESPONSE_RUN_MODEL_FAILED = 3;

    protected ProgressDialog pbLoadModel = null;
    protected ProgressDialog pbRunModel = null;

    protected Handler receiver = null; // receive messages from worker thread
    protected Handler sender = null; // send command to worker thread
    protected HandlerThread worker = null; // worker thread to load&run model

    protected Bitmap selectedImage;
    protected ImageView imageView;


    private List<Bitmap> bitmaps = new ArrayList<>();

    // model config
    Config config = new Config();

    //protected Predictor predictor = new Predictor();

    Preprocess preprocess = new Preprocess();

    //Visualize visualize = new Visualize();

    Native predictor = new Native();

    private String selectedImagePath;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            //判断是否拥有权限
            if (ContextCompat.checkSelfPermission(
                    this, Manifest.permission.WRITE_EXTERNAL_STORAGE
            ) != PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(
                    this, Manifest.permission.READ_EXTERNAL_STORAGE
            ) != PackageManager.PERMISSION_GRANTED
            ) {
                Log.i("MainActivity","onCreate >= Build.VERSION_CODES.M denied");
                //没有授权，编写申请权限代码
                ActivityCompat.requestPermissions(
                        this,new String[]{ Manifest.permission.WRITE_EXTERNAL_STORAGE,
                                Manifest.permission.READ_EXTERNAL_STORAGE}, 100);
            } else {
                Log.i("MainActivity","onCreate >= Build.VERSION_CODES.M has");
                initData();
            }
        } else {
            Log.i("MainActivity","onCreate < Build.VERSION_CODES.M denied");
            initData();
        }

        imageView = findViewById(R.id.imageView);

        Button buttonSelectImage = findViewById(R.id.button_select_image);
        Button btnGenerateVideo = findViewById(R.id.button_generate_video);

        buttonSelectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openImageChooser();
            }
        });

        btnGenerateVideo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (selectedImage != null) {
                    Log.d("Button Generate","find image and video ");
                    loadModel();
                }
                else{
                    Log.d("Button Generate","Image or video is null");
                }
            }
        });
    }


    private void openImageChooser() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, PICK_IMAGE_REQUEST);
        Log.d("openImageChooser","ImageURI : " + MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
    }



    public boolean onLoadModel() {
        String realModelDir = getCacheDir() + "/" + config.modelPath;
        Utils.copyDirectoryFromAssets(this, config.modelPath, realModelDir);
        String realFomModelDir = getCacheDir() + "/" + config.fommodelPath;
        Utils.copyDirectoryFromAssets(this, config.fommodelPath, realFomModelDir);
        String realLabelPath = getCacheDir() + "/" + config.labelPath;
        Utils.copyDirectoryFromAssets(this, config.labelPath, realLabelPath);
        return predictor.init(realModelDir,
                realFomModelDir, realLabelPath, config.cpuThreadNum, config.cpuPowerMode,
                config.inputShape[2], config.inputShape[3], config.inputMean, config.inputStd);
    }


    public boolean onRunModel() {
        //return predictor.isLoaded() && predictor.runFomModel(preprocess);
        //return predictor.isLoaded() && predictor.runModel(preprocess,visualize);
        if (!new File(getCacheDir() + "/assets/").exists()) {
            new File(getCacheDir() + "/assets/").mkdirs();
        }
        String realvideoPath = getCacheDir() + "/assets/" +config.videoPath.substring(config.videoPath.lastIndexOf("/") + 1);
        Utils.copyFileFromAssets(this, config.videoPath, realvideoPath);
//        String realimagePath = getCacheDir() + "/assets/" + config.imagePath.substring(config.videoPath.lastIndexOf("/") + 1);
//        Utils.copyFileFromAssets(this, config.imagePath, realimagePath);

        String realimagePath = selectedImagePath;
        Log.d(TAG, "Image Path: " + realimagePath);
        Log.d(TAG, "Video Path: " + realvideoPath);

        predictor.fomProcess(realimagePath, realvideoPath);

        File file = new File("/storage/emulated/0/fom_inference001.mp4");
        if (file.exists()) {file.delete();}

        String[] command = {"-i", "/storage/emulated/0/fom_inference001.avi", "/storage/emulated/0/fom_inference001.mp4"};

        final FFtask task = FFmpeg.getInstance(this).execute(command, new ExecuteBinaryResponseHandler() {
        });
        return true;
    }

    public void onLoadModelFailed() {

    }
    public void onRunModelFailed() {
    }

    public void loadModel() {
        pbLoadModel = ProgressDialog.show(this, "", "Loading model...", false, false);
        sender.sendEmptyMessage(REQUEST_LOAD_MODEL);
    }

    public void runModel() {
        pbRunModel = ProgressDialog.show(this, "", "Running model...", false, false);
        sender.sendEmptyMessage(REQUEST_RUN_MODEL);
    }

    public List<Bitmap>  getBitmapsFromVideo(String datapath) {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        retriever.setDataSource(datapath);
        int fps = 25;
        int US_OF_S = 1;//1000 * 1000;
        String time = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
        // 取得视频的长度(单位为秒)
        int seconds = Integer.valueOf(time) / 1000;
        double inc = 1;//US_OF_S / fps;
        // 得到每一秒时刻的bitmap比如第一秒,第二秒
        int num = 0;
        for (double i = 0; i < seconds*US_OF_S; i+=inc) {
            Bitmap bitmap = retriever.getFrameAtTime((long)i,MediaMetadataRetriever.OPTION_CLOSEST);
            if (bitmap != null) {
                Bitmap scaleImage = Bitmap.createScaledBitmap(bitmap, 256, 256, true);
                bitmaps.add(scaleImage);
            }
            num += 1;
        }

        return bitmaps;
    }


    public void onLoadModelSuccessed() {
        // load test image from file_paths and run model
        //try {
//        if (config.imagePath.isEmpty()) {
//            return;
//        }
        if (config.videoPath.isEmpty()) {
            return;
        }
            /*
            Bitmap image = null;
            List<Bitmap> videos = null;
            // read test image file from custom file_paths if the first character of mode file_paths is '/', otherwise read test
            // image file from assets
            if (!config.imagePath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(config.imagePath);
                image = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(config.imagePath).exists()) {
                    return;
                }
                image = BitmapFactory.decodeFile(config.imagePath);
            }
            if (!config.videoPath.substring(0, 1).equals("/")) {
                videos = getBitmapsFromVideo(predictor.videoPath);
            } else {
                if (!new File(config.videoPath).exists()) {
                    return;
                }
                videos = getBitmapsFromVideo(config.videoPath);
            }
            if (image != null && predictor.isLoaded()) {
                predictor.setInputImage(image, videos);
                runModel();
            }

            runModel();
        } catch (IOException e) {
            Toast.makeText(MainActivity.this, "Load image failed!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }*/
        requestAllPermissions();
        runModel();
    }

    public void onRunModelSuccessed() {
        // obtain results and update UI
        /*
        tvInferenceTime.setText("kp Inference time: " + predictor.kpInferenceTime() + " ms\n" +
                "generator Inference time: " + predictor.generatorInferenceTime() + " ms\n" );
        Bitmap outputImage = predictor.outputImage();
        if (outputImage != null) {
            ivInputImage.setImageBitmap(outputImage);
        }
        tvOutputResult.setText(predictor.outputResult());
        tvOutputResult.scrollTo(0, 0);

         */
    }


    public void onImageChanged(Bitmap image, List<Bitmap> videos) {
        // rerun model if users pick test image from gallery or camera
        if (image != null && predictor.isLoaded()) {
            //predictor.setInputImage(image, videos);
            runModel();
        }
    }


    /*
    public void onImageChanged(String path) {
        Bitmap image = BitmapFactory.decodeFile(path);
        predictor.setInputImage(image);
        runModel();
    }
    */

    public void onSettingsClicked() {
        startActivity(new Intent(MainActivity.this, SettingsActivity.class));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_action_options, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                break;
            case R.id.settings:
                if (requestAllPermissions()) {
                    // make sure we have SDCard r&w permissions to load model from SDCard
                    onSettingsClicked();
                }
                break;
        }
        return super.onOptionsItemSelected(item);
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
        }
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            Uri selectedMediaUri = data.getData();
            if (requestCode == PICK_IMAGE_REQUEST) {
                if (selectedMediaUri != null) {
                    try {
                        InputStream imageStream = getContentResolver().openInputStream(selectedMediaUri);
                        Log.d("fom onActivityResult","selectedMediaUri"+ selectedMediaUri);
                        selectedImage = BitmapFactory.decodeStream(imageStream);
                        imageView.setImageBitmap(selectedImage);
                        // 获取图片路径
                        selectedImagePath = getPathFromUri(selectedMediaUri);
                        Log.d("fom onActivityResult","selectedImagePath:" + selectedImagePath);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }else if(requestCode==100){
            initData();
        }
    }
    private boolean requestAllPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.CAMERA},
                    0);
            return false;
        }
        return true;
    }


    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        boolean isLoaded = predictor.isLoaded();
        menu.findItem(R.id.open_gallery).setEnabled(isLoaded);
        menu.findItem(R.id.take_photo).setEnabled(isLoaded);
        return super.onPrepareOptionsMenu(menu);
    }
    protected void onResume() {
        super.onResume();

        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        boolean settingsChanged = false;
        String model_path = sharedPreferences.getString(getString(R.string.MODEL_PATH_KEY),
                getString(R.string.MODEL_PATH_DEFAULT));
        String fom_model_path = sharedPreferences.getString(getString(R.string.FOM_MODEL_PATH_KEY),
                getString(R.string.FOM_MODEL_PATH_MOBILE));
        String label_path = sharedPreferences.getString(getString(R.string.LABEL_PATH_KEY),
                getString(R.string.LABEL_PATH_DEFAULT));
//        String image_path = sharedPreferences.getString(getString(R.string.IMAGE_PATH_KEY),
//                selectedImagePath);
        String image_path = selectedImagePath;
        String video_path = sharedPreferences.getString(getString(R.string.VIDEO_PATH_KEY),
                getString(R.string.VIDEO_PATH_DEFAULT));
//        video_path = "fom/model/fom_mobile/mayiyahei.mp4";

        settingsChanged |= !model_path.equalsIgnoreCase(config.modelPath);
        settingsChanged |= !label_path.equalsIgnoreCase(config.labelPath);
//        settingsChanged |= !image_path.equalsIgnoreCase(config.imagePath);
        int cpu_thread_num = Integer.parseInt(sharedPreferences.getString(getString(R.string.CPU_THREAD_NUM_KEY),
                getString(R.string.CPU_THREAD_NUM_DEFAULT)));
        settingsChanged |= cpu_thread_num != config.cpuThreadNum;
        String cpu_power_mode =
                sharedPreferences.getString(getString(R.string.CPU_POWER_MODE_KEY),
                        getString(R.string.CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpu_power_mode.equalsIgnoreCase(config.cpuPowerMode);
        String input_color_format =
                sharedPreferences.getString(getString(R.string.INPUT_COLOR_FORMAT_KEY),
                        getString(R.string.INPUT_COLOR_FORMAT_DEFAULT));
        settingsChanged |= !input_color_format.equalsIgnoreCase(config.inputColorFormat);
        int[] input_shape =
                Utils.parseLongsFromString(sharedPreferences.getString(getString(R.string.INPUT_SHAPE_KEY),
                        getString(R.string.INPUT_SHAPE_DEFAULT)), ",");
        float[] input_mean =
                Utils.parseFloatsFromString(sharedPreferences.getString(getString(R.string.INPUT_MEAN_KEY),
                        getString(R.string.INPUT_MEAN_DEFAULT)), ",");
        float[] input_std =
                Utils.parseFloatsFromString(sharedPreferences.getString(getString(R.string.INPUT_STD_KEY)
                        , getString(R.string.INPUT_STD_DEFAULT)), ",");
        settingsChanged |= input_shape.length != config.inputShape.length;
        settingsChanged |= input_mean.length != config.inputMean.length;
        settingsChanged |= input_std.length != config.inputStd.length;
        if (!settingsChanged) {
            for (int i = 0; i < input_shape.length; i++) {
                settingsChanged |= input_shape[i] != config.inputShape[i];
            }
            for (int i = 0; i < input_mean.length; i++) {
                settingsChanged |= input_mean[i] != config.inputMean[i];
            }
            for (int i = 0; i < input_std.length; i++) {
                settingsChanged |= input_std[i] != config.inputStd[i];
            }
        }

        if (settingsChanged) {
            config.init(model_path, fom_model_path, label_path, image_path, video_path, cpu_thread_num, cpu_power_mode,
                    input_color_format, input_shape, input_mean, input_std);
            preprocess.init(config);
            // reload model if configure has been changed
//            loadModel();  // -- 我注释的，放到按钮中触发
        }
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.release();
        }
        worker.quit();
        super.onDestroy();
    }

    public void initData() {
        receiver = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case RESPONSE_LOAD_MODEL_SUCCESSED:
                        pbLoadModel.dismiss();
                        onLoadModelSuccessed();
                        break;
                    case RESPONSE_LOAD_MODEL_FAILED:
                        pbLoadModel.dismiss();
                        Toast.makeText(MainActivity.this, "Load model failed!", Toast.LENGTH_SHORT).show();
                        onLoadModelFailed();
                        break;
                    case RESPONSE_RUN_MODEL_SUCCESSED:
                        pbRunModel.dismiss();
                        onRunModelSuccessed();
                        break;
                    case RESPONSE_RUN_MODEL_FAILED:
                        pbRunModel.dismiss();
                        Toast.makeText(MainActivity.this, "Run model failed!", Toast.LENGTH_SHORT).show();
                        onRunModelFailed();
                        break;
                    default:
                        break;
                }
            }
        };

        worker = new HandlerThread("Predictor Worker");
        worker.start();
        sender = new Handler(worker.getLooper()) {
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case REQUEST_LOAD_MODEL:
                        // load model and reload test image
                        if (onLoadModel()) {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_FAILED);
                        }
                        break;
                    case REQUEST_RUN_MODEL:
                        // run model if model is loaded
                        if (onRunModel()) {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_FAILED);
                        }
                        break;
                    default:
                        break;
                }
            }
        };
    }
    private String getPathFromUri(Uri uri) {
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
        if (cursor != null) {
            cursor.moveToFirst();
            int columnIndex = cursor.getColumnIndex(projection[0]);
            String path = cursor.getString(columnIndex);
            cursor.close();
            return path;
        }
        return null;
    }
}
