/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.adam.ocr;

import android.app.ProgressDialog;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.SparseArray;
import android.util.TypedValue;
import android.view.View;


import com.example.adam.ocr.env.ImageUtils;
import com.example.adam.ocr.env.Logger;
import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.barcode.Barcode;
import com.google.android.gms.vision.barcode.BarcodeDetector;
import com.google.android.gms.vision.text.Text;
import com.google.android.gms.vision.text.TextBlock;
import com.google.android.gms.vision.text.TextRecognizer;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.util.List;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();



  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private long lastProcessingTimeMs;

  // These are the settings for the original v1 Inception model. If you want to
  // use a model that's been produced from the TensorFlow for Poets codelab,
  // you'll need to set IMAGE_SIZE = 299, IMAGE_MEAN = 128, IMAGE_STD = 128,
  // INPUT_NAME = "Mul", and OUTPUT_NAME = "final_result".
  // You'll also need to update the MODEL_FILE and LABEL_FILE paths to point to
  // the ones you produced.
  //
  // To use v3 Inception model, strip the DecodeJpeg Op from your retrained
  // model first:
  //
  private static final int INPUT_SIZE_HEIGHT_SEGMENTATION = 512;
  private static final int INPUT_SIZE_WIDTH_SEGMENTATION = 384;
  private static final String INPUT_NAME_SEGMENTATION = "input";
  private static final String OUTPUT_NAME_SEGMENTATION_1 = "mask_1";
  private static final String OUTPUT_NAME_SEGMENTATION_2 = "mask_2";
  private static final String TAG_SEGMANTATION = "TensorFlowImageSegmentation";
  private static final String MODEL_FILE_SEGMENTATION = "file:///android_asset/saved_model/frozen_ocr.pb";


  private static final int INPUT_SIZE_HEIGHT = 1024;
  private static final int INPUT_SIZE_WIDTH = 768;
  private static final int IMAGE_MEAN = 125;
  private static final float IMAGE_STD = 255;
  private static final String INPUT_NAME = "input";
  private static final String OUTPUT_NAME = "output";

  private static final String TAG = "TensorFlowImageClassify";


  private static final String MODEL_FILE = "file:///android_asset/saved_model_classify/opt_frozen_ocr.pb";
  private static final String LABEL_FILE =
      "file:///android_asset/imagenet_comp_graph_label_strings.txt";


  private static final boolean MAINTAIN_ASPECT = true;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(1920, 1080);


  private Integer sensorOrientation;
  private Classifier classifier;
  private Classifier segmentation;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  private boolean isProcessingFrame = false;
  BarcodeDetector detector;
  TextRecognizer textRecognizer;
  public static final String BAR_CODE = "com.example.ocr.barcode";
  public static final String DATE = "com.example.ocr.date";
  public static final String BAR_CODE_RECT = "com.example.ocr.barcode.rect";
  public static final String DATE_RECT = "com.example.ocr.date.rect";
  public static final String BAR_CODE_IMAGE = "com.example.ocr.barcode.image";
  public static final String DATE_IMAGE = "com.example.ocr.date.image";


  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  private static final float TEXT_SIZE_DIP = 10;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());



    segmentation =
            TensorFlowImageSegmentation.create(
                    getAssets(),
                    MODEL_FILE_SEGMENTATION,
                    LABEL_FILE,
                    INPUT_SIZE_WIDTH_SEGMENTATION,
                    INPUT_SIZE_HEIGHT_SEGMENTATION,
                    IMAGE_MEAN,
                    IMAGE_STD,
                    INPUT_NAME_SEGMENTATION,
                    new String[] {OUTPUT_NAME_SEGMENTATION_1, OUTPUT_NAME_SEGMENTATION_2});

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();
    LOGGER.i("camera size: %s",  size.getHeight());

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

    frameToCropTransform = ImageUtils.getTransformationMatrix(
        previewWidth, previewHeight,
            INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT,
        sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);
    detector =
            new BarcodeDetector.Builder(getApplicationContext())
                    .setBarcodeFormats(Barcode.DATA_MATRIX | Barcode.EAN_13)
                    .build();
    textRecognizer = new TextRecognizer.Builder(getApplicationContext()).build();



  }


  public void takePicture(View view) {

      isProcessingFrame = true;

    }

    public void test(){}

  @Override
  protected void processImage() {
    if(isProcessingFrame) {
      final ProgressDialog progress = new ProgressDialog(this);
      progress.setTitle("Processing Image");
      progress.setMessage("Wait while loading...");
      progress.setCancelable(false); // disable dismiss by tapping outside of the dialog
      progress.show();

      final AssetManager am = this.getAssets();
      final Intent intent = new Intent(this, PreviewActivity.class);
      final ByteArrayOutputStream bStream = new ByteArrayOutputStream();
      runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                      final long startTime = SystemClock.uptimeMillis();
                      String barCode = null;
                      String date = null;
                      rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
                        List<Bitmap> segments = segmentation.recognizeImage(rgbFrameBitmap, am);
                        if (segments != null) {
                          Bitmap bitmap = segments.get(0);
                          Mat input = new Mat();
                          Utils.bitmapToMat(bitmap, input, true);
                          Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2GRAY);
                          Core.transpose(input, input);
                          Core.flip(input, input, +1);
                          Imgproc.threshold(input, input, 0, 255, Imgproc.THRESH_OTSU);
                          bitmap = Bitmap.createBitmap(input.cols(), input.rows(), Bitmap.Config.ARGB_8888);

                          Utils.matToBitmap(input, bitmap);

                          Frame frame = new Frame.Builder().setBitmap(bitmap).build();
                          SparseArray<Barcode> barcodes = detector.detect(frame);
                          if (barcodes.size() > 0) {
                            bitmap.compress(Bitmap.CompressFormat.PNG, 100, bStream);

                            byte[] byteArray = bStream.toByteArray();
                            intent.putExtra(BAR_CODE_IMAGE, byteArray);
                            bStream.reset();
                            Barcode thisCode = barcodes.valueAt(0);
                            RectF rect = new RectF(thisCode.getBoundingBox());
                            intent.putExtra(BAR_CODE_RECT, rect);
                            //Compress it before sending it to minimize the size and quality of bitmap.
                              Imgcodecs.imwrite(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES) + "/filename4.jpg", input);



                            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                            LOGGER.i("Detect: %s", lastProcessingTimeMs);
                            LOGGER.i("Value: %s", thisCode.rawValue);
                            barCode = thisCode.rawValue;
                          }
                        }
                      RectF rect = null;
                      if (barCode != null) {
                        Bitmap bitmap = segments.get(1);
                        Mat input = new Mat();
                        Utils.bitmapToMat(bitmap, input, true);


                        Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2GRAY);
                        Imgproc.adaptiveThreshold(input, input, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 5);
                        bitmap = Bitmap.createBitmap(input.cols(), input.rows(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(input, bitmap);
                        Frame frame = new Frame.Builder().setBitmap(bitmap).build();
                        Imgcodecs.imwrite(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES) + "/filename4.jpg", input);

                        SparseArray<TextBlock> text = textRecognizer.detect(frame);

                        if (text.size() > 0) {
                          bitmap.compress(Bitmap.CompressFormat.PNG, 100, bStream);
                          byte[] byteArray = bStream.toByteArray();
                          intent.putExtra(DATE_IMAGE, byteArray);
//                          TextBlock blok = text.get(text.size() - 1);
                          LOGGER.i("find text");
                          for (int i = 0; i < text.size(); i++){
                            TextBlock blok = text.get(i);

                            if (blok != null) {
                              for (Text currentText : blok.getComponents()) {
                                LOGGER.i("Value: %s", currentText.getValue());
                                if (currentText.getValue() != null && currentText.getValue().matches("[0-9 .]+")) {
                                  date = currentText.getValue();
                                  rect = new RectF(currentText.getBoundingBox());
                                }
                              }

                              LOGGER.i("Value: %s", date);
                              lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                              LOGGER.i("Detect: %s", lastProcessingTimeMs);
                            }
                        }
                        }
                      }
                      intent.putExtra(BAR_CODE, barCode);
                      intent.putExtra(DATE, date);
                      intent.putExtra(DATE_RECT, rect);

                      isProcessingFrame = false;
                      progress.dismiss();
                      if(intent.getStringExtra(BAR_CODE) != null) {
                        startActivity(intent);
                      }
                      else {
                        readyForNextImage();
                      }
                    }
                });
    }
    else {
        readyForNextImage();
    }
  }

  public Bitmap transformImage(Bitmap bitmap, boolean adaptive){
    Mat input = new Mat();
    Utils.bitmapToMat(bitmap, input, true);

    Imgproc.cvtColor(input, input, Imgproc.COLOR_RGB2GRAY);
    if(adaptive){
      Imgproc.adaptiveThreshold(input, input, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 5);
    }
    else {
      Core.transpose(input, input);
      Core.flip(input, input, +1);
      Imgproc.threshold(input, input, 0, 255, Imgproc.THRESH_OTSU);
    }
    bitmap = Bitmap.createBitmap(input.cols(), input.rows(), Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(input, bitmap);
    return bitmap;
  }


  @Override
  public void onResume()
  {
    super.onResume();
    if (!OpenCVLoader.initDebug()) {
      Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
      OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, null);
    } else {
      Log.d(TAG, "OpenCV library found inside package. Using it!");
    }
  }

}
