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

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;


import com.example.adam.ocr.env.ImageUtils;
import com.example.adam.ocr.env.Logger;

import org.opencv.android.OpenCVLoader;

import java.util.List;
import java.util.Vector;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  protected static final boolean SAVE_PREVIEW_BITMAP = false;

  private ResultsView resultsView;

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
  private static final String OUTPUT_NAME_SEGMENTATION_2 = "mask_1";
  private static final String TAG_SEGMANTATION = "TensorFlowImageSegmentation";
  private static final String MODEL_FILE_SEGMENTATION = "file:///android_asset/saved_model/frozen_ocr.pb";


  private static final int INPUT_SIZE_HEIGHT = 1024;
  private static final int INPUT_SIZE_WIDTH = 768;
  private static final int IMAGE_MEAN = 117;
  private static final float IMAGE_STD = 255;
  private static final String INPUT_NAME = "input";
  private static final String OUTPUT_NAME = "output";
//  private static final String OUTPUT_NAME_2 = "mask_2";

  private static final String TAG = "TensorFlowImageClassify";


  private static final String MODEL_FILE = "file:///android_asset/saved_model_classify/opt_frozen_ocr.pb";
  private static final String LABEL_FILE =
      "file:///android_asset/imagenet_comp_graph_label_strings.txt";


  private static final boolean MAINTAIN_ASPECT = true;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(1080, 1920);


  private Integer sensorOrientation;
  private Classifier classifier;
  private Classifier segmentation;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;




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


    classifier =
            ImageClassification.create(
            getAssets(),
            MODEL_FILE,
            LABEL_FILE,
            INPUT_SIZE_WIDTH,
            INPUT_SIZE_HEIGHT,
            IMAGE_MEAN,
            IMAGE_STD,
            INPUT_NAME,
            new String[] {OUTPUT_NAME});

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


  }

  @Override
  protected void processImage() {

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // For examining the actual TF input.
    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            final long startTime = SystemClock.uptimeMillis();
            final boolean results = classifier.classifyImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            LOGGER.i("Detect: %s", lastProcessingTimeMs);
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);

            if(results) {
              LOGGER.i("Activation: %s");

              segmentation.recognizeImage(croppedBitmap);
            }

            readyForNextImage();
          }
        });
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


//  @Override
//  public void onSetDebug(boolean debug) {
//    classifier.enableStatLogging(debug);
//  }

  private void renderDebug(final Canvas canvas) {
    if (!isDebug()) {
      return;
    }
    final Bitmap copy = cropCopyBitmap;
    if (copy != null) {
      final Matrix matrix = new Matrix();
      final float scaleFactor = 2;
      matrix.postScale(scaleFactor, scaleFactor);
      matrix.postTranslate(
          canvas.getWidth() - copy.getWidth() * scaleFactor,
          canvas.getHeight() - copy.getHeight() * scaleFactor);
      canvas.drawBitmap(copy, matrix, new Paint());

      final Vector<String> lines = new Vector<String>();
//      if (classifier != null) {
//        String statString = classifier.getStatString();
//        String[] statLines = statString.split("\n");
//        for (String line : statLines) {
//          lines.add(line);
//        }
//      }

      lines.add("Frame: " + previewWidth + "x" + previewHeight);
      lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
      lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
      lines.add("Rotation: " + sensorOrientation);
      lines.add("Inference time: " + lastProcessingTimeMs + "ms");

    }
  }
}
