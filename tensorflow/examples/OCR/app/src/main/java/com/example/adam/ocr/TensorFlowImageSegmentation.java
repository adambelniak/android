/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.adam.ocr;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Environment;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;


/** A classifier specialized to label images using TensorFlow. */
public class TensorFlowImageSegmentation implements Classifier {
  private static final String TAG = "TensorFlowImageSegmentation";

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 3;
  private static final float THRESHOLD = 0.1f;
  private Mat transformed;

  // Config values.
  private String inputName;
  private String keep_prob;

  private int inputWidth;
  private int inputHeight;
  private int imageMean;
  private float imageStd;

  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  private float[] floatValues;
  private double[] outputsMask1;
    private double[] outputsMask2;

    private String[] outputNames;
  private float[] prob_value;
  private boolean logStats = false;

  private TensorFlowInferenceInterface inferenceInterface;

  private TensorFlowImageSegmentation() {}

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param imageMean The assumed mean of the image values.
   * @param imageStd The assumed std of the image values.
   * @param inputName The label of the image input node.
   * @param outputName The label of the output node.
   * @throws IOException
   */
  public static Classifier create(
      AssetManager assetManager,
      String modelFilename,
      String labelFilename,
      int inputWidth,
      int inputHeight,
      int imageMean,
      float imageStd,
      String inputName,
      String[] outputName) {
    TensorFlowImageSegmentation c = new TensorFlowImageSegmentation();
    c.inputName = inputName;
    c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

    // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
    final Operation test = c.inferenceInterface.graphOperation(inputName);
    // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
    // the placeholder node for input in the graphdef typically used does not specify a shape, so it
    // must be passed in as a parameter.
    c.inputWidth = inputWidth;
    c.inputHeight = inputHeight;

    c.imageMean = imageMean;
    c.imageStd = imageStd;

    // Pre-allocate buffers.
    c.outputNames = outputName;
    c.intValues = new int[inputWidth * inputHeight];
    c.floatValues = new float[inputWidth * inputHeight * 3];
    c.outputsMask1 =  new double[inputWidth * inputHeight];
    c.outputsMask2 =  new double[inputWidth * inputHeight];

      c.prob_value = new float[1];
    return c;
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    Log.i("Size: %s", Double.toString(bitmap.getWidth()));
    Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap,inputWidth, inputHeight, true);
    scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());
    for (int i = 0; i < intValues.length; ++i) {
      final int val = intValues[i];
      floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
      floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
      floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
    }
    Trace.endSection();
    prob_value[0] = 1;
    keep_prob = "keep_prob";
    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    inferenceInterface.feed(inputName, floatValues, 1, inputHeight, inputWidth, 3);
    inferenceInterface.feed(keep_prob, prob_value, 1);

    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    inferenceInterface.run(outputNames, logStats);
    Trace.endSection();

    // Copy the output Tensor back into the output array.
    Trace.beginSection("fetch");
    inferenceInterface.fetch(outputNames[0], outputsMask1);
    inferenceInterface.fetch(outputNames[1], outputsMask2);

      Trace.endSection();
//    INDArray nd = Nd4j.create(outputs, new int[]{inputHeight, inputWidth, 1});
//      for (int i = 0; i < 3; i++) {
//          INDArray segmentation = nd.getColumn(0);
//          INDArray image = segmentation.reshape(new int[]{inputHeight, inputWidth});
//          BooleanIndexing.replaceWhere(image, 1.0, Conditions.greaterThan(0.33));
//          BooleanIndexing.replaceWhere(image, 0.0, Conditions.lessThan(0.33));
//          image = image.mul(255);
//    NativeImageLoader loader = new NativeImageLoader();
//              Log.i(TAG, "Saved:");
    final long startTime = SystemClock.uptimeMillis();

    Mat cv_image = new Mat(new Size(inputWidth, inputHeight), CvType.CV_8UC1, new Scalar(outputsMask2));
    double lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
    Log.i("Process: %s", Double.toString(lastProcessingTimeMs));
//          boolean saved = opencv_imgcodecs.imwrite("faceDetection.png", cv_image);
    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    Mat mHierarchy = new Mat();
    Imgproc.findContours(cv_image, contours, mHierarchy, Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
    Log.i("amount of contours: %s", String.valueOf(contours.size()));
    Mat cropped;
    if (isExternalStorageWritable()) {
      for (MatOfPoint contour : contours) {
        cropped = new Mat(cv_image, Imgproc.boundingRect(contour));
        Size sz = new Size(inputWidth * 4, inputHeight * 4);
        Imgproc.resize(cropped, cropped, sz);
        boolean saved = Imgcodecs.imwrite(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES) + "/filename.jpg", cropped);
        Log.i("saved: %s", String.valueOf(saved));
      }
    }



      // Find the best classifications.
    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();

    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

    @Override
    public boolean classifyImage(Bitmap bitmap) {
        return false;
    }

    @Override
    public boolean segmentationImage(Bitmap bitmap) {
        return false;
    }

    public boolean isExternalStorageWritable() {
    String state = Environment.getExternalStorageState();
    if (Environment.MEDIA_MOUNTED.equals(state)) {
      return true;
    }
    return false;
  }

  @Override
  public void enableStatLogging(boolean logStats) {
    this.logStats = logStats;
  }

  @Override
  public String getStatString() {
    return inferenceInterface.getStatString();
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
