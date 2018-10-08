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
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Environment;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import com.example.adam.ocr.env.Logger;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
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


/**
 * A classifier specialized to label images using TensorFlow.
 */
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

    private TensorFlowImageSegmentation() {
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param imageMean     The assumed mean of the image values.
     * @param imageStd      The assumed std of the image values.
     * @param inputName     The label of the image input node.
     * @param outputName    The label of the output node.
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
        c.outputsMask1 = new double[inputWidth * inputHeight];
        c.outputsMask2 = new double[inputWidth * inputHeight];

        c.prob_value = new float[1];
        return c;
    }

    @Override
    public List<Bitmap> recognizeImage(Bitmap bitmap, AssetManager am) {
        // Log this method so that it can be analyzed with systrace.
        final long startTime = SystemClock.uptimeMillis();

        Trace.beginSection("preprocessBitmap");

        Mat input = new Mat();
        Mat outputImage = new Mat();

        Utils.bitmapToMat(bitmap, outputImage, true);
        Utils.bitmapToMat(bitmap, input, true);


        Core.transpose(input, input);
        Core.flip(input, input, +1);
        input.copyTo(outputImage);
        Imgproc.resize(input, input, new Size(inputWidth, inputHeight), 0, 0, Imgproc.INTER_CUBIC); //resize image

        Bitmap scaledBitmap = Bitmap.createBitmap(input.cols(), input.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(input, scaledBitmap);
//        Imgcodecs.imwrite(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES) + "/filename2.jpg", input);

        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF)) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF)) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF)) / imageStd;
        }
        Trace.endSection();
        prob_value[0] = 1;
        float[] thresh = new float[]{0.1f};
        keep_prob = "keep_prob";
        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputHeight, inputWidth, 3);
        inferenceInterface.feed(keep_prob, prob_value, 1);
        inferenceInterface.feed("thresh", thresh, 1);

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
        long lastProcessingTime = SystemClock.uptimeMillis() - startTime;
        Log.i("czas sameh sieci: %s", String.valueOf(lastProcessingTime));
//        retrievingImage(outputsMask1, bitmap, outputImage, "/filename8.jpg");
//
        Mat croppedDigits = retrievingImage(outputsMask2, bitmap, outputImage, "/filename7.jpg");
//        Log.i("zapisano %s", Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).toString());
//        Imgcodecs.imwrite(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES) + "/filename10.jpg", sample);

        Mat cv_image = new Mat(inputHeight, inputWidth, CvType.makeType(CvType.CV_8UC1, 1));
        cv_image.put(0, 0, outputsMask1);


        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat mHierarchy = new Mat();
        Size sz = new Size(bitmap.getHeight(), bitmap.getWidth());
        Imgproc.resize(cv_image, cv_image, sz);
        Imgproc.findContours(cv_image, contours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Mat cropped;
        Rect largestRect = null;
        if (isExternalStorageWritable()) {
            for (MatOfPoint contour : contours) {
                Rect boundingRect = Imgproc.boundingRect(contour);
                if (largestRect == null) {
                    largestRect = boundingRect;
                } else if (largestRect.area() < boundingRect.area()) {
                    largestRect = boundingRect;
                }
            }
        }
        if (largestRect == null) {
            return null;
        }

        cropped = new Mat(outputImage, largestRect);
        Mat croppedLabel;

        if (croppedDigits == null) {
            int height = outputImage.height() - largestRect.y - (largestRect.height) + 200;
            int x = largestRect.x - 150;
            int y = largestRect.y + largestRect.height - 200;
            Rect labelData = new Rect(x, y,
                    x + largestRect.width + 500 >= outputImage.width() ? outputImage.width() - 10 - x : largestRect.width + 500, y + height >= outputImage.height() ?
                    outputImage.height() - y - 10 : height);

            croppedLabel = new Mat(outputImage, labelData);
        }
        else {
            croppedLabel = croppedDigits;
        }

        scaledBitmap = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(cropped, scaledBitmap);

        Bitmap scaledLabelBitmap = Bitmap.createBitmap(croppedLabel.cols(), croppedLabel.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(croppedLabel, scaledLabelBitmap);

//        Imgcodecs.imwrite(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES) + "/filename6.jpg", cropped);
////
//        Imgcodecs.imwrite(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES) + "/filename.jpg", croppedLabel);

        // Find the best classifications.
        final ArrayList<Bitmap> recognitions = new ArrayList<Bitmap>();
        recognitions.add(scaledBitmap);
        recognitions.add(scaledLabelBitmap);
//        long lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
//        Log.i("czas sieci: %s", String.valueOf(lastProcessingTimeMs));
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }


    public Mat retrievingImage(double[] outputsMask, Bitmap inputBitmap, Mat outputImage, String name) {

        Mat cv_image = new Mat(inputHeight, inputWidth, CvType.makeType(CvType.CV_8UC1, 1));
        cv_image.put(0, 0, outputsMask);


        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat mHierarchy = new Mat();
        Size sz = new Size(inputBitmap.getHeight(), inputBitmap.getWidth());
        Imgproc.resize(cv_image, cv_image, sz);
        Imgproc.findContours(cv_image, contours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Mat cropped;
        Rect largestRect = null;
        if (isExternalStorageWritable()) {
            for (MatOfPoint contour : contours) {
                Rect boundingRect = Imgproc.boundingRect(contour);
                if (largestRect == null) {
                    largestRect = boundingRect;
                } else if (largestRect.area() < boundingRect.area()) {
                    largestRect = boundingRect;
                }
            }
        }
        if (largestRect == null) {
//            Log.i("Error %s", Long.toString(10));
            return null;
        }

        cropped = new Mat(outputImage, largestRect);
//        Bitmap scaledBitmap = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(cropped, scaledBitmap);
//        Log.i("zapisano %s", Long.toString(10));
//
//        Imgcodecs.imwrite(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES) + "/filename7.jpg", cropped);

        return cropped;
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
