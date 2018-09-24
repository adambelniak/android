package com.example.adam.ocr;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

public class ImageClassification implements Classifier {
    private static final String TAG = "ImageClassification";

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 2;
    private static final float THRESHOLD = 0.5f;

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
    private float[] outputs;
    private String[] outputNames;
    private float[] prob_value;
    private boolean logStats = true;

    private TensorFlowInferenceInterface inferenceInterface;

    private ImageClassification() {}

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
        ImageClassification c = new ImageClassification();
        c.inputName = inputName;
        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
//        final Operation test = c.inferenceInterface.graphOperation(inputName);
        c.labels.add("good_quality");
        c.labels.add("bad_quality");

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
        c.outputs =  new float[2];
        c.prob_value = new float[1];
        return c;
    }

    @Override
    public List<Bitmap> recognizeImage(Bitmap bitmap, AssetManager am) {
        return null;
    }

    @Override
    public boolean classifyImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");
        final long startTime = SystemClock.uptimeMillis();

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap,inputWidth, inputHeight, true);
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
        }
        long lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
        Log.i("Detect: %s", Long.toString(lastProcessingTimeMs));
        Trace.endSection();
        prob_value[0] = 1;
        keep_prob = "keep_prob";
        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputHeight, inputWidth, 3);
        inferenceInterface.feed(keep_prob, prob_value, 1);

        Trace.endSection();
        Log.i(TAG, "Processing ");

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();
        Log.i(TAG, "finished ");

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputNames[0], outputs);
        Trace.endSection();

        // Find the best classifications.

//        PriorityQueue<Recognition> pq =
//                new PriorityQueue<Recognition>(
//                        2,
//                        new Comparator<Recognition>() {
//                            @Override
//                            public int compare(Recognition lhs, Recognition rhs) {
//                                // Intentionally reversed to put high confidence at the head of the queue.
//                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
//                            }
//                        });
//        for (int i = 0; i < outputs.length; ++i) {
//            if (outputs[i] > THRESHOLD) {
//                pq.add(
//                        new Recognition(
//                                "" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
//            }
//        }
        Log.i(TAG, Float.toString(outputs[1]));
//        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
//        for (int i = 0; i < recognitionsSize; ++i) {
//            recognitions.add(pq.poll());
//        }
        Trace.endSection(); // "recognizeImage"
        return outputs[1] > 0.95;
    }

    @Override
    public boolean segmentationImage(Bitmap bitmap) {
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
