package com.example.adam.ocr;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.adam.ocr.env.Logger;
import com.google.android.gms.vision.CameraSource;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;

import static com.example.adam.ocr.ClassifierActivity.BAR_CODE;
import static com.example.adam.ocr.ClassifierActivity.BAR_CODE_IMAGE;
import static com.example.adam.ocr.ClassifierActivity.BAR_CODE_RECT;
import static com.example.adam.ocr.ClassifierActivity.DATE;
import static com.example.adam.ocr.ClassifierActivity.DATE_IMAGE;
import static com.example.adam.ocr.ClassifierActivity.DATE_RECT;

public class PreviewActivity extends AppCompatActivity {
    private static final Logger LOGGER = new Logger();

    private int imageCounter;
    private Bitmap dateImage;
    private String date;
    private String barCode;

    private RectF dateRect;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_preview);
        Intent intent = getIntent();
        Bitmap barcodeImage = BitmapFactory.decodeByteArray(intent.getByteArrayExtra(BAR_CODE_IMAGE),0,intent.getByteArrayExtra(BAR_CODE_IMAGE).length);
        barCode = intent.getStringExtra(BAR_CODE);

        date = intent.getStringExtra(DATE);


        ImageView barCodeView = (ImageView)findViewById(R.id.preview);
        RectF barcodeRect = intent.getParcelableExtra(BAR_CODE_RECT);
        barcodeImage = this.drawFrame(barcodeImage, barcodeRect, barCode);
        barcodeImage = this.rotateImage(barcodeImage);

        barCodeView.setImageBitmap(barcodeImage);
    }

    public void nextImage(View view){
        if (imageCounter > 0  || date == null) {
            final Intent intent = new Intent(this, ShowOutput.class);
            intent.putExtra(BAR_CODE, barCode);
            intent.putExtra(DATE, date);
            startActivity(intent);
            finish();
        }
        else if (imageCounter == 0 ){
            imageCounter++;
            Intent intent = getIntent();
            dateRect = intent.getParcelableExtra(DATE_RECT);

            dateImage = BitmapFactory.decodeByteArray(intent.getByteArrayExtra(DATE_IMAGE),0,intent.getByteArrayExtra(DATE_IMAGE).length);

            dateImage = this.drawFrame(dateImage, dateRect, date);
            ImageView barCodeView = (ImageView)findViewById(R.id.preview);
            barCodeView.setImageBitmap(dateImage);

        }

    }


    private Bitmap drawFrame(final Bitmap bitmap, RectF rect, String text) {
        Bitmap copy = bitmap.copy(bitmap.getConfig(), true);
        Canvas cnvs = new Canvas(copy);
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4);
        paint.setColor(Color.RED);
        cnvs.drawRect(rect.left, rect.top, rect.right, rect.bottom, paint);
        paint.setTextSize(35);
        if(rect.top > bitmap.getHeight() - rect.bottom) {
            cnvs.drawText(text, rect.left, rect.top - 35, paint);
        }
        else {
            cnvs.drawText(text, rect.left, rect.bottom + 35, paint);

        }

        return copy;
    }

    private Bitmap rotateImage(Bitmap bitmap) {
        Mat input = new Mat();
        Utils.bitmapToMat(bitmap, input, true);
        Core.transpose(input, input);
        Core.flip(input, input, Core.ROTATE_90_CLOCKWISE);
        Bitmap rotatedBitmap = Bitmap.createBitmap(input.cols(), input.rows(), Bitmap.Config.ARGB_8888);

        Utils.matToBitmap(input, rotatedBitmap, true);
        return rotatedBitmap;
    }


//    @Override
//    public void onBackPressed() {
//        Intent setIntent = new Intent(Intent.ACTION_MAIN);
//        setIntent.addCategory(Intent.CATEGORY_HOME);
//        setIntent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
//        startActivity(setIntent);
//    }

}


