package com.example.adam.ocr;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

import static com.example.adam.ocr.ClassifierActivity.BAR_CODE;
import static com.example.adam.ocr.ClassifierActivity.DATE;

public class ShowOutput extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Intent intent = getIntent();
        String barcode = intent.getStringExtra(BAR_CODE);

        TextView barCodeView = (TextView)findViewById(R.id.bar_code);
        barCodeView.setText(barcode);

        String date = intent.getStringExtra(DATE);

        TextView dateView = (TextView)findViewById(R.id.date);
        dateView.setText(date);
    }

    @Override
    public void onBackPressed() {
        startActivity(new Intent(this, ClassifierActivity.class));
        finish();

    }
}
