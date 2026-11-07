package com.example.fitlensartest

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter

data class PersonMask(
    val width: Int,
    val height: Int,
    val confidence: FloatArray
)

class ImageSegmenterHelper(context: Context) {

    private val segmenter: ImageSegmenter

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("selfie_segmenter.tflite")
            .build()

        val options = ImageSegmenter.ImageSegmenterOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.IMAGE)
            .setOutputCategoryMask(false)
            .setOutputConfidenceMasks(true)
            .build()

        segmenter = ImageSegmenter.createFromOptions(
            context,
            options
        )
    }

    fun segmentImage(bitmap: Bitmap): PersonMask {

        val mpImage = BitmapImageBuilder(bitmap).build()

        val result = segmenter.segment(mpImage)

        val masks = result.confidenceMasks().orElse(null)
            ?: throw IllegalStateException("No segmentation mask returned")

        require(masks.isNotEmpty()) {
            "Expected at least one confidence mask"
        }

        // Handles both single-mask (person at index 0) and multi-mask (person at index 1) segmenters
        val personMask = if (masks.size > 1) masks[1] else masks[0]

        val width = personMask.width
        val height = personMask.height

        val buffer = ByteBufferExtractor
            .extract(personMask)
            .asFloatBuffer()

        val confidence = FloatArray(width * height)

        buffer.rewind()
        buffer.get(confidence)

        return PersonMask(
            width,
            height,
            confidence
        )
    }

    fun close() {
        segmenter.close()
    }
}
