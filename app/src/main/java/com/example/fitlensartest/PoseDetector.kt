package com.example.fitlensartest

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

class PoseDetector(
    private val context: Context,
    private val onResult: (PoseLandmarkerResult) -> Unit,
    private val onError: ((Throwable) -> Unit)? = null
) {

    private var poseLandmarker: PoseLandmarker? = null

    // Initialize MediaPipe Pose Landmarker with Full precision model
    fun initialize() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker_full.task")
                .build()

            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumPoses(1)
                .setResultListener { result, _ ->
                    onResult(result)
                }
                .setErrorListener { error ->
                    Log.e("FitLensPose", "Pose detection error callback", error)
                    onError?.invoke(error)
                }
                .build()

            poseLandmarker = PoseLandmarker.createFromOptions(
                context,
                options
            )

            Log.d("FitLensPose", "Pose detector initialized with full model")

        } catch (e: Exception) {
            Log.e("FitLensPose", "Failed to initialize pose detector", e)
            onError?.invoke(e)
        }
    }

    // Detect body landmarks asynchronously from camera frame
    fun detect(bitmap: Bitmap, timestampMs: Long) {
        try {
            val detector = poseLandmarker ?: return
            val mpImage = BitmapImageBuilder(bitmap).build()
            detector.detectAsync(mpImage, timestampMs)
        } catch (e: Exception) {
            Log.e("FitLensPose", "Pose detection failed", e)
            onError?.invoke(e)
        }
    }

    // Release MediaPipe resources
    fun close() {
        try {
            poseLandmarker?.close()
            poseLandmarker = null
            Log.d("FitLensPose", "Pose detector closed")
        } catch (e: Exception) {
            Log.e("FitLensPose", "Failed to close pose detector", e)
        }
    }
}
