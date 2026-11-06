package com.example.fitlensartest

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.fitlensartest.databinding.ActivityMainBinding
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    // Stores the dimensions of the image being processed
    @Volatile
    private var processedImageWidth = 0

    @Volatile
    private var processedImageHeight = 0

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private var poseDetector: PoseDetector? = null

    // User height calibration (default: 175 cm / 5'9")
    private var userHeightCm = 175

    // Sliding window measurement histories for outlier rejection (Trimmed Mean)
    private val shoulderHistory = ArrayDeque<Double>()
    private val chestHistory = ArrayDeque<Double>()
    private val maxHistorySize = 15

    // Smoothed measurement values (Exponential Moving Average)
    private var smoothedShoulderWidthCm = 0.0
    private var smoothedChestCircumferenceCm = 0.0

    // Camera permission request
    private val cameraPermission =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            if (isGranted) {
                startCamera()
            } else {
                binding.statusText.text = "Camera permission denied"
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        setupHeightCalibrationUI()
        initPoseDetector()
        checkCameraPermission()
    }

    private fun setupHeightCalibrationUI() {
        updateHeightDisplay()

        binding.btnHeightMinus.setOnClickListener {
            if (userHeightCm > 120) {
                userHeightCm -= 1
                resetMeasurementHistory()
                updateHeightDisplay()
            }
        }

        binding.btnHeightPlus.setOnClickListener {
            if (userHeightCm < 230) {
                userHeightCm += 1
                resetMeasurementHistory()
                updateHeightDisplay()
            }
        }
    }

    private fun updateHeightDisplay() {
        val totalInches = (userHeightCm / 2.54).roundToInt()
        val feet = totalInches / 12
        val inches = totalInches % 12
        binding.userHeightText.text = String.format(
            Locale.US, "%d cm (%d'%d\")", userHeightCm, feet, inches
        )
    }

    private fun resetMeasurementHistory() {
        shoulderHistory.clear()
        chestHistory.clear()
        smoothedShoulderWidthCm = 0.0
        smoothedChestCircumferenceCm = 0.0
    }

    // Initialize MediaPipe
    private fun initPoseDetector() {
        try {
            poseDetector = PoseDetector(
                context = this,
                onResult = ::onPoseDetected,
                onError = { error ->
                    Log.e("FitLensPose", "Pose detection error", error)
                }
            ).also {
                it.initialize()
            }
        } catch (e: Exception) {
            Log.e("FitLensPose", "MediaPipe initialization failed", e)
            binding.statusText.text = "MediaPipe initialization failed"
        }
    }

    // Handle detected body pose and estimate measurements with anatomical precision
    private fun onPoseDetected(result: PoseLandmarkerResult) {
        val landmarks = result.landmarks().firstOrNull()
        val worldLandmarks = result.worldLandmarks().firstOrNull()

        runOnUiThread {
            if (landmarks != null && landmarks.size > 24 && worldLandmarks != null && worldLandmarks.size > 24) {

                // Landmark visibility check (ensure key joints are not occluded)
                val leftShoulderVis = landmarks[11].visibility().orElse(1.0f)
                val rightShoulderVis = landmarks[12].visibility().orElse(1.0f)
                val leftHipVis = landmarks[23].visibility().orElse(1.0f)
                val rightHipVis = landmarks[24].visibility().orElse(1.0f)

                if (leftShoulderVis < 0.6f || rightShoulderVis < 0.6f || leftHipVis < 0.5f || rightHipVis < 0.5f) {
                    binding.statusText.text = "Body partially blocked — Stand in clear view"
                    return@runOnUiThread
                }

                // Image space landmarks (0..1)
                val leftShoulder = landmarks[11]
                val rightShoulder = landmarks[12]
                val leftHip = landmarks[23]
                val rightHip = landmarks[24]

                // 3D World space landmarks (in real-world meters)
                val leftShoulderWorld = worldLandmarks[11]
                val rightShoulderWorld = worldLandmarks[12]
                val leftHipWorld = worldLandmarks[23]
                val rightHipWorld = worldLandmarks[24]

                // Calculate 3D Euclidean distance between acromion shoulder points (meters)
                val dxS = (leftShoulderWorld.x() - rightShoulderWorld.x()).toDouble()
                val dyS = (leftShoulderWorld.y() - rightShoulderWorld.y()).toDouble()
                val dzS = (leftShoulderWorld.z() - rightShoulderWorld.z()).toDouble()

                val rawAcromion3DMeters = sqrt(dxS * dxS + dyS * dyS + dzS * dzS)

                // 3D Torso Height (midpoint of shoulders to midpoint of hips)
                val midShoulderX = (leftShoulderWorld.x() + rightShoulderWorld.x()) / 2.0
                val midShoulderY = (leftShoulderWorld.y() + rightShoulderWorld.y()) / 2.0
                val midShoulderZ = (leftShoulderWorld.z() + rightShoulderWorld.z()) / 2.0

                val midHipX = (leftHipWorld.x() + rightHipWorld.x()) / 2.0
                val midHipY = (leftHipWorld.y() + rightHipWorld.y()) / 2.0
                val midHipZ = (leftHipWorld.z() + rightHipWorld.z()) / 2.0

                val dxT = midShoulderX - midHipX
                val dyT = midShoulderY - midHipY
                val dzT = midShoulderZ - midHipZ
                val torsoHeightMeters = sqrt(dxT * dxT + dyT * dyT + dzT * dzT)

                // Check body yaw rotation (angle of body turning relative to camera view)
                val sinYaw = (abs(dzS) / (rawAcromion3DMeters + 1e-6)).coerceIn(0.0, 1.0)
                val yawAngleDegrees = Math.toDegrees(asin(sinYaw))

                if (yawAngleDegrees > 22.0) {
                    binding.statusText.text = "Please face straight toward the camera"
                    binding.shoulderOverlay.updateBody(
                        leftShoulderX = leftShoulder.x(),
                        leftShoulderY = leftShoulder.y(),
                        rightShoulderX = rightShoulder.x(),
                        rightShoulderY = rightShoulder.y(),
                        leftHipX = leftHip.x(),
                        leftHipY = leftHip.y(),
                        rightHipX = rightHip.x(),
                        rightHipY = rightHip.y(),
                        imageWidth = processedImageWidth,
                        imageHeight = processedImageHeight,
                        isAligned = false
                    )
                    return@runOnUiThread
                }

                // Apply trigonometric yaw correction for minor body turns
                val yawCorrectionFactor = 1.0 / cos(Math.toRadians(yawAngleDegrees)).coerceAtLeast(0.85)
                val correctedAcromionMeters = rawAcromion3DMeters * yawCorrectionFactor

                // Height Scale Factor relative to standard reference model
                val heightScaleFactor = userHeightCm / 170.0

                // Outer Bideltoid Shoulder Width Calibration:
                // MediaPipe measures inner acromion distance; outer bideltoid breadth incorporates deltoid muscle thickness (~1.30 anatomical ratio)
                val rawShoulderWidthCm = correctedAcromionMeters * 100.0 * 1.30 * heightScaleFactor

                // Dynamic chest depth ratio based on Torso Aspect Ratio (torsoHeight / shoulderWidth)
                val torsoAspectRatio = torsoHeightMeters / (correctedAcromionMeters + 1e-6)
                val chestDepthRatio = when {
                    torsoAspectRatio > 1.25 -> 0.62 // Slender frame
                    torsoAspectRatio < 1.05 -> 0.72 // Broad frame
                    else -> 0.67 // Standard frame
                }

                // Pectoral & Latissimus Anatomical Model:
                // Chest Circumference C ~ 2.22 * Bideltoid Shoulder Width
                val chestWidthCm = rawShoulderWidthCm * 0.94
                val chestDepthCm = chestWidthCm * chestDepthRatio

                val a = chestWidthCm / 2.0
                val b = chestDepthCm / 2.0
                val h = ((a - b) * (a - b)) / ((a + b) * (a + b))

                // Ramanujan's approximation for perimeter of an ellipse
                val rawChestCircumferenceCm = Math.PI * (a + b) * (1.0 + (3.0 * h) / (10.0 + sqrt(4.0 - 3.0 * h)))

                // Add sample to sliding window
                shoulderHistory.addLast(rawShoulderWidthCm)
                if (shoulderHistory.size > maxHistorySize) shoulderHistory.removeFirst()

                chestHistory.addLast(rawChestCircumferenceCm)
                if (chestHistory.size > maxHistorySize) chestHistory.removeFirst()

                // Calculate Trimmed Mean (discard outliers for high stability)
                val medianShoulder = calculateTrimmedMean(shoulderHistory)
                val medianChest = calculateTrimmedMean(chestHistory)

                // Exponential Moving Average filter
                if (smoothedShoulderWidthCm == 0.0) {
                    smoothedShoulderWidthCm = medianShoulder
                    smoothedChestCircumferenceCm = medianChest
                } else {
                    val alpha = 0.15
                    smoothedShoulderWidthCm += alpha * (medianShoulder - smoothedShoulderWidthCm)
                    smoothedChestCircumferenceCm += alpha * (medianChest - smoothedChestCircumferenceCm)
                }

                val shoulderInches = smoothedShoulderWidthCm / 2.54
                val chestInches = smoothedChestCircumferenceCm / 2.54

                // Update overlay lines and landmark positions
                binding.shoulderOverlay.updateBody(
                    leftShoulderX = leftShoulder.x(),
                    leftShoulderY = leftShoulder.y(),
                    rightShoulderX = rightShoulder.x(),
                    rightShoulderY = rightShoulder.y(),
                    leftHipX = leftHip.x(),
                    leftHipY = leftHip.y(),
                    rightHipX = rightHip.x(),
                    rightHipY = rightHip.y(),
                    imageWidth = processedImageWidth,
                    imageHeight = processedImageHeight,
                    isAligned = true
                )

                // Update HUD display
                binding.statusText.text = "Body Aligned — Measuring"
                binding.shoulderWidthText.text = String.format(
                    Locale.US, "%.1f cm (%.1f in)", smoothedShoulderWidthCm, shoulderInches
                )
                binding.chestCircumferenceText.text = String.format(
                    Locale.US, "%.1f cm (%.1f in)", smoothedChestCircumferenceCm, chestInches
                )

            } else {
                shoulderHistory.clear()
                chestHistory.clear()
                smoothedShoulderWidthCm = 0.0
                smoothedChestCircumferenceCm = 0.0

                binding.shoulderOverlay.clear()
                binding.statusText.text = "No body detected"
                binding.shoulderWidthText.text = "-- cm"
                binding.chestCircumferenceText.text = "-- cm"
            }
        }
    }

    // Trims top 20% and bottom 20% outliers to prevent measurement spikes
    private fun calculateTrimmedMean(history: Collection<Double>): Double {
        if (history.isEmpty()) return 0.0
        val sorted = history.sorted()
        if (sorted.size < 5) return sorted.average()
        val trimCount = sorted.size / 5
        val trimmed = sorted.subList(trimCount, sorted.size - trimCount)
        return trimmed.average()
    }

    // Check camera permission
    private fun checkCameraPermission() {
        val permissionGranted =
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED

        if (permissionGranted) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    // Start CameraX
    private fun startCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

                if (!cameraProvider.hasCamera(cameraSelector)) {
                    binding.statusText.text = "Front camera unavailable"
                    return@addListener
                }

                // Camera preview
                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.surfaceProvider = binding.previewView.surfaceProvider
                    }

                // Camera frame analysis
                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                    )
                    .setOutputImageFormat(
                        ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
                    )
                    .build()

                // Process camera frames
                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    try {
                        // Convert camera frame to Bitmap
                        val bitmap = imageProxy.toBitmap()

                        // Rotate image according to camera orientation
                        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

                        val matrix = Matrix().apply {
                            postRotate(rotationDegrees.toFloat())
                        }

                        val rotatedBitmap = Bitmap.createBitmap(
                            bitmap,
                            0,
                            0,
                            bitmap.width,
                            bitmap.height,
                            matrix,
                            true
                        )

                        // Save image dimensions for shoulder overlay
                        processedImageWidth = rotatedBitmap.width
                        processedImageHeight = rotatedBitmap.height

                        // Strictly monotonic timestamp in milliseconds
                        val timestampMs = SystemClock.uptimeMillis()

                        // Detect body landmarks asynchronously
                        poseDetector?.detect(
                            rotatedBitmap,
                            timestampMs
                        )

                    } catch (e: Exception) {
                        Log.e("FitLensCamera", "Frame processing failed", e)
                    } finally {
                        imageProxy.close()
                    }
                }

                // Remove previous camera connections
                cameraProvider.unbindAll()

                // Connect camera
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )

                binding.statusText.text = "Camera running — detecting body..."

            } catch (e: Exception) {
                Log.e("FitLensCamera", "Camera setup failed", e)
                binding.statusText.text = "Camera failed to start"
            }

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        poseDetector?.close()
        if (::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
        }
    }
}
