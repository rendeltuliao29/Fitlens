package com.example.fitlensartest

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.CountDownTimer
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.fitlensartest.analysis.DetectedOrientation
import com.example.fitlensartest.analysis.DistanceStatus
import com.example.fitlensartest.analysis.MeasurementEngine
import com.example.fitlensartest.analysis.PoseOrientationDetector
import com.example.fitlensartest.data.AppScanState
import com.example.fitlensartest.data.Gender
import com.example.fitlensartest.data.MeasurementResult
import com.example.fitlensartest.data.PreferredFit
import com.example.fitlensartest.data.UserInfo
import com.example.fitlensartest.databinding.ActivityMainBinding
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    @Volatile
    private var processedImageWidth = 0

    @Volatile
    private var processedImageHeight = 0

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    // Processing Helpers
    private var poseDetector: PoseDetector? = null
    private var imageSegmenter: ImageSegmenterHelper? = null
    private val orientationDetector = PoseOrientationDetector()
    private val measurementEngine = MeasurementEngine()

    // State Tracking
    private var currentState = AppScanState.USER_INPUT_FORM
    private var currentUserInfo: UserInfo? = null

    @Volatile
    private var latestPersonMask: PersonMask? = null

    @Volatile
    private var latestFrameBitmap: Bitmap? = null

    @Volatile
    private var latestPoseResult: PoseLandmarkerResult? = null

    // Captured Scans
    private var frontCapturedBitmap: Bitmap? = null
    private var frontCapturedLandmarks: List<NormalizedLandmark>? = null
    private var frontCapturedMask: PersonMask? = null

    private var sideCapturedBitmap: Bitmap? = null
    private var sideCapturedLandmarks: List<NormalizedLandmark>? = null
    private var sideCapturedMask: PersonMask? = null

    // Countdown Timer
    private var countdownTimer: CountDownTimer? = null
    private var isCountingDown = false

    private var segmentationFrameCounter = 0

    // Camera permission
    private val cameraPermission =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            if (isGranted) {
                if (currentState == AppScanState.FRONT_SCANNING || currentState == AppScanState.SIDE_SCANNING) {
                    startCamera()
                }
            } else {
                Toast.makeText(this, "Camera permission required for scanning", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        setupFormUI()
        setupScanUI()
        setupResultsUI()
        updateUiForState(AppScanState.USER_INPUT_FORM)

        initPoseDetector()
        initImageSegmenter()

        checkCameraPermission()
    }

    // ----------------------------------------------------
    // FORM SETUP & VALIDATION
    // ----------------------------------------------------

    private fun setupFormUI() {
        binding.btnStartScan.setOnClickListener {
            val ageText = binding.etAge.text.toString().trim()
            val heightText = binding.etHeight.text.toString().trim()
            val weightText = binding.etWeight.text.toString().trim()

            if (ageText.isEmpty() || heightText.isEmpty() || weightText.isEmpty()) {
                Toast.makeText(this, "Please complete all required fields (*)", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val age = ageText.toIntOrNull()
            val height = heightText.toDoubleOrNull()
            val weight = weightText.toDoubleOrNull()

            if (age == null || age <= 0 || height == null || height <= 0.0 || weight == null || weight <= 0.0) {
                Toast.makeText(this, "Please enter valid numeric values for Age, Height, and Weight", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val gender = when (binding.rgGender.checkedRadioButtonId) {
                R.id.rbGenderFemale -> Gender.FEMALE
                R.id.rbGenderOther -> Gender.OTHER
                else -> Gender.MALE
            }

            val preferredFit = when (binding.rgFit.checkedRadioButtonId) {
                R.id.rbFitSlim -> PreferredFit.SLIM
                R.id.rbFitOversized -> PreferredFit.OVERSIZED
                else -> PreferredFit.REGULAR
            }

            val knownChest = binding.etKnownChest.text.toString().trim().toDoubleOrNull()
            val usualShirt = binding.etShirtSize.text.toString().trim().ifEmpty { null }

            currentUserInfo = UserInfo(
                gender = gender,
                age = age,
                heightCm = height,
                weightKg = weight,
                preferredFit = preferredFit,
                knownChestCircumferenceCm = knownChest,
                usualShirtSize = usualShirt
            )

            // Reset scans and proceed to Front Scan
            resetScanCaptures()
            updateUiForState(AppScanState.FRONT_SCANNING)
        }
    }

    private fun setupScanUI() {
        // Instant Manual Capture Button
        binding.btnManualCapture.setOnClickListener {
            cancelCountdown()
            if (currentState == AppScanState.FRONT_SCANNING) {
                captureFrontScan()
            } else if (currentState == AppScanState.SIDE_SCANNING) {
                captureSideScan()
            }
        }
    }

    private fun setupResultsUI() {
        binding.btnRestartScan.setOnClickListener {
            cancelCountdown()
            resetScanCaptures()
            updateUiForState(AppScanState.USER_INPUT_FORM)
        }
    }

    private fun resetScanCaptures() {
        frontCapturedBitmap = null
        frontCapturedLandmarks = null
        frontCapturedMask = null

        sideCapturedBitmap = null
        sideCapturedLandmarks = null
        sideCapturedMask = null

        orientationDetector.resetHistory()
        cancelCountdown()
    }

    // ----------------------------------------------------
    // STATE TRANSITION UI MANAGEMENT
    // ----------------------------------------------------

    private fun updateUiForState(newState: AppScanState) {
        currentState = newState

        binding.formScrollView.visibility = if (newState == AppScanState.USER_INPUT_FORM) View.VISIBLE else View.GONE
        binding.cameraContainer.visibility = if (newState == AppScanState.FRONT_SCANNING || newState == AppScanState.SIDE_SCANNING) View.VISIBLE else View.GONE
        binding.layoutAnalyzing.visibility = if (newState == AppScanState.ANALYZING) View.VISIBLE else View.GONE
        binding.resultsScrollView.visibility = if (newState == AppScanState.RESULTS) View.VISIBLE else View.GONE

        when (newState) {
            AppScanState.USER_INPUT_FORM -> {
                binding.shoulderOverlay.clear()
            }
            AppScanState.FRONT_SCANNING -> {
                binding.statusText.text = "Front Scan — Face Camera"
                binding.orientationPromptText.text = "Face camera straight at ~2.0m or tap 'Capture Scan Now'."
                binding.stabilityProgressBar.progress = 0
                binding.previewView.post {
                    startCamera()
                }
            }
            AppScanState.SIDE_SCANNING -> {
                binding.statusText.text = "Side Scan — Turn Sideways"
                binding.orientationPromptText.text = "Turn sideways to camera at ~2.0m or tap 'Capture Scan Now'."
                binding.stabilityProgressBar.progress = 0
                binding.previewView.post {
                    startCamera()
                }
            }
            AppScanState.ANALYZING -> {
                performAnalysisInBackground()
            }
            AppScanState.RESULTS -> {
                // UI populated after analysis finishes
            }
        }
    }

    // ----------------------------------------------------
    // POSE & SCAN ANALYSIS LOOP
    // ----------------------------------------------------

    private fun onPoseDetected(result: PoseLandmarkerResult) {
        latestPoseResult = result

        if (currentState != AppScanState.FRONT_SCANNING && currentState != AppScanState.SIDE_SCANNING) {
            return
        }

        val landmarks = result.landmarks().firstOrNull() ?: run {
            runOnUiThread {
                binding.shoulderOverlay.clear()
                binding.statusText.text = "No body detected"
                cancelCountdown()
            }
            return
        }

        val detectedOrientation = orientationDetector.detectOrientation(result)
        val stability = orientationDetector.checkStability(result)
        val distanceStatus = orientationDetector.checkDistanceStatus(result)

        runOnUiThread {
            if (landmarks.size > 24) {
                val leftShoulder = landmarks[11]
                val rightShoulder = landmarks[12]
                val leftHip = landmarks[23]
                val rightHip = landmarks[24]

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
                    isAligned = (stability > 0.5f),
                    distanceStatus = distanceStatus
                )
            }

            binding.stabilityProgressBar.progress = (stability * 100).toInt()

            if (distanceStatus == DistanceStatus.TOO_CLOSE) {
                binding.statusText.text = "Step Back (Too Close)"
                cancelCountdown()
                return@runOnUiThread
            } else if (distanceStatus == DistanceStatus.TOO_FAR) {
                binding.statusText.text = "Step Closer (Too Far)"
                cancelCountdown()
                return@runOnUiThread
            }

            when (currentState) {
                AppScanState.FRONT_SCANNING -> {
                    if (detectedOrientation == DetectedOrientation.FRONT) {
                        if (stability >= 0.65f) {
                            binding.statusText.text = "Front Position Held — Capturing..."
                            startCountdownTimer {
                                captureFrontScan()
                            }
                        } else {
                            binding.statusText.text = "Front Detected — Hold Still..."
                            cancelCountdown()
                        }
                    } else {
                        binding.statusText.text = "Please face straight toward the camera"
                        cancelCountdown()
                    }
                }
                AppScanState.SIDE_SCANNING -> {
                    if (detectedOrientation == DetectedOrientation.SIDE) {
                        if (stability >= 0.50f) {
                            binding.statusText.text = "Side Position Held — Capturing..."
                            startCountdownTimer {
                                captureSideScan()
                            }
                        } else {
                            binding.statusText.text = "Side Detected — Hold Still..."
                            cancelCountdown()
                        }
                    } else {
                        binding.statusText.text = "Please turn sideways to the camera"
                        cancelCountdown()
                    }
                }
                else -> {}
            }
        }
    }

    private fun startCountdownTimer(onFinish: () -> Unit) {
        if (isCountingDown) return
        isCountingDown = true

        binding.countdownText.visibility = View.VISIBLE
        binding.countdownText.text = "2"

        countdownTimer?.cancel()
        countdownTimer = object : CountDownTimer(2000, 1000) {
            override fun onTick(millisUntilFinished: Long) {
                val secondsLeft = (millisUntilFinished / 1000) + 1
                binding.countdownText.text = secondsLeft.toString()
            }

            override fun onFinish() {
                binding.countdownText.visibility = View.GONE
                isCountingDown = false
                onFinish()
            }
        }.start()
    }

    private fun cancelCountdown() {
        if (isCountingDown) {
            countdownTimer?.cancel()
            isCountingDown = false
            binding.countdownText.visibility = View.GONE
        }
    }

    // ----------------------------------------------------
    // AUTOMATIC SCAN CAPTURES
    // ----------------------------------------------------

    private fun captureFrontScan() {
        val bitmap = latestFrameBitmap
        val landmarks = latestPoseResult?.landmarks()?.firstOrNull()
        val mask = latestPersonMask

        if (bitmap == null || landmarks == null || mask == null) {
            Toast.makeText(this, "Capture failed — retrying", Toast.LENGTH_SHORT).show()
            return
        }

        frontCapturedBitmap = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)
        frontCapturedLandmarks = landmarks
        frontCapturedMask = mask

        Toast.makeText(this, "Front scan captured!", Toast.LENGTH_SHORT).show()
        orientationDetector.resetHistory()
        updateUiForState(AppScanState.SIDE_SCANNING)
    }

    private fun captureSideScan() {
        val bitmap = latestFrameBitmap
        val landmarks = latestPoseResult?.landmarks()?.firstOrNull()
        val mask = latestPersonMask

        if (bitmap == null || landmarks == null || mask == null) {
            Toast.makeText(this, "Capture failed — retrying", Toast.LENGTH_SHORT).show()
            return
        }

        sideCapturedBitmap = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)
        sideCapturedLandmarks = landmarks
        sideCapturedMask = mask

        Toast.makeText(this, "Side scan captured!", Toast.LENGTH_SHORT).show()
        updateUiForState(AppScanState.ANALYZING)
    }

    // ----------------------------------------------------
    // BACKGROUND IMAGE ANALYSIS & RESULTS
    // ----------------------------------------------------

    private fun performAnalysisInBackground() {
        val userInfo = currentUserInfo
        val frontBmp = frontCapturedBitmap
        val frontLms = frontCapturedLandmarks
        val frontMsk = frontCapturedMask

        val sideBmp = sideCapturedBitmap
        val sideLms = sideCapturedLandmarks
        val sideMsk = sideCapturedMask

        if (userInfo == null || frontBmp == null || frontLms == null || frontMsk == null || sideBmp == null || sideLms == null || sideMsk == null) {
            Toast.makeText(this, "Analysis failed: missing scan data", Toast.LENGTH_LONG).show()
            updateUiForState(AppScanState.USER_INPUT_FORM)
            return
        }

        cameraExecutor.execute {
            try {
                val result = measurementEngine.calculateMeasurements(
                    userInfo = userInfo,
                    frontBitmap = frontBmp,
                    frontLandmarks = frontLms,
                    frontMask = frontMsk,
                    sideBitmap = sideBmp,
                    sideLandmarks = sideLms,
                    sideMask = sideMsk
                )

                runOnUiThread {
                    displayFinalResults(userInfo, result)
                    updateUiForState(AppScanState.RESULTS)
                }

            } catch (e: Exception) {
                Log.e("FitLensAnalysis", "Error calculating measurements", e)
                runOnUiThread {
                    Toast.makeText(this, "Analysis error: ${e.message}", Toast.LENGTH_LONG).show()
                    updateUiForState(AppScanState.USER_INPUT_FORM)
                }
            }
        }
    }

    private fun displayFinalResults(userInfo: UserInfo, result: MeasurementResult) {
        val swInches = result.shoulderWidthCm / 2.54
        val cwInches = result.chestWidthCm / 2.54
        val cdInches = result.chestDepthCm / 2.54
        val ccInches = result.estimatedChestCircumferenceCm / 2.54

        binding.tvResShoulderWidth.text = String.format(Locale.US, "%.1f cm (%.1f in)", result.shoulderWidthCm, swInches)
        binding.tvResChestWidth.text = String.format(Locale.US, "%.1f cm (%.1f in)", result.chestWidthCm, cwInches)
        binding.tvResChestDepth.text = String.format(Locale.US, "%.1f cm (%.1f in)", result.chestDepthCm, cdInches)
        binding.tvResChestCircumference.text = String.format(Locale.US, "%.1f cm (%.1f in)", result.estimatedChestCircumferenceCm, ccInches)

        // Known Chest Reference Validation Comparison
        if (userInfo.knownChestCircumferenceCm != null && result.knownChestDifferenceCm != null) {
            binding.cardValidation.visibility = View.VISIBLE
            val knownCm = userInfo.knownChestCircumferenceCm
            val diffCm = result.knownChestDifferenceCm
            val pctDiff = (diffCm / knownCm) * 100.0

            val sign = if (diffCm >= 0) "+" else ""
            binding.tvResChestValidation.text = String.format(
                Locale.US,
                "Calculated: %.1f cm | User Reference: %.1f cm | Variance: %s%.1f cm (%s%.1f%%)",
                result.estimatedChestCircumferenceCm, knownCm, sign, diffCm, sign, pctDiff
            )
        } else {
            binding.cardValidation.visibility = View.GONE
        }

        // User Info Summary Card
        val fitText = userInfo.preferredFit.name.lowercase().replaceFirstChar { it.uppercase() }
        val shirtText = userInfo.usualShirtSize ?: "N/A"
        binding.tvUserInfoSummary.text = String.format(
            Locale.US,
            "Gender: %s | Age: %d yrs | Height: %.0f cm | Weight: %.0f kg\nPreferred Fit: %s | Usual Shirt Size: %s",
            userInfo.gender.name, userInfo.age, userInfo.heightCm, userInfo.weightKg, fitText, shirtText
        )
    }

    // ----------------------------------------------------
    // INITIALIZATION & PROCESS SILHOUETTE
    // ----------------------------------------------------

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
        }
    }

    private fun initImageSegmenter() {
        try {
            imageSegmenter = ImageSegmenterHelper(this)
        } catch (e: Exception) {
            Log.e("FitLensSegmentation", "Image Segmenter initialization failed", e)
        }
    }

    private fun processSilhouette(bitmap: Bitmap) {
        try {
            val segmenter = imageSegmenter ?: return
            val personMask = segmenter.segmentImage(bitmap)
            latestPersonMask = personMask

            val maskBitmap = SilhouetteProcessor.createBinaryMask(personMask, threshold = 0.5f)

            runOnUiThread {
                if (!isFinishing && !isDestroyed) {
                    binding.segmentationPreview.setImageBitmap(maskBitmap)
                }
            }
        } catch (e: Exception) {
            Log.e("FitLensSegmentation", "Segmentation failed", e)
        }
    }

    // ----------------------------------------------------
    // CAMERA SETUP & FRAME ANALYSIS
    // ----------------------------------------------------

    private fun checkCameraPermission() {
        val permissionGranted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (permissionGranted) {
            if (currentState == AppScanState.FRONT_SCANNING || currentState == AppScanState.SIDE_SCANNING) {
                startCamera()
            }
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

                if (!cameraProvider.hasCamera(cameraSelector)) {
                    binding.statusText.text = "Front camera unavailable"
                    return@addListener
                }

                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.surfaceProvider = binding.previewView.surfaceProvider
                    }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    try {
                        val bitmap = imageProxy.toBitmap()
                        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

                        val matrix = Matrix().apply {
                            postRotate(rotationDegrees.toFloat())
                        }

                        val rotatedBitmap = Bitmap.createBitmap(
                            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
                        )

                        processedImageWidth = rotatedBitmap.width
                        processedImageHeight = rotatedBitmap.height
                        latestFrameBitmap = rotatedBitmap

                        segmentationFrameCounter++
                        if (segmentationFrameCounter % 2 == 0) {
                            processSilhouette(rotatedBitmap)
                        }

                        val timestampMs = SystemClock.uptimeMillis()
                        poseDetector?.detect(rotatedBitmap, timestampMs)

                    } catch (e: Exception) {
                        Log.e("FitLensCamera", "Frame processing failed", e)
                    } finally {
                        imageProxy.close()
                    }
                }

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )

            } catch (e: Exception) {
                Log.e("FitLensCamera", "Camera setup failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        cancelCountdown()
        poseDetector?.close()
        poseDetector = null

        imageSegmenter?.close()
        imageSegmenter = null

        if (::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
        }

        super.onDestroy()
    }
}
