package com.example.fitlensartest.analysis

import android.graphics.Bitmap
import com.example.fitlensartest.PersonMask
import com.example.fitlensartest.SilhouetteProcessor
import com.example.fitlensartest.data.MeasurementResult
import com.example.fitlensartest.data.UserInfo
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.abs
import kotlin.math.sqrt

class MeasurementEngine {

    fun calculateMeasurements(
        userInfo: UserInfo,
        frontBitmap: Bitmap,
        frontLandmarks: List<NormalizedLandmark>,
        frontMask: PersonMask,
        sideBitmap: Bitmap,
        sideLandmarks: List<NormalizedLandmark>,
        sideMask: PersonMask
    ): MeasurementResult {

        val userHeightCm = userInfo.heightCm

        // 1. Calculate Normalized Height Scale (cm per 1.0 normalized Y unit in image)
        val frontBodyHeightNorm = calculateNormalizedBodyHeight(frontLandmarks)
        val cmPerNormY = userHeightCm / frontBodyHeightNorm.coerceIn(0.40f, 0.95f)

        // 2. Front Image Calculations: Shoulder Width and Chest Width
        val frontShoulderY = (frontLandmarks[11].y() + frontLandmarks[12].y()) / 2f
        val frontShoulderX = (frontLandmarks[11].x() + frontLandmarks[12].x()) / 2f

        val frontHipY = (frontLandmarks[23].y() + frontLandmarks[24].y()) / 2f
        val frontChestY = frontShoulderY + (frontHipY - frontShoulderY) * 0.28f

        // Mask pixel coordinates
        val shoulderRowPx = (frontShoulderY * frontMask.height).toInt().coerceIn(0, frontMask.height - 1)
        val chestRowPxFront = (frontChestY * frontMask.height).toInt().coerceIn(0, frontMask.height - 1)
        val centerPxXFront = (frontShoulderX * frontMask.width).toInt().coerceIn(0, frontMask.width - 1)

        // Landmark-based acromion span in normalized units
        val acromionSpanNorm = abs(frontLandmarks[11].x() - frontLandmarks[12].x())

        // --- SHOULDER WIDTH (ANTERIOR FRONT MEASUREMENT) ---
        // Search bounds restricted around shoulders to isolate deltoids (+/- 80% of acromion span)
        val minPxShoulder = ((frontShoulderX - acromionSpanNorm * 0.8f).coerceAtLeast(0f) * frontMask.width).toInt()
        val maxPxShoulder = ((frontShoulderX + acromionSpanNorm * 0.8f).coerceAtMost(1f) * frontMask.width).toInt()

        val deltoidSpanMaskPx = SilhouetteProcessor.findTorsoSpan(
            mask = frontMask,
            y = shoulderRowPx,
            centerX = centerPxXFront,
            minX = minPxShoulder,
            maxX = maxPxShoulder
        )?.width ?: (acromionSpanNorm * 1.22f * frontMask.width).toInt()

        val deltoidSpanNorm = deltoidSpanMaskPx.toFloat() / frontMask.width
        val deltoidWidthCm = deltoidSpanNorm * cmPerNormY

        // Convert outer deltoid silhouette width to anterior front shoulder width (0.91 factor)
        val rawAnteriorShoulderWidthCm = deltoidWidthCm * 0.91

        // Anatomical clamp for Front Shoulder Width (14.5 in to 19.5 in for adults)
        val minShoulderCm = userHeightCm * 0.205
        val maxShoulderCm = userHeightCm * 0.260
        val shoulderWidthCm = rawAnteriorShoulderWidthCm.coerceIn(minShoulderCm, maxShoulderCm)

        // --- CHEST WIDTH (FRONT) ---
        // Bound search tightly between inner shoulders to exclude arms completely
        val minPxChestFront = ((frontShoulderX - acromionSpanNorm * 0.55f).coerceAtLeast(0f) * frontMask.width).toInt()
        val maxPxChestFront = ((frontShoulderX + acromionSpanNorm * 0.55f).coerceAtMost(1f) * frontMask.width).toInt()

        val chestSpanMaskPxFront = SilhouetteProcessor.findTorsoSpan(
            mask = frontMask,
            y = chestRowPxFront,
            centerX = centerPxXFront,
            minX = minPxChestFront,
            maxX = maxPxChestFront
        )?.width ?: (acromionSpanNorm * 0.88f * frontMask.width).toInt()

        val chestSpanNormFront = chestSpanMaskPxFront.toFloat() / frontMask.width
        val rawChestWidthCm = chestSpanNormFront * cmPerNormY * 0.96 // Anatomical chest boundary factor

        // Anatomical clamp for Front Chest Width (11.5 in to 16.0 in)
        val minChestWidthCm = userHeightCm * 0.165
        val maxChestWidthCm = userHeightCm * 0.215
        val chestWidthCm = rawChestWidthCm.coerceIn(minChestWidthCm, maxChestWidthCm)

        // --- CHEST DEPTH (SIDE) ---
        val sideShoulderY = (sideLandmarks[11].y() + sideLandmarks[12].y()) / 2f
        val sideHipY = (sideLandmarks[23].y() + sideLandmarks[24].y()) / 2f
        val sideChestY = sideShoulderY + (sideHipY - sideShoulderY) * 0.28f

        val sideCenterX = (sideLandmarks[11].x() + sideLandmarks[12].x()) / 2f

        val chestRowPxSide = (sideChestY * sideMask.height).toInt().coerceIn(0, sideMask.height - 1)
        val centerPxXSide = (sideCenterX * sideMask.width).toInt().coerceIn(0, sideMask.width - 1)

        // Bound side search around torso (+/- 15% image width)
        val minPxChestSide = ((sideCenterX - 0.15f).coerceAtLeast(0f) * sideMask.width).toInt()
        val maxPxChestSide = ((sideCenterX + 0.15f).coerceAtMost(1f) * sideMask.width).toInt()

        val rawSideProfileSpanPx = SilhouetteProcessor.findTorsoSpan(
            mask = sideMask,
            y = chestRowPxSide,
            centerX = centerPxXSide,
            minX = minPxChestSide,
            maxX = maxPxChestSide
        )?.width ?: (chestSpanMaskPxFront * 0.70f).toInt()

        // Apply Arm-Occlusion Correction Factor (0.70) to isolate true thoracic cage depth from arm
        val trueChestDepthSpanNorm = (rawSideProfileSpanPx.toFloat() / sideMask.width) * 0.70f
        val rawChestDepthCm = trueChestDepthSpanNorm * cmPerNormY

        // Anatomical clamp for Side Chest Depth (7.0 in to 10.0 in)
        val minChestDepthCm = userHeightCm * 0.098
        val maxChestDepthCm = userHeightCm * 0.138
        val chestDepthCm = rawChestDepthCm.coerceIn(minChestDepthCm, maxChestDepthCm)

        // --- CHEST CIRCUMFERENCE (SNUG TAPE-MEASURE RAMANUJAN MODEL) ---
        val a = chestWidthCm / 2.0
        val b = chestDepthCm / 2.0
        val h = ((a - b) * (a - b)) / ((a + b) * (a + b))

        val ramanujanPerimeter = Math.PI * (a + b) * (1.0 + (3.0 * h) / (10.0 + sqrt(4.0 - 3.0 * h)))

        // Snug Tape Path Factor (0.89x) converts outer bounding ellipse perimeter to tape tension path
        val rawSnugChestCircumferenceCm = ramanujanPerimeter * 0.89

        // Anatomical clamp for Chest Circumference (30 in to 42 in for standard adult heights)
        val minChestCircCm = userHeightCm * 0.42
        val maxChestCircCm = userHeightCm * 0.54
        val estimatedChestCircumferenceCm = rawSnugChestCircumferenceCm.coerceIn(minChestCircCm, maxChestCircCm)

        // Optional Reference Comparison
        val knownChest = userInfo.knownChestCircumferenceCm
        val knownDiff = knownChest?.let { estimatedChestCircumferenceCm - it }

        return MeasurementResult(
            shoulderWidthCm = shoulderWidthCm,
            chestWidthCm = chestWidthCm,
            chestDepthCm = chestDepthCm,
            estimatedChestCircumferenceCm = estimatedChestCircumferenceCm,
            knownChestCircumferenceCm = knownChest,
            knownChestDifferenceCm = knownDiff
        )
    }

    private fun calculateNormalizedBodyHeight(landmarks: List<NormalizedLandmark>): Float {
        val noseY = landmarks.getOrNull(0)?.y() ?: 0.15f
        val eyeY = landmarks.getOrNull(2)?.y() ?: noseY
        val headTopY = (noseY - abs(noseY - eyeY) * 2.5f).coerceIn(0f, 1f)

        val leftAnkleY = landmarks.getOrNull(27)?.y() ?: 0.88f
        val rightAnkleY = landmarks.getOrNull(28)?.y() ?: 0.88f
        val feetBottomY = maxOf(leftAnkleY, rightAnkleY).coerceIn(0f, 1f)

        val bodyHeightNorm = abs(feetBottomY - headTopY)
        return if (bodyHeightNorm > 0.35f) bodyHeightNorm else 0.70f
    }
}
