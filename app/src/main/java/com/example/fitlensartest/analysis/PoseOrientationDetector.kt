package com.example.fitlensartest.analysis

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.sqrt

enum class DetectedOrientation {
    FRONT,
    SIDE,
    UNKNOWN
}

enum class DistanceStatus {
    TOO_CLOSE,
    TOO_FAR,
    OK,
    UNKNOWN
}

class PoseOrientationDetector {

    private val landmarkHistory = ArrayDeque<List<NormalizedLandmark>>()
    private val maxHistorySize = 5

    fun resetHistory() {
        landmarkHistory.clear()
    }

    // Estimate user distance from camera in meters
    fun estimateDistanceMeters(result: PoseLandmarkerResult): Double {
        val worldLandmarks = result.worldLandmarks().firstOrNull() ?: return 0.0
        if (worldLandmarks.size <= 24) return 0.0

        // MediaPipe worldLandmarks z-coordinate of hip center gives distance in meters
        val leftHipZ = worldLandmarks[23].z().toDouble()
        val rightHipZ = worldLandmarks[24].z().toDouble()
        val avgHipZ = abs((leftHipZ + rightHipZ) / 2.0)

        // Mid-torso depth offset
        val midShoulderZ = abs((worldLandmarks[11].z().toDouble() + worldLandmarks[12].z().toDouble()) / 2.0)
        val estimatedDistanceMeters = (avgHipZ + midShoulderZ) / 2.0 + 1.25 // Base distance offset

        return estimatedDistanceMeters
    }

    fun checkDistanceStatus(result: PoseLandmarkerResult): DistanceStatus {
        val landmarks = result.landmarks().firstOrNull() ?: return DistanceStatus.UNKNOWN
        if (landmarks.size <= 24) return DistanceStatus.UNKNOWN

        // Check normalized height span of shoulder-to-ankle
        val shoulderY = (landmarks[11].y() + landmarks[12].y()) / 2f
        val ankleY = (landmarks[27].y() + landmarks[28].y()) / 2f
        val torsoAnkleSpan = abs(ankleY - shoulderY)

        // Ideal distance at 2.0m fills ~0.55 to 0.75 of vertical screen height
        return when {
            torsoAnkleSpan > 0.82f -> DistanceStatus.TOO_CLOSE
            torsoAnkleSpan < 0.42f -> DistanceStatus.TOO_FAR
            else -> DistanceStatus.OK
        }
    }

    fun detectOrientation(result: PoseLandmarkerResult): DetectedOrientation {
        val landmarks = result.landmarks().firstOrNull() ?: return DetectedOrientation.UNKNOWN
        val worldLandmarks = result.worldLandmarks().firstOrNull() ?: return DetectedOrientation.UNKNOWN

        if (landmarks.size <= 24 || worldLandmarks.size <= 24) {
            return DetectedOrientation.UNKNOWN
        }

        val leftShoulderVis = landmarks[11].visibility().orElse(1.0f)
        val rightShoulderVis = landmarks[12].visibility().orElse(1.0f)

        if (leftShoulderVis < 0.35f && rightShoulderVis < 0.35f) {
            return DetectedOrientation.UNKNOWN
        }

        // Image space 2D distance between shoulders (normalized 0..1)
        val dx2D = abs(landmarks[11].x() - landmarks[12].x())

        // 3D World space shoulder z-depth difference (meters)
        val dz3D = abs(worldLandmarks[11].z() - worldLandmarks[12].z())

        // Front Facing: Both shoulders visible (>0.5), broad 2D span (>0.15) and small z-depth (<0.14m)
        if (leftShoulderVis > 0.5f && rightShoulderVis > 0.5f && dx2D > 0.15f && dz3D < 0.14f) {
            return DetectedOrientation.FRONT
        }

        // Side Facing: Narrow 2D shoulder span (<0.18) OR significant z-depth difference (>0.10m)
        val isNarrow2D = dx2D < 0.18f
        val isDeep3D = dz3D > 0.10f
        val isOneShoulderOccluded = (leftShoulderVis < 0.5f && rightShoulderVis > 0.5f) || (rightShoulderVis < 0.5f && leftShoulderVis > 0.5f)

        if (isNarrow2D || isDeep3D || isOneShoulderOccluded) {
            return DetectedOrientation.SIDE
        }

        return DetectedOrientation.UNKNOWN
    }

    fun checkStability(result: PoseLandmarkerResult): Float {
        val landmarks = result.landmarks().firstOrNull() ?: run {
            resetHistory()
            return 0f
        }

        if (landmarks.size <= 24) {
            resetHistory()
            return 0f
        }

        landmarkHistory.addLast(landmarks)
        if (landmarkHistory.size > maxHistorySize) {
            landmarkHistory.removeFirst()
        }

        if (landmarkHistory.size < 3) {
            return 0.5f
        }

        val firstFrame = landmarkHistory.first()
        val latestFrame = landmarkHistory.last()

        val indicesToCheck = intArrayOf(11, 12, 23, 24)
        var maxShift = 0f

        for (idx in indicesToCheck) {
            val p1 = firstFrame[idx]
            val p2 = latestFrame[idx]
            val shift = sqrt(((p1.x() - p2.x()) * (p1.x() - p2.x()) + (p1.y() - p2.y()) * (p1.y() - p2.y())).toDouble()).toFloat()
            if (shift > maxShift) {
                maxShift = shift
            }
        }

        val maxAllowedShift = 0.05f
        val stabilityFactor = (1f - (maxShift / maxAllowedShift)).coerceIn(0f, 1f)

        return stabilityFactor
    }
}
