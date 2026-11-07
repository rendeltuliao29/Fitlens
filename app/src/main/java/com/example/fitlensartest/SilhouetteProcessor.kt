package com.example.fitlensartest

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.abs

data class PixelSpan(val left: Int, val right: Int) {
    val width: Int get() = right - left + 1
}

object SilhouetteProcessor {

    // Convert segmentation confidence mask into a clear, high-contrast silhouette
    fun createBinaryMask(
        mask: PersonMask,
        threshold: Float = 0.4f
    ): Bitmap {

        val pixels = IntArray(mask.width * mask.height)
        val personColor = Color.parseColor("#34D399") // Vibrant Emerald Green
        val bgColor = Color.parseColor("#FF111827")     // Dark slate background

        for (i in mask.confidence.indices) {
            pixels[i] = if (mask.confidence[i] >= threshold) {
                personColor
            } else {
                bgColor
            }
        }

        return Bitmap.createBitmap(
            pixels,
            mask.width,
            mask.height,
            Bitmap.Config.ARGB_8888
        )
    }

    // Find the person's left and right silhouette boundaries at a selected height,
    // with optional minX and maxX bounds to exclude arms.
    fun findTorsoSpan(
        mask: PersonMask,
        y: Int,
        centerX: Int,
        minX: Int = 0,
        maxX: Int = mask.width - 1,
        threshold: Float = 0.5f
    ): PixelSpan? {

        if (y !in 0 until mask.height) return null

        val startX = minX.coerceIn(0, mask.width - 1)
        val endX = maxX.coerceIn(startX, mask.width - 1)

        val runs = mutableListOf<PixelSpan>()
        var start = -1

        for (x in startX..endX) {
            val index = y * mask.width + x
            val isPerson = mask.confidence[index] >= threshold

            if (isPerson && start == -1) {
                start = x
            }

            if (!isPerson && start != -1) {
                runs.add(PixelSpan(start, x - 1))
                start = -1
            }
        }

        if (start != -1) {
            runs.add(PixelSpan(start, endX))
        }

        if (runs.isEmpty()) return null

        // Select the torso run that contains or is closest to the body center X
        return runs.minByOrNull { span ->
            if (centerX in span.left..span.right) {
                0
            } else {
                val runCenter = (span.left + span.right) / 2
                abs(runCenter - centerX)
            }
        }
    }
}
