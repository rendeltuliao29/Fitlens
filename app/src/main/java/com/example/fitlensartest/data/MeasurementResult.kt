package com.example.fitlensartest.data

data class MeasurementResult(
    val shoulderWidthCm: Double,
    val chestWidthCm: Double,
    val chestDepthCm: Double,
    val estimatedChestCircumferenceCm: Double,
    val knownChestCircumferenceCm: Double? = null,
    val knownChestDifferenceCm: Double? = null
)
